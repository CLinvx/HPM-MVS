#include "HPM.h"

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

HPM::HPM() {}

HPM::~HPM()
{
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }

    if (params.planar_prior) {
        delete[] prior_planes_host;
        delete[] plane_masks_host;

        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }
}

Camera ReadCamera_short(const std::string& cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    file >> camera.depth_min >> camera.depth_max;
    return camera;
}


Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

float3 Get3DPointonRefCam_factor(const int x, const int y, const float depth, const Camera camera, float factor)
{
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2] * factor) / (camera.K[0] * factor);
    pointX.y = depth * (y - camera.K[5] * factor) / (camera.K[4] * factor);
    pointX.z = depth;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)depth.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

int readDepthDmb(const std::string file_path, cv::Mat_<float>& depth)
{
    FILE* inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float* data;
    data = (float*)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    depth = cv::Mat(h, w, CV_32F);

    memcpy(depth.data, data, sizeof(float) * dataSize);
    free(data);

    fclose(inimage);
    return 0;
}

int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f>& normal)
{
    FILE* inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type, sizeof(int32_t), 1, inimage);
    fread(&h, sizeof(int32_t), 1, inimage);
    fread(&w, sizeof(int32_t), 1, inimage);
    fread(&nb, sizeof(int32_t), 1, inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h * w * nb;

    float* data;
    data = (float*)malloc(sizeof(float) * dataSize);
    fread(data, sizeof(float), dataSize, inimage);

    normal = cv::Mat(h, w, CV_32FC3);


    memcpy(normal.data, data, sizeof(float) * dataSize);
    free(data);


    fclose(inimage);
    return 0;
}


int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)normal.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(int i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[3] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

void HPM::SetGeomConsistencyParams(bool multi_geometry=false)
{
    params.geom_consistency = true;
    params.max_iterations = 2;
    if (multi_geometry) {
        params.multi_geometry = true;
    }
}

void HPM::SetPlanarPriorParams()
{
    params.planar_prior = true;
}

void HPM::InuputInitialization(const std::string &dense_folder, const Problem &problem)
{
    images.clear();
    cameras.clear();

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << "_cam.txt";
    //Camera camera = ReadCamera(cam_path.str());
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);
    image_uint.release();
    image_float.release();
    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << "_cam.txt";
        //Camera camera = ReadCamera(cam_path.str());
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width = image_float.cols;
        cameras.push_back(camera);
        image_uint.release();
        image_float.release();
    }

    // Scale cameras and images
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].cols <= params.max_image_size && images[i].rows <= params.max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(params.max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(params.max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
        scaled_image_float.release();
    }

    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    std::cout << "depthe range: " << params.depth_min << " " << params.depth_max << std::endl;
    params.num_images = (int)images.size();
    std::cout << "num images: " << params.num_images << std::endl;
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();

        std::stringstream result_path;
        result_path << dense_folder << "/HPM" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        ref_depth.release();

        size_t num_src_images = problem.src_image_ids.size();
        for (size_t i = 0; i < num_src_images; ++i) {
            std::stringstream result_path;
            result_path << dense_folder << "/HPM" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            cv::Mat_<float> depth;
            readDepthDmb(depth_path, depth);
            depths.push_back(depth);
            depth.release();
        }
    }
}

void HPM::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem)
{
    num_images = (int)images.size();

    for (int i = 0; i < num_images; ++i) {
        int rows = images[i].rows;
        int cols = images[i].cols;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width)); // Updated by Qingshan 2020-01-15

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        std::stringstream result_path;
        result_path << dense_folder << "/HPM" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center] = ref_cost(row, col);
            }
        }

        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
        ref_depth.release();
        ref_normal.release();
        ref_cost.release();
    }
}

void HPM::CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks)
{
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
        }
    }

    cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

int HPM::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int HPM::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat HPM::GetReferenceImage()
{
    return images[0];
}

float4 HPM::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float HPM::GetCost(const int index)
{
    return costs_host[index];
}

float HPM::GetMinDepth()
{
    return params.depth_min;
}

float HPM::GetMaxDepth()
{
    return params.depth_max;
}

void HPM::GetSupportPoints(std::vector<cv::Point>& support2DPoints)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    int center = r * width + c;
                    if (GetCost(center) < 2.0f && min_cost > GetCost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost = GetCost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

std::vector<Triangle> HPM::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points)
{
    if (points.empty()) {
        return std::vector<Triangle>();
    }

    std::vector<Triangle> results;

    std::vector<cv::Vec6f> temp_results;
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    subdiv2d.getTriangleList(temp_results);

    for (const auto temp_vec : temp_results) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

float4 HPM::GetPriorPlaneParams_factor(const Triangle triangle, const cv::Mat_<float> depths, float factor)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = Get3DPointonRefCam_factor(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0], factor);
    float3 ptX2 = Get3DPointonRefCam_factor(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0], factor);
    float3 ptX3 = Get3DPointonRefCam_factor(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0], factor);

    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 = make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}

float HPM::GetDepthFromPlaneParam_factor(const float4 plane_hypothesis, const int x, const int y, float factor)
{
    return -plane_hypothesis.w * (cameras[0].K[0] * factor) / ((x - cameras[0].K[2] * factor) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5] * factor) * plane_hypothesis.y + cameras[0].K[0] * factor * plane_hypothesis.z);
}

int HPM::ComputingMultiScaleSettings(int width, int height) {
    int size_bound = 1000;
    int max_size = std::max(width, height);
    int k = 0;
    while (max_size > size_bound) {
        max_size /= 2;
        k++;
    }
    return k;
}

void HPM::WritePointCloud(PointCloudPtr& cloud, std::vector<cv::Point>points) {
    pcl::PointXY tmp;
    for (const auto point : points) {
        tmp.x = point.x;
        tmp.y = point.y;
        cloud->push_back(tmp);
    }
}


float4 HPM::TransformNormal(float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = cameras[0].R[0] * plane_hypothesis.x + cameras[0].R[3] * plane_hypothesis.y + cameras[0].R[6] * plane_hypothesis.z;
    transformed_normal.y = cameras[0].R[1] * plane_hypothesis.x + cameras[0].R[4] * plane_hypothesis.y + cameras[0].R[7] * plane_hypothesis.z;
    transformed_normal.z = cameras[0].R[2] * plane_hypothesis.x + cameras[0].R[5] * plane_hypothesis.y + cameras[0].R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

float4 HPM::TransformNormal2RefCam(float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = cameras[0].R[0] * plane_hypothesis.x + cameras[0].R[1] * plane_hypothesis.y + cameras[0].R[2] * plane_hypothesis.z;
    transformed_normal.y = cameras[0].R[3] * plane_hypothesis.x + cameras[0].R[4] * plane_hypothesis.y + cameras[0].R[5] * plane_hypothesis.z;
    transformed_normal.z = cameras[0].R[6] * plane_hypothesis.x + cameras[0].R[7] * plane_hypothesis.y + cameras[0].R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

float HPM::GetDistance2Origin_factor(const int2 p, const float depth, const float4 normal, float factor)
{
    float X[3];
    X[0] = depth * (p.x - cameras[0].K[2] * 1) / (cameras[0].K[0] * 1);
    X[1] = depth * (p.y - cameras[0].K[5] * 1) / (cameras[0].K[4] * 1);
    X[2] = depth;
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

void HPM::ReloadPlanarPriorInitialization(const cv::Mat_<float>& masks, float4* prior_plane_parameters)
{
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));


    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = prior_plane_parameters[center];
            }
        }
    }
    cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

float Distance(int x1, int y1, int x2, int y2) {
    float tmp = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    return sqrtf(tmp);
}

bool JudgeNotOnline(cv::Point point1, cv::Point point2, cv::Point point3) {
    float line1 = Distance(point1.x, point1.y, point2.x, point2.y);
    float line2 = Distance(point1.x, point1.y, point3.x, point3.y);
    float line3 = Distance(point3.x, point3.y, point2.x, point2.y);
    float area = 0.5 * (line1 + line2 + line3);
    if (area * (area - line1) * (area - line2) * (area - line3) > 0) {
        return true;
    }
    return false;
}

void HPM::DepthsPredictSupplement(PointCloudPtr cloud, cv::Mat_<float>& depth, cv::Mat_<cv::Vec3f>& normal_reliable, float4* plane_parameters, cv::Mat_<float>& mask, float factor) {
    int height = depth.rows;
    int width = depth.cols;
    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    kdtree.setInputCloud(cloud);
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
#pragma omp parallel for
        for (int j = 0; j < width; j++) {
            if (mask(i, j) > 0) {
                continue;
            }
            pcl::PointXY search_point;
            float distance;
            std::vector<int> pointSearch;
            std::vector<float> pointDistance;

            search_point.x = j;
            search_point.y = i;
            pointSearch.clear();
            pointDistance.clear();


            if (kdtree.nearestKSearch(search_point, 6, pointSearch, pointDistance) > 0) {
                int vertex[12];
                for (int k = 0; k < pointSearch.size(); k++) {

                    vertex[k * 2] = cloud->points[pointSearch[k]].x;
                    vertex[2 * k + 1] = cloud->points[pointSearch[k]].y;

                }

                cv::Point tmp1(vertex[0], vertex[1]);
                cv::Point tmp2(vertex[2], vertex[3]);
                cv::Point tmp3(vertex[4], vertex[5]);
                cv::Point tmp4(vertex[6], vertex[7]);
                cv::Point tmp5(vertex[8], vertex[9]);
                cv::Point tmp6(vertex[10], vertex[11]);
                cv::Point pt1;
                cv::Point pt2;
                cv::Point pt3;

                cv::Point process_now(j, i);
                if (JudgeNotOnline(tmp1, tmp2, tmp3)) {
                    pt1 = tmp1;
                    pt2 = tmp2;
                    pt3 = tmp3;
                }
                else if (JudgeNotOnline(tmp1, tmp2, tmp4)) {
                    pt1 = tmp1;
                    pt2 = tmp2;
                    pt3 = tmp4;
                }
                else if (JudgeNotOnline(tmp1, tmp2, tmp5)) {
                    pt1 = tmp1;
                    pt2 = tmp2;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp1, tmp3, tmp4)) {
                    pt1 = tmp1;
                    pt2 = tmp3;
                    pt3 = tmp4;
                }
                else if (JudgeNotOnline(tmp1, tmp3, tmp5)) {
                    pt1 = tmp1;
                    pt2 = tmp3;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp1, tmp4, tmp5)) {
                    pt1 = tmp1;
                    pt2 = tmp4;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp2, tmp3, tmp4)) {
                    pt1 = tmp2;
                    pt2 = tmp3;
                    pt3 = tmp4;
                }
                else if (JudgeNotOnline(tmp2, tmp3, tmp5)) {
                    pt1 = tmp2;
                    pt2 = tmp3;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp2, tmp4, tmp5)) {
                    pt1 = tmp2;
                    pt2 = tmp4;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp3, tmp4, tmp5)) {
                    pt1 = tmp3;
                    pt2 = tmp4;
                    pt3 = tmp5;
                }
                else if (JudgeNotOnline(tmp1, tmp2, tmp6)) {
                    pt1 = tmp1;
                    pt2 = tmp2;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp1, tmp3, tmp6)) {
                    pt1 = tmp1;
                    pt2 = tmp3;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp1, tmp4, tmp6)) {
                    pt1 = tmp1;
                    pt2 = tmp4;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp1, tmp5, tmp6)) {
                    pt1 = tmp1;
                    pt2 = tmp5;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp2, tmp3, tmp6)) {
                    pt1 = tmp2;
                    pt2 = tmp3;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp2, tmp4, tmp6)) {
                    pt1 = tmp2;
                    pt2 = tmp4;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp2, tmp5, tmp6)) {
                    pt1 = tmp2;
                    pt2 = tmp5;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp3, tmp4, tmp6)) {
                    pt1 = tmp3;
                    pt2 = tmp4;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp3, tmp5, tmp6)) {
                    pt1 = tmp3;
                    pt2 = tmp5;
                    pt3 = tmp6;
                }
                else if (JudgeNotOnline(tmp4, tmp5, tmp6)) {
                    pt1 = tmp4;
                    pt2 = tmp5;
                    pt3 = tmp6;
                }
                else {
                    continue;
                }

                float4 n4 = GetPriorPlaneParams_factor(Triangle(pt1, pt2, pt3), depth, factor);
                int center = i * width + j;
                float d = GetDepthFromPlaneParam_factor(n4, j, i, factor);
                if (d <= GetMaxDepth() && d >= GetMinDepth()) {
                    mask(i, j) = -233.0;
                    plane_parameters[center] = n4;
                }
            }

        }
    }
}

void HPM::CudaPlanarPriorInitializationSupplement(const std::vector<float4>& PlaneParams, cv::Mat_<float>& masks, const float4* plane_hypothesis)
{
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
            else if (masks(j, i) < -200.0f) {
                prior_planes_host[center] = plane_hypothesis[center];
                masks(j, i) = 1;
            }
        }
    }

    cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

void HPM::GetSupportPointsScale1(std::vector<cv::Point>& support2DPoints, cv::Mat_<float>costs)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = std::round(GetReferenceImageWidth() * 0.5);
    const int height = std::round(GetReferenceImageHeight() * 0.5);
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {

                    if (costs(r, c) < 2.0f && min_cost > costs(r, c)) {
                        temp_point = cv::Point(c, r);
                        min_cost = costs(r, c);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

void HPM::GetSupportPointsScale0(std::vector<cv::Point>& support2DPoints, cv::Mat_<float>costs)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = std::round(GetReferenceImageWidth() * 0.25);
    const int height = std::round(GetReferenceImageHeight() * 0.25);
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    
                    if (costs(r, c) < 2.0f && min_cost > costs(r, c)) {
                        temp_point = cv::Point(c, r);
                        min_cost = costs(r, c);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

void JBUAddImageToTextureFloatGray(std::vector<cv::Mat_<float>>& imgs, cudaTextureObject_t texs[], cudaArray* cuArray[], const int& numSelViews)
{
    for (int i = 0; i < numSelViews; i++) {
        int index = i;
        int rows = imgs[index].rows;
        int cols = imgs[index].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray(cuArray[i], 0, 0, imgs[index].ptr<float>(), imgs[index].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL);
    }
    return;
}


void HPM::JointBilateralUpsampling(const cv::Mat_<float>& scaled_image_float, const cv::Mat_<float>& src_depthmap, cv::Mat_<float>& upsample_depthmap, const cv::Mat_<cv::Vec3f>& src_normal, cv::Mat_<cv::Vec3f>& upsample_normal) {
    uint32_t rows = scaled_image_float.rows;
    uint32_t cols = scaled_image_float.cols;
    int Imagescale = std::max(scaled_image_float.rows / src_depthmap.rows, scaled_image_float.cols / src_depthmap.cols);
    if (Imagescale == 1) {
        std::cout << "Image.rows = Depthmap.rows" << std::endl;
        return;
    }
    std::vector<cv::Mat_<float> > imgs(JBU_NUM);
    imgs[0] = scaled_image_float.clone();
    imgs[1] = src_depthmap.clone();

    JBU jbu;
    jbu.jp_h.height = rows;
    jbu.jp_h.width = cols;
    jbu.jp_h.s_height = src_depthmap.rows;
    jbu.jp_h.s_width = src_depthmap.cols;
    jbu.jp_h.Imagescale = Imagescale;

    JBUAddImageToTextureFloatGray(imgs, jbu.jt_h.imgs, jbu.cuArray, JBU_NUM);
    jbu.normal_origin_host = new float4[src_depthmap.rows * src_depthmap.cols];
    for (int i = 0; i < src_depthmap.rows; i++) {
        for (int j = 0; j < src_depthmap.cols; j++) {
            int center = i * src_depthmap.cols + j;
            jbu.normal_origin_host[center].x = src_normal(i, j)[0];
            jbu.normal_origin_host[center].y = src_normal(i, j)[1];
            jbu.normal_origin_host[center].z = src_normal(i, j)[2];
            jbu.normal_origin_host[center].w = src_depthmap(i, j);
            //std::cout << jbu.normal_origin_host[center].x << " " << jbu.normal_origin_host[center].y << " " << jbu.normal_origin_host[center].z << " " << jbu.normal_origin_host[center].w << std::endl;
        }
    }
    jbu.InitializeParameters(rows * cols, src_depthmap.rows * src_depthmap.cols);
    jbu.CudaRun();


    for (uint32_t i = 0; i < cols; ++i) {
        for (uint32_t j = 0; j < rows; ++j) {
            int center = i + cols * j;
            if (jbu.depth_h[center] != jbu.depth_h[center]) {
                //std::cout << "wrong!" << std::endl;
                upsample_depthmap(j, i) = src_depthmap(j / 2, i / 2);
                upsample_normal(j, i)[0] = src_normal(j / 2, i / 2)[0];
                upsample_normal(j, i)[1] = src_normal(j / 2, i / 2)[1];
                upsample_normal(j, i)[2] = src_normal(j / 2, i / 2)[2];
            }
            upsample_depthmap(j, i) = jbu.depth_h[center];
            upsample_normal(j, i)[0] = jbu.normal_h[center].x;
            upsample_normal(j, i)[1] = jbu.normal_h[center].y;
            upsample_normal(j, i)[2] = jbu.normal_h[center].z;
        }
    }

    for (int i = 0; i < JBU_NUM; i++) {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(jbu.jt_h.imgs[i]));
        CUDA_SAFE_CALL(cudaFreeArray(jbu.cuArray[i]));
    }
    //jbu.~JBU();
    jbu.ReleaseJBUCudaMemory();
    imgs[0].release();
    imgs[1].release();
    //delete(jbu.depth_h);
    //delete(jbu.normal_h);
    //free(jbu.depth_h);
    //free(jbu.normal_h);
    delete(jbu.normal_origin_host);
    cudaDeviceSynchronize();
}

void HPM::ReleasePriorCudaMemory() {
    cudaFree(prior_planes_cuda);
    cudaFree(plane_masks_cuda);
}

void HPM::ReleaseProblemCudaMemory() {
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);
    cudaFree(texture_depths_cuda);
    std::cout << "Releasing CUDA memory..." << std::endl;
}

void HPM::ReleaseProblemHostMemory() {
    //delete(plane_hypotheses_host);
    //delete(costs_host);
    images.swap(std::vector<cv::Mat>());
    cameras.swap(std::vector<Camera>());
    depths.swap(std::vector<cv::Mat>());
    std::cout << "Releasing Host memory..." << std::endl;
}

void JBU::ReleaseJBUHostMemory() {
    delete(depth_h);
    delete(normal_h);
}

void JBU::ReleaseJBUCudaMemory() {
    cudaFree(depth_d);
    cudaFree(normal_d);
    cudaFree(normal_origin_cuda);
    cudaFree(jp_d);
    cudaFree(jt_d);
}

void JBU::InitializeParameters(int n, int origin_n)
{
    depth_h = (float*)malloc(sizeof(float) * n);
    normal_h = new float4[n];


    cudaMalloc((void**)&depth_d, sizeof(float) * n);
    cudaMalloc((void**)&normal_d, sizeof(float4) * n);
    cudaMalloc((void**)&normal_origin_cuda, sizeof(float4) * origin_n);
    cudaMemcpy(normal_origin_cuda, normal_origin_host, sizeof(float4) * origin_n, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&jp_d, sizeof(JBUParameters) * 1);
    cudaMemcpy(jp_d, &jp_h, sizeof(JBUParameters) * 1, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&jt_d, sizeof(JBUTexObj) * 1);
    cudaMemcpy(jt_d, &jt_h, sizeof(JBUTexObj) * 1, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

JBU::JBU() {}

JBU::~JBU()
{
    free(depth_h);

    cudaFree(depth_d);
    cudaFree(jp_d);
    cudaFree(jt_d);
}
