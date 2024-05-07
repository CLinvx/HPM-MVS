#include "main.h"
#include "HPM.h"

void GenerateSampleList(const std::string& dense_folder, std::vector<Problem>& problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");

    problems.clear();

    std::ifstream file(cluster_list_path);

    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

void ProcessProblem(const std::string& dense_folder, const Problem& problem, bool geom_consistency, bool planar_prior, bool multi_geometrty = false)
{
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << dense_folder << "/HPM" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str());

    HPM hpm;
    if (geom_consistency) {
        hpm.SetGeomConsistencyParams(multi_geometrty);
    }
    hpm.InuputInitialization(dense_folder, problem);

    hpm.CudaSpaceInitialization(dense_folder, problem);
    hpm.RunPatchMatch();

    const int width = hpm.GetReferenceImageWidth();
    const int height = hpm.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    float4 plane_hypothesis;
    float4 tmp_n4;

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            plane_hypothesis = hpm.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = hpm.GetCost(center);
        }
    }

    if (planar_prior) {
        std::cout << "Run HPM with Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        hpm.SetPlanarPriorParams();
        int scale = hpm.ComputingMultiScaleSettings(width, height);
        PointCloudPtr pointcloud(new pcl::PointCloud<pcl::PointXY>);
        for (int i = 2 - scale; i <= 2; i++) {
            if (i == 0) {
                std::cout << "Scale 0 prior generating..." << std::endl;
                cv::Mat_<float>depths_scale0;
                cv::Mat_<float>costs_scale0;
                cv::Mat_<cv::Vec3f>normals_scale0;
                cv::resize(depths, depths_scale0, cv::Size(std::round(width * 0.25), std::round(height * 0.25)), 0, 0, cv::INTER_LINEAR);
                cv::resize(costs, costs_scale0, cv::Size(std::round(width * 0.25), std::round(height * 0.25)), 0, 0, cv::INTER_LINEAR);
                cv::resize(normals, normals_scale0, cv::Size(std::round(width * 0.25), std::round(height * 0.25)), 0, 0, cv::INTER_LINEAR);

                std::vector<cv::Point> support2DPoints_scale0;
                hpm.GetSupportPointsScale0(support2DPoints_scale0, costs_scale0);
                const cv::Rect imageRC(0, 0, std::round(width * 0.25), std::round(height * 0.25));
                const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints_scale0);
                cv::Mat_<float> mask_tri_scale0 = cv::Mat::zeros(std::round(height * 0.25), std::round(width * 0.25), CV_32FC1);
                std::vector<float4> planeParams_tri_scale0;
                planeParams_tri_scale0.clear();
                uint32_t idx = 0;
                for (const auto triangle : triangles) {
                    if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;

                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri_scale0(y, x) = idx + 1.0;
                            }
                        }
                        float4 n4 = hpm.GetPriorPlaneParams_factor(triangle, depths_scale0, 0.25);
                        planeParams_tri_scale0.push_back(n4);
                        idx++;
                    }
                }

                float4* prior_supplement_scale0 = new float4[std::round(height * 0.25) * std::round(width * 0.25)];
                pointcloud->clear();
                hpm.WritePointCloud(pointcloud, support2DPoints_scale0);
                hpm.DepthsPredictSupplement(pointcloud, depths_scale0, normals_scale0, prior_supplement_scale0, mask_tri_scale0, 0.25);
                cv::Mat_<float>priordepths_scale0 = cv::Mat::zeros(std::round(height * 0.25), std::round(width * 0.25), CV_32FC1);
                cv::Mat_<cv::Vec3f>priornormals_scale0 = cv::Mat::zeros(std::round(height * 0.25), std::round(width * 0.25), CV_32FC3);
                for (int i = 0; i < std::round(width * 0.25); ++i) {
                    for (int j = 0; j < std::round(height * 0.25); ++j) {
                        if (mask_tri_scale0(j, i) > 0) {
                            float d = hpm.GetDepthFromPlaneParam_factor(planeParams_tri_scale0[mask_tri_scale0(j, i) - 1], i, j, 0.25);

                            if (d <= hpm.GetMaxDepth() * 1.2f && d >= hpm.GetMinDepth() * 0.6f) {
                                priordepths_scale0(j, i) = d;
                                float4 tmp_n4 = hpm.TransformNormal(planeParams_tri_scale0[mask_tri_scale0(j, i) - 1]);
                                priornormals_scale0(j, i)[0] = tmp_n4.x;
                                priornormals_scale0(j, i)[1] = tmp_n4.y;
                                priornormals_scale0(j, i)[2] = tmp_n4.z;
                            }
                            else {
                                mask_tri_scale0(j, i) = 0;
                            }
                        }
                        else {
                            if (mask_tri_scale0(j, i) < -200) {
                                int width_scale0 = std::round(width * 0.25);
                                float4 tmp_n4 = prior_supplement_scale0[j * width_scale0 + i];
                                //std::cout << tmp_n4.x << tmp_n4.y << tmp_n4.z << tmp_n4.w << std::endl;
                                float d = hpm.GetDepthFromPlaneParam_factor(tmp_n4, i, j, 0.25);
                                if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
                                    priordepths_scale0(j, i) = d;
                                    float4 tmp = hpm.TransformNormal(tmp_n4);
                                    priornormals_scale0(j, i)[0] = tmp.x;
                                    priornormals_scale0(j, i)[1] = tmp.y;
                                    priornormals_scale0(j, i)[2] = tmp.z;
                                }
                                else {
                                    mask_tri_scale0(j, i) = 0;
                                }
                            }
                            else {
                                mask_tri_scale0(j, i) = 0;
                            }
                        }
                    }
                }


                std::stringstream image_path;
                image_path << dense_folder << "/images" << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
                cv::Mat_<uint8_t> image_uint;
                cv::resize(cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE), image_uint, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
                cv::Mat image_float;
                image_uint.convertTo(image_float, CV_32FC1);
                cv::Mat_<float>priordepth_upsample_scale0 = cv::Mat::zeros(height, width, CV_32FC1);
                cv::Mat_<cv::Vec3f>priornormal_upsample_scale0 = cv::Mat::zeros(height, width, CV_32FC3);
                std::cout << "Running JBU..." << std::endl;
                hpm.JointBilateralUpsampling(image_float, priordepths_scale0, priordepth_upsample_scale0, priornormals_scale0, priornormal_upsample_scale0);
                image_float.release();
                image_uint.release();
                cv::Mat_<float> mask_tri_scale0_upsample = cv::Mat::zeros(height, width, CV_32FC1);
                float4* prior_plane_parameters = new float4[height * width];
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        if (priordepth_upsample_scale0(j, i) != priordepth_upsample_scale0(j, i)) {
                            mask_tri_scale0_upsample(j, i) = 0;
                        }
                        if (priordepth_upsample_scale0(j, i) <= hpm.GetMaxDepth() && priordepth_upsample_scale0(j, i) >= hpm.GetMinDepth()) {
                            mask_tri_scale0_upsample(j, i) = 1;
                            int center = j * width + i;
                            float4 tmp_reload;
                            tmp_reload.x = priornormal_upsample_scale0(j, i)[0];
                            tmp_reload.y = priornormal_upsample_scale0(j, i)[1];
                            tmp_reload.z = priornormal_upsample_scale0(j, i)[2];
                            tmp_reload.w = priordepth_upsample_scale0(j, i);
                            tmp_reload = hpm.TransformNormal2RefCam(tmp_reload);
                            float depth_now = tmp_reload.w;
                            int2 p = make_int2(i, j);
                            tmp_reload.w = hpm.GetDistance2Origin_factor(p, depth_now, tmp_reload, 0.25);
                            prior_plane_parameters[center] = tmp_reload;

                        }
                        else {
                            mask_tri_scale0_upsample(j, i) = 0;
                        }
                    }
                }
                hpm.ReloadPlanarPriorInitialization(mask_tri_scale0_upsample, prior_plane_parameters);
                hpm.RunPatchMatch();

                depths_scale0.release();
                costs_scale0.release();
                normals_scale0.release();
                mask_tri_scale0.release();
                delete(prior_supplement_scale0);
                pointcloud->clear();
                priordepths_scale0.release();
                priornormals_scale0.release();
                image_uint.release();
                image_float.release();
                priordepth_upsample_scale0.release();
                priordepth_upsample_scale0.release();
                mask_tri_scale0_upsample.release();
                delete(prior_plane_parameters);
                hpm.ReleasePriorCudaMemory();

                for (int col = 0; col < width; ++col) {
                    for (int row = 0; row < height; ++row) {
                        int center = row * width + col;
                        plane_hypothesis = hpm.GetPlaneHypothesis(center);
                        depths(row, col) = plane_hypothesis.w;
                        normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                        costs(row, col) = hpm.GetCost(center);
                    }
                }
            }
            else if (i == 1) {
                std::cout << "Scale 1 prior generating..." << std::endl;
                cv::Mat_<float>depths_scale1;
                cv::Mat_<float>costs_scale1;
                cv::Mat_<cv::Vec3f>normals_scale1;
                cv::resize(depths, depths_scale1, cv::Size(std::round(width * 0.5), std::round(height * 0.5)), 0, 0, cv::INTER_LINEAR);
                cv::resize(costs, costs_scale1, cv::Size(std::round(width * 0.5), std::round(height * 0.5)), 0, 0, cv::INTER_LINEAR);
                cv::resize(normals, normals_scale1, cv::Size(std::round(width * 0.5), std::round(height * 0.5)), 0, 0, cv::INTER_LINEAR);
                std::vector<cv::Point> support2DPoints_scale1;
                hpm.GetSupportPointsScale1(support2DPoints_scale1, costs_scale1);
                const cv::Rect imageRC(0, 0, std::round(width * 0.5), std::round(height * 0.5));
                const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints_scale1);
                cv::Mat_<float> mask_tri_scale1 = cv::Mat::zeros(std::round(height * 0.5), std::round(width * 0.5), CV_32FC1);
                std::vector<float4> planeParams_tri_scale1;
                planeParams_tri_scale1.clear();
                uint32_t idx = 0;
                for (const auto triangle : triangles) {
                    if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;

                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri_scale1(y, x) = idx + 1.0;
                            }
                        }
                        float4 n4 = hpm.GetPriorPlaneParams_factor(triangle, depths_scale1, 0.5);
                        planeParams_tri_scale1.push_back(n4);
                        idx++;
                    }
                }
                float4* prior_supplement_scale1 = new float4[std::round(height * 0.5) * std::round(width * 0.5)];
                pointcloud->clear();
                hpm.WritePointCloud(pointcloud, support2DPoints_scale1);
                hpm.DepthsPredictSupplement(pointcloud, depths_scale1, normals_scale1, prior_supplement_scale1, mask_tri_scale1, 0.5);
                cv::Mat_<float>priordepths_scale1 = cv::Mat::zeros(std::round(height * 0.5), std::round(width * 0.5), CV_32FC1);
                cv::Mat_<cv::Vec3f>priornormals_scale1 = cv::Mat::zeros(std::round(height * 0.5), std::round(width * 0.5), CV_32FC3);

                for (int i = 0; i < std::round(width * 0.5); ++i) {
                    for (int j = 0; j < std::round(height * 0.5); ++j) {
                        if (mask_tri_scale1(j, i) > 0) {
                            float d = hpm.GetDepthFromPlaneParam_factor(planeParams_tri_scale1[mask_tri_scale1(j, i) - 1], i, j, 0.5);

                            if (d <= hpm.GetMaxDepth() * 1.2f && d >= hpm.GetMinDepth() * 0.6f) {
                                priordepths_scale1(j, i) = d;
                                tmp_n4 = hpm.TransformNormal(planeParams_tri_scale1[mask_tri_scale1(j, i) - 1]);
                                priornormals_scale1(j, i)[0] = tmp_n4.x;
                                priornormals_scale1(j, i)[1] = tmp_n4.y;
                                priornormals_scale1(j, i)[2] = tmp_n4.z;
                            }
                            else {
                                mask_tri_scale1(j, i) = 0;
                            }
                        }
                        else {
                            if (mask_tri_scale1(j, i) < -200) {
                                int width_scale1 = std::round(width * 0.5);
                                tmp_n4 = prior_supplement_scale1[j * width_scale1 + i];
                                float d = hpm.GetDepthFromPlaneParam_factor(tmp_n4, i, j, 0.5);
                                if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
                                    priordepths_scale1(j, i) = d;
                                    float4 tmp = hpm.TransformNormal(tmp_n4);
                                    priornormals_scale1(j, i)[0] = tmp.x;
                                    priornormals_scale1(j, i)[1] = tmp.y;
                                    priornormals_scale1(j, i)[2] = tmp.z;
                                }
                                else {
                                    mask_tri_scale1(j, i) = 0;
                                }
                            }
                            else {
                                mask_tri_scale1(j, i) = 0;
                            }
                        }
                    }
                }
                std::stringstream image_path;
                image_path << dense_folder << "/images" << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
                cv::Mat_<uint8_t> image_uint;
                cv::resize(cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE), image_uint, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
                cv::Mat image_float;
                image_uint.convertTo(image_float, CV_32FC1);
                cv::Mat_<float>priordepth_upsample_scale1 = cv::Mat::zeros(height, width, CV_32FC1);
                cv::Mat_<cv::Vec3f>priornormal_upsample_scale1 = cv::Mat::zeros(height, width, CV_32FC3);
                std::cout << "Running JBU..." << std::endl;
                hpm.JointBilateralUpsampling(image_float, priordepths_scale1, priordepth_upsample_scale1, priornormals_scale1, priornormal_upsample_scale1);
                image_float.release();
                image_uint.release();
                cv::Mat_<float> mask_tri_scale1_upsample = cv::Mat::zeros(height, width, CV_32FC1);
                float4* prior_plane_parameters = new float4[height * width];
                float4 tmp_reload;
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        if (priordepth_upsample_scale1(j, i) != priordepth_upsample_scale1(j, i)) {
                            mask_tri_scale1_upsample(j, i) = 0;
                        }
                        if (priordepth_upsample_scale1(j, i) <= hpm.GetMaxDepth() && priordepth_upsample_scale1(j, i) >= hpm.GetMinDepth()) {
                            mask_tri_scale1_upsample(j, i) = 1;
                            int center = j * width + i;

                            tmp_reload.x = priornormal_upsample_scale1(j, i)[0];
                            tmp_reload.y = priornormal_upsample_scale1(j, i)[1];
                            tmp_reload.z = priornormal_upsample_scale1(j, i)[2];
                            tmp_reload.w = priordepth_upsample_scale1(j, i);
                            tmp_reload = hpm.TransformNormal2RefCam(tmp_reload);
                            float depth_now = tmp_reload.w;
                            int2 p = make_int2(i, j);
                            tmp_reload.w = hpm.GetDistance2Origin_factor(p, depth_now, tmp_reload, 0.5);
                            prior_plane_parameters[center] = tmp_reload;
                        }
                        else {
                            mask_tri_scale1_upsample(j, i) = 0;
                        }
                    }
                }
                hpm.ReloadPlanarPriorInitialization(mask_tri_scale1_upsample, prior_plane_parameters);
                hpm.RunPatchMatch();

                depths_scale1.release();
                costs_scale1.release();
                normals_scale1.release();
                mask_tri_scale1.release();
                delete(prior_supplement_scale1);
                pointcloud->clear();
                priordepths_scale1.release();
                priornormals_scale1.release();
                image_uint.release();
                image_float.release();
                priordepth_upsample_scale1.release();
                priordepth_upsample_scale1.release();
                mask_tri_scale1_upsample.release();
                delete(prior_plane_parameters);
                hpm.ReleasePriorCudaMemory();


                for (int col = 0; col < width; ++col) {
                    for (int row = 0; row < height; ++row) {
                        int center = row * width + col;
                        plane_hypothesis = hpm.GetPlaneHypothesis(center);
                        depths(row, col) = plane_hypothesis.w;
                        normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                        costs(row, col) = hpm.GetCost(center);
                    }
                }
            }
            else {

                std::cout << "Scale 2 prior generating..." << std::endl;
                const cv::Rect imageRC(0, 0, width, height);
                std::vector<cv::Point> support2DPoints;

                hpm.GetSupportPoints(support2DPoints);
                const auto triangles = hpm.DelaunayTriangulation(imageRC, support2DPoints);


                cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
                std::vector<float4> planeParams_tri;
                planeParams_tri.clear();

                uint32_t idx = 0;
                for (const auto triangle : triangles) {
                    if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                        float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                        float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                        float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                        float max_edge_length = std::max(L01, std::max(L02, L12));
                        float step = 1.0 / max_edge_length;

                        for (float p = 0; p < 1.0; p += step) {
                            for (float q = 0; q < 1.0 - p; q += step) {
                                int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                                int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                                mask_tri(y, x) = idx + 1.0;
                            }
                        }

                        float4 n4 = hpm.GetPriorPlaneParams_factor(triangle, depths, 1);
                        planeParams_tri.push_back(n4);
                        idx++;
                    }
                }
                float4* prior_supplement_origin = new float4[width * height];
                pointcloud->clear();
                hpm.WritePointCloud(pointcloud, support2DPoints);
                hpm.DepthsPredictSupplement(pointcloud, depths, normals, prior_supplement_origin, mask_tri, 1);

                cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
                for (int i = 0; i < width; ++i) {
                    for (int j = 0; j < height; ++j) {
                        if (mask_tri(j, i) > 0) {
                            float d = hpm.GetDepthFromPlaneParam_factor(planeParams_tri[mask_tri(j, i) - 1], i, j, 1);
                            if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
                                priordepths(j, i) = d;
                            }
                            else {
                                mask_tri(j, i) = 0;
                            }
                        }
                        else {
                            if (mask_tri(j, i) < -200) {
                                tmp_n4 = prior_supplement_origin[j * width + i];
                                float d = hpm.GetDepthFromPlaneParam_factor(tmp_n4, i, j, 1);
                                if (d <= hpm.GetMaxDepth() && d >= hpm.GetMinDepth()) {
                                    priordepths(j, i) = d;
                                }
                                else {
                                    mask_tri(j, i) = 0;
                                }
                            }
                            else {
                                mask_tri(j, i) = 0;
                            }
                        }
                    }
                }
                std::string depth_path = result_folder + "/depths_prior.dmb";
                writeDepthDmb(depth_path, priordepths);

                hpm.CudaPlanarPriorInitializationSupplement(planeParams_tri, mask_tri, prior_supplement_origin);
                hpm.RunPatchMatch();

                //ÊÍ·ÅÄÚ´æ
                mask_tri.release();
                delete(prior_supplement_origin);
                pointcloud->clear();
                priordepths.release();
                hpm.ReleasePriorCudaMemory();


                for (int col = 0; col < width; ++col) {
                    for (int row = 0; row < height; ++row) {
                        int center = row * width + col;
                        plane_hypothesis = hpm.GetPlaneHypothesis(center);
                        depths(row, col) = plane_hypothesis.w;
                        normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                        costs(row, col) = hpm.GetCost(center);
                    }
                }
            }
        }
    }
    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    hpm.ReleaseProblemCudaMemory();
    hpm.ReleaseProblemHostMemory();
    depths.release();
    normals.release();
    costs.release();
    

    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}


void RunFusion(std::string& dense_folder, const std::vector<Problem>& problems, bool geom_consistency)
{
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        std::stringstream result_path;
        result_path << dense_folder << "/HPM" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = problems[i].src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = { (float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2] };
                int num_consistent = 0;
                float dynamic_consistency = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = problems[i].src_image_ids[j];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                            float cons = exp(-tmp_index);
                            dynamic_consistency += exp(-tmp_index);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[problems[i].src_image_ids[j]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = dense_folder + "/HPM/HPM_model.ply";
    StoreColorPlyFileBinaryPointCloud(ply_path, PointCloud);
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "USAGE: HPM dense_folder" << std::endl;
        return -1;
    }

    std::string dense_folder = argv[1];
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);

    std::string output_folder = dense_folder + std::string("/HPM");
    mkdir(output_folder.c_str());

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    bool geom_consistency = false;
    bool planar_prior = true;
    bool multi_geometry = false;
    int geom_iterations = 2;
    
    for (size_t i = 0; i < num_images; ++i) {
        ProcessProblem(dense_folder, problems[i], geom_consistency, planar_prior);

    }
    geom_consistency = true;
    planar_prior = false;
    for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
        if (geom_iter == 0) {
            multi_geometry = false;
        }
        else {
            multi_geometry = true;
        }
        for (size_t i = 0; i < num_images; ++i) {
            ProcessProblem(dense_folder, problems[i], geom_consistency, planar_prior, multi_geometry);
        }
    }
    RunFusion(dense_folder, problems, geom_consistency);
    return 0;
}
