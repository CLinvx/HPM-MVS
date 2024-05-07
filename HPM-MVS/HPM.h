#ifndef _HPM_H_
#define _HPM_H_

#include "main.h"

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera_short(const std::string& cam_path);
Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size = 3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;
    bool geom_consistency = false;
    bool multi_geometry = false;
    bool planar_prior = false;
};

class HPM {
public:
    HPM();
    ~HPM();

    void InuputInitialization(const std::string &dense_folder, const Problem &problem);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    void GetSupportPoints(std::vector<cv::Point>& support2DPoints);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    float4 GetPriorPlaneParams_factor(const Triangle triangle, const cv::Mat_<float> depths, float factor);
    float GetDepthFromPlaneParam_factor(const float4 plane_hypothesis, const int x, const int y, float factor);
    float GetMinDepth();
    float GetMaxDepth();
    void CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks);

    int ComputingMultiScaleSettings(int width, int height);
    void WritePointCloud(PointCloudPtr& cloud, std::vector<cv::Point>points);
    float4 TransformNormal(float4 plane_hypothesis);
    float4 TransformNormal2RefCam(float4 plane_hypothesis); 
    float GetDistance2Origin_factor(const int2 p, const float depth, const float4 normal, float factor);
    void ReloadPlanarPriorInitialization(const cv::Mat_<float>& masks, float4* prior_plane_parameters);
    void DepthsPredictSupplement(PointCloudPtr cloud, cv::Mat_<float>& depth, cv::Mat_<cv::Vec3f>& normal_reliable, float4* plane_parameters, cv::Mat_<float>& mask, float factor);
    void CudaPlanarPriorInitializationSupplement(const std::vector<float4>& PlaneParams, cv::Mat_<float>& masks, const float4* plane_hypothesis);
    void GetSupportPointsScale0(std::vector<cv::Point>& support2DPoints, cv::Mat_<float>costs);
    void GetSupportPointsScale1(std::vector<cv::Point>& support2DPoints, cv::Mat_<float>costs);
    void JointBilateralUpsampling(const cv::Mat_<float>& scaled_image_float, const cv::Mat_<float>& src_depthmap, cv::Mat_<float>& upsample_depthmap, const cv::Mat_<cv::Vec3f>& src_normal, cv::Mat_<cv::Vec3f>& upsample_normal);
    void ReleasePriorCudaMemory();
    void ReleaseProblemCudaMemory();
    void ReleaseProblemHostMemory();


private:
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float *costs_host;
    float4 *prior_planes_host;
    unsigned int *plane_masks_host;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float *costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
    float4 *prior_planes_cuda;
    unsigned int *plane_masks_cuda;
};

#endif // _HPM_H_
