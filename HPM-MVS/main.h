#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include<direct.h>
#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

//#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef pcl::PointCloud<pcl::PointXY>::Ptr PointCloudPtr;

#define MAX_IMAGES 512
#define M_PI 3.14159265358979323846
#define JBU_NUM 2

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

struct TexObj {
    cudaTextureObject_t imgs[MAX_IMAGES];
};

struct JBUParameters {
    int height;
    int width;
    int s_height;
    int s_width;
    int Imagescale;
};

struct JBUTexObj {
    cudaTextureObject_t imgs[JBU_NUM];
};

class JBU {
public:
    JBU();
    ~JBU();

    // Host Parameters
    float* depth_h;
    float4* normal_h;
    JBUTexObj jt_h;
    JBUParameters jp_h;
    float4* normal_origin_host;

    // Device Parameters
    float* depth_d;
    cudaArray* cuArray[JBU_NUM]; // The first for reference image, and the second for stereo depth image
    JBUTexObj* jt_d;
    JBUParameters* jp_d;
    float4* normal_d;
    float4* normal_origin_cuda;

    void InitializeParameters(int n, int origin_n);
    void CudaRun();
    void ReleaseJBUCudaMemory();
    void ReleaseJBUHostMemory();
};

#endif // _MAIN_H_
