# HPM-MVS
This is the official repo for the implementation of Hierarchical Prior Mining for Non-local Multi-View Stereo (Chunlin Ren, Qingshan Xu, Shikun Zhang, Jiaqi Yang, ICCV 2023).

## Introduction
In this work, we propose a Hierarchical Prior Mining for Non-local Multi-View Stereo (HPM-MVS). The key characteristics are the following techniques that exploit non-local information to assist MVS: 1) A Non-local Extensible Sampling Pattern (NESP), which is able to adaptively change the size of sampled areas without becoming snared in locally optimal solutions. 2) A new approach to leverage non-local reliable points and construct a planar prior model based on K-Nearest Neighbor (KNN), to obtain potential hypotheses for the regions where prior construction is challenging. 3) A Hierarchical Prior Mining (HPM) framework, which is used to mine extensive non-local prior information at different scales to assist 3D model recovery, this strategy can achieve a considerable balance between the reconstruction of details and low-textured areas. Experimental results on the ETH3D and Tanks & Temples have verified the superior performance and strong generalization capability of our method.
<div align=center>
<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/pipeline.png" width="800">
</div>

## NEWS！！！
* The initial version for [HPM-MVS++](https://github.com/CLinvx/HPM-MVS_plusplus) has been released.

## TO DO LIST (The code of HPM-MVS will be available alongside HPM-MVS++, which ranks 2nd in the ETH3D benchmark.)
* Stage 1: Realse the code of NESP+ACMM, NESP+ACMP, NESP+ACMMP. 
* Stage 2: Realse the code of HPM-MVS.

## Dependencies
The code has been tested on Windows 10 with RTX 3070.
* NESP+ACMM, NESP+ACMP, NESP+ACMMP<br />
  [cmake](https://cmake.org/)<br />
  [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 6.0<br />
  [OpenCV](https://opencv.org/) >=2.4
* HPM-MVS

## Useage
* Compile
```
mkdir build
cd build
cmake ..
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to MVS input   
Run ./xxx $data_folder to get reconstruction results (./xxx represents the project name)
```

## Results
* Benchmark Performance

1. ETH3D benchmark

<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/ETH3D.png" width="400">
2. Tanks & Temples benchmark

<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/Tanks_Temples.png" width="400">

* Ablation Study
<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/abaltion.png" width="700">

* Generalization Performance of NESP
<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/generalize_NESP.png" width="400">

* Runtime Performance (Resolution: 3200*3130) 
<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/Runtime.png" width="180">

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Ren_2023_ICCV,
    author    = {Ren, Chunlin and Xu, Qingshan and Zhang, Shikun and Yang, Jiaqi},
    title     = {Hierarchical Prior Mining for Non-local Multi-View Stereo},
    booktitle = {Proc. IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2023},
    pages     = {3611-3620}
}
```

## Acknowledgemets
This code largely benefits from the following repositories: [ACMH](https://github.com/GhiXu/ACMH), [ACMP](https://github.com/GhiXu/ACMP), [ACMMP](https://github.com/GhiXu/ACMMP). Thanks to their authors for opening source of their excellent works.
