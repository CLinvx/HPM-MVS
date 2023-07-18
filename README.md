# HPM-MVS
Hierarchical Prior Mining for Non-local Multi-View Stereo (ICCV 2023)

## Abstract
As a fundamental problem in computer vision, multiview stereo (MVS) aims at recovering the 3D geometry of a target from a set of 2D images. Recent advances in MVS have shown that it is important to perceive non-local structured information for recovering geometry in low-textured areas. In this work, we propose a Hierarchical Prior Mining for Non-local Multi-View Stereo (HPM-MVS). The key characteristics are the following techniques that exploit non-local information to assist MVS: 1) A Non-local Extensible Sampling Pattern (NESP), which is able to adaptively change the size of sampled areas without becoming snared in locally optimal solutions. 2) A new approach to leverage non-local reliable points and construct a planar prior model based on K-Nearest Neighbor (KNN), to obtain potential hypotheses for the regions where prior construction is challenging. 3) A Hierarchical Prior Mining (HPM) framework, which is used to mine extensive non-local prior information at different scales to assist 3D model recovery, this strategy can achieve a considerable balance between the reconstruction of details and low-textured areas. Experimental results on the ETH3D and Tanks & Temples have verified the superior performance and strong generalization capability of our method.
<div align=center>
<img src="https://github.com/CLinvx/HPM-MVS/blob/main/figures/pipeline.png" width="800">
</div>

## TO DO LIST:
* Stage 1: Realse the code of NESP+ACMM, NESP+ACMP, NESP+ACMMP. 
* Stage 2: Realse the code of HPM-MVS. (The code of HPM-MVS will be released as soon as I get back from my summer vacation.)
