# CAMS_Net

To address the poor performance of anterior rib segmentation in chest X-rays, as well as the issues of incomplete and discontinuous rib segmentation, a rib and clavicle segmentation algorithm based on attention and multi-scale features is proposed. 

The algorithm uses a ResUNet-like structure as the basic framework and introduces a collaborative attention skip connection module and an attention-guided multi-scale feature selection module. By emphasizing the feature representation of the ribs and using multi-scale information to jointly determine the pixel classification of regions with abnormal grayscale values, this method improves the segmentation performance of ribs in low-contrast and abnormal grayscale regions.


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4