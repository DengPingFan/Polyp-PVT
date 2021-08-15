# Polyp-PVT

by Bo Dong, Wenhai Wang, Deng Ping Fan,Jinpeng Li, Huazhu Fu, & Ling Shao.

This repo is the official implementation of ["Polyp-PVT: Polyp Segmentation with PyramidVision Transformers"](https://arxiv.org/pdf/xxxx.pdf). 
<img src="./Figs/visual.gif" width="100%" />

## Introduction
**Polyp-PVT** is initially described in [arxiv](https://arxiv.org/pdf/xxxx.pdf).
Most polyp segmentation methods use CNNs as their backbone, leading to two key issues when exchanging information between the encoder and decoder: 1) taking into account the differences in contribution between different-level features; and 2) designing effective mechanism for fusing these features.
Different from existing CNN-based methods, we adopt a transformer encoder, which learns more powerful and robust representations. 
In addition, considering the image acquisition influence and elusive properties of polyps, we introduce three novel modules, including a cascaded fusion module (CFM), a camouflage identification module (CIM), a and similarity aggregation module (SAM).
Among these, the CFM is used to collect the semantic and location information of polyps from high-level features, while the CIM is applied to capture polyp information disguised in low-level features. 
With the help of the SAM, we extend the pixel features of the polyp area with high-level semantic position information to the entire polyp area, thereby effectively fusing cross-level features.
The proposed model, named **Polyp-PVT** , effectively suppresses noises in the features and significantly improves their expressive capabilities. 


Polyp-PVT achieves strong performance on image-level polyp segmentation (`0.808 mean Dice` and `0.727 mean IoU` on ColonDB) and
video polyp segmentation (`0.880 mean dice` and `0.802 mean IoU` on CVC-300-TV), surpassing previous models by a large margin.



## Framework Overview
![](https://github.com/DengPingFan/Polyp-PVT/blob/main/Figs/network.png)

## Usage:
### Requirement:
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```

### Preprocessing:
Clone the repository:
```
git clone https://github.com/DengPingFan/Polyp-PVT.git
cd Polyp-PVT 
bash train.sh
bash test.sh
```

### Data preparation:
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Google Drive](xxxxxx)/[Baidu Drive](https://pan.baidu.com/s/1OBVivLJAs9ZpnB5I2s3lNg) [code:dr1h].

### Pretrained model:
You should download the pretrained model from [Google Drive](xxxxxx)/[Baidu Drive](https://pan.baidu.com/s/1Vez7iT2v_g7VYsDxRGE8HA) [code:w4vk], and then put it in the './pretrained_pth' folder for initialization. 

### Well trained model:
You could download the trained model from [Google Drive](xxxxxx)/[Baidu Drive](https://pan.baidu.com/s/1csPvdWqtbPBGrUWYO346Ug) [code:9rpy] and put the model in directory './model_pth'.

### Pre-computed maps:
[Google Drive](xxxxxx)/[Baidu Drive](https://pan.baidu.com/s/1UO1VaqXRRFNq23ku9yfMaw) [code:x3jc]

### Compared results:
We also provide some result of baseline methods, You could download from [Google Drive](xxxxxx)/[Baidu Drive](https://pan.baidu.com/s/15Ay_gv3W-ktXvE6UkrsjTA) [code:5tek].

### Results:


## Citation:
```
xxxxx
```

## Questions:
Please contact "xx@gmail.com" 
