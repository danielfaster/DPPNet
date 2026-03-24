# DPPNet: A Depth Pixel-wise Potential-aware Network for RGB-D Salient Object Detection

> **Authors:** 
> Junbin Yuan,
> Yiqi Wang,
> Zhoutao Wang,
> Qingzhen Xu,
> Bharadwaj Veeravalli,
> Xulei Yang

## Preface

- This repository provides code for "_**DPPNet: A Depth Pixel-Wise Potential-Aware Network for RGB-D Salient Object Detection**_" [IEEE Transactions on Multimedia, 2025](URL "[title](https://ieeexplore.ieee.org/abstract/document/10882929)").


# Framework
![image](https://github.com/danielfaster/DPPNet/blob/main/figures/network.png)


# Experiment
1. Visual comparison results
![image](https://github.com/danielfaster/DPPNet/blob/main/figures/visual_comparsion.png)

2. Quantitative comparison results
![image](https://github.com/danielfaster/DPPNet/blob/main//figures/quantitative_comparsion.png)




# Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA Tesla V100 GPU with 32 GB memory. 


## ⚙️ Environment Setup

We recommend **Python 3.8–3.11** and **Miniconda** (or any virtualenv). **PyTorch must be installed before** `pip install -r requirements.txt`, because CUDA/CPU wheels depend on your machine.

### 1. Clone the repository

```bash
git clone https://github.com/danielfaster/DPPNet.git
cd DPPNet
```

### 2. Create and activate a conda environment (optional)

```bash
conda create -n dppnet python=3.10
conda activate dppnet
```

### 3. Install PyTorch and torchvision

Choose **one** command that matches your GPU driver / CUDA, or use the CPU line. See also the [official PyTorch install guide](https://pytorch.org/get-started/locally/).

**CUDA 11.8** (common on Linux and Windows with recent NVIDIA drivers):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**

```bash
pip install torch torchvision
```

### 4. Install remaining dependencies

`requirements.txt` pins the rest of the stack (e.g. `timm`, `numpy`, `opencv-python`, `Pillow`, `PyYAML`). It is a normal pip list—**not** a conda `file://` export—so it should install on any OS after PyTorch is in place.

```bash
pip install -r requirements.txt
```
## 📂 Dataset Preparation
### 1. Training Dataset

 Download from  [Google Drive RGB+depth](https://drive.google.com/file/d/1Orss85k3wEUgDhItwT1goEN6WQFA1SOw/view?usp=sharing)  
 [Google Drive depth_quality_pseudo_label](https://drive.google.com/file/d/1KXbmMYO_TWEuWdEChOv2w0wI2m17RSnn/view?usp=sharing)
 
Place it into:  ../Data/TrainDataset/
```
├──DPPNet/
├── Data/
│   └── TrainDataset/
│       └──  RGB/
│       └──  depth/
│       └──  depth_quality_pseudo_label/
```
### 2. Testing Datasets

Download from [Google Drive](https://drive.google.com/file/d/1sWJqCg2dAKSSkfrvB7zkwwsW6Ybd4Gd1/view?usp=sharing)  
Place it into: ../Data/TestDataset/
Directory structure:
```
├──DPPNet/
├── Data/
│   └── TestDataset/
│       ├── NLPR/
│       ├── NJUD/
│       ├── STERE/
│       └── SIP/
```
We evaluate our method on four widely-used RGB-D SOD benchmark datasets:

 
## 🧾 Pretrained Model

Download ViT-L pretrained weights: [download](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth)


Place it into:
```
DPPNet/
├── pretrain/
│   └── mae_pretrain_vit_large.pth
```

## 🚀 Training
```
python train.py
```
## 🔍 Inference
```
python inference.py
```
## Evaluating your trained model


## ⭐ Acknowledgement

This project is built upon several excellent open-source works, including:

- [SPNet](https://github.com/taozh2017/SPNet)

We sincerely thank the authors for making their code publicly available.

## Citation

Please cite our paper if you find the work useful: 
	
     @article{yuan2025dppnet,
     title={DPPNet: A Depth Pixel-wise Potential-aware Network for RGB-D Salient Object Detection},
     author={Yuan, Junbin and Wang, Yiqi and Wang, Zhoutao and Xu, Qingzhen and Veeravalli, Bharadwaj and Yang, Xulei},
     journal={IEEE Transactions on Multimedia},
     year={2025},
     publisher={IEEE}
      }  



## ✅ TODO List
- [x] Release prediction results
- [x] Release the full model code (architecture & configs)
- [x] Open-source training and inference pipelines
- [ ] Publish evaluation scripts and metrics
- [ ] Upload pretrained checkpoints




