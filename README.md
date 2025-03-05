# EchoCardMAE: Video Masked Auto-Encoders Customized for Echocardiography

## Introduction

![framework](https://github.com/user-attachments/assets/7b3f0c04-ab75-4038-8efa-c1d92e9eece9)

## Visualization

### Reconstruction

EchoCardMAE reconstruction results on the EchoNet-Dynamic dataset.

<img src="https://github.com/user-attachments/assets/6ee03655-620c-49d7-a92a-e0150c04befd" width="50%">

### Segmentation

Segmentation results on the EchoNet-Dynamic and CAMUS dataset.

![echonet-camus-seg](https://github.com/user-attachments/assets/e938ec59-dbc5-4e21-96bd-cafb4dda9b60)

## Installation

```bash
# remove GIT_LFS_SKIP_SMUDGE=1 if you want to download the pretraining weights
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/m1dsolo/EchoCardMAE.git
cd EchoCardMAE
conda create -n EchoCardMAE python=3.10
conda activate EchoCardMAE
pip install -r requirements.txt
git submodule add --depth=1 https://github.com/m1dsolo/yangdl.git yangdl
cd yangdl
pip install -e .
```

Experimental environment:
- PyTorch 2.5.1
- Python 3.10.15
- GPU memory 24GB

## Usage

### Data Preparation

1. EchoNet-Dynamic: [Download](https://echonet.github.io/dynamic/index.html#dataset) to `EchoCardMAE/dataset/EchoNet-Dynamic`
2. CAMUS: [Download](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) to `EchoCardMAE/dataset/CAMUS`
3. HMC-QU: [Download](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset) to `EchoCardMAE/dataset/hmcqu-dataset`

### Data preprocessing

```bash
python -m echonet.avi2npy
```

### Pre-training

You can use [pretraining weights](EchoCardMAE.pt) provided by us.
Or you can pretrain the model by yourself:
```bash
python pretrain.py
```

### Fine-tuning

```bash
python -m echonet.train_ef
```

## TODO

- [ ] upload the code of CAMUS and HMC-QU

## Citation

TODO
