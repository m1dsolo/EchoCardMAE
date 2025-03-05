# EchoCardMAE: Video Masked Auto-Encoders Customized for Echocardiography

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
./echonet/avi2npy.py
```

### Pre-training

You can use [pretraining weights](EchoCardMAE.pt) provided by us.
Or you can pretrain the model by yourself:
```bash
python pretrain.py
```

### Fine-tuning

```bash
cd echonet
python train_ef.py
```

## TODO

- [ ] upload the code of CAMUS and HMC-QU

## Citation

TODO
