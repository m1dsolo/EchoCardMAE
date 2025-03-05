import resource

import torch
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.optim import AdamW
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd

import yangdl as yd
from ..model.vit_seg import ViTSeg2D
from ..loss.dice_loss import DiceLoss
from ..utils import (
    NpyDataset,
    get_file_names
)

yd.env.seed = 0
yd.env.exp_path = '../res/echonet_seg'

df = pd.read_csv(f'/home/yang/dataset/EchoNet-Dynamic/FileList.csv', index_col='FileName')
NPY_PATH = '../npy/echonet/edes'
file_names = set(get_file_names(f'{NPY_PATH}/image'))
df = df.loc[df.index.isin(file_names)]

BATCH_SIZE = 64
CACHE = True
CKPT_NAME = f'../res/pretrain/ckpt/1/best.pt'

MEAN = (0.1257, 0.1271, 0.1292)
STD = (0.1951, 0.1957, 0.1974)

task_hparams = {
    'early_stop_params': {
        'monitor': {'loss.val': 'min'},
        'patience': 25,
        'min_stop_epoch': 25,
        'max_stop_epoch': 1000,
    },
    'benchmark': True,
    'deterministic': True,
}


class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True
        )
        self.loss = yd.ValueMetric()
        self.metric = yd.SegMetric(num_classes=2)
        self.scaler = GradScaler()

    def __iter__(self):
        self.model = ViTSeg2D(
            img_size=112,
            patch_size=8,
            encoder_in_chans=3,
            encoder_embed_dim=384,
            encoder_depth=12,
            encoder_num_heads=6,
            decoder_embed_dim=192,
            decoder_depth=4,
            decoder_num_heads=3,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            init_values=0.,
            cos_attn=False,
        )
        if CKPT_NAME is not None:
            sd = torch.load(CKPT_NAME, weights_only=True)['model']
            sdd = {}
            for key, val in sd.items():
                if key == 'encoder.patch_embed.proj.weight':
                    continue
                elif key[:12] == 'decoder.head':
                    continue
                sdd[key] = val
            missing, unexpected = self.model.load_state_dict(sdd, strict=False)
            print(missing)
            print(unexpected)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))

        yield

    def train_step(self, batch):
        loss = self._step(batch)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {
            'loss': loss,
            'dice': self.metric.dice,
        }

    def val_step(self, batch):
        loss = self._step(batch)

        return {
            'loss': loss,
            'dice': self.metric.dice,
        }

    def test_step(self, batch):
        loss = self._step(batch)

        return {
            'loss': loss,
            'dice': self.metric.dice,
        }

    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            dice=self.metric.dice,
        )

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            dice=self.metric.dice,
        )

    def test_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            dice=self.metric.dice,
        )

    def _step(self, batch):
        x = batch['image'] # (B, C, H, W)
        y = batch['mask'].long() # (B, H, W)

        with autocast('cuda'):
            logits = self.model(x) # (B, 2, H, W)
            probs = F.softmax(logits, dim=1)
            loss = self.criterion(logits, y[:, None])

        self.metric.update(probs, y)
        self.loss.update(loss, len(x))

        return loss


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

    def train_loader(self):
        trans = A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            A.PadIfNeeded(min_height=112 + 12 * 2, min_width=112 + 12 * 2, value=0, border_mode=cv2.BORDER_CONSTANT, p=1.),
            A.RandomCrop(112, 112, p=1.),
            ToTensorV2(),
        ])
        def transform(d):
            t = trans(image=d['image'], mask=d['mask'].astype(np.uint8))
            return {
                'image': t['image'],
                'mask': t['mask'],
            }

        dataset = NpyDataset(
            NPY_PATH,
            df=df[df['Split'] == 'TRAIN'],
            transform=transform,
            rets=['image', 'mask'],
            cache=CACHE
        )
        yield DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    def val_loader(self):
        trans = A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        def transform(d):
            t = trans(image=d['image'], mask=d['mask'])
            return {
                'image': t['image'],
                'mask': t['mask'],
            }

        dataset = NpyDataset(
            NPY_PATH,
            df=df[df['Split'] == 'VAL'],
            transform=transform,
            rets=['image', 'mask'],
            cache=CACHE
        )
        yield DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

    def test_loader(self):
        trans = A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        def transform(d):
            t = trans(image=d['image'], mask=d['mask'])
            return {
                'image': t['image'],
                'mask': t['mask'],
            }

        dataset = NpyDataset(
            NPY_PATH,
            df=df[df['Split'] == 'TEST'],
            transform=transform,
            rets=['image', 'mask'],
            cache=False
        )
        yield DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        )


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    task_module = yd.TaskModule(
        **task_hparams,
        model_module=MyModelModule(),
        data_module=MyDataModule()
    )
    task_module.do()
