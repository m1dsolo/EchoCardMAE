import os
import resource
from functools import partial

import torch
import cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import yangdl as yd
from model.vit import VisionTransformer
from utils import (
    NpyDataset,
    replay_transform
)

yd.env.exp_path = './res/echonet_ef'
yd.env.seed = 0

DATASET_PATH = os.path.join(os.environ['HOME'], 'dataset/EchoNet-Dynamic')
df = pd.read_csv(
    os.path.join(DATASET_PATH, 'FileList.csv'),
    index_col='FileName',
    usecols=['FileName', 'EF', 'Split']
)

NPY_PATH = './npy/echonet/main'

BATCH_SIZE = 16
CACHE = True
CLIP_LENGTH = 16
SAMPLING_RATE = 4
CKPT_NAME = f'./res/pretrain/ckpt/1/best.pt'

MEAN = (0.1257, 0.1271, 0.1292)
STD = (0.1951, 0.1957, 0.1974)

task_hparams = {
    'early_stop_params': {
        'monitor': {'metric.mae': 'min'},
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

        self.criterion = nn.L1Loss(reduction='mean')
        self.loss = yd.ValueMetric()
        self.metric = yd.RegMetric()
        self.scaler = GradScaler()

    def __iter__(self):
        self.model = VisionTransformer(
            num_frames=CLIP_LENGTH,
            img_size=112,
            patch_size=8,
            in_chans=3,
            num_classes=1,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            head_drop_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.,
            init_scale=0.,
            tubelet_size=2,
            cos_attn=False,
        )
        if CKPT_NAME is not None:
            sd = torch.load(CKPT_NAME, weights_only=True)['model']
            sdd = {}
            for key, val in sd.items():
                if key[:8] == 'encoder.':
                    if key[8:] == 'patch_embed.proj.weight':
                        val = F.interpolate(
                            val,
                            size=(2, 8, 8),
                            mode='trilinear',
                            align_corners=False,
                        )
                    sdd[key[8:]] = val
                else:
                    sdd[key] = val
            missing, unexpected = self.model.load_state_dict(sdd, strict=False)
            print(missing)
            print(unexpected)

        self.model.head.bias.data[0] = 0.556

        self.optimizer = AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.999))

        yield

    def train_step(self, batch):
        loss = self._step(batch['video'], batch['EF'])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {
            'loss': loss,
            'mae': self.metric.mae,
            'rmse': self.metric.rmse,
            'r2': self.metric.r2,
        }

    def val_step(self, batch):
        loss = self._step(batch['video'], batch['EF'])

        return {
            'loss': loss,
            'mae': self.metric.mae,
            'rmse': self.metric.rmse,
            'r2': self.metric.r2,
        }

    def test_step(self, batch):
        loss = self._step(batch['video'], batch['EF'])

        return {
            'loss': loss,
            'mae': self.metric.mae,
            'rmse': self.metric.rmse,
            'r2': self.metric.r2,
        }

    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            mae=self.metric.mae,
            rmse=self.metric.rmse,
            r2=self.metric.r2,
        )

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            mae=self.metric.mae,
            rmse=self.metric.rmse,
            r2=self.metric.r2,
        )

    def test_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            mae=self.metric.mae,
            rmse=self.metric.rmse,
            r2=self.metric.r2,
        )

    # (B, C, T, H, W)
    def _step(self, x, y):
        with autocast('cuda'):
            preds = self.model(x)[:, 0] * 100 # (B,)
            loss = self.criterion(preds, y)

        self.metric.update(preds, y)
        self.loss.update(loss, len(x))

        return loss


def video_transform(trans, clip_length, sampling_rate):
    def transform(res):
        # maxd
        video = res['video']
        D, H, W, C = video.shape
        maxd = (clip_length - 1) * sampling_rate + 1
        if D < maxd:
            video = np.concatenate((video, np.zeros((maxd - D, H, W, C), dtype=video.dtype)), axis=0)
            D = maxd
            
        start_frame = np.random.choice(D - (clip_length - 1) * sampling_rate)
        video = video[start_frame + np.arange(clip_length) * sampling_rate] # (self.clip_length, H, W, C)

        return {
            'video': replay_transform(trans, video),   # (C, clip_length, H, W)
            'EF': torch.tensor(res['EF'], dtype=torch.float32),
        }

    return transform


class MyDataModule(yd.DataModule):
    def __init__(
        self,
        clip_length: int = 16,
        sampling_rate: int = 4,
    ):
        super().__init__()

        self.clip_length = clip_length
        self.sampling_rate = sampling_rate

    def train_loader(self):
        trans = A.ReplayCompose([
            A.Normalize(mean=MEAN, std=STD),
            A.PadIfNeeded(min_height=112 + 12 * 2, min_width=112 + 12 * 2, value=0, border_mode=cv2.BORDER_CONSTANT, p=1.),
            A.RandomCrop(112, 112, p=1.),
            ToTensorV2(),
        ])

        dataset = NpyDataset(
            NPY_PATH,
            df[df['Split'] == 'TRAIN'],
            transform=video_transform(trans, self.clip_length, self.sampling_rate),
            rets=['video', 'EF'],
            cache=CACHE,
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
        trans = A.ReplayCompose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        dataset = NpyDataset(
            NPY_PATH,
            df[df['Split'] == 'VAL'],
            transform=video_transform(trans, self.clip_length, self.sampling_rate),
            rets=['video', 'EF'],
            cache=CACHE,
        )
        yield DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)

    def test_loader(self):
        trans = A.ReplayCompose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
        dataset = NpyDataset(
            NPY_PATH,
            df[df['Split'] == 'TEST'],
            transform=video_transform(trans, self.clip_length, self.sampling_rate),
            rets=['video', 'EF'],
            cache=False,
        )
        yield DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    task_module = yd.TaskModule(
        **task_hparams,
        model_module=MyModelModule(),
        data_module=MyDataModule()
    )
    task_module.do()
