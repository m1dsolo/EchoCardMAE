from functools import partial
import os
import resource

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import yangdl as yd
from ..model.vit import VisionTransformer
from ..utils import (
    NpyDataset,
    replay_transform,
)

yd.env.seed = 0
yd.env.exp_path = '../res/echonet_ef/mean5'
CKPT_NAME = '../res/echonet_ef/ckpt/1/best.pt'
NPY_PATH = '../npy/echonet/main'
DATASET_PATH = '../dataset/EchoNet-Dynamic'

MEAN = (0.1257, 0.1271, 0.1292)
STD = (0.1951, 0.1957, 0.1974)
CLIP_LENGTH = 16
SAMPLING_RATE = 4
BATCH_SIZE = 8

df = pd.read_csv(
    os.path.join(DATASET_PATH, 'FileList.csv'),
    index_col='FileName',
    usecols=['FileName', 'EF', 'Split']
)

REPEAT_NUM = 5

class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = nn.L1Loss(reduction='mean')

        self.loss = yd.ValueMetric()
        self.metric = yd.RegMetric()

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
        sd = torch.load(CKPT_NAME, map_location='cpu', weights_only=True)
        self.model.load_state_dict(sd['model'], strict=True)

    def val_step(self, batch):
        loss = self._step(batch)

        return {
            'loss': loss,
            'mae': self.metric.mae,
            'rmse': self.metric.rmse,
            'r2': self.metric.r2,
        }

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            mae=self.metric.mae,
            rmse=self.metric.rmse,
            r2=self.metric.r2,
        )

    def test_step(self, batch):
        loss = self._step(batch)

        return {
            'loss': loss,
            'mae': self.metric.mae,
            'rmse': self.metric.rmse,
            'r2': self.metric.r2,
        }

    def test_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            mae=self.metric.mae,
            rmse=self.metric.rmse,
            r2=self.metric.r2,
        )

    def _step(self, batch):
        x = batch['video']  # (B, 5, C, D, H, W)
        y = batch['EF']  # (B,)

        B, _, C, D, H, W = x.shape
        x = x.reshape(B * REPEAT_NUM, C, D, H, W)

        with autocast('cuda'):
            preds = self.model(x)[:, 0] * 100 # (B * 5,)
            preds = preds.reshape(B, REPEAT_NUM)
            preds_mean = preds.mean(dim=1)

            loss = self.criterion(preds_mean, y)

        self.metric.update(preds_mean, y)
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
            
        videos = []
        for i in range(REPEAT_NUM):
            start_frame = (D - (clip_length - 1) * sampling_rate - 1) * i // (REPEAT_NUM - 1)  # mean5
            clip = video[start_frame + np.arange(clip_length) * sampling_rate] # (T, H, W, C)
            videos.append(replay_transform(trans, clip))

        return {
            'video': torch.stack(videos, dim=0),  # (5, C, T, H, W)
            'EF': torch.tensor(res['EF'], dtype=torch.float32),
        }

    return transform


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

    def val_loader(self):
        trans = A.ReplayCompose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

        dataset = NpyDataset(
            NPY_PATH,
            df[df['Split'] == 'VAL'],
            transform=video_transform(trans, CLIP_LENGTH, SAMPLING_RATE),
            rets=['video', 'EF'],
            mmap_mode='r',
            cache=False,
        )
        print(f'val dataset size: {len(dataset)}')
        yield DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)

    def test_loader(self):
        trans = A.ReplayCompose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])

        dataset = NpyDataset(
            NPY_PATH,
            df[df['Split'] == 'TEST'],
            transform=video_transform(trans, CLIP_LENGTH, SAMPLING_RATE),
            rets=['video', 'EF'],
            mmap_mode='r',
            cache=False,
        )
        print(f'test dataset size: {len(dataset)}')
        yield DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=False, pin_memory=True)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    task_module = yd.TaskModule(
        model_module=MyModelModule(),
        data_module=MyDataModule(),
        benchmark=True,
    )
    task_module.do()
