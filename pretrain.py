import os
from functools import partial

import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.optim import  AdamW
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

import yangdl as yd

from model.videomae_vit_align import VideoMAEViT
from loss.infonce_loss import infoNCE_loss
from utils import (
    patchify,
    random_mask_in_roi,
    replay_transform,
    get_roi,
    NpyDataset,
)

yd.env.exp_path = './res/pretrain'
yd.env.seed = 0

NPY_PATH = './npy/echonet/main'
BATCH_SIZE = 16
ACCUM = 1
CACHE = True
NUM_FRAMES = 16
SAMPLING_RATE = 4
MASK_RATIO = 0.75
ALIGN_LOSS_RATIO = 0.2
TEMPERATURE = 0.1

MASK_METHOD = partial(random_mask_in_roi, mask_ratio=MASK_RATIO)
roi = get_roi()
fg_mask = torch.tensor(roi.reshape(-1).copy(), device='cuda:0')
# (14*14,) to (8*14*14,)
fg_mask = fg_mask.tile(8)
# (8*14*14,) to (B, 8*14*14)
fg_mask = fg_mask.expand(BATCH_SIZE, -1)
print(fg_mask.shape)
CKPT_NAME = f'{os.environ["HOME"]}/ckpt/videomae_vit_s.pth'

MEAN = (0.1257, 0.1271, 0.1292)
STD = (0.1951, 0.1957, 0.1974)
x_trans = A.ReplayCompose([
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
# y_trans = None
y_trans = A.ReplayCompose([
    A.MedianBlur(blur_limit=3, p=1.),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


DATASET_PATH = os.path.join(os.environ['HOME'], 'dataset/EchoNet-Dynamic')
df = pd.read_csv(
    os.path.join(DATASET_PATH, 'FileList.csv'),
    index_col='FileName',
    usecols=['FileName', 'EF', 'Split']
)
df = df[df['Split'] == 'TRAIN']

task_hparams = {
    'early_stop_params': {
        'max_stop_epoch': 1600,
    },
    'benchmark': True,
    'deterministic': True,
    'save_ckpt_period': 0,
}

class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = nn.MSELoss(reduction='mean')

        self.model = VideoMAEViT(
            num_frames=NUM_FRAMES,
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.,
            tubelet_size=2,
            cos_attn=False,
        )
        if CKPT_NAME is not None:
            sd = torch.load(CKPT_NAME, weights_only=True)['model']
            sdd = {}
            for key, val in sd.items():
                if key == 'encoder.patch_embed.proj.weight':
                    val = F.interpolate(
                        val,
                        size=(2, 8, 8),
                        mode='trilinear',
                        align_corners=False,
                    )
                    sdd[key] = val
                elif key[:12] == 'decoder.head':
                    continue
                else:
                    sdd[key] = val
            self.model.load_state_dict(sdd, strict=False)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=5e-2, betas=(0.9, 0.95))

        self.loss = yd.ValueMetric()
        self.loss_rec = yd.ValueMetric()
        self.loss_align = yd.ValueMetric()
        self.scaler = GradScaler()

    def train_step(self, batch):
        loss_all = 0
        loss_rec_all = 0
        loss_align_all = 0
        for i in range(0, len(batch['x1']), BATCH_SIZE):
            x1 = batch['x1'][i:i+BATCH_SIZE]
            x2 = batch['x2'][i:i+BATCH_SIZE]
            y1 = batch['y1'][i:i+BATCH_SIZE]
            y2 = batch['y2'][i:i+BATCH_SIZE]
            mask = batch['mask'][i:i+BATCH_SIZE]
            loss, loss_rec, loss_align = self._step(x1, x2, y1, y2, mask)
            loss /= ACCUM
            loss_rec /= ACCUM
            loss_align /= ACCUM
            loss_all += loss.item()
            loss_rec_all += loss_rec.item()
            loss_align_all += loss_align.item()
            self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {'loss': loss_all, 'loss_rec': loss_rec_all, 'loss_align': loss_align_all}
    
    def _step(self, x1, x2, y1, y2, mask):
        y1 = patchify(y1)  # (B, N, C)
        y2 = patchify(y2)  # (B, N, C)
        B, _, C = y1.shape
        y1 = y1[mask].reshape(B, -1, C)
        y2 = y2[mask].reshape(B, -1, C)

        with autocast('cuda'):
            output1, feat1 = self.model(x1, mask, fg_mask)  # (B, N_mask, C)
            output2, feat2 = self.model(x2, mask, fg_mask)  # (B, N_mask, C)

            loss_rec1 = self.criterion(output1, y1)
            loss_rec2 = self.criterion(output2, y2)
            loss_rec = (loss_rec1 + loss_rec2) / 2
            loss_align1 = infoNCE_loss(feat1, feat2, TEMPERATURE)
            loss_align2 = infoNCE_loss(feat2, feat1, TEMPERATURE)
            loss_align = (loss_align1 + loss_align2) / 2
            loss = (1 - ALIGN_LOSS_RATIO) * loss_rec + ALIGN_LOSS_RATIO * loss_align

        self.loss.update(loss, B)
        self.loss_rec.update(loss_rec, B)
        self.loss_align.update(loss_align, B)

        return loss, loss_rec, loss_align

    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            loss_rec=self.loss_rec.val,
            loss_align=self.loss_align.val,
        )


def video_transform(x_trans, y_trans, clip_length: int = 16, sampling_rate: int = 4):
    def transform(d):
        video = d['video']
        T, H, W, C = video.shape

        # maxt
        maxt = (clip_length - 1) * sampling_rate + 1
        if T < maxt:
            video = np.concatenate((video, np.zeros((maxt - T, H, W, C), dtype=video.dtype)), axis=0)
            T = maxt
            
        # random pick video clip
        start_frame1 = np.random.choice(T - (clip_length - 1) * sampling_rate)
        x1 = video[start_frame1 + np.arange(clip_length) * sampling_rate]  # (T, H, W, 3)
        start_frame2 = np.random.choice(T - (clip_length - 1) * sampling_rate)
        x2 = video[start_frame2 + np.arange(clip_length) * sampling_rate]  # (T, H, W, 3)

        return {
            'x1': replay_transform(x_trans, x1),  # (3, T, H, W)
            'x2': replay_transform(x_trans, x2),  # (3, T, H, W)
            'y1': replay_transform(y_trans, x1),  # (3, T, H, W)
            'y2': replay_transform(y_trans, x2),  # (3, T, H, W)
            'mask': torch.from_numpy(MASK_METHOD()),  # (N_mask,)
        }

    return transform


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

    def train_loader(self):
        dataset = NpyDataset(
            NPY_PATH,
            df,
            transform=video_transform(x_trans, y_trans, NUM_FRAMES, SAMPLING_RATE),
            rets=['video'],
            mmap_mode='r',
            cache=CACHE,
        )
        yield DataLoader(
            dataset,
            batch_size=BATCH_SIZE * ACCUM,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )


if __name__ == '__main__':
    task_module = yd.TaskModule(**task_hparams, model_module=MyModelModule(), data_module=MyDataModule())
    res = task_module.do()
