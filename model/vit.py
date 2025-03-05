# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial
from typing import Optional

import torch
from torch import nn
from timm.layers.weight_init import trunc_normal_
from .utils import (
    get_sinusoid_encoding_table,
    PatchEmbed,
    Block,
)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage. """
    def __init__(
        self,
        num_frames: int = 16,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        head_drop_rate: float = 0.,
        norm_layer=nn.LayerNorm,
        init_values: float = 0.,
        init_scale: float = 0.,
        tubelet_size: int = 2,
        cos_attn: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            num_frames=num_frames,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn
            ) for i in range(depth)
        ])
        self.fc_norm = norm_layer(embed_dim)
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # (B, C, D, H, W), (DHW/p0p1p2)
    def forward_features(self, x, mask=None):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, DHW/p0p1p2, C)

        x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        if mask is not None:
            x = x[:, ~mask, :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return self.fc_norm(x.mean(1))  # (B, C)

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head_dropout(x)
        x = self.head(x)
        return x

    def get_parameter_groups(self, wd: float = 1e-5):
        parameter_no_wd_groups = {'weight_decay': 0., 'params': []}
        parameter_wd_groups = {'weight_decay': wd, 'params': []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias') or name == 'pos_embed':
                parameter_no_wd_groups['params'].append(param)
            else:
                parameter_wd_groups['params'].append(param)

        return [parameter_no_wd_groups, parameter_wd_groups]

    def freeze(self, patch_embed: bool = False, num_block: int = 0, fc_norm: bool = False):
        # freeze
        for name, param in self.named_parameters():
            if patch_embed and name.startswith('patch_embed.'):
                param.requires_grad = False
            for block in range(num_block):
                # name的前缀是'encoder.:
                if name.startswith(f'blocks.{block}.'):
                    param.requires_grad = False
            if fc_norm and name.startswith('fc_norm.'):
                param.requires_grad = False
