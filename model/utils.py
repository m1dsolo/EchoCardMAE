import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import (
    drop_path,
    to_2tuple,
)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))),
                requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        attn_head_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    # def forward(self, x):
    #     B, N, C = x.shape
    #     qkv_bias = None
    #     if self.q_bias is not None:
    #         qkv_bias = torch.cat(
    #             (self.q_bias,
    #              torch.zeros_like(self.v_bias,
    #                               requires_grad=False), self.v_bias))
    #     qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    #     qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[
    #         2]  # make torchscript happy (cannot use tensor as tuple)
    #
    #     q = q * self.scale
    #     attn = (q @ k.transpose(-2, -1))
    #
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #
    #     x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    #
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # (B, N, C*3) -> qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, N, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        cos_attn=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cos_attn:
            self.attn = CosAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(
        self,
        num_frames=16,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        tubelet_size=2
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1])
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed2D(nn.Module):
    """ Image to Patch Embedding """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
