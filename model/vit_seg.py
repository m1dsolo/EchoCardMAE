# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .utils import (
    get_sinusoid_encoding_table,
    PatchEmbed2D,
    Block,
)
from einops import rearrange


class ViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage. """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        cos_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.patch_embed.num_patches, embed_dim))
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
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

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

    # (B, C, D, H, W), (N,)
    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, N, C)
        B, _, C = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)

    def forward(self, x):
        return self.forward_features(x)


class ViTDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer=nn.LayerNorm,
        init_values=None,
    ):
        super().__init__()

        self.num_classes = num_classes * patch_size ** 2
        self.embed_dim = embed_dim
        self.patch_size = patch_size

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
                init_values=init_values
            )
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.head(self.norm(x))
        x = rearrange(x, 'B (h w) (p0 p1 c) -> B c (h p0) (w p1)', p0=8, p1=8, h=14, w=14)

        return x


class ViTSeg2D(nn.Module):
    def __init__(
        self,
        num_classes=2,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        cos_attn=False,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer,
            init_values=init_values,
            cos_attn=cos_attn,
        )
        self.decoder = ViTDecoder(
            num_classes=num_classes,
            patch_size=patch_size, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer,
            init_values=init_values,
        )
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim))

    # (B, C, D, H, W), (B, dhw)
    def forward(self, x):
        x = self.encoder(x) # [B, N_vis, C_e]
        x = self.encoder_to_decoder(x) # [B, N_vis, C_d]
        x = self.decoder(x + self.pos_embed) # [B, N_mask, C]

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
