"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from functools import partial
import torch.nn as nn
from .modules import ConvBlock, GCBlock, CBAM


class StyleEncoder(nn.Module):
    def __init__(self, layers, out_shape):
        super().__init__()

        self.layers = nn.Sequential(*layers)
        self.out_shape = out_shape

    def forward(self, x):
        style_feat = self.layers(x)
        return style_feat


def style_enc_builder(C_in, C, norm='none', activ='relu', pad_type='reflect', skip_scale_var=False):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='none', activ='none'),
        ConvBlk(C*1, C*2, 3, 1, 1, downsample=True),
        GCBlock(C*2),
        ConvBlk(C*2, C*4, 3, 1, 1, downsample=True),
        CBAM(C*4)
    ]

    out_shape = (C*4, 32, 32)

    return StyleEncoder(layers, out_shape)
