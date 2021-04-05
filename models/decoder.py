"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock


class Integrator(nn.Module):
    def __init__(self, C_in, C_out, norm='none', activ='none'):
        super().__init__()
        self.integrate_layer = ConvBlock(C_in, C_out, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, x, integrated):
        out = self.integrate_layer(integrated)
        out = torch.cat([x, out], dim=1)

        return out


class Decoder(nn.Module):
    def __init__(self, layers, skip_idx=None, skip_layer=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.skip_idx = skip_idx
        self.skip_layer = skip_layer

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, last, skip=None):
        for i, layer in enumerate(self.layers):
            if i == self.skip_idx:
                last = self.skip_layer(last, integrated=skip.flatten(1, 2))

            if i == 0:
                last = last.flatten(1, 2)
            last = layer(last)

        return self.out(last)


def dec_builder(C, C_out, n_experts, norm='IN', activ='relu', pad_type='reflect', out='sigmoid'):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type)
    IntegrateBlk = partial(Integrator, norm='none', activ='none')

    layers = [
        ConvBlk(C*8*n_experts, C*8, 1, 1, 0, norm="none", activ="none"),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ResBlk(C*8, C*8, 3, 1),
        ConvBlk(C*8, C*4, 3, 1, 1, upsample=True),   # 32x32
        ConvBlk(C*8, C*2, 3, 1, 1, upsample=True),   # 64x64
        ConvBlk(C*2, C*1, 3, 1, 1, upsample=True),   # 128x128
        ConvBlk(C*1, C_out, 3, 1, 1)
    ]
    skip_idx = 5
    skip_layer = IntegrateBlk(C*4*n_experts, C*4)

    return Decoder(layers, skip_idx, skip_layer, out=out)
