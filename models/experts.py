"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from functools import partial

import torch
import torch.nn as nn

from .modules import ResBlock, CBAM


class SingleExpert(nn.Module):
    def __init__(self, layers, skip_idx=None):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.skip_idx = skip_idx

    def forward(self, x):
        ret = {}

        for lidx, layer in enumerate(self.layers):
            x = layer(x)
            if lidx == self.skip_idx:
                ret.update({"skip": x})

        ret.update({"last": x})

        return ret


class Experts(nn.Module):
    def __init__(self, out_shape, experts, skip_shape=None):
        super(Experts, self).__init__()
        self.out_shape = out_shape
        self.skip_shape = skip_shape

        self.n_experts = len(experts)
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        outs = [expert(x) for expert in self.experts]
        last = torch.stack([out["last"] for out in outs], 1)
        ret = {"last": last}
        if self.skip_shape is not None:
            skip = torch.stack([out["skip"] for out in outs], 1)
            ret.update({"skip": skip})

        return ret


def exp_builder(C, n_experts, norm='none', activ='relu', pad_type='reflect', skip_scale_var=False):

    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, scale_var=skip_scale_var)

    experts = [[
            ResBlk(C*4, C*4, 3, 1),
            CBAM(C*4),
            ResBlk(C*4, C*4, 3, 1),
            ResBlk(C*4, C*8, 3, 1, downsample=True),  # 16x16
            CBAM(C*8),
            ResBlk(C*8, C*8)] for _ in range(n_experts)]

    out_shape = (C*8, 16, 16)
    skip_idx = 2
    skip_shape = (C*4, 32, 32)

    experts = [SingleExpert(exp, skip_idx) for exp in experts]

    return Experts(out_shape, experts, skip_shape)
