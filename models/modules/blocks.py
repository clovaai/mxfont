"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from functools import partial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .frn import TLU, FilterResponseNorm1d, FilterResponseNorm2d
from .modules import spectral_norm


# NOTE for nsml pytorch 1.1 docker
class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = 'none'

        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        'none': nn.Identity,
        'in': partial(nn.InstanceNorm2d, affine=False),  # false as default
        'bn': nn.BatchNorm2d,
        'frn': FilterResponseNorm2d
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        'spectral': spectral_norm,
        'none': lambda x: x
    }[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    if norm_dispatch(norm) == FilterResponseNorm2d:
        # use TLU for FRN
        activ = 'tlu'

    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
        "tlu": TLU
    }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d
    }[pad_type.lower()]


class LinearBlock(nn.Module):
    """ pre-active linear block """
    def __init__(self, C_in, C_out, norm='none', activ='relu', bias=True, w_norm='none',
                 dropout=0.):
        super().__init__()
        activ = activ_dispatch(activ, norm)
        if norm.lower() == 'bn':
            norm = nn.BatchNorm1d
        elif norm.lower() == 'frn':
            norm = FilterResponseNorm1d
        elif norm.lower() == 'none':
            norm = nn.Identity
        else:
            raise ValueError(f"LinearBlock supports BN only (but {norm} is given)")
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):
    """ pre-active conv block """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                 activ='relu', bias=True, upsample=False, downsample=False, w_norm='none',
                 pad_type='zero', dropout=0., size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample

        assert ((norm == FilterResponseNorm2d) == (activ == TLU)), "Use FRN and TLU together"

        if norm == FilterResponseNorm2d and size == 1:
            self.norm = norm(C_in, learnable_eps=True)
        else:
            self.norm = norm(C_in)
        if activ == TLU:
            self.activ = activ(C_in)
        else:
            self.activ = activ()
        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """
    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                 norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.,
                 scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                               upsample=upsample, w_norm=w_norm, pad_type=pad_type,
                               dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                               w_norm=w_norm, pad_type=pad_type, dropout=dropout)

        # XXX upsample / downsample needs skip conv?
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out
