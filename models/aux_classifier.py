"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from functools import partial
import torch.nn as nn
from .modules import ResBlock, Flatten


class AuxClassifier(nn.Module):
    def __init__(self, layers, heads, cam=False):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(heads)
        self.extractor = None

    def extract_cam(self, x, labels):
        if self.extractor is None:
            self.extractor = self.layers[:2]
        feature_map = self.extractor(x)
        cam_weights = self.heads["comp"].weight[labels]
        cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) * feature_map).mean(1, keepdim=False)
        return cams

    def forward(self, x):
        feat = self.layers(x)

        logit_s = self.heads["style"](feat)
        logit_c = self.heads["comp"](feat)

        return logit_s, logit_c


def aux_clf_builder(in_shape, num_s, num_c, norm='IN', gap_size=8, activ='relu', pad_type='reflect',
                    conv_dropout=0., clf_dropout=0., last_type="linear", w_norm="spectral", cam=False):

    ResBlk = partial(ResBlock, norm=norm, activ=activ, pad_type=pad_type, dropout=conv_dropout)

    assert in_shape[1] == in_shape[2]
    C = in_shape[0]

    layers = [
        ResBlk(C, C*2, 3, 1, downsample=True),
        ResBlk(C*2, C*2, 3, 1),
        nn.AdaptiveAvgPool2d(1),
        Flatten(1),
        nn.Dropout(clf_dropout),
    ]

    heads = {"style": nn.Linear(C*2, num_s), "comp": nn.Linear(C*2, num_c)}

    aux_clf = AuxClassifier(layers, heads, cam=cam)
    return aux_clf
