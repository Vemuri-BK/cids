"""⑥ One independent council expert = a small 3D U-Net for one band group.

Input  (half resolution d = D/2): its group's Haar bands of the image (3 ch x k bands)
                                  + finder map + checker map (avg-pooled to d)
Output (full resolution D):       its own tumour logit (B,1,D,H,W) and its last features (for the gate)
The full-resolution maps are concatenated again before the last layer, so boundaries stay sharp.
Experts share no weights: each learns only from its own frequency group (plus the two maps).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True))


class BandExpert(nn.Module):
    def __init__(self, in_ch: int, w: int = 24, n_maps: int = 2):
        super().__init__()
        self.e1 = conv_block(in_ch, w)                  # d
        self.e2 = conv_block(w, 2 * w)                  # d/2
        self.e3 = conv_block(2 * w, 4 * w)              # d/4
        self.u2 = nn.ConvTranspose3d(4 * w, 2 * w, 2, stride=2)
        self.d2 = conv_block(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose3d(2 * w, w, 2, stride=2)
        self.d1 = conv_block(2 * w, w)
        self.u0 = nn.ConvTranspose3d(w, w // 2, 2, stride=2)                 # back to full resolution
        self.head = nn.Sequential(nn.Conv3d(w // 2 + n_maps, w // 2, 3, padding=1), nn.LeakyReLU(0.01, inplace=True))
        self.out = nn.Conv3d(w // 2, 1, 1)

    def forward(self, bands: torch.Tensor, maps_half: torch.Tensor, maps_full: torch.Tensor):
        x1 = self.e1(torch.cat([bands, maps_half], 1))
        x2 = self.e2(F.max_pool3d(x1, 2))
        x3 = self.e3(F.max_pool3d(x2, 2))
        y = self.d2(torch.cat([self.u2(x3), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        f = self.head(torch.cat([self.u0(y), maps_full], 1))
        return self.out(f), f
