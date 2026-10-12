"""⑤ Frequency adapter for SAM-Med3D (no metadata yet).

    SAM input crop (B,1,128^3) ──Haar──► 8 bands (B,8,64^3) ──× band weights w (8)──►
        conv s2 (32^3) ─► conv s2 (16^3) ─► conv s2 (8^3) ─► 1x1 conv (zero-init) ─► residual (B,384,8,8,8)
    SAM embedding' = SAM embedding + residual

- Band weights w_b = 2·sigmoid(a_b): start at 1 (all bands equal), learned. They are the interpretable
  "which frequency bands does the checker need" readout, and the place where acquisition geometry will act
  later (cond -> per-band weights; reserved `cond_dim`).
- Zero-initialised output: at step 0 SAM behaves exactly as SAM-S2b.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cids.models.haar import BAND_NAMES, HaarDWT3D


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True),
                         nn.LeakyReLU(0.01, inplace=True))


class SamFreqAdapter(nn.Module):
    def __init__(self, emb_dim: int = 384, width: int = 32, cond_dim: int = 0):
        super().__init__()
        self.dwt = HaarDWT3D()
        self.band_logit = nn.Parameter(torch.zeros(8))
        self.enc = nn.Sequential(_block(8, width), _block(width, 2 * width), _block(2 * width, 4 * width))
        self.out = nn.Conv3d(4 * width, emb_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.cond = nn.Linear(cond_dim, 8) if cond_dim else None
        if self.cond is not None:
            nn.init.zeros_(self.cond.weight); nn.init.zeros_(self.cond.bias)

    def band_weights(self, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.band_logit[None]                                       # (1, 8)
        if self.cond is not None and cond is not None:
            a = a + self.cond(cond)                                     # (B, 8), metadata stage
        return 2 * torch.sigmoid(a)

    def forward(self, crop: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = self.dwt(crop)[:, 0]                                        # (B, 8, 64, 64, 64)
        b = b * self.band_weights(cond)[..., None, None, None]
        return self.out(self.enc(b))                                    # (B, 384, 8, 8, 8)

    def readout(self) -> dict:
        return dict(zip(BAND_NAMES, self.band_weights().detach().cpu()[0].tolist()))
