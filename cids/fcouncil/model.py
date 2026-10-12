"""⑥ FreqCouncil: 4 independent band experts + gate -> final mask and uncertainty.

Input x (B, 5, D, H, W) = [pre, post1, post1-pre, finder prob (SegResNet), checker prob (SAM-S2b + adapter)].

    image (3 ch) ──Haar──► 8 bands (B,3,8,D/2,...) ──► groups: E1 LLL | E2 LLH,LHL,HLL | E3 LHH,HLH,HHL | E4 HHH
    maps (2 ch)  ──avg-pool──► half resolution
    expert k: (its bands, maps_half, maps_full) ──► logit_k (B,1,D,H,W)          [independent weights]
    gate: (4 logits, maps) ──► w (B,4,D,H,W)       final = Σ w_k logit_k
    uncertainty = standard deviation of the 4 expert probabilities
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cids.models.haar import HaarDWT3D

from .bands import BAND_GROUPS, split_groups
from .council import CouncilGate
from .experts import BandExpert

N_IMG, N_MAPS = 3, 2


class FreqCouncil(nn.Module):
    def __init__(self, width: int = 24, cond_dim: int = 0):
        super().__init__()
        self.dwt = HaarDWT3D()
        self.experts = nn.ModuleList([BandExpert(N_IMG * len(g) + N_MAPS, width, N_MAPS) for g in BAND_GROUPS])
        self.gate = CouncilGate(len(BAND_GROUPS), N_MAPS, cond_dim=cond_dim)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        img, maps = x[:, :N_IMG], x[:, N_IMG:N_IMG + N_MAPS]
        groups = split_groups(self.dwt(img))
        maps_half = F.avg_pool3d(maps, 2)
        logits = torch.cat([e(groups[k], maps_half, maps)[0] for k, e in enumerate(self.experts)], 1)
        w = self.gate(logits, maps, cond)
        final = (w * logits).sum(1, keepdim=True)
        unc = torch.sigmoid(logits.float()).std(1, keepdim=True)
        return dict(final=final, experts=logits, weights=w, uncertainty=unc)
