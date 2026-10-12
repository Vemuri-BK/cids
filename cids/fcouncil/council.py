"""⑥ Council gate: combines the 4 independent expert masks into the final mask.

    gate input (full res) = 4 expert logits + finder map + checker map      [+ cond, later]
    w = softmax over experts (per voxel)        final logit = Σ_k w_k · logit_k
The last layer starts at zero -> equal weights (plain average of the experts) at step 0.
`cond_dim` reserves the slot for acquisition-geometry tokens (metadata stage); 0 = image-only.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CouncilGate(nn.Module):
    def __init__(self, n_experts: int = 4, n_maps: int = 2, width: int = 16, cond_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(n_experts + n_maps, width, 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv3d(width, n_experts, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.cond = nn.Linear(cond_dim, n_experts) if cond_dim else None
        if self.cond is not None:
            nn.init.zeros_(self.cond.weight); nn.init.zeros_(self.cond.bias)

    def forward(self, expert_logits: torch.Tensor, maps: torch.Tensor, cond: Optional[torch.Tensor] = None):
        g = self.net(torch.cat([expert_logits, maps], 1))
        if self.cond is not None:
            if cond is None:
                raise ValueError("gate built with cond_dim > 0 needs `cond`")
            g = g + self.cond(cond)[..., None, None, None]
        return torch.softmax(g, 1)
