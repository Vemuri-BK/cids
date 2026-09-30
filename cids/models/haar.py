"""3D Haar wavelet transform and instruction-conditioned wavelet shrinkage.

Band order along dim 2: index = 4*d + 2*h + w with 0=low, 1=high per axis, i.e.
0=LLL, 1=LLH, 2=LHL, 3=LHH, 4=HLL, 5=HLH, 6=HHL, 7=HHH.
"""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_NAMES = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]


class HaarDWT3D(nn.Module):
    """x: (B, C, D, H, W) with even D, H, W  ->  (B, C, 8, D/2, H/2, W/2)."""

    def __init__(self):
        super().__init__()
        lo = torch.tensor([1.0, 1.0]) / 2 ** 0.5
        hi = torch.tensor([1.0, -1.0]) / 2 ** 0.5
        k = [torch.einsum("i,j,k->ijk", a, b, c) for a, b, c in itertools.product([lo, hi], repeat=3)]
        self.register_buffer("w", torch.stack(k).unsqueeze(1))  # (8, 1, 2, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        y = F.conv3d(x.reshape(B * C, 1, D, H, W), self.w.to(x.dtype), stride=2)
        return y.reshape(B, C, 8, D // 2, H // 2, W // 2)


class HaarIDWT3D(nn.Module):
    """Exact inverse of HaarDWT3D (the Haar filters are orthonormal)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", HaarDWT3D().w.clone())

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, C, K, d, h, w = y.shape
        x = F.conv_transpose3d(y.reshape(B * C, K, d, h, w), self.w.to(y.dtype), stride=2)
        return x.reshape(B, C, 2 * d, 2 * h, 2 * w)


class InstructedShrink(nn.Module):
    """Learnable soft-thresholding of one wavelet band, threshold set by the text.

    sigma = median(|band|) / 0.6745         (robust noise estimate, per sample & channel)
    tau   = softplus(a + MLP(t)) * sigma
    out   = sign(band) * relu(|band| - tau)
    """

    def __init__(self, channels: int, text_dim: int = 256, hidden: int = 64):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.mlp = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, band: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = band.abs().flatten(2).median(dim=-1).values / 0.6745          # (B, C)
        tau = F.softplus(self.a + self.mlp(t)[..., None, None, None]) * sigma[..., None, None, None]
        return torch.sign(band) * F.relu(band.abs() - tau)
