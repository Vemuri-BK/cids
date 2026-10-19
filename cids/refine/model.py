"""Residual boundary refiner around a frozen mask M0 (design: docs/review_2026-10-14/04_...txt).

    z_final = z0 + allow * g * delta,     z0 = logit(0.25 + 0.5 * M0) = +-log(3)

delta (signed correction) is ZERO-initialised and g (spatial gate) starts at 0.5, so before training the
output is exactly M0 (threshold 0.5). `allow` limits corrections to a band around M0's boundary
(|signed distance| <= R voxels): no new blobs far away, no holes deep inside.

Inputs: image (3 ch: pre, post1, post1-pre) + M0 + clipped signed distance of M0 (no finder map: SegResNet's
maps are in-sample on training cases).

Two variants with IDENTICAL architecture and parameter count except one fixed/learned layer:
  mode="freq"     the image is split by a fixed 3D Haar transform into LLL (context, 3 ch) and 7 signed
                  high-frequency bands x 3 ch (detail, 21 ch) at half resolution
  mode="freq_lll" the same Haar split, but the 7 high-frequency bands are zeroed (low-frequency context only;
                  identical parameter count): tests whether the high-frequency detail adds anything
  mode="spatial"  the same 24 half-resolution channels come from a learned stride-2 convolution
                  (spatial-only twin: same capacity, no frequency decomposition)
gate=False drops the output gate (ablation):  z_final = z0 + allow * delta  (delta still zero-initialised).
Detail features enter the encoder through a context-guided gate (high frequency is evidence, not a label):
    c = proj([down(features), ctx]);  h = proj(detail);  e2 = c + sigmoid(W[c, h]) * h
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from cids.models.haar import HaarDWT3D

LOG3 = math.log(3.0)


def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01))


class BoundaryRefiner(nn.Module):
    def __init__(self, mode: str = "freq", width: int = 16, img_ch: int = 3, gate: bool = True):
        super().__init__()
        assert mode in ("freq", "freq_lll", "spatial")
        self.mode, w, self.use_gate = mode, width, gate
        self.n_ctx, self.n_det = img_ch, img_ch * 7
        self.enc1 = block(img_ch + 2, w)
        self.down1 = nn.Conv3d(w, 2 * w, 3, stride=2, padding=1)
        if mode in ("freq", "freq_lll"):
            self.split = HaarDWT3D()
        else:
            self.split = nn.Conv3d(img_ch, self.n_ctx + self.n_det, 2, stride=2)
        self.ctx_proj = nn.Conv3d(2 * w + self.n_ctx, 2 * w, 1)
        self.det_proj = nn.Sequential(nn.Conv3d(self.n_det, 2 * w, 3, padding=1), nn.LeakyReLU(0.01))
        self.fuse = nn.Conv3d(4 * w, 2 * w, 1)
        self.enc2 = block(2 * w, 2 * w)
        self.down2 = nn.Conv3d(2 * w, 4 * w, 3, stride=2, padding=1)
        self.enc3 = block(4 * w, 4 * w)
        self.up2 = nn.ConvTranspose3d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = block(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose3d(2 * w, w, 2, stride=2)
        self.dec1 = block(2 * w, w)
        self.delta = nn.Conv3d(w, 1, 1)
        self.gate = nn.Conv3d(w, 1, 1) if gate else None
        for m in (self.delta, self.gate):          # delta = 0, g = 0.5 at start -> output == M0
            if m is not None:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def bands(self, img):
        s = self.split(img)
        if self.mode == "freq":                    # (B, C, 8, ...): band 0 = LLL
            return s[:, :, 0], s[:, :, 1:].flatten(1, 2)
        if self.mode == "freq_lll":                # high-frequency bands removed
            return s[:, :, 0], torch.zeros_like(s[:, :, 1:].flatten(1, 2))
        return s[:, :self.n_ctx], s[:, self.n_ctx:]

    def forward(self, x):
        """x: (B, 6, D, H, W) = [img 3 ch, M0, signed distance (clipped, /R), allow]; D, H, W divisible by 4."""
        img, m0, sdf, allow = x[:, :3], x[:, 3:4], x[:, 4:5], x[:, 5:6]
        e1 = self.enc1(torch.cat([img, m0, sdf], 1))
        ctx, det = self.bands(img)
        c = self.ctx_proj(torch.cat([self.down1(e1), ctx], 1))
        h = self.det_proj(det)
        e2 = self.enc2(c + torch.sigmoid(self.fuse(torch.cat([c, h], 1))) * h)
        e3 = self.enc3(self.down2(e2))
        u2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        u1 = self.dec1(torch.cat([self.up1(u2), e1], 1))
        z0 = (2 * m0 - 1) * LOG3
        r = self.delta(u1)
        if self.gate is not None:
            r = torch.sigmoid(self.gate(u1)) * r
        return z0 + allow * r
