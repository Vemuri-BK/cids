"""Band groups of the 3D Haar transform.

HaarDWT3D (cids/models/haar.py) gives 8 sub-bands per channel, index = 4*d + 2*h + w (0 = low,
1 = high along that axis): 0 LLL, 1 LLH, 2 LHL, 3 LHH, 4 HLL, 5 HLH, 6 HHL, 7 HHH.
The council has one expert per group:

  E1 shape        LLL               low frequency on all axes: blob shape, context, contrast
  E2 edges        LLH, LHL, HLL     high frequency along ONE axis: edges / walls in one direction
  E3 corners      LHH, HLH, HHL     high frequency along TWO axes: thin structures, corners
  E4 texture      HHH               high frequency on all axes: fine texture and noise

Directional bands matter later: an acquisition with thick slices along one axis has little real
high-frequency content along that axis (the metadata stage will use this).
"""
from __future__ import annotations

from typing import List

import torch

BAND_GROUPS: List[List[int]] = [[0], [1, 2, 4], [3, 5, 6], [7]]
GROUP_NAMES = ["E1_shape_LLL", "E2_edges_1axis", "E3_corners_2axes", "E4_texture_HHH"]


def split_groups(bands: torch.Tensor) -> List[torch.Tensor]:
    """bands (B, C, 8, d, h, w) -> list of 4 tensors (B, C*len(group), d, h, w)."""
    B, C, _, d, h, w = bands.shape
    return [bands[:, :, g].reshape(B, C * len(g), d, h, w) for g in BAND_GROUPS]


def band_energy(bands: torch.Tensor) -> torch.Tensor:
    """Mean absolute coefficient per group -> (B, 4), log-compressed. A cheap global descriptor of how
    much signal each frequency group carries in this patch (the gate sees it)."""
    e = bands.abs().mean(dim=(1, 3, 4, 5))                         # (B, 8)
    g = torch.stack([e[:, idx].mean(1) for idx in BAND_GROUPS], 1)  # (B, 4)
    return torch.log1p(g)
