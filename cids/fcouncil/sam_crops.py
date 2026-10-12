"""Rebuild SAM-Med3D's input crop exactly as S0 did, so the adapter sees the same crop as the cached embedding.

S0 (scripts/s0_sam_embeddings.py, cids.data.cache.make_sam_crops): 1 mm cache -> resample to 1.5 mm ->
128^3 crop centred at `origin + 64` (zero-padded) of the post-contrast channel -> z-normalised over the
crop's foreground (post1 > 0).
"""
from __future__ import annotations

import numpy as np

from cids.data.cache import SAM_SIZE, crop_pad, resample, znorm


def sam_crop(img15: np.ndarray, origin) -> np.ndarray:
    """img15 (2, X, Y, Z) at 1.5 mm, origin (3,) -> (128,128,128) float32 SAM input (post1, z-normalised)."""
    c = np.asarray(origin, int) + SAM_SIZE // 2
    fg, _ = crop_pad((img15[1] > 0).astype(np.uint8), c)
    vc, _ = crop_pad(img15[1], c)
    return znorm(vc, fg)


def sam_crop_from_case(img: np.ndarray, lbl: np.ndarray, origin) -> np.ndarray:
    """img (2, X, Y, Z) at 1 mm (cache) -> SAM input crop at `origin` (1.5 mm grid)."""
    img15, _ = resample(img, lbl)
    return sam_crop(img15, origin)
