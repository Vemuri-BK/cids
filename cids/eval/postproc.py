"""Connected-component post-processing (26-connectivity), as used for every reported SegResNet number.

rel20: keep every component whose volume is at least 20% of the largest component.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCT = np.ones((3, 3, 3))


def rel20(mask: np.ndarray, frac: float = 0.2) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= frac * sizes.max()) + 1
    return np.isin(lab, keep)
