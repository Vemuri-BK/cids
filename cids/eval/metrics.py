"""3D segmentation metrics on binary masks (spacing in mm): Dice, HD95, NSD."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

SPACING = (1.0, 1.0, 1.0)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2.0 * (p & g).sum() / s)


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), border_value=0)


def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing=SPACING):
    """Distances (mm) from each surface voxel of pred to gt's surface, and vice versa.
    Computed inside the joint bounding box (+margin) for speed."""
    both = pred | gt
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, both.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=SPACING) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")          # undefined; reported separately as a miss
    a, b = _surface_distances(p, g, spacing)
    return float(np.percentile(np.concatenate([a, b]), 95))


def nsd(pred: np.ndarray, gt: np.ndarray, tol_mm: float = 2.0, spacing=SPACING) -> float:
    """Normalised surface Dice: fraction of both surfaces within tol_mm of the other."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    a, b = _surface_distances(p, g, spacing)
    return float(((a <= tol_mm).sum() + (b <= tol_mm).sum()) / (len(a) + len(b)))


def all_metrics(pred: np.ndarray, gt: np.ndarray, spacing=SPACING) -> dict:
    return dict(dice=dice(pred, gt), hd95=hd95(pred, gt, spacing), nsd=nsd(pred, gt, 2.0, spacing),
                pred_ml=float(pred.sum() * np.prod(spacing) / 1000), gt_ml=float(gt.sum() * np.prod(spacing) / 1000))
