"""Read the FADC 2-channel MAMA-MIA cache and build SAM-Med3D input crops.

Cache layout (scripts/preprocess_to_cache.py in FADC-main):
    <root>/{train,val}/<cache_id>.npz
    image: (2, X, Y, Z) float16, RAS, 1 mm isotropic, ch0 = pre-contrast, ch1 = post-contrast 1,
           each channel scaled to [0, 1] by its own 1-99 percentiles
    label: (1, X, Y, Z) uint8

SAM-Med3D (turbo) expects 128^3 crops at 1.5 mm, z-normalised over voxels > 0
of the crop (utils/infer_utils.py: tio.Resample(1.5) -> ToCanonical -> CropOrPad -> ZNormalization).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CACHE_SPACING = 1.0
SAM_SPACING = 1.5
SAM_SIZE = 128
SAM_MODES = ("subtraction", "post1")


# ---------------------------------------------------------------------------
def find_cache_root(explicit: str = "", search_root: str = "/kaggle/input") -> Path:
    if explicit and Path(explicit, "train").is_dir():
        return Path(explicit)
    hits = sorted({p.parent for p in Path(search_root).rglob("train")
                   if p.is_dir() and (p.parent / "val").is_dir() and next(p.glob("*.npz"), None)})
    if len(hits) != 1:
        raise FileNotFoundError(f"Set cache_root explicitly; candidates: {hits}")
    return hits[0]


def list_cases(root: Path) -> List[Tuple[str, str, Path]]:
    """[(cache_id, split, path)]; split is 'train' or 'test' (FADC 'val' = official test)."""
    out = []
    for sub, split in (("train", "train"), ("val", "test")):
        for p in sorted((root / sub).glob("*.npz")):
            out.append((p.stem, split, p))
    return out


def load_case(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    img = d["image"].astype(np.float32)          # (2, X, Y, Z)
    lbl = d["label"]
    lbl = (lbl[0] if lbl.ndim == 4 else lbl).astype(np.uint8)
    return img, lbl


# ---------------------------------------------------------------------------
def resample(img: np.ndarray, lbl: np.ndarray, src: float = CACHE_SPACING,
             dst: float = SAM_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """1 mm -> 1.5 mm. Image trilinear, label trilinear + 0.5 threshold."""
    shape = [max(1, int(round(s * src / dst))) for s in lbl.shape]
    t = torch.from_numpy(img)[None]
    img2 = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)[0].numpy()
    l = torch.from_numpy(lbl.astype(np.float32))[None, None]
    lbl2 = (F.interpolate(l, size=shape, mode="trilinear", align_corners=False)[0, 0] > 0.5).numpy()
    return img2, lbl2.astype(np.uint8)


def sam_channel(img: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (volume, foreground mask) for one SAM input mode."""
    pre, post = img[0], img[1]
    fg = post > 0
    if mode == "post1":
        return post, fg
    if mode == "subtraction":
        # NOTE: pre/post were percentile-scaled separately in the cache, so this is an
        # approximate enhancement map (see configs/cids.yaml).
        return post - pre, fg
    raise ValueError(mode)


def tumor_center(lbl: np.ndarray) -> np.ndarray:
    idx = np.argwhere(lbl > 0)
    if len(idx) == 0:
        return np.array(lbl.shape) // 2
    return ((idx.min(0) + idx.max(0)) // 2).astype(int)


def crop_pad(vol: np.ndarray, center: Iterable[int], size: int = SAM_SIZE,
             fill: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a size^3 cube centred at `center`, zero-padding outside. Returns (crop, origin)."""
    center = np.asarray(center, int)
    origin = center - size // 2
    out = np.full((size,) * 3, fill, dtype=vol.dtype)
    src_lo = np.maximum(origin, 0)
    src_hi = np.minimum(origin + size, vol.shape)
    dst_lo = src_lo - origin
    dst_hi = dst_lo + (src_hi - src_lo)
    if np.all(src_hi > src_lo):
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out, origin


def znorm(crop: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """SAM-Med3D style z-normalisation: statistics over foreground voxels, applied to all."""
    m = fg > 0
    if m.sum() < 10:
        m = np.ones_like(m, dtype=bool)
    v = crop[m]
    return ((crop - v.mean()) / max(float(v.std()), 1e-6)).astype(np.float32)


def crop_centers(lbl15: np.ndarray, n_jitter: int, max_shift: int, seed: int) -> np.ndarray:
    """First centre = tumour bbox centre; then n_jitter random shifts of up to +-max_shift voxels."""
    c0 = tumor_center(lbl15)
    rng = np.random.default_rng(seed)
    cs = [c0] + [c0 + rng.integers(-max_shift, max_shift + 1, size=3) for _ in range(n_jitter)]
    return np.stack(cs).astype(int)


def make_sam_crops(img: np.ndarray, lbl: np.ndarray, centers: np.ndarray,
                   modes=SAM_MODES) -> Dict[str, np.ndarray]:
    """img/lbl already at 1.5 mm. Returns {'<mode>': (K,1,128,128,128) float32, 'label': (K,128,128,128) uint8,
    'origin': (K,3)}."""
    out: Dict[str, list] = {m: [] for m in modes}
    out["label"], out["origin"] = [], []
    fg_full = img[1] > 0
    for c in centers:
        fg_c, origin = crop_pad(fg_full.astype(np.uint8), c)
        for m in modes:
            vol, _ = sam_channel(img, m)
            vc, _ = crop_pad(vol, c)
            out[m].append(znorm(vc, fg_c)[None])
        lc, _ = crop_pad(lbl, c)
        out["label"].append(lc)
        out["origin"].append(origin)
    return {k: np.stack(v) for k, v in out.items()}


def seed_for(cache_id: str, base: int = 42) -> int:
    return (base * 1_000_003 + sum((i + 1) * ord(ch) for i, ch in enumerate(cache_id))) % (2 ** 31)
