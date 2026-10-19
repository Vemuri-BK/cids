"""Council inputs: image (3 ch) + finder map + checker map, as 96^3 training patches or whole volumes.

Map files (scripts/s5_export_maps.py): <maps_dir>/<cache_id>.npz with
  seg  uint8 (X,Y,Z)  SegResNet-100 probability x 255 (1 mm, same grid as the cache)
  sam  uint8 (X,Y,Z)  SAM-S2b(+adapter) probability x 255, max over SegResNet's candidates (0 elsewhere)

Training cases are seen by SegResNet during its own training, so their finder maps are cleaner than on new
cases. To keep the council from trusting the maps blindly, training patches get MAP AUGMENTATION:
the checker map is dropped (p=0.2), the finder map is dropped (p=0.1), and both get a random gamma
(p^g, g in [0.6, 1.6]) and light noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cids.data.cache import load_case
from cids.data.volumes import pad_to, three_channel


def load_maps(maps_dir, cid: str, shape) -> np.ndarray:
    f = Path(maps_dir) / f"{cid}.npz"
    if not f.exists():
        raise FileNotFoundError(f"no finder/checker maps for {cid} in {maps_dir} (run s5_export_maps.py)")
    d = np.load(f)
    m = np.stack([d["seg"], d["sam"]]).astype(np.float32) / 255.0
    assert m.shape[1:] == tuple(shape), (cid, m.shape, shape)
    return m


def council_input(path: Path, maps_dir, cid: str):
    img2, lbl = load_case(path)
    return np.concatenate([three_channel(img2), load_maps(maps_dir, cid, lbl.shape)]), lbl


def augment_maps(x: np.ndarray, rng) -> np.ndarray:
    """x (5, ...) patch; channels 3, 4 = finder, checker maps."""
    for ch, p_drop in ((3, 0.1), (4, 0.2)):
        if rng.random() < p_drop:
            x[ch] = 0.0
            continue
        g = rng.uniform(0.6, 1.6)
        x[ch] = np.clip(x[ch] ** g + rng.normal(0, 0.02, x[ch].shape).astype(np.float32), 0, 1)
    return x


class CouncilPatchDataset(Dataset):
    def __init__(self, cases: List[Tuple[str, Path]], maps_dir, patch: int = 96, n_patches: int = 4,
                 pos_ratio: float = 0.67, augment: bool = True, map_aug: bool = True, seed: int = 42):
        self.cases, self.maps_dir, self.patch, self.n = cases, maps_dir, patch, n_patches
        self.pos_ratio, self.augment, self.map_aug, self.seed = pos_ratio, augment, map_aug, seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, path = self.cases[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        x5, lbl = council_input(path, self.maps_dir, cid)
        P = self.patch
        x5, _ = pad_to(x5, (P, P, P))
        lbl, _ = pad_to(lbl, (P, P, P))
        tum = np.argwhere(lbl > 0)
        body = None
        xs, ys, ts = [], [], []
        for _ in range(self.n):
            if len(tum) and rng.random() < self.pos_ratio:
                c = tum[rng.integers(len(tum))]
            else:
                if body is None:
                    body = np.argwhere(x5[1][::4, ::4, ::4] > 0) * 4
                    if len(body) == 0:
                        body = np.array([[s // 2 for s in lbl.shape]])
                c = body[rng.integers(len(body))]
            lo = np.clip(c - P // 2, 0, np.array(lbl.shape) - P)
            sl = tuple(slice(a, a + P) for a in lo)
            x = x5[(slice(None),) + sl].copy()
            y = lbl[sl].astype(np.float32)
            if self.augment:
                for ax in range(3):
                    if rng.random() < 0.5:
                        x = np.flip(x, ax + 1); y = np.flip(y, ax)
                x = np.ascontiguousarray(x)
                scale = rng.uniform(0.9, 1.1, size=(3, 1, 1, 1)).astype(np.float32)
                shift = rng.uniform(-0.05, 0.05, size=(3, 1, 1, 1)).astype(np.float32)
                x[:3] = x[:3] * scale + shift
            ts.append(x[3:5].copy())                  # distillation targets: the maps BEFORE map augmentation
            if self.map_aug:
                x = augment_maps(x, rng)
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
        return dict(cid=cid, image=torch.from_numpy(np.stack(xs)), label=torch.from_numpy(np.stack(ys)[:, None]),
                    teacher=torch.from_numpy(np.ascontiguousarray(np.stack(ts))))


class CouncilVolumeDataset(Dataset):
    """Whole volumes (5 channels) for validation / test."""

    def __init__(self, cases: List[Tuple[str, Path]], maps_dir, min_size: int = 96):
        self.cases, self.maps_dir, self.min_size = cases, maps_dir, min_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, path = self.cases[i]
        x5, lbl = council_input(path, self.maps_dir, cid)
        x5, pads = pad_to(x5, (self.min_size,) * 3)
        return dict(cid=cid, image=torch.from_numpy(x5), label=torch.from_numpy(lbl), pads=pads)
