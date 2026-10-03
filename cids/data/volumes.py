"""Full-resolution (1 mm) volumes from the FADC 2-channel cache for SegResNet (council member B).

Input channels: [pre, post1, post1 - pre]. Label: binary tumour mask.
Training: each item = one patient -> n random patches (tumour-centred with prob pos_ratio,
otherwise centred on a random body voxel), light augmentation (flips, intensity jitter).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_case


def three_channel(img2: np.ndarray) -> np.ndarray:
    pre, post = img2[0], img2[1]
    return np.stack([pre, post, post - pre]).astype(np.float32)


def pad_to(vol: np.ndarray, size: Tuple[int, int, int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Pad the last 3 dims up to `size` (symmetric). Returns padded array and pad widths."""
    pads = []
    for s, t in zip(vol.shape[-3:], size):
        d = max(t - s, 0)
        pads.append((d // 2, d - d // 2))
    widths = [(0, 0)] * (vol.ndim - 3) + pads
    return np.pad(vol, widths), pads


def unpad(vol: np.ndarray, pads: List[Tuple[int, int]]) -> np.ndarray:
    sl = [slice(None)] * (vol.ndim - 3)
    for (a, b), s in zip(pads, vol.shape[-3:]):
        sl.append(slice(a, s - b))
    return vol[tuple(sl)]


class PatchDataset(Dataset):
    def __init__(self, cases: List[Tuple[str, Path]], patch: int = 96, n_patches: int = 4,
                 pos_ratio: float = 0.67, augment: bool = True, seed: int = 42):
        self.cases, self.patch, self.n = cases, patch, n_patches
        self.pos_ratio, self.augment, self.seed = pos_ratio, augment, seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, path = self.cases[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img2, lbl = load_case(path)
        img = three_channel(img2)
        P = self.patch
        img, _ = pad_to(img, (P, P, P))
        lbl, _ = pad_to(lbl, (P, P, P))
        tum = np.argwhere(lbl > 0)
        body = None
        xs, ys = [], []
        for _ in range(self.n):
            if len(tum) and rng.random() < self.pos_ratio:
                c = tum[rng.integers(len(tum))]
            else:
                if body is None:
                    body = np.argwhere(img[1][::4, ::4, ::4] > 0) * 4
                    if len(body) == 0:
                        body = np.array([[s // 2 for s in lbl.shape]])
                c = body[rng.integers(len(body))]
            lo = np.clip(c - P // 2, 0, np.array(lbl.shape) - P)
            sl = tuple(slice(a, a + P) for a in lo)
            x = img[(slice(None),) + sl].copy()
            y = lbl[sl].copy()
            if self.augment:
                for ax in range(3):
                    if rng.random() < 0.5:
                        x = np.flip(x, ax + 1)
                        y = np.flip(y, ax)
                scale = rng.uniform(0.9, 1.1, size=(3, 1, 1, 1)).astype(np.float32)
                shift = rng.uniform(-0.05, 0.05, size=(3, 1, 1, 1)).astype(np.float32)
                x = x * scale + shift
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
        return dict(cid=cid, image=torch.from_numpy(np.stack(xs)),
                    label=torch.from_numpy(np.stack(ys)[:, None].astype(np.float32)))


class VolumeDataset(Dataset):
    """Whole volumes for validation / test (sliding-window inference)."""

    def __init__(self, cases: List[Tuple[str, Path]], min_size: int = 96):
        self.cases, self.min_size = cases, min_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, path = self.cases[i]
        img2, lbl = load_case(path)
        img, pads = pad_to(three_channel(img2), (self.min_size,) * 3)
        return dict(cid=cid, image=torch.from_numpy(img), label=torch.from_numpy(lbl), pads=pads)
