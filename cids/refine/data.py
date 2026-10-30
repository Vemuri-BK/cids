"""Refiner data: one region of interest (ROI) per case around the memory-jury mask M0.

prepare_case(): ROI = bounding box of M0 + margin (from M0 only -- never from ground truth), cropped once to
  <work>/<cid>.npz: img (3, ...) float16, m0 uint8, gt uint8 (training / evaluation only), plus meta
  (bbox, full shape, gt voxels outside the ROI) so full-volume Dice is exact without the full volume.
RefinePatchDataset: training patches; M0 is randomly perturbed (smooth signed-distance noise = grow /
  shrink / wobble) so training masks are as imperfect as masks of NEW patients (SegResNet saw the training
  patients). Patches are centred on M0's boundary (75 %) or anywhere in the ROI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from cids.data.cache import load_case
from cids.data.volumes import three_channel

R_ALLOW = 8.0          # corrections only within 8 voxels (8 mm) of M0's boundary
SDF_CLIP = 8.0


def signed_distance(m: np.ndarray) -> np.ndarray:
    """> 0 outside, < 0 inside (voxels)."""
    if not m.any():
        return np.full(m.shape, SDF_CLIP * 4, np.float32)
    return (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)).astype(np.float32)


def channels(img3: np.ndarray, m0: np.ndarray, r_allow: float = R_ALLOW, prob: np.ndarray = None) -> np.ndarray:
    """(6 or 7, ...) float32: img 3 ch, M0, sdf/clip (clipped to [-1, 1]), allow band
    [+ SegResNet probability in [0, 1] (refiner v2): where the finder hesitated]."""
    sdf = signed_distance(m0.astype(bool))
    allow = (np.abs(sdf) <= r_allow).astype(np.float32)
    parts = [img3.astype(np.float32), m0[None].astype(np.float32), np.clip(sdf / SDF_CLIP, -1, 1)[None], allow[None]]
    if prob is not None:
        parts.append(np.asarray(prob, np.float32)[None])
    return np.concatenate(parts)


def roi_channels(d, m0=None) -> np.ndarray:
    """Channels of a prepared ROI file (d = np.load(<work>/<cid>.npz)); the probability is used if stored."""
    m0 = d["m0"].astype(bool) if m0 is None else m0
    prob = d["prob"].astype(np.float32) / 255.0 if "prob" in d.files else None
    return channels(d["img"], m0, prob=prob)


def prepare_case(cid, cache_path, mask_path, work: Path, margin: int = 16, prob_path=None):
    """prob_path: optional SegResNet map (<cid>.npz with 'prob' = probability x 255, full cache grid), cropped
    to the same ROI and stored as 'prob' (refiner v2)."""
    out = work / f"{cid}.npz"
    meta_f = work / f"{cid}.json"
    if out.exists() and meta_f.exists():
        return json.loads(meta_f.read_text())
    m = np.load(mask_path)["mask"].astype(bool)
    img2, lbl = load_case(cache_path)
    gt = lbl > 0
    meta = dict(cid=cid, shape=list(m.shape), gt_total=int(gt.sum()), m0_vox=int(m.sum()))
    if not m.any():
        meta.update(empty=True, bbox=None, gt_out=int(gt.sum()))
    else:
        nz = np.argwhere(m)
        lo = np.maximum(nz.min(0) - margin, 0); hi = np.minimum(nz.max(0) + 1 + margin, m.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        g = gt[sl]
        tmp = work / f"{cid}.partial.npz"
        extra = {}
        if prob_path is not None:
            pr = np.load(prob_path)["prob"]
            if pr.shape != m.shape:
                raise ValueError(f"{cid}: probability map {pr.shape} vs mask {m.shape}")
            extra["prob"] = pr[sl].astype(np.uint8)
        np.savez(tmp, img=three_channel(img2[(slice(None),) + sl]).astype(np.float16), m0=m[sl].astype(np.uint8),
                 gt=g.astype(np.uint8), **extra)
        tmp.replace(out)
        mc = m[sl]
        den = int(mc.sum()) + int(gt.sum())
        meta.update(empty=False, bbox=[lo.tolist(), hi.tolist()], gt_out=int(gt.sum() - g.sum()),
                    m0_dice=float(2 * (mc & g).sum() / den) if den else 1.0)
    meta_f.write_text(json.dumps(meta))
    return meta


def perturb_mask(m: np.ndarray, rng, p: float = 0.8, scale: float = 1.0) -> np.ndarray:
    """With probability p: smooth random boundary change -- shift (grow/shrink up to 2*scale voxels) +
    low-frequency wobble (amplitude 0.5-2 voxels * scale). v1 used p = 0.8, scale = 1."""
    if not m.any() or rng.random() >= p:
        return m
    sdf = signed_distance(m)
    noise = ndimage.gaussian_filter(rng.standard_normal(m.shape).astype(np.float32), sigma=rng.uniform(3, 6))
    noise *= scale * rng.uniform(0.5, 2.0) / (noise.std() + 1e-6)
    out = (sdf + noise + scale * rng.uniform(-2.0, 2.0)) < 0
    return out if out.any() else m


def pad_min(x: np.ndarray, size: int):
    """Pad the last 3 dims to >= size and to a multiple of 4 (symmetric). Returns array, pads."""
    assert size % 4 == 0, f"patch size must be divisible by 4 (got {size})"
    pads = []
    for s in x.shape[-3:]:
        t = max(size, int(np.ceil(s / 4) * 4))
        d = t - s
        pads.append((d // 2, d - d // 2))
    return np.pad(x, [(0, 0)] * (x.ndim - 3) + pads), pads


class RefinePatchDataset(Dataset):
    def __init__(self, cids, work: Path, patch: int = 64, n_patches: int = 4, augment: bool = True, seed: int = 42,
                 p_perturb: float = 0.8, perturb_scale: float = 1.0, error_sampling: bool = False):
        """error_sampling: patch centres 25 % on correctable missed tumour (GT & ~M0 & band), 25 % on correctable
        extra (M0 & ~GT & band), 50 % on M0's boundary (so it also learns where NOT to edit); an empty
        category goes to the boundary. Uses the ground truth of TRAINING cases only."""
        self.cids, self.work, self.patch, self.n, self.augment, self.seed = list(cids), Path(work), patch, n_patches, augment, seed
        self.p_perturb, self.perturb_scale, self.error_sampling = p_perturb, perturb_scale, error_sampling
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.cids)

    def __getitem__(self, i):
        cid = self.cids[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        d = np.load(self.work / f"{cid}.npz")
        m0 = d["m0"].astype(bool)
        if self.augment:
            m0 = perturb_mask(m0, rng, self.p_perturb, self.perturb_scale)
        x = roi_channels(d, m0)
        y = d["gt"][None].astype(np.float32)
        P = self.patch
        x, _ = pad_min(x, P); y, _ = pad_min(y, P)
        mp = x[3] > 0.5
        edge = np.argwhere(mp ^ ndimage.binary_erosion(mp))
        if self.error_sampling:
            band, g = x[5] > 0.5, y[0] > 0.5
            pools = [(0.25, np.argwhere(g & ~mp & band)), (0.25, np.argwhere(mp & ~g & band)), (0.5, edge)]
        else:
            pools = [(0.75, edge)]
        xs, ys = [], []
        for _ in range(self.n):
            u, c, acc = rng.random(), None, 0.0
            for p_, pool in pools:
                acc += p_
                if u < acc:
                    c = pool[rng.integers(len(pool))] if len(pool) else (edge[rng.integers(len(edge))] if len(edge) else None)
                    break
            if c is None:
                c = np.array([rng.integers(s) for s in x.shape[1:]])
            lo = np.clip(c - P // 2, 0, np.array(x.shape[1:]) - P)
            sl = (slice(None),) + tuple(slice(a, a + P) for a in lo)
            xp, yp = x[sl].copy(), y[sl].copy()
            if self.augment:
                for ax in range(3):
                    if rng.random() < 0.5:
                        xp = np.flip(xp, ax + 1); yp = np.flip(yp, ax + 1)
                xp = np.ascontiguousarray(xp); yp = np.ascontiguousarray(yp)
                # intensity jitter on pre and post only, then the subtraction is RECOMPUTED (keeps sub = post - pre)
                xp[:2] = xp[:2] * rng.uniform(0.9, 1.1, (2, 1, 1, 1)).astype(np.float32) \
                    + rng.uniform(-0.05, 0.05, (2, 1, 1, 1)).astype(np.float32)
                xp[2] = xp[1] - xp[0]
            xs.append(xp); ys.append(yp)
        return dict(cid=cid, x=torch.from_numpy(np.stack(xs)), y=torch.from_numpy(np.stack(ys)))
