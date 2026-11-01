import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cids.jury import JURY13, MemoryJury
from cids.radiomics import (RAD_ALL, REL, add_relative, candidate_features, case_features, icc_31, lins_ccc,
                            shape_features)

ROOT = Path(__file__).resolve().parents[1]


def _ball(shape, c, r):
    g = np.indices(shape)
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r ** 2


def _tube(shape, c, r, length):
    g = np.indices(shape)
    return ((g[1] - c[1]) ** 2 + (g[2] - c[2]) ** 2 <= r ** 2) & (np.abs(g[0] - c[0]) <= length // 2)


def test_shape_sphere_vs_tube():
    s = shape_features(_ball((40, 40, 40), (20, 20, 20), 8))
    t = shape_features(_tube((60, 20, 20), (30, 10, 10), 2, 50))
    assert s["rad_elongation"] > 0.9 and s["rad_flatness"] > 0.9
    assert 0.8 < s["rad_sphericity"] <= 1.05 and s["rad_solidity"] > 0.9
    assert t["rad_elongation"] < 0.3 and t["rad_sphericity"] < s["rad_sphericity"]
    assert t["rad_surf_to_vol"] > s["rad_surf_to_vol"]


def _image(shape, lesion, inside_post, ring_post=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pre = 0.2 + 0.02 * rng.standard_normal(shape)
    post = ring_post + 0.02 * rng.standard_normal(shape)
    post[lesion] = inside_post + 0.02 * rng.standard_normal(lesion.sum())
    return np.stack([pre, post]).astype(np.float32)


def test_contrast_bright_lesion_vs_blended():
    shape = (48, 48, 48)
    les = _ball(shape, (24, 24, 24), 7)
    lab = les.astype(np.int32)
    bright = candidate_features(_image(shape, les, 0.9), lab, 1)
    flat = candidate_features(_image(shape, les, 0.2), lab, 1)
    assert bright["rad_ring_cnr"] > 10 and abs(flat["rad_ring_cnr"]) < 2
    assert bright["rad_ring_ratio"] > 3 and 0.8 < flat["rad_ring_ratio"] < 1.25
    assert bright["rad_local_z_sub"] > 10 and bright["rad_rel_enh"] > 0.5
    assert bright["rad_edge_sharpness"] > flat["rad_edge_sharpness"]
    assert all(np.isfinite(bright[c]) for c in bright)


def test_padding_zeros_excluded_from_reference():
    shape = (48, 48, 48)
    les = _ball(shape, (24, 24, 24), 7)
    img = _image(shape, les, 0.9)
    img2 = img.copy()
    img2[:, :, :, 36:] = 0.0                          # padding slab inside the 15 mm box
    a = candidate_features(img, les.astype(np.int32), 1)
    b = candidate_features(img2, les.astype(np.int32), 1)
    assert abs(a["rad_local_z_sub"] - b["rad_local_z_sub"]) / a["rad_local_z_sub"] < 0.2


def test_case_features_distance_to_dominant():
    shape = (64, 40, 40)
    lab = np.zeros(shape, np.int32)
    lab[_ball(shape, (15, 20, 20), 6)] = 1
    lab[_ball(shape, (45, 20, 20), 3)] = 2
    rows = case_features("duke_001", _image(shape, lab > 0, 0.8), lab, [1, 2])
    assert rows[0]["rad_dist_to_dom_mm"] == 0.0 and abs(rows[1]["rad_dist_to_dom_mm"] - 30) < 1
    assert set(RAD_ALL) <= set(rows[0])


def test_add_relative_within_patient_only():
    df = pd.DataFrame(dict(cache_id=["a", "a", "b", "b"], k=[1, 2, 1, -1], seg_prob_mean=[0.9, 0.45, 0.6, 0],
                           sam_soft_ml=[2.0, 1.0, 1.0, 0], agree=[0.8, 0.3, 0.5, 0], mem_knn=[0.9, 0.2, 0.4, 0],
                           rad_ring_cnr=[5.0, 1.0, 2.0, 0]))
    r = add_relative(df)
    assert np.allclose(r.rel_seg_prob, [1.0, 0.5, 1.0, 0.0])
    assert np.allclose(r.rel_agree, [0, -0.5, 0, 0]) and np.allclose(r.rel_ring_cnr, [0, -4, 0, 0])
    assert r.loc[3, REL].abs().sum() == 0              # 'no candidate' row untouched


def test_jury_post_hook_and_default_unchanged():
    from tests.test_jury import _table
    df, D = _table()
    for c in ("sam_soft_ml", "rad_ring_cnr"):
        df[c] = np.abs(np.random.default_rng(1).normal(1, 0.3, len(df)))
    base = MemoryJury(k=5).fit(df, D).proba(df, D)
    again = MemoryJury(k=5, post=None).fit(df, D).proba(df, D)
    assert np.array_equal(base, again)                 # default jury unchanged by the new hook
    j = MemoryJury(k=5, features=JURY13 + REL, post=add_relative).fit(df, D)
    assert np.isfinite(j.proba(df, D)).all()
    old = MemoryJury(k=5).fit(df, D); del old.post     # juries pickled before S3h
    assert np.array_equal(old.proba(df, D), base)


def test_icc_ccc():
    a = np.array([1.0, 2, 3, 4, 5])
    assert icc_31(a, a) == pytest.approx(1.0) and lins_ccc(a, a) == pytest.approx(1.0)
    assert icc_31(a, a + 1) == pytest.approx(1.0)       # consistency ignores a constant shift
    assert lins_ccc(a, a + 1) == pytest.approx(0.8)     # agreement does not: 2*2 / (2 + 2 + 1)
    assert icc_31(a, a[::-1]) < 0


# ---------------------------------------------------------------------------- end to end (synthetic S3f)
def _fake_s3f(tmp: Path, n=(40, 10, 10)):
    cache, s3f = tmp / "cache", tmp / "s3f"
    for d in (cache / "train", cache / "val", s3f / "cases"):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    split_rows = []
    shape = (90, 36, 36)
    for split, m in zip(("train", "val", "test"), n):
        rows, descs = [], []
        for i in range(m):
            cid = f"{['duke', 'ispy2'][i % 2]}_{split}{i:03d}"
            lab = np.zeros(shape, np.int32)
            # sizes random per patient -> size / rank clues are uninformative; only shape + contrast separate
            lab[_ball(shape, (16, 18, 18), int(rng.integers(4, 8)))] = 1          # tumour: round, bright
            lab[_tube(shape, (56, 18, 18), int(rng.integers(2, 4)), int(rng.integers(16, 44)))] = 2  # false
            img = _image(shape, lab == 1, 0.9, seed=i)
            img[1][lab == 2] = 0.35
            gt = (lab == 1).astype(np.uint8)
            off = "val" if split == "test" else "train"
            np.savez_compressed(cache / off / f"{cid}.npz", image=img.astype(np.float16), label=gt[None])
            np.savez_compressed(s3f / "cases" / f"{cid}.npz", seg=np.zeros(shape, np.uint8), lab=lab)
            split_rows.append(dict(cache_id=cid, split=split, official="test" if split == "test" else "train",
                                   collection=cid.split("_")[0].upper()))
            sizes = {k: int((lab == k).sum()) for k in (1, 2)}
            for rank, k in enumerate(sorted(sizes, key=lambda x: -sizes[x])):
                real = int(k == 1)
                rows.append(dict(cache_id=cid, k=k, rank=rank, n_cand=2, size_vox=sizes[k],
                                 inter_vox=sizes[k] if real else 0, gt_vox=int(gt.sum()), on_tumour=real,
                                 rel20_inter=int(gt.sum()), rel20_size=sizes[1] + sizes[2],
                                 log_seg_ml=np.log(sizes[k] / 1000), rel_size=sizes[k] / max(sizes.values()),
                                 is_largest=int(rank == 0), seg_prob_mean=rng.uniform(0.6, 0.9),
                                 seg_prob_max=0.95, agree=rng.uniform(0.2, 0.8), sam_ml=1.0,
                                 sam_soft_ml=rng.uniform(0.5, 2), log_sam_ml=0.0, log_sam_soft_ml=0.0,
                                 log_sam_over_seg=0.0))
                d = rng.normal(0, 1, 16); descs.append(d / np.linalg.norm(d))
        pd.DataFrame(rows).to_csv(s3f / f"{split}_features.csv", index=False)
        np.save(s3f / f"{split}_desc.npy", np.stack(descs).astype(np.float16))
    sp = tmp / "splits.csv"
    pd.DataFrame(split_rows).to_csv(sp, index=False)
    return cache, s3f, sp


def test_s3h_end_to_end(tmp_path):
    cache, s3f, sp = _fake_s3f(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "s3h_radiomics_jury.py"), "--s3f_dir", str(s3f),
           "--cache_root", str(cache), "--splits", str(sp), "--out", str(out), "--workers", "2", "--k", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    v = pd.read_csv(out / "variants.csv").set_index("variant")
    assert {"jury13", "finder_rad", "jury13_rad_rel", "-shape"} <= set(v.index)
    # the false tube is the largest candidate and the finder / SAM clues are random: only radiomics can tell
    assert v.loc["finder_rad", "auc_test"] > 0.95 and v.loc["finder_rad", "test_dice"] == pytest.approx(1.0)
    assert v.loc["jury13", "auc_test"] < 0.9                         # without radiomics it cannot
    rad = pd.read_csv(out / "test_radiomics.csv")
    assert len(rad) == 20 and rad[RAD_ALL].notna().all().all()
    # second run re-uses the cached radiomics tables
    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0 and "using cached test_radiomics.csv" in r2.stdout
