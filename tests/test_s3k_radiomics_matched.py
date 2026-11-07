"""Tests for scripts/s3k_radiomics_matched.py (synthetic MRI cache + component maps; CPU, ~1 min)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3k_radiomics_matched as K  # noqa: E402
from cids.radiomics import add_relative  # noqa: E402

C = json.loads((ROOT / "configs" / "s3i_subset_jury.json").read_text())
C_FAST = {**C, "hgb_grid": C["hgb_grid"][:1], "inner_folds": 2, "outer_folds": 2,
          "prob_threshold_grid": {"start": 0.1, "stop": 0.9, "step": 0.2}}


def _ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r ** 2


def make_split(tmp, prefix, n, official, seed, empty=0):
    """Cache npz (image + label) and component maps; candidate table with the S3f/S3g columns."""
    rng = np.random.default_rng(seed)
    shape = (40, 40, 40)
    cache, cases = tmp / "cache" / official, tmp / f"cases_{prefix}"
    cache.mkdir(parents=True, exist_ok=True); cases.mkdir(parents=True, exist_ok=True)
    rows, split_rows = [], []
    for i in range(n + empty):
        cid = f"{prefix}_{i:03d}"
        img = (0.2 + 0.05 * rng.standard_normal((2,) + shape)).astype(np.float32)
        gt = _ball(shape, (10, 10, 20), 4)
        img[1][gt] += 0.5
        lab = np.zeros(shape, np.int32)
        if i < n:
            K_ = int(rng.integers(1, 4))
            centres = [(10, 10, 20), (30, 30, 20), (30, 10, 10)][:K_]
            for r, c in enumerate(centres):
                comp = _ball(shape, c, 3 + r)
                lab[comp] = r + 3
                inter = int((comp & gt).sum())
                rows.append(dict(cache_id=cid, k=r + 3, rank=r, n_cand=K_, size_vox=int(comp.sum()), inter_vox=inter,
                                 gt_vox=int(gt.sum()), log_seg_ml=np.log(comp.sum() / 1000), rel_size=1.0 / (r + 1),
                                 is_largest=int(r == 0), seg_prob_mean=rng.uniform(.5, 1), seg_prob_max=rng.uniform(.8, 1),
                                 agree=rng.uniform(), sam_soft_ml=1.0))
        else:
            rows.append(dict(cache_id=cid, k=-1, gt_vox=int(gt.sum()), inter_vox=0, size_vox=0))
        np.savez(cache / f"{cid}.npz", image=img.astype(np.float16), label=gt[None].astype(np.uint8))
        np.savez(cases / f"{cid}.npz", lab=lab)
        split_rows.append(dict(cache_id=cid, official=official, split="train" if official == "train" else "val"))
    t = tmp / f"{prefix}_features.csv"
    pd.DataFrame(rows).to_csv(t, index=False)
    return t, cases, split_rows


def test_rel2_matches_add_relative():
    df = pd.DataFrame(dict(cache_id=["a", "a", "b", "c"], k=[1, 2, 1, -1], seg_prob_mean=[.9, .6, .7, np.nan],
                           rad_ring_cnr=[2.0, 0.5, 1.0, np.nan], sam_soft_ml=[1, 1, 1, np.nan],
                           agree=[.5, .5, .5, np.nan]))
    a, b = K.add_rel2(df), add_relative(df)
    for c in K.REL2:
        assert np.allclose(a[c], b[c], equal_nan=True)


def test_rule_routes():
    A = dict(dice=0.7398, dice_lt_0_1=69, dice_lt_0_5=152)
    r = K.rule_check(A, dict(dice=0.745, dice_lt_0_1=69, dice_lt_0_5=167), ci_lo=0.001)
    assert r["route1_better_average"] and r["radiomics_qualify"] and r["fail05_max"] == 167
    r = K.rule_check(A, dict(dice=0.745, dice_lt_0_1=70, dice_lt_0_5=150), ci_lo=0.001)   # failures increased
    assert not r["route1_better_average"] and not r["radiomics_qualify"]
    r = K.rule_check(A, dict(dice=0.735, dice_lt_0_1=51, dice_lt_0_5=160), ci_lo=-0.01)
    assert r["route2_fewer_failures"] and r["fail01_route2_max"] == 51
    r = K.rule_check(A, dict(dice=0.735, dice_lt_0_1=52, dice_lt_0_5=160), ci_lo=-0.01)
    assert not r["route2_fewer_failures"]
    r = K.rule_check(A, dict(dice=0.7398, dice_lt_0_1=40, dice_lt_0_5=168), ci_lo=0.01)   # too many < 0.5
    assert not r["radiomics_qualify"]


def test_end_to_end(tmp_path):
    tr, trc, s1 = make_split(tmp_path, "duke", 24, "train", 0, empty=1)
    va, vac, s2 = make_split(tmp_path, "nact", 10, "train", 1)
    sp = tmp_path / "splits.csv"
    s2 = [dict(r, split="val") for r in s2]
    pd.DataFrame(s1 + s2).to_csv(sp, index=False)
    (tmp_path / "cache" / "val").mkdir(exist_ok=True)
    cfg = tmp_path / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "s3k_radiomics_matched.py"), "--cache_root", str(tmp_path / "cache"),
           "--splits", str(sp), "--train", str(tr), "--train_cases", str(trc), "--val", str(va), "--val_cases", str(vac),
           "--config", str(cfg), "--out", str(out), "--workers", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "VERDICT" in r.stdout
    assert json.loads((out / "decision_rule.json").read_text())["fail_cut"] == 0.25
    s = pd.read_csv(out / "arms_summary.csv")
    assert set(s.policy) == {"B0", "B0_rad"} and set(s.split) == {"train_oof", "val"}
    pp = pd.read_csv(out / "per_patient.csv")
    e = pp[pp.cache_id == "duke_024"]
    assert len(e) == 1 and e.K.iloc[0] == 0 and e.B0_dice.iloc[0] == 0
    rad = pd.read_csv(out / "train_oof_radiomics.csv")
    assert len(rad) == (pd.read_csv(tr).k >= 0).sum() and rad.rad_sphericity.notna().all()
    r2 = subprocess.run(cmd, capture_output=True, text=True)                 # second run reuses the extraction
    assert r2.returncode == 0 and "using cached" in r2.stdout
