"""Tests for scripts/s3v_review_agent.py (synthetic; CPU; the end-to-end test uses random SAM weights)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3v_review_agent as V  # noqa: E402

SHAPE = (60, 56, 44)


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def test_offset_click():
    gt = ball(SHAPE, (30, 28, 20), 7)
    p = (30, 28, 20)
    q = V.offset_click(gt, p, "duke_001")
    d = np.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
    assert 3 <= d <= 5 and gt[q] and q == V.offset_click(gt, p, "duke_001")
    tiny = np.zeros(SHAPE, bool); tiny[5, 5, 5] = True
    assert V.offset_click(tiny, (5, 5, 5), "x") == (5, 5, 5)


def test_select_and_outcome():
    sel = V.select(np.array([0.1, 0.9, 0.9, 0.5]), np.array(["d", "c", "b", "a"]), 0.5)
    assert list(sel) == [False, True, True, False]                          # ties by cache_id
    d = pd.DataFrame(dict(B_dice=[0.0, 0.8], Cn_dice=[0.7, 0.6], S_dice=[0.5, 0.9],
                          B_ml=[0.0, 2.0], Cn_ml=[1.0, 1.5], S_ml=[1.2, 2.2], E_ml=[1.0, 2.0]))
    dice, verr = V.outcome(d, np.array([True, False]), "Cn")
    assert list(dice) == [0.7, 0.8] and list(verr) == [0.0, 0.0]
    dice, _ = V.outcome(d, np.array([True, True]), pd.Series(["S", "Cn"]))
    assert list(dice) == [0.5, 0.6]


def _project(tmp):
    root = tmp / "cache"; (root / "train").mkdir(parents=True); (root / "val").mkdir()
    for d in ("tc", "vc", "s3k", "s3n"):
        (tmp / d).mkdir()
    rng = np.random.default_rng(0)
    sp, pp, tab, folds, rev = [], [], [], [], []
    specs = [("train", f"duke_{i:03d}", i % 3) for i in range(8)] + [("val", f"nact_{i:03d}", i % 3) for i in range(4)]
    for split, cid, kind in specs:
        img = (0.2 + 0.02 * rng.standard_normal((2,) + SHAPE)).astype(np.float32)
        gt = ball(SHAPE, (30, 28, 20), 6)
        img[1][gt] += 0.6
        lab = np.zeros(SHAPE, np.int32)
        if kind != 2:
            lab[ball(SHAPE, (30, 28, 20), 5)] = 3                              # tumour blob
        lab[ball(SHAPE, (50, 45, 35), 3)] = 4                                  # false blob
        keep = [3] if kind == 0 else [4]                                        # kind 1/2: B0_rad wrong pick
        np.savez(root / split / f"{cid}.npz", image=img.astype(np.float16), label=gt[None].astype(np.uint8))
        np.savez(tmp / ("tc" if split == "train" else "vc") / f"{cid}.npz", lab=lab)
        sp.append(dict(patient_id=cid.upper(), official="train" if split == "train" else "test", split=split,
                       cache_id=cid, collection=cid.split("_")[0].upper()))
        s = "train_oof" if split == "train" else "val"
        B = np.isin(lab, keep)
        pp.append(dict(split=s, cache_id=cid, B0_rad_dice=V.M.dice_count(B, gt), B0_rad_subset="+".join(map(str, keep))))
        for k in np.unique(lab[lab > 0]):
            tab.append(dict(split=s, cache_id=cid, k=int(k), seg_prob_mean=0.7))
        if split == "train":
            folds.append(dict(cache_id=cid, outer_fold=len(folds) % 2))
        rev.append(dict(split=s, cache_id=cid, is_empty=0, risk_baseline=0.9 if kind else 0.1,
                        **{c: float(rng.random()) for c in V.BASE}))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    pd.DataFrame(pp).to_csv(tmp / "s3k" / "per_patient.csv", index=False)
    t = pd.DataFrame(tab)
    t[t.split == "train_oof"].drop(columns="split").to_csv(tmp / "s3k" / "train_oof_table.csv", index=False)
    t[t.split == "val"].drop(columns="split").to_csv(tmp / "s3k" / "val_table.csv", index=False)
    pd.DataFrame(folds).to_csv(tmp / "s3k" / "folds.csv", index=False)
    r = pd.DataFrame(rev)
    r[r.split == "train_oof"].drop(columns="split").to_csv(tmp / "s3n" / "train_oof_review.csv", index=False)
    r[r.split == "val"].drop(columns="split").to_csv(tmp / "s3n" / "val_review.csv", index=False)
    return [sys.executable, str(ROOT / "scripts" / "s3v_review_agent.py"), "--s3k_out", str(tmp / "s3k"),
            "--out", str(tmp / "out")], ["--cache_root", str(root), "--splits", str(tmp / "splits.csv"),
                                         "--train_cases", str(tmp / "tc"), "--val_cases", str(tmp / "vc")]


def test_end_to_end(tmp_path):
    base, data = _project(tmp_path)
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "s3n")], capture_output=True, text=True)
    assert r.returncode != 0 and "run --step outcomes first" in r.stdout + r.stderr
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PARITY OK" in r.stdout and "OUTCOMES DONE" in r.stdout
    out = tmp_path / "out"
    oc = pd.read_csv(out / "outcomes.csv")
    assert len(oc) == 24 and set(oc.click) == {"deep", "offset"}
    d = oc[oc.click == "deep"].set_index("cache_id")
    assert d.loc["duke_001", "Cn_dice"] > 0.5 and d.loc["duke_001", "B_dice"] == 0      # click fixes a wrong pick
    assert d.loc["duke_002", "Cn_dice"] == 0 and d.loc["duke_002", "cn_dist_mm"] > 0     # no tumour blob
    assert (d.E_dice == 1).all() and d.loc["duke_000", "dice_b_cn"] == 1
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "s3n")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads((out / "decision_rule.json").read_text())["stage1"]["features"] == V.BASE
    v = json.loads((out / "rules.json").read_text())
    assert {"R1_primary", "R2_agent_beats_qc", "rules"} <= set(v) and set(v["rules"]) == {"train_oof", "val"}
    c = pd.read_csv(out / "curves.csv")
    assert set(c.policy) == {"QC", "agent", "oracle", "random"} and set(c.click) == {"deep", "offset"}
    full = c[(c.split == "train_oof") & (c.click == "deep") & (c.budget == 1.0) & (c.tool == "Cn")]
    assert np.allclose(full.mean_dice, full.mean_dice.iloc[0])                # everyone clicked: policy irrelevant
    zero = c[(c.split == "train_oof") & (c.click == "deep") & (c.budget == 0.0)]
    assert np.isclose(zero.mean_dice, d.loc[d.split == "train_oof", "B_dice"].mean()).all()
    assert (out / "choices_val_deep.csv").exists()


def test_test_files_refused(tmp_path):
    base, _ = _project(tmp_path)
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "test_s3n")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "test files are refused" in r.stdout + r.stderr
