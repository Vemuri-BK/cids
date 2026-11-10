"""Tests for scripts/s3w_composite_prompts.py (synthetic; CPU; random SAM weights for the end-to-end run)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3w_composite_prompts as W  # noqa: E402
from test_s3v_review_agent import _project  # noqa: E402

SHAPE = (60, 56, 44)


def ball(c, r, shape=SHAPE):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def test_region_and_prompts():
    prob = np.zeros(SHAPE, np.float32)
    prob[ball((30, 28, 20), 7)] = 0.35                      # weak tumour (below 0.5: no SegResNet blob)
    prob[ball((30, 28, 20), 3)] = 0.45
    prob[ball((50, 45, 35), 3)] = 0.9                       # strong false blob
    lab = np.zeros(SHAPE, np.int32); lab[ball((50, 45, 35), 3)] = 4
    pr = W.build_prompts(prob, lab, (30, 28, 20))
    assert pr["region"][30, 28, 20] and not pr["region"][50, 45, 35]
    assert pr["pos"][0] == (30, 28, 20) and 1 <= len(pr["pos"]) <= 3
    for q in pr["pos"][1:]:
        assert pr["region"][q] and np.sqrt(sum((a - b) ** 2 for a, b in zip(q, (30, 28, 20)))) >= 5
    assert pr["neg"] == [(50, 45, 35)]
    near = W.region_at(prob, (30, 28, 29))                  # 2 mm outside the region -> nearest one
    assert near[30, 28, 20]
    assert not W.region_at(prob, (5, 5, 5)).any()           # nothing within 10 mm
    lab[ball((30, 28, 20), 2)] = 3                          # a blob inside the region is not a negative
    assert W.build_prompts(prob, lab, (30, 28, 20))["neg"] == [(50, 45, 35)]


def test_extra_positives_respect_spacing():
    prob = np.zeros(SHAPE, np.float32)
    reg = ball((30, 28, 20), 10)
    prob[reg] = np.random.default_rng(0).uniform(0.3, 1.0, reg.sum())
    pts = W.extra_positives(prob, reg, (30, 28, 20), n=2)
    allp = [(30, 28, 20)] + pts
    assert len(pts) == 2
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.sqrt(sum((a - b) ** 2 for a, b in zip(allp[i], allp[j]))) >= 5


def test_end_to_end(tmp_path):
    base, data = _project(tmp_path)
    # the S3v project's component maps need SegResNet probabilities ('seg') for S3w
    for d in ("tc", "vc"):
        for f in (tmp_path / d).glob("*.npz"):
            lab = np.load(f)["lab"]
            np.savez(f, lab=lab, seg=np.where(lab > 0, 230, 40).astype(np.uint8))
    folds = ["--sam_fold_ckpt", str(tmp_path / "f{fold}" / "best.pt"), "--folds", str(tmp_path / "s3k" / "folds.csv")]
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"] + folds, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "s3n")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s3v = tmp_path / "out"
    w = [sys.executable, str(ROOT / "scripts" / "s3w_composite_prompts.py"), "--s3v_out", str(s3v),
         "--out", str(tmp_path / "w")]
    r = subprocess.run(w + ["--step", "analyse"], capture_output=True, text=True)
    assert r.returncode != 0 and "run --step masks first" in r.stdout + r.stderr
    r = subprocess.run(w + data + ["--step", "masks", "--random_weights"] + folds, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "parity S1 vs S3v" in r.stdout and "MASKS DONE" in r.stdout
    det = pd.read_csv(tmp_path / "w" / "detail.csv")
    assert len(det) == 24 and {"S1_dice", "SP_dice", "SC_dice"} <= set(det.columns)
    assert (det.S1_dice - det.S3v_S_dice).abs().max() < 1e-6
    pol = pd.read_csv(tmp_path / "w" / "policy_SC" / "outcomes.csv")
    assert len(pol) == 24 and np.allclose(pol.sort_values(["split", "cache_id", "click"]).S_dice.values,
                                          det.sort_values(["split", "cache_id", "click"]).SC_dice.values)
    assert json.loads((tmp_path / "w" / "decision_rule.json").read_text())["config"]["mask_tau"] == 0.25
    r = subprocess.run(w + ["--step", "analyse"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "R5a" in json.loads((tmp_path / "w" / "rules.json").read_text())
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3v_review_agent.py"), "--step", "policy",
                        "--s3k_out", str(tmp_path / "s3k"), "--s3n_out", str(tmp_path / "s3n"),
                        "--out", str(tmp_path / "w" / "policy_SC")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]                # R5b runs on the composite outcomes


def test_requires_oof_s3v(tmp_path):
    base, data = _project(tmp_path)
    for d in ("tc", "vc"):
        for f in (tmp_path / d).glob("*.npz"):
            lab = np.load(f)["lab"]
            np.savez(f, lab=lab, seg=(lab > 0).astype(np.uint8) * 200)
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]                # NOT out-of-fold
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3w_composite_prompts.py"), "--s3v_out",
                        str(tmp_path / "out"), "--out", str(tmp_path / "w"), "--step", "masks", "--random_weights",
                        "--sam_fold_ckpt", "x{fold}", "--folds", str(tmp_path / "s3k" / "folds.csv")] + data,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "OUT-OF-FOLD" in r.stdout + r.stderr
