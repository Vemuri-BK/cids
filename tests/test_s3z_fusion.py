"""Tests for scripts/s3z_fusion.py (synthetic; CPU; random SAM weights for the end-to-end run)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3z_fusion as Z  # noqa: E402
from test_s3v_review_agent import _project  # noqa: E402

SHAPE = (60, 56, 44)


def ball(c, r, shape=SHAPE):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def test_fuse_rules():
    p = (30, 28, 20)
    sam = np.zeros(SHAPE, np.float32); sam[ball(p, 6)] = 0.9           # SAM: full tumour
    seg = np.zeros(SHAPE, np.float32); seg[ball(p, 3)] = 0.8            # SegResNet: only the core
    seg[ball((50, 45, 35), 3)] = 0.95                                   # + a far false blob
    m = Z.fuse(sam, seg, p)
    assert m["S"].sum() == ball(p, 6).sum()
    assert m["FA"].sum() == ball(p, 3).sum()                            # (0.9 + 0) / 2 < 0.5 outside the core...
    seg2 = seg.copy(); seg2[ball(p, 6) & ~ball(p, 3)] = 0.3             # ...unless SegResNet is mildly positive there
    assert Z.fuse(sam, seg2, p)["FA"].sum() == ball(p, 6).sum()
    fm = m["FM"]
    assert fm[p] and not fm[50, 45, 35]                                 # union, but only the piece at the click
    empty = Z.fuse(np.zeros(SHAPE, np.float32), seg, p)
    assert empty["S"].sum() == 0 and empty["FM"].sum() == ball(p, 3).sum()


def test_choose():
    pr = pd.DataFrame(dict(S=[0.1, -0.2, 0.05], FA=[0.2, -0.1, -0.3], FM=[-0.1, -0.05, 0.04]))
    assert list(Z.choose(pr)) == ["FA", "Cn", "S"]


def test_end_to_end(tmp_path):
    base, data = _project(tmp_path)
    for d in ("tc", "vc"):
        for f in (tmp_path / d).glob("*.npz"):
            lab = np.load(f)["lab"]
            np.savez(f, lab=lab, seg=np.where(lab > 0, 230, 40).astype(np.uint8))
    folds = ["--sam_fold_ckpt", str(tmp_path / "f{fold}" / "best.pt"), "--folds", str(tmp_path / "s3k" / "folds.csv")]
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"] + folds, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "s3n")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    z = [sys.executable, str(ROOT / "scripts" / "s3z_fusion.py"), "--s3v_out", str(tmp_path / "out"),
         "--out", str(tmp_path / "z")]
    r = subprocess.run(z + ["--step", "analyse", "--folds", str(tmp_path / "s3k" / "folds.csv")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "run --step masks first" in r.stdout + r.stderr
    r = subprocess.run(z + data + ["--step", "masks", "--random_weights"] + folds, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "parity S vs S3v" in r.stdout and "MASKS DONE" in r.stdout
    det = pd.read_csv(tmp_path / "z" / "detail.csv")
    assert len(det) == 24 and set(Z.FUS) <= set(det.columns)
    assert (det.S_dice - det.S3v_S_dice).abs().max() < 1e-6
    assert json.loads((tmp_path / "z" / "decision_rule.json").read_text())["masks"]["F_avg"].startswith("(P_sam")
    r = subprocess.run(z + ["--step", "analyse", "--folds", str(tmp_path / "s3k" / "folds.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    v = json.loads((tmp_path / "z" / "rules.json").read_text())
    assert {"R7", "R7a"} <= set(v["rules"]["train_oof"])
    s = pd.read_csv(tmp_path / "z" / "summary.csv")
    assert {"A2", "A2_ext", "A4", "always F_avg"} <= set(s.policy)


def test_requires_oof_s3v(tmp_path):
    base, data = _project(tmp_path)
    for d in ("tc", "vc"):
        for f in (tmp_path / d).glob("*.npz"):
            lab = np.load(f)["lab"]
            np.savez(f, lab=lab, seg=(lab > 0).astype(np.uint8) * 200)
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]                            # NOT out-of-fold
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3z_fusion.py"), "--s3v_out", str(tmp_path / "out"),
                        "--out", str(tmp_path / "z"), "--step", "masks", "--random_weights",
                        "--sam_fold_ckpt", "x{fold}", "--folds", str(tmp_path / "s3k" / "folds.csv")] + data,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "OUT-OF-FOLD" in r.stdout + r.stderr
