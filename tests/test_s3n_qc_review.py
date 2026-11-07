"""Tests for scripts/s3n_qc_review.py (synthetic S3k-format tables; CPU)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3n_qc_review as N  # noqa: E402
from test_s3l_sam_matched import C_FAST, make_table  # noqa: E402


def test_review_budget_and_ties():
    risk = np.array([0.1, np.inf, 0.9, 0.9, 0.2])
    cid = np.array(["e", "a", "c", "b", "d"])
    sel = N.review(risk, cid, 0.4)                       # 2 reviews: the empty one, then the first 0.9 by id
    assert sel.tolist() == [False, True, False, True, False]
    assert N.caught(np.array([1, 1, 1, 0, 0]), risk, cid, 0.4) == 1


def test_kept_from_subset():
    assert N.kept_from_subset("3+5") == {3, 5} and N.kept_from_subset(np.nan) == set() and N.kept_from_subset("4") == {4}


def _s3k(tmp):
    """S3k-like folder: tables + per_patient.csv produced by S3l's identical B0_rad fit."""
    src = tmp / "s3k"; src.mkdir()
    make_table(src / "train_oof_table.csv", "duke", 40, 0, empty=1)
    make_table(src / "val_table.csv", "nact", 12, 1)
    cfg = tmp / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    o = tmp / "s3l"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3l_sam_matched.py"), "--s3k_out", str(src),
                        "--config", str(cfg), "--out", str(o)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    shutil.copy(o / "per_patient.csv", src / "per_patient.csv")
    return src, cfg


def test_scores_reproduce_and_evaluate(tmp_path):
    src, cfg = _s3k(tmp_path)
    out = tmp_path / "out"
    base = [sys.executable, str(ROOT / "scripts" / "s3n_qc_review.py"), "--s3k_out", str(src), "--config", str(cfg),
            "--out", str(out)]
    r = subprocess.run(base + ["--step", "scores"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "REPRODUCED" in r.stdout
    js = pd.read_csv(out / "jury_scores.csv")
    assert set(js.split) == {"train_oof", "val"} and (js[js.split == "val"].outer_fold == -1).all()
    # synthetic overlap: SAM disagrees more on failures
    pp = pd.read_csv(src / "per_patient.csv")
    rng = np.random.default_rng(0)
    ov = pd.DataFrame(dict(cache_id=pp.cache_id, split=pp.split.map({"train_oof": "train", "val": "val"}),
                           dice_ab=np.clip(np.where(pp.B0_rad_dice < 0.5, 0.3, 0.9) + 0.05 * rng.standard_normal(len(pp)), 0, 1),
                           sam_seconds=0.1))
    ov.to_csv(tmp_path / "overlap.csv", index=False)
    r = subprocess.run(base + ["--step", "evaluate", "--overlap", str(tmp_path / "overlap.csv")], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "VERDICT" in r.stdout
    v = json.loads((out / "decision_rule_check.json").read_text())
    assert v["budget_reviews"] == round(0.1 * 41) and v["caught_sam"] >= v["caught_baseline"]
    s = pd.read_csv(out / "review_summary.csv")
    assert set(s.budget) == {0.05, 0.10, 0.20} and (s.caught + s.missed == s.failures).all()
    tr = pd.read_csv(out / "train_oof_review.csv")
    assert tr.loc[tr.is_empty == 1, "risk_baseline"].eq(np.inf).all()
    assert json.loads((out / "decision_rule.json").read_text())["min_gain_caught"] == 10


def test_scores_detect_mismatch(tmp_path):
    src, cfg = _s3k(tmp_path)
    pp = pd.read_csv(src / "per_patient.csv")
    i = pp.index[(pp.split == "train_oof") & (pp.K >= 2)][0]
    pp.loc[i, "B0_rad_subset"] = "99"
    pp.to_csv(src / "per_patient.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3n_qc_review.py"), "--step", "scores", "--s3k_out",
                        str(src), "--config", str(cfg), "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0 and "REPRODUCTION FAILED" in r.stdout + r.stderr
