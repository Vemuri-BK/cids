"""Tests for scripts/s3o_final_test.py (synthetic cache, S3k run on synthetic data, then freeze + test once)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3o_final_test as O  # noqa: E402
from test_s3k_radiomics_matched import C_FAST, make_split  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "s3o_final_test.py")


def test_oracle_dice():
    assert O.oracle_dice([0, 0], [10, 10], 100) == 0.0
    assert O.oracle_dice([80, 0], [100, 50], 100) == 2 * 80 / 200
    assert np.isnan(O.oracle_dice([], [], 0))


def _project(tmp):
    tr, trc, s1 = make_split(tmp, "duke", 24, "train", 0, empty=1)
    va, vac, s2 = make_split(tmp, "nact", 10, "train", 1)
    te, tec, s3 = make_split(tmp, "ispy2", 8, "val", 2, empty=1)          # test files live in cache/val/
    s2 = [dict(r, split="val") for r in s2]
    s3 = [dict(r, split="test", official="test") for r in s3]
    sp = tmp / "splits.csv"
    pd.DataFrame(s1 + s2 + s3).to_csv(sp, index=False)
    cfg = tmp / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    k = tmp / "s3k"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3k_radiomics_matched.py"), "--cache_root",
                        str(tmp / "cache"), "--splits", str(sp), "--train", str(tr), "--train_cases", str(trc),
                        "--val", str(va), "--val_cases", str(vac), "--config", str(cfg), "--out", str(k),
                        "--workers", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return sp, cfg, k, te, tec


def test_freeze_then_test_once(tmp_path):
    sp, cfg, k, te, tec = _project(tmp_path)
    out = tmp_path / "out"
    base = [sys.executable, SCRIPT, "--s3k_out", str(k), "--config", str(cfg), "--out", str(out)]
    test_args = ["--step", "test", "--cache_root", str(tmp_path / "cache"), "--splits", str(sp), "--test", str(te),
                 "--test_cases", str(tec), "--workers", "1"]
    r = subprocess.run(base + test_args, capture_output=True, text=True)
    assert r.returncode != 0 and "run --step freeze first" in r.stdout + r.stderr
    r = subprocess.run(base + ["--step", "freeze"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "FROZEN" in r.stdout and (out / "test_plan.json").exists()
    man = json.loads((out / "frozen" / "manifest.json").read_text())
    assert set(man["models"]) == {"B0_rad", "B0"} and len(man["models"]["B0_rad"]["features"]) == 24
    r = subprocess.run(base + test_args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "TEST DONE" in r.stdout
    pp = pd.read_csv(out / "test" / "per_patient.csv")
    assert len(pp) == 9 and pp.cache_id.str.startswith("ispy2").all()
    e = pp[pp.K == 0]
    assert len(e) == 1 and e.B0_rad_dice.iloc[0] == 0 and np.isnan(e.B0_rad_hd95.iloc[0])
    assert (pp.oracle_dice >= pp.B0_rad_dice - 1e-9).all()
    s = pd.read_csv(out / "test" / "summary.csv")
    assert {"all", "ISPY2"} <= set(s.group) and set(s.policy) == {"B0_rad", "B0", "rel20", "keep_all"}
    assert len(list((out / "test" / "masks").glob("*.npz"))) == 9
    r = subprocess.run(base + test_args, capture_output=True, text=True)              # second run refused
    assert r.returncode != 0 and "evaluated once" in r.stdout + r.stderr


def test_tampered_model_refused(tmp_path):
    sp, cfg, k, te, tec = _project(tmp_path)
    out = tmp_path / "out"
    base = [sys.executable, SCRIPT, "--s3k_out", str(k), "--config", str(cfg), "--out", str(out)]
    assert subprocess.run(base + ["--step", "freeze"], capture_output=True, text=True).returncode == 0
    f = out / "frozen" / "b0_rad.pkl"
    f.write_bytes(f.read_bytes() + b"x")
    r = subprocess.run(base + ["--step", "test", "--cache_root", str(tmp_path / "cache"), "--splits", str(sp),
                               "--test", str(te), "--test_cases", str(tec), "--workers", "1"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "md5 mismatch" in r.stdout + r.stderr


def test_freeze_detects_val_mismatch(tmp_path):
    sp, cfg, k, te, tec = _project(tmp_path)
    pp = pd.read_csv(k / "per_patient.csv")
    i = pp.index[pp.split == "val"][0]
    pp.loc[i, "B0_rad_dice"] += 0.01
    pp.to_csv(k / "per_patient.csv", index=False)
    r = subprocess.run([sys.executable, SCRIPT, "--s3k_out", str(k), "--config", str(cfg), "--out", str(tmp_path / "o"),
                        "--step", "freeze"], capture_output=True, text=True)
    assert r.returncode != 0 and "differ from S3k" in r.stdout + r.stderr
