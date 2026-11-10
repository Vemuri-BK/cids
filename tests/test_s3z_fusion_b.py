"""Tests for scripts/s3z_fusion_b.py (synthetic S3z choices)."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "scripts" / "s3z_fusion_b.py")


def _choices(tmp, split, n, n_sam):
    d = pd.DataFrame(dict(cache_id=[f"{split}_{i:03d}" for i in range(n)],
                          tool2=["S"] * n_sam + ["Cn"] * (n - n_sam),
                          qc=[1.0 - i / n for i in range(n)],
                          S_dice=[0.4] * n, FM_dice=[0.5] * n_sam + [0.3] * (n - n_sam), Cn_dice=[0.7] * n))
    d.to_csv(tmp / f"choices_{split}_deep.csv", index=False)


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def test_freeze_then_val(tmp_path):
    z = tmp_path / "s3z"; z.mkdir()
    _choices(z, "train_oof", 50, 5)
    _choices(z, "val", 20, 3)
    out = tmp_path / "b"
    r = run("--step", "val", "--s3z_out", str(z), "--out", str(out))
    assert r.returncode != 0 and "freeze first" in r.stdout + r.stderr
    r = run("--step", "freeze", "--s3z_out", str(z), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    rule = json.loads((out / "decision_rule.json").read_text())
    assert "R9" in rule and rule["development_train_oof"]["all_clicked"]["n"] == 5
    r = run("--step", "freeze", "--s3z_out", str(z), "--out", str(out))
    assert r.returncode != 0 and "already exists" in r.stdout + r.stderr     # cannot be re-frozen silently
    r = run("--step", "val", "--s3z_out", str(z), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads((out / "val_result.json").read_text())["val"]
    assert v["all_clicked"]["n"] == 3 and abs(v["all_clicked"]["diff"] - 0.1) < 1e-9
    assert v["all_clicked"]["better"] == 3 and v["all_clicked"]["worse"] == 0
    assert v["qc10_clicked"]["n"] == 2                                       # top 10% by qc = first 2 patients
    assert len(pd.read_csv(out / "val_patients.csv")) == 3


def test_refuses_test_folder(tmp_path):
    z = tmp_path / "s3z_test"; z.mkdir()
    r = run("--step", "freeze", "--s3z_out", str(z), "--out", str(tmp_path / "b"))
    assert r.returncode != 0 and "refused" in r.stdout + r.stderr
