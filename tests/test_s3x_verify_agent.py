"""Tests for scripts/s3x_verify_agent.py (synthetic S3z choices; CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3x_verify_agent as X  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "s3x_verify_agent.py")


def _choices(n, seed, prefix):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(dict(cache_id=[f"{prefix}_{i:04d}" for i in range(n)]))
    for c in X.VER:
        if c not in X.DELIV:
            d[c] = rng.random(n)
    bad = rng.random(n) < 0.3                               # learnable: bad results have a small 'cn_log_ml'
    d["cn_log_ml"] = np.where(bad, rng.uniform(-6, -4, n), rng.uniform(0, 2, n))
    d["qc"] = rng.random(n)
    d.loc[:3, "risk_baseline"] = np.inf                      # empty automatic masks (S3n: reviewed first)
    d.loc[:3, "qc"] = np.inf
    d["tool2"] = np.where(rng.random(n) < 0.2, "S", "Cn")
    d["B_dice"] = rng.uniform(0, 1, n)
    d["Cn_dice"] = np.where(bad, 0.2, 0.8)
    d["FM_dice"] = np.where(bad, 0.6, 0.7)
    for t in ("Cn", "FM"):
        d[f"{t}_ml"], d[f"{t}_diam"], d[f"{t}_enh"] = rng.uniform(1, 20, n), rng.uniform(5, 40, n), rng.random(n)
    return d


def test_prepare_delivered_mask():
    d = pd.DataFrame(dict(cache_id=["a", "b"], tool2=["S", "Cn"], Cn_dice=[0.1, 0.9], FM_dice=[0.7, 0.3],
                          Cn_ml=[1.0, 2.0], FM_ml=[3.0, 4.0], Cn_diam=[1, 2], FM_diam=[3, 4], Cn_enh=[0, 0],
                          FM_enh=[1, 1]))
    p = X.prepare(d)
    assert list(p.deliv_dice) == [0.7, 0.9] and list(p.alt_dice) == [0.1, 0.3]
    assert list(p.d_is_fm) == [1, 0] and list(p.y) == [0, 0] and np.isclose(p.d_log_ml[0], np.log(3.001))


def test_end_to_end(tmp_path):
    z = tmp_path / "s3z"; z.mkdir()
    tr = _choices(600, 0, "duke")
    tr.to_csv(z / "choices_train_oof_deep.csv", index=False)
    _choices(80, 1, "nact").to_csv(z / "choices_val_deep.csv", index=False)
    pd.DataFrame(dict(cache_id=tr.cache_id, outer_fold=np.arange(600) % 5)).to_csv(tmp_path / "folds.csv", index=False)
    r = subprocess.run([sys.executable, SCRIPT, "--s3z_out", str(z), "--folds", str(tmp_path / "folds.csv"),
                        "--out", str(tmp_path / "x")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    v = json.loads((tmp_path / "x" / "rules.json").read_text())
    t = v["results"]["train_oof"]
    assert t["n_clicked"] == 60 and t["R8b"]["n_flagged"] == 12
    assert t["R8"]["auroc"] > 0.9 and v["R8_primary"]                   # the signal is learnable
    assert t["R8b"]["diff"] > 0                                         # bad blobs switched to the better F_max
    assert t["oracle_fallback_dice"] >= t["R8b"]["dice_fallback"] - 1e-12
    assert json.loads((tmp_path / "x" / "decision_rule.json").read_text())["R8"].startswith("PRIMARY")


def test_refuses_test(tmp_path):
    z = tmp_path / "s3z_test"; z.mkdir()
    r = subprocess.run([sys.executable, SCRIPT, "--s3z_out", str(z), "--folds", "x", "--out", str(tmp_path / "x")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refused" in r.stdout + r.stderr
