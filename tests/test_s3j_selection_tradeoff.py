"""Tests for scripts/s3j_selection_tradeoff.py (synthetic data; CPU, seconds)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import s3j_selection_tradeoff as J  # noqa: E402
from test_s3i_subset_jury import synth_table  # noqa: E402

C = json.loads((ROOT / "configs" / "s3i_subset_jury.json").read_text())
C_FAST = {**C, "hgb_grid": C["hgb_grid"][:1], "inner_folds": 2, "outer_folds": 2,
          "prob_threshold_grid": {"start": 0.1, "stop": 0.9, "step": 0.2}}
F = C["finder_features"]


def with_empty(df, n_empty=2):
    """Add patients that have no candidate: one k = -1 row each (as S3f / S3g write them)."""
    extra = [dict(cache_id=f"ispy2_{900 + j}", k=-1, gt_vox=5000.0, inter_vox=0, size_vox=0) for j in range(n_empty)]
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def test_no_candidate_patients_keep_real_id_and_k0(tmp_path):
    p = tmp_path / "train_features.csv"
    with_empty(synth_table(n_pat=10)).to_csv(p, index=False)
    pats = J.load_patients(p, F, C["forbidden_features"])
    empty = [q for q in pats if q["cid"].startswith("ispy2_9")]
    assert len(pats) == 12 and len(empty) == 2
    assert all(q["K"] == 0 and q["status"] == "no_candidate" and q["tgt"]["G"] == 5000.0 for q in empty)
    assert all(J.keep_all(q).size == 0 and J.rel20(q).size == 0 for q in empty)


def test_mixed_real_and_placeholder_rows_rejected(tmp_path):
    df = synth_table(n_pat=3)
    bad = pd.concat([df, pd.DataFrame([dict(cache_id=df.cache_id.iloc[0], k=-1, gt_vox=df.gt_vox.iloc[0],
                                            inter_vox=0, size_vox=0)])], ignore_index=True)
    p = tmp_path / "train_features.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        J.load_patients(p, F, C["forbidden_features"])


def test_rel20():
    p = dict(K=4, size=np.array([1000., 300., 199., 200.]))
    assert J.rel20(p).tolist() == [True, True, False, True]


def test_patient_weighting(monkeypatch, tmp_path):
    seen = {}

    class Rec:
        def fit(self, X, y, sample_weight=None):
            seen["w"] = sample_weight
            return self

    monkeypatch.setattr(J.S, "_hgb", lambda *a, **k: Rec())
    p = tmp_path / "train_features.csv"
    synth_table(n_pat=8).to_csv(p, index=False)
    pats = J.load_patients(p, F, C["forbidden_features"])
    J.BinaryJury("B0w", F, C_FAST, weighted=True)._fit(pats, C_FAST["hgb_grid"][0], 0)
    w = seen["w"]
    i = 0
    for q in pats:                                   # each patient's rows sum to 1
        assert abs(w[i:i + q["K"]].sum() - 1) < 1e-12
        i += q["K"]
    J.BinaryJury("B0", F, C_FAST, weighted=False)._fit(pats, C_FAST["hgb_grid"][0], 0)
    assert np.all(seen["w"] == 1.0)


def test_threshold_rules():
    tab = pd.DataFrame(dict(config_index=[0] * 5 + [1] * 2, t=[.1, .3, .5, .7, .9, .5, .7],
                            dice=[.800, .810, .812, .806, .790, .811, .805],
                            fail01=[10, 14, 20, 30, 40, 18, 25], kept_n=[1.9, 1.5, 1.2, 1.05, 1.0, 1.2, 1.0]))
    best = J.pick_best(tab)
    assert best.t == .5 and best.config_index == 0
    assert J.pick_match(tab, 0, 1.45).t == .3
    safe = J.pick_safe(tab, 0, 0.005)               # within 0.005 of 0.812: t = .3 (.810), .5, .7 (.806 > .807? no)
    assert safe.t == .3 and safe.fail01 == 14


def test_rule_check():
    s = pd.DataFrame([dict(policy="U0", dice=0.810, dice_lt_0_1=15, dice_lt_0_5=60)] +
                     [dict(policy=v, dice=0.812, dice_lt_0_1=30, dice_lt_0_5=58) for v in J.FAIR_B])
    r, ok = J.rule_check(s, 0.005, 0.25, 0.10)
    assert ok and r.passes.all()
    s.loc[s.policy == "B0_safe", "dice_lt_0_1"] = 18      # a permissive binary jury nearly matches U0
    r, ok = J.rule_check(s, 0.005, 0.25, 0.10)
    assert not ok and not r.set_index("versus").loc["B0_safe"].passes


def test_end_to_end_synthetic(tmp_path):
    tr, va = tmp_path / "train_features.csv", tmp_path / "val_features.csv"
    with_empty(synth_table(n_pat=40, seed=1)).to_csv(tr, index=False)
    v = with_empty(synth_table(n_pat=16, seed=2), 1)
    v["cache_id"] = v.cache_id.str.replace("duke_", "nact_").str.replace("ispy2_9", "ispy2_8")
    v.to_csv(va, index=False)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(C_FAST))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3j_selection_tradeoff.py"), "--train", str(tr),
                        "--val", str(va), "--config", str(cfg), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "VERDICT" in r.stdout
    s = pd.read_csv(out / "policies_summary.csv")
    assert set(s.policy) >= {"keep_all", "rel20", "B0", "B0w", "B0_match", "B0_safe", "U0", "ORACLE"}
    pp = pd.read_csv(out / "per_patient.csv")
    e = pp[pp.cache_id.str.startswith("ispy2_9")]
    assert len(e) == 2 and (e.K == 0).all() and (e.U0_dice == 0).all() and (e.keep_all_dice == 0).all()
    ka = s[(s.split == "train_oof") & (s.policy == "keep_all")].iloc[0]
    orc = s[(s.split == "train_oof") & (s.policy == "ORACLE")].iloc[0]
    assert orc.dice >= s[s.split == "train_oof"].dice.max() - 1e-12 and ka.kept_n >= s.kept_n.min()


def test_refuses_test_files(tmp_path):
    t = tmp_path / "test_features.csv"
    synth_table(n_pat=4).to_csv(t, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3j_selection_tradeoff.py"), "--train", str(t),
                        "--val", str(t), "--config", str(ROOT / "configs" / "s3i_subset_jury.json"),
                        "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0 and "test file" in (r.stdout + r.stderr)
