"""Tests for scripts/s3t_review_test.py (synthetic; CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3n_qc_review as N  # noqa: E402
import s3t_review_test as T  # noqa: E402
import s3v_review_agent as V  # noqa: E402


def test_rules_reproduce_s3v(tmp_path):
    from test_s3v_review_agent import _project
    base, data = _project(tmp_path)
    r = subprocess.run(base + data + ["--step", "outcomes", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    r = subprocess.run(base + ["--step", "policy", "--s3n_out", str(tmp_path / "s3n")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "out"
    ref = json.loads((out / "rules.json").read_text())["rules"]
    for split in ("train_oof", "val"):
        ch = pd.read_csv(out / f"choices_{split}_deep.csv")
        mine = T.rules(ch.sort_values("cache_id").reset_index(drop=True))
        for k in ("removed_QC", "diff"):
            assert abs(mine["R1"][k] - ref[split]["R1"][k]) < 1e-9
        for k in ("sam_chosen", "gain_on_chosen", "gain_10pct_vs_always_cn"):
            assert abs(mine["R4"][k] - ref[split]["R4"][k]) < 1e-9
    tr, va = pd.read_csv(out / "choices_train_oof_deep.csv"), pd.read_csv(out / "choices_val_deep.csv")
    assert np.allclose(T.stage2(tr).predict(va[V.POST].fillna(-1)), va.sort_values("cache_id").agent2)


def test_jury_on_frozen_s3o(tmp_path):
    from test_s3o_final_test import _project
    sp, cfg, k, te, tec = _project(tmp_path)
    out = tmp_path / "s3o"
    base = [sys.executable, str(ROOT / "scripts" / "s3o_final_test.py"), "--s3k_out", str(k), "--config", str(cfg),
            "--out", str(out)]
    assert subprocess.run(base + ["--step", "freeze"], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--step", "test", "--cache_root", str(tmp_path / "cache"), "--splits", str(sp),
                               "--test", str(te), "--test_cases", str(tec), "--workers", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    T.ROOT = tmp_path                                          # the frozen jury's config lives in cfg.json here
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "s3i_subset_jury.json").write_text(cfg.read_text())
    f = dict(b0_rad=out / "frozen" / "b0_rad.pkl", s3o_test_table=out / "test" / "test_table.csv",
             s3o_test_per_patient=out / "test" / "per_patient.csv")
    js, ref = T.jury_test(f)
    assert set(js.cache_id) <= set(ref.index) and len(ref) == 9
    pp = ref.reset_index()[["cache_id", "B0_rad_dice"]]
    table = pd.read_csv(f["s3o_test_table"])
    feats = N.patient_features(table, js, pd.DataFrame(dict(cache_id=pp.cache_id, dice_ab=np.nan)), pp)
    assert len(feats) == 9 and feats.is_empty.sum() == 1 and set(N.BASE) <= set(feats.columns)
    pp2 = pd.read_csv(f["s3o_test_per_patient"], dtype={"B0_rad_subset": str})
    pp2.loc[pp2.K > 1, "B0_rad_subset"] = "999"                # a tampered reference must be caught
    pp2.to_csv(tmp_path / "bad.csv", index=False)
    try:
        T.jury_test(dict(f, s3o_test_per_patient=tmp_path / "bad.csv"))
        assert False, "tampered kept sets not detected"
    except SystemExit as e:
        assert "does not reproduce" in str(e)
    T.ROOT = ROOT


def test_analyse(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for click in ("deep", "offset"):
        for i in range(40):
            rows.append(dict(cache_id=f"t_{i:03d}", click=click, collection="DUKE" if i % 2 else "NACT",
                             qc=float(rng.random()), B_dice=0.0 if i < 6 else 0.8, Cn_dice=0.6 if i < 6 else 0.8,
                             S_dice=0.7 if i < 3 else 0.5, FM_dice=0.75 if i < 3 else 0.5,
                             tool2="S" if i < 3 else "Cn"))
    oc = pd.DataFrame(rows)
    va = pd.DataFrame(dict(cache_id=["v1", "v2"], tool2=["S", "Cn"], S_dice=[0.6, 0.5], FM_dice=[0.7, 0.4],
                           Cn_dice=[0.3, 0.8]))
    v = T.analyse(oc, va, tmp_path)
    r9 = v["results"]["R9"]
    assert r9["n"] == 4 and r9["n_val"] == 1 and r9["n_test"] == 3 and r9["worse"] == 0
    assert abs(r9["diff"] - 0.0625) < 1e-9
    assert v["results"]["deep"]["R4"]["sam_chosen"] == 3
    cv = pd.read_csv(tmp_path / "curves_deep.csv")
    assert set(cv.tool) == {"Cn", "agent", "agent_F_max"} and (tmp_path / "per_hospital.csv").exists()


def test_freeze_needs_inputs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3t_review_test.py"), "--step", "freeze",
                        "--s3v_out", str(tmp_path), "--s3n_out", str(tmp_path), "--s3z_out", str(tmp_path),
                        "--s3z_b", str(tmp_path), "--s3o_out", str(tmp_path), "--s3k_out", str(tmp_path),
                        "--sam_ckpt", str(tmp_path / "x.pt"), "--out", str(tmp_path / "t")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "missing frozen inputs" in r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3t_review_test.py"), "--step", "test",
                        "--s3v_out", str(tmp_path), "--s3n_out", str(tmp_path), "--s3z_out", str(tmp_path),
                        "--s3z_b", str(tmp_path), "--s3o_out", str(tmp_path), "--s3k_out", str(tmp_path),
                        "--sam_ckpt", "x", "--turbo_ckpt", "y", "--test_cases", "z", "--out", str(tmp_path / "t")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "freeze first" in r.stdout + r.stderr
