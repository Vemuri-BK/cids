"""Tests for scripts/s3r_sam_memory_matched.py (synthetic S3k-format tables + descriptors; CPU, ~1 min)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3r_sam_memory_matched as R  # noqa: E402
from cids.radiomics import RAD_ALL  # noqa: E402
from test_s3l_sam_matched import C, C_FAST, make_table  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "s3r_sam_memory_matched.py")


def add_geometry(path, seed, dim=32):
    """Add cx/cy/cz + on_tumour to a make_table() table and write unit descriptors (tumours share a direction)."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(path)
    real = df.k >= 0
    df["on_tumour"] = ((df.inter_vox > 0) & real).astype(int)
    for c in R.CENTRE:
        df[c] = np.where(real, rng.uniform(0, 200, len(df)), np.nan)
    df.to_csv(path, index=False)
    t = np.zeros(dim); t[0] = 1
    D = rng.standard_normal((len(df), dim)) * 0.5 + np.outer(df.on_tumour.to_numpy(), t) * 2
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    D[~real.to_numpy()] = 0
    np.save(str(path).replace("_table.csv", "_desc.npy"), D.astype(np.float16))
    df[["cache_id", "k"]].to_csv(str(path).replace("_table.csv", "_features.csv"), index=False)


def project(tmp):
    src = tmp / "s3k"; src.mkdir()
    make_table(src / "train_oof_table.csv", "duke", 40, 0, empty=1)
    make_table(src / "val_table.csv", "nact", 12, 1)
    add_geometry(src / "train_oof_table.csv", 0)
    add_geometry(src / "val_table.csv", 1)
    cfg = tmp / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    return src, cfg


def cmd(src, cfg, out, order_check=True):
    c = [sys.executable, SCRIPT, "--s3k_out", str(src), "--config", str(cfg), "--out", str(out),
         "--train_desc", str(src / "train_oof_desc.npy"), "--val_desc", str(src / "val_desc.npy")]
    if order_check:
        c += ["--train_features", str(src / "train_oof_features.csv"), "--val_features", str(src / "val_features.csv")]
    return c


def test_feature_sets():
    FA = list(C["finder_features"]) + RAD_ALL + R.K.REL2
    assert len(FA) == 24 and len(FA + R.CTRL3) == 27 and len(FA + R.MEM6) == 30
    assert not set(R.CTRL3 + R.MEM6) & set(FA)
    assert not set(FA + R.CTRL3 + R.MEM6) & set(C["forbidden_features"])


def test_cross_fit_never_sees_own_patient():
    """One-hot descriptor per patient; even patients all tumour. A leak would give mem_knn ~1 for them."""
    rows, n = [], 30
    for i in range(n):
        for k in (1, 2):
            rows.append(dict(cache_id=f"p{i:02d}", k=k, on_tumour=int(i % 2 == 0)))
    df = pd.DataFrame(rows)
    D = np.zeros((len(df), n)); D[np.arange(len(df)), np.repeat(np.arange(n), 2)] = 1
    m = R.memory_columns(df, D)
    assert m[df.on_tumour == 1].mem_knn.max() < 0.95
    other = df.iloc[:20].reset_index(drop=True)
    m2 = R.memory_columns(df.iloc[20:].reset_index(drop=True), D[20:], other, D[:20])
    assert m2.mem_knn.max() < 0.95                                         # memory = other patients only


def test_anchor_features():
    df = pd.DataFrame(dict(cache_id=["a"] * 3 + ["b"], k=[1, 2, 3, -1], size_vox=[10, 20, 30, 0],
                           seg_prob_mean=[0.9, 0.5, 0.5, 0], seg_prob_max=[0.8, 1.0, 0.9, 0],
                           cx=[0, 10, 0, np.nan], cy=[0, 0, 20, np.nan], cz=[0, 0, 0, np.nan],
                           mem_knn=[0.1, 0.2, 0.9, 0], mem_probe=[0.3, 0.3, 0.2, 0]))
    c = R.add_control(df)
    assert list(c.dist_to_peak_anchor_mm[:3]) == [10, 0, np.hypot(10, 20)]
    assert (c.peak_anchor_shift_mm[:3] == 10).all() and c.peak_anchor_shift_mm[3] == 0
    assert np.allclose(c.rel_seg_prob_max[:3], [-0.2, 0, -0.1])
    s = R.add_sam_derived(df)
    assert list(s.dist_to_sam_anchor_mm[:3]) == [20, np.hypot(10, 20), 0] and (s.sam_anchor_shift_mm[:3] == 20).all()
    assert np.allclose(s.rel_mem_knn[:3], [-0.8, -0.7, 0]) and np.allclose(s.rel_mem_probe[:3], [0, 0, -0.1])
    t = R.add_sam_derived(df.assign(mem_knn=[0.5, 0.5, 0.1, 0]))            # tie -> larger candidate (k=2)
    assert t.dist_to_sam_anchor_mm[1] == 0


def test_end_to_end_with_parity(tmp_path):
    src, cfg = project(tmp_path)
    ref = tmp_path / "s3l"                                                   # S3l arm B0_rad == S3k B0_rad
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3l_sam_matched.py"), "--s3k_out", str(src),
                        "--config", str(cfg), "--out", str(ref)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    shutil.copy(ref / "per_patient.csv", src / "per_patient.csv")
    out = tmp_path / "out"
    r = subprocess.run(cmd(src, cfg, out), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "VERDICT" in r.stdout and "B0_rad 24, B0_rad_anchor 27, B0_rad_mem 30" in r.stdout
    v = json.loads((out / "decision_rule_check.json").read_text())
    assert v["parity_A_vs_S3k"]["checked"] and v["parity_A_vs_S3k"]["max_abs_diff"] <= 1e-12
    assert {"primary_C_vs_A", "attribution_C_vs_B", "control_B_vs_A", "gain_attributable_to_sam"} <= set(v)
    s = pd.read_csv(out / "arms_summary.csv")
    assert set(s.policy) == {"B0_rad", "B0_rad_anchor", "B0_rad_mem"} and set(s.split) == {"train_oof", "val"}
    assert len(pd.read_csv(out / "paired_comparison.csv")) == 6 and len(pd.read_csv(out / "auc.csv")) == 6
    pp = pd.read_csv(out / "per_patient.csv")
    assert len(pp) == 41 + 12 and pp[pp.cache_id == "duke_040"].B0_rad_mem_dice.iloc[0] == 0
    m = pd.read_csv(out / "train_oof_memory_features.csv")
    assert m.outer_fold.nunique() == 2 and len(m) == len(pd.read_csv(src / "train_oof_table.csv"))
    auc = pd.read_csv(out / "val_memory_features.csv").query("k >= 0")
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(auc.on_tumour, auc.mem_knn) > 0.8                    # informative synthetic memory works
    # tampered reference -> parity failure
    p = pd.read_csv(src / "per_patient.csv"); p.loc[0, "B0_rad_dice"] += 0.01; p.to_csv(src / "per_patient.csv", index=False)
    r = subprocess.run(cmd(src, cfg, tmp_path / "out2"), capture_output=True, text=True)
    assert r.returncode != 0 and "PARITY FAILED" in r.stdout + r.stderr


def test_input_guards(tmp_path):
    src, cfg = project(tmp_path)
    f = pd.read_csv(src / "val_features.csv"); f.iloc[[0, 1]] = f.iloc[[1, 0]].values
    f.to_csv(src / "val_features.csv", index=False)
    r = subprocess.run(cmd(src, cfg, tmp_path / "o"), capture_output=True, text=True)
    assert r.returncode != 0 and "misaligned" in r.stdout + r.stderr
    np.save(src / "val_desc.npy", np.ones((3, 32), np.float16))
    r = subprocess.run(cmd(src, cfg, tmp_path / "o", order_check=False), capture_output=True, text=True)
    assert r.returncode != 0 and "descriptor rows" in r.stdout + r.stderr
    D = np.load(src / "train_oof_desc.npy").astype(np.float32) * 3
    np.save(src / "train_oof_desc.npy", D.astype(np.float16))
    r = subprocess.run(cmd(src, cfg, tmp_path / "o", order_check=False), capture_output=True, text=True)
    assert r.returncode != 0 and "unit length" in r.stdout + r.stderr
    r = subprocess.run(cmd(src, cfg, tmp_path / "o")[:-4] + ["--train_features", str(src / "test_features.csv")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refusing to read a test file" in r.stdout + r.stderr
