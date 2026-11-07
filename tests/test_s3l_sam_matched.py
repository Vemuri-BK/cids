"""Tests for scripts/s3l_sam_matched.py (synthetic S3k-format tables; CPU, < 1 min)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3l_sam_matched as L  # noqa: E402
from cids.radiomics import RAD_ALL  # noqa: E402

C = json.loads((ROOT / "configs" / "s3i_subset_jury.json").read_text())
C_FAST = {**C, "hgb_grid": C["hgb_grid"][:1], "inner_folds": 2, "outer_folds": 2,
          "prob_threshold_grid": {"start": 0.1, "stop": 0.9, "step": 0.2}}


def make_table(path, prefix, n, seed, empty=0, nan_sam=False):
    """One row per candidate with every column the S3k tables carry for these arms; SAM 'agree' is informative."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n + empty):
        cid = f"{prefix}_{i:03d}"
        G = 1000
        if i >= n:
            rows.append(dict(cache_id=cid, k=-1, gt_vox=G, inter_vox=0, size_vox=0))
            continue
        K_ = int(rng.integers(1, 4))
        true = int(rng.integers(0, K_))
        for r in range(K_):
            on = r == true
            size = int(rng.integers(200, 1500))
            row = dict(cache_id=cid, k=r + 1, rank=r, n_cand=K_, size_vox=size, gt_vox=G,
                       inter_vox=int(0.8 * min(size, G)) if on else 0, log_seg_ml=np.log(size / 1000),
                       rel_size=1.0 / (r + 1), is_largest=int(r == 0), seg_prob_mean=rng.uniform(.5, 1),
                       seg_prob_max=rng.uniform(.8, 1),
                       agree=(0.8 if on else 0.2) + 0.1 * rng.standard_normal(),
                       log_sam_ml=rng.standard_normal(), log_sam_soft_ml=rng.standard_normal(),
                       log_sam_over_seg=rng.standard_normal(), rel_seg_prob=rng.uniform(), rel_ring_cnr=rng.uniform())
            row.update({c: rng.standard_normal() for c in RAD_ALL})
            if nan_sam and i == 0 and r == 0:
                row["agree"] = np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_feature_sets():
    FA = list(C["finder_features"]) + RAD_ALL + L.K.REL2
    assert len(FA) == 24 and len(FA + L.SAM4) == 28 and not set(L.SAM4) & set(FA)
    assert not set(FA + L.SAM4) & set(C["forbidden_features"])


def test_end_to_end(tmp_path):
    src = tmp_path / "s3k"; src.mkdir()
    make_table(src / "train_oof_table.csv", "duke", 30, 0, empty=1, nan_sam=True)
    make_table(src / "val_table.csv", "nact", 12, 1)
    cfg = tmp_path / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "s3l_sam_matched.py"), "--s3k_out", str(src),
           "--config", str(cfg), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "VERDICT" in r.stdout and "missing SAM value: 1" in r.stdout
    assert json.loads((out / "decision_rule.json").read_text())["B"].startswith("B0_rad + 4 SAM")
    s = pd.read_csv(out / "arms_summary.csv")
    assert set(s.policy) == {"B0_rad", "B0_rad_sam"} and set(s.split) == {"train_oof", "val"}
    pp = pd.read_csv(out / "per_patient.csv")
    e = pp[pp.cache_id == "duke_030"]
    assert len(e) == 1 and e.K.iloc[0] == 0 and e.B0_rad_dice.iloc[0] == 0
    assert len(pd.read_csv(out / "auc.csv")) == 2
    prov = json.loads((out / "provenance.json").read_text())
    assert len(prov["features_B"]) == 28 and prov["inputs"]["train"]["md5"]


def test_missing_tables(tmp_path):
    cfg = tmp_path / "cfg.json"; cfg.write_text(json.dumps(C_FAST))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3l_sam_matched.py"), "--s3k_out", str(tmp_path),
                        "--config", str(cfg), "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0 and "run S3k first" in (r.stderr + r.stdout)
