"""Tests for scripts/s3q_location_audit.py (synthetic project; CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3q_location_audit as Q  # noqa: E402
from test_s3m_contour import _project  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "s3q_location_audit.py")


def _setup(tmp):
    base = _project(tmp)                                     # ispy1_003 keeps blob 4 (wrong), blob 3 touches tumour
    pp = pd.read_csv(tmp / "s3k" / "per_patient.csv")
    rows = []
    for cid in pp[pp.split == "train_oof"].cache_id:
        rows += [dict(cache_id=cid, k=3, rank=0, size_vox=515, inter_vox=515, gt_vox=925),
                 dict(cache_id=cid, k=4, rank=1, size_vox=123, inter_vox=0, gt_vox=925)]
    pd.DataFrame(rows).to_csv(tmp / "s3k" / "train_oof_table.csv", index=False)
    i = base.index("--cache_root"); j = base.index("--splits"); k = base.index("--train_cases")
    return [sys.executable, SCRIPT, "--cache_root", base[i + 1], "--splits", base[j + 1], "--s3k_out", str(tmp / "s3k"),
            "--train_cases", base[k + 1]]


def test_wrong_picks_selection(tmp_path):
    _setup(tmp_path)
    pats, key = Q.wrong_picks(tmp_path / "s3k")
    assert set(pats.cache_id) == {"duke_002", "ispy1_003"}                 # duke_002: synthetic 'nothing kept'
    k = key[key.cache_id == "ispy1_003"]
    assert k.loc[k.k == 4, "kept"].item() and not k.loc[k.k == 3, "kept"].item()


def test_images_then_tally(tmp_path):
    base = _setup(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(base + ["--step", "images", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert (out / "stageA" / "ispy1_003.png").exists() and (out / "stageB" / "ispy1_003.png").exists()
    t = pd.read_csv(out / "annotation_template.csv")
    assert "kept" not in t.columns and "single_dice" not in t.columns            # blind
    r = subprocess.run(base + ["--step", "tally", "--annotations", str(out / "annotation_template.csv"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "valid location" in r.stdout + r.stderr
    t["location"] = t.k.map({3: "inside_breast", 4: "outside_breast"})
    t.to_csv(out / "annotation_filled.csv", index=False)
    r = subprocess.run(base + ["--step", "tally", "--annotations", str(out / "annotation_filled.csv"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "audit_summary.json").read_text())
    assert s["rescuable_01"] == 1 and s["rescuable_05"] == 1 and not s["proceed_to_breast_mask_experiment"]
    assert s["categories"] == {"clear_outside": 1, "nothing_kept": 1}
