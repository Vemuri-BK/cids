"""Tests for scripts/nnunet_baseline.py (synthetic cache; no nnU-Net needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import nnunet_baseline as N  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "nnunet_baseline.py")
SHAPE = (30, 28, 20)


def _cache(tmp):
    root = tmp / "cache"; (root / "train").mkdir(parents=True); (root / "val").mkdir()
    rng = np.random.default_rng(0)
    sp = []
    for split, official, cid in [("train", "train", "duke_001"), ("train", "train", "ispy2_002"),
                                 ("val", "train", "nact_003"), ("test", "test", "duke_004"), ("test", "test", "ispy1_005")]:
        img = rng.random((2,) + SHAPE).astype(np.float16)
        gt = np.zeros(SHAPE, np.uint8); gt[10:15, 10:14, 5:9] = 1
        np.savez(root / ("train" if official == "train" else "val") / f"{cid}.npz", image=img, label=gt[None])
        sp.append(dict(patient_id=cid.upper(), official=official, split=split, cache_id=cid,
                       collection=cid.split("_")[0].upper()))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    return root


def test_convert(tmp_path):
    root = _cache(tmp_path)
    raw = tmp_path / "raw"
    cmd = [sys.executable, SCRIPT, "--step", "convert", "--cache_root", str(root), "--splits",
           str(tmp_path / "splits.csv"), "--nnunet_raw", str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    ds = raw / "Dataset501_MAMAMIA3ch"
    assert "imagesTr 3 (train 2 + val 1) | imagesTs 2" in r.stdout
    assert len(list((ds / "imagesTr").glob("*.nii.gz"))) == 9 and len(list((ds / "labelsTr").glob("*"))) == 3
    assert len(list((ds / "imagesTs").glob("*.nii.gz"))) == 6
    d = np.load(root / "train" / "duke_001.npz")
    pre, post = d["image"][0].astype(np.float32), d["image"][1].astype(np.float32)
    assert np.allclose(N.load_nii(ds / "imagesTr" / "duke_001_0002.nii.gz"), post - pre)
    assert np.allclose(N.load_nii(ds / "imagesTr" / "duke_001_0000.nii.gz"), pre)
    assert (N.load_nii(ds / "labelsTr" / "duke_001.nii.gz") == d["label"][0]).all()
    dj = json.loads((ds / "dataset.json").read_text())
    assert dj["numTraining"] == 3 and dj["channel_names"] == {"0": "pre", "1": "post1", "2": "sub"}
    spl = json.loads((ds / "splits_final.json").read_text())
    assert spl == [dict(train=["duke_001", "ispy2_002"], val=["nact_003"])]
    assert (ds / "test_ids.txt").read_text().split() == ["duke_004", "ispy1_005"]
    r = subprocess.run(cmd, capture_output=True, text=True)                    # resume: nothing re-converted
    assert r.returncode == 0 and "converted now 0" in r.stdout


def test_evaluate_and_test_guard(tmp_path):
    root = _cache(tmp_path)
    pred = tmp_path / "pred"; pred.mkdir()
    for cid, off in (("nact_003", "train"), ("duke_004", "val"), ("ispy1_005", "val")):
        gt = np.load(root / off / f"{cid}.npz")["label"][0]
        p = gt.copy()
        if cid == "ispy1_005":
            p[:] = 0                                                           # a complete miss
        N.save_nii(p, pred / f"{cid}.nii.gz", np.uint8)
    ref = pd.DataFrame(dict(split=["val", "test", "test"], cache_id=["nact_003", "duke_004", "ispy1_005"],
                            B0_rad_dice=[0.5, 0.8, 0.9]))
    ref.to_csv(tmp_path / "ref.csv", index=False)
    base = [sys.executable, SCRIPT, "--step", "evaluate", "--cache_root", str(root), "--splits",
            str(tmp_path / "splits.csv"), "--pred_dir", str(pred), "--compare", str(tmp_path / "ref.csv")]
    r = subprocess.run(base + ["--split", "val", "--out", str(tmp_path / "ev")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "ev" / "result.json").read_text())
    assert res["n"] == 1 and abs(res["paired_nnunet_minus_ref"]["diff"] - 0.5) < 1e-9
    r = subprocess.run(base + ["--split", "test", "--out", str(tmp_path / "et")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = pd.read_csv(tmp_path / "et" / "summary.csv").set_index("group")
    assert s.loc["all", "n"] == 2 and s.loc["all", "dice_lt_0_1"] == 1 and abs(s.loc["all", "dice"] - 0.5) < 1e-9
    r = subprocess.run(base + ["--split", "test", "--out", str(tmp_path / "et")], capture_output=True, text=True)
    assert r.returncode != 0 and "evaluated once" in r.stdout + r.stderr
    (pred / "nact_003.nii.gz").unlink()
    r = subprocess.run(base + ["--split", "val", "--out", str(tmp_path / "ev2")], capture_output=True, text=True)
    assert r.returncode != 0 and "predictions missing" in r.stdout + r.stderr


def test_partial_leftovers_removed(tmp_path):
    root = _cache(tmp_path)
    ds = tmp_path / "raw" / "Dataset501_MAMAMIA3ch" / "imagesTr"
    ds.mkdir(parents=True)
    (ds / "duke_001_0000.partial.nii.gz").write_bytes(b"x")
    r = subprocess.run([sys.executable, SCRIPT, "--step", "convert", "--cache_root", str(root), "--splits",
                        str(tmp_path / "splits.csv"), "--nnunet_raw", str(tmp_path / "raw")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert not list(ds.glob("*.partial.nii.gz"))
