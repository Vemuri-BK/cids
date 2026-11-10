"""Tests for scripts/reader_package.py (synthetic cache, 24 validation patients)."""
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import reader_package as R  # noqa: E402

SHAPE = (24, 22, 16)


def _project(tmp):
    root = tmp / "cache"; (root / "train").mkdir(parents=True); (root / "val").mkdir()
    for d in ("vc", "s3k", "s3n"):
        (tmp / d).mkdir()
    rng = np.random.default_rng(0)
    sp, pp, rev = [], [], []
    for i in range(24):
        cid = f"ispy2_{i:03d}"
        img = rng.random((2,) + SHAPE).astype(np.float16)
        gt = np.zeros(SHAPE, np.uint8); gt[8:14, 8:12, 4:9] = 1
        lab = np.zeros(SHAPE, np.int32); lab[8:14, 8:12, 4:9] = 3; lab[18:21, 15:18, 10:13] = 4
        keep = [4] if i % 4 == 0 else [3]
        np.savez(root / "train" / f"{cid}.npz", image=img, label=gt[None])
        np.savez(tmp / "vc" / f"{cid}.npz", lab=lab)
        sp.append(dict(patient_id=cid.upper(), official="train", split="val", cache_id=cid, collection="ISPY2"))
        B = np.isin(lab, keep)
        pp.append(dict(split="val", cache_id=cid, B0_rad_dice=R.M.dice_count(B, gt > 0), B0_rad_subset="+".join(map(str, keep))))
        rev.append(dict(cache_id=cid, is_empty=0, risk_baseline=float(i) / 24))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    pd.DataFrame(pp).to_csv(tmp / "s3k" / "per_patient.csv", index=False)
    pd.DataFrame(rev).to_csv(tmp / "s3n" / "val_review.csv", index=False)
    return [sys.executable, str(ROOT / "scripts" / "reader_package.py"), "--cache_root", str(root),
            "--splits", str(tmp / "splits.csv"), "--s3k_out", str(tmp / "s3k"), "--val_cases", str(tmp / "vc"),
            "--s3n_out", str(tmp / "s3n"), "--out", str(tmp / "out")]


def test_select_cases():
    rev = pd.DataFrame(dict(cache_id=[f"p{i:02d}" for i in range(30)], is_empty=[0] * 29 + [1],
                            risk_baseline=np.linspace(0, 1, 30)))
    c = R.select_cases(rev, rev.cache_id.tolist())
    flagged = {x for x, f in c if f}
    assert len(c) == 20 and len(flagged) == 10 and "p29" in flagged and "p28" in flagged      # empty mask first
    assert c == R.select_cases(rev, rev.cache_id.tolist())                                     # deterministic


def test_build(tmp_path):
    cmd = _project(tmp_path)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    pkg = tmp_path / "out" / "reader_pilot_package"
    cases = sorted(p.name for p in pkg.iterdir() if p.is_dir())
    assert cases == [f"case{i:02d}" for i in range(1, 21)]
    files = sorted(p.name for p in (pkg / "case01").iterdir())
    assert files == ["case01_auto_mask.seg.nii.gz", "case01_post.nii.gz", "case01_pre.nii.gz", "case01_sub.nii.gz"]
    key = pd.read_csv(tmp_path / "out" / "reader_key_DO_NOT_SEND.csv")
    assert key.flagged_by_qc.sum() == 10 and len(key) == 20
    z = zipfile.ZipFile(tmp_path / "out" / "reader_pilot_package.zip")
    names = z.namelist()
    assert not any("key" in n.lower() or "ispy2_" in n for n in names)                         # nothing identifying
    assert "reader_pilot_package/README.txt" in names and "reader_pilot_package/responses.csv" in names
    import nibabel as nib
    m = np.asarray(nib.load(str(pkg / "case01" / "case01_auto_mask.seg.nii.gz")).dataobj)
    cid = key.set_index("case").loc["case01", "cache_id"]
    lab = np.load(tmp_path / "vc" / f"{cid}.npz")["lab"]
    kept = 4 if int(cid[-3:]) % 4 == 0 else 3
    assert (m.astype(bool) == (lab == kept)).all()
    r = subprocess.run(cmd, capture_output=True, text=True)                                    # built once
    assert r.returncode != 0 and "already exists" in r.stdout + r.stderr
