"""Tests for scripts/s3u_click_biomarkers.py (synthetic; CPU; the end-to-end test uses random SAM weights)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3u_click_biomarkers as U  # noqa: E402

SHAPE = (70, 66, 48)


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def test_click_and_component():
    gt = ball(SHAPE, (30, 30, 20), 6) | ball(SHAPE, (55, 50, 35), 2)       # largest component wins
    assert U.click_point(gt) == (30, 30, 20) and U.click_point(np.zeros(SHAPE, bool)) is None
    m = ball(SHAPE, (30, 30, 20), 4) | ball(SHAPE, (55, 50, 35), 3)
    assert U.component_at(m, (30, 30, 20)).sum() == ball(SHAPE, (30, 30, 20), 4).sum()
    near = U.component_at(m, (60, 50, 35))                                   # outside -> nearest component
    assert near[55, 50, 35] and not near[30, 30, 20]
    assert not U.component_at(np.zeros(SHAPE, bool), (1, 1, 1)).any()


def test_build_masks_routes():
    gt = ball(SHAPE, (30, 30, 20), 6)
    lab = np.zeros(SHAPE, np.int32)
    lab[ball(SHAPE, (30, 30, 20), 5)] = 3
    lab[ball(SHAPE, (55, 50, 35), 3)] = 4
    fake = lambda img, p: ball(SHAPE, p, 7)                                  # noqa: E731
    m, info = U.build_masks(None, gt, lab, [4], fake)
    assert info["k_click"] == 3 and not info["h_used_sam"]
    assert (m["C"] == (lab == 3)).all() and (m["H"] == m["C"]).all() and (m["B"] == (lab == 4)).all()
    lab2 = np.where(lab == 3, 0, lab)                                         # no candidate under the click
    m2, info2 = U.build_masks(None, gt, lab2, [], fake)
    assert info2["h_used_sam"] and not m2["C"].any() and (m2["H"] == m2["S"]).all() and not m2["B"].any()


def test_features():
    assert U.diameter_mm(ball(SHAPE, (30, 30, 20), 6)) == 12.0
    img = np.zeros((2,) + SHAPE, np.float32) + 0.2
    m = ball(SHAPE, (30, 30, 20), 6)
    img[1][m] += 0.5
    f = U.mask_features(img, m)
    assert set(f) == set(U.FEATS) and len(U.FEATS) == 17
    assert abs(f["volume_ml"] - m.sum() / 1000) < 1e-9 and abs(f["enh_mean"] - 0.5) < 1e-5
    assert f["rad_ring_cnr"] > 1 and np.isnan(U.mask_features(img, np.zeros(SHAPE, bool))["volume_ml"])


def test_analyse_and_pcr(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    cids = [f"ispy2_{i:03d}" for i in range(n)]
    base = rng.lognormal(1, 0.5, (n, len(U.FEATS)))
    noise = dict(E=0, B=0.3, C=0.05, S=0.15, H=0.05)
    feats, pp = [], []
    for i, c in enumerate(cids):
        r = dict(cache_id=c, collection="ISPY2", h_used_sam=False)
        for s in U.SOURCES:
            v = base[i] * (1 + noise[s] * rng.standard_normal(len(U.FEATS)))
            feats.append(dict(cache_id=c, source=s, **dict(zip(U.FEATS, v))))
            r[f"{s}_dice"] = 0.05 if (s == "B" and i < 5) else 0.8
            r[f"{s}_ml"] = 1.0
        pp.append(r)
    out = tmp_path / "o"; out.mkdir()
    pd.DataFrame(pp).to_csv(out / "per_patient.csv", index=False)
    pd.DataFrame(feats).to_csv(out / "features.csv", index=False)
    y = (base[:, 0] > np.median(base[:, 0])).astype(int)
    pd.DataFrame(dict(patient_id=[c.upper() for c in cids], pcr=y)).to_csv(tmp_path / "clin.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3u_click_biomarkers.py"), "--step", "analyse",
                        "--s3k_out", str(tmp_path), "--clinical", str(tmp_path / "clin.csv"), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["claims"]
    assert s["one_click_sam_removes_gross_failures"] and s["sam_better_biomarkers_than_automatic"]
    assert not s["sam_beats_non_sam_click_control"]
    a = pd.read_csv(out / "agreement_summary.csv").set_index("source")
    assert a.loc["C", "median_ccc"] > a.loc["S", "median_ccc"] > a.loc["B", "median_ccc"]
    pc = pd.read_csv(out / "pcr_auc.csv").set_index("source")
    assert pc.loc["E", "auc_mean"] > 0.9 and set(pc.index) == set(U.SOURCES)


def _project(tmp):
    root = tmp / "cache"; (root / "train").mkdir(parents=True); (root / "val").mkdir()
    (tmp / "vc").mkdir(); (tmp / "s3k").mkdir()
    rng = np.random.default_rng(0)
    sp, pp = [], []
    for i, (cid, on_tumour, keep) in enumerate([("nact_000", True, [3]), ("nact_001", False, [4]),
                                                ("ispy2_002", True, [])]):
        img = (0.2 + 0.02 * rng.standard_normal((2,) + SHAPE)).astype(np.float32)
        gt = ball(SHAPE, (30, 30, 20), 6)
        img[1][gt] += 0.6
        lab = np.zeros(SHAPE, np.int32)
        if on_tumour:
            lab[ball(SHAPE, (30, 30, 20), 5)] = 3
        lab[ball(SHAPE, (55, 50, 35), 3)] = 4
        np.savez(root / "val" / f"{cid}.npz", image=img.astype(np.float16), label=gt[None].astype(np.uint8))
        np.savez(tmp / "vc" / f"{cid}.npz", lab=lab)
        sp.append(dict(patient_id=cid.upper(), official="test", split="val", cache_id=cid,
                       collection=cid.split("_")[0].upper()))
        B = np.isin(lab, keep) if keep else np.zeros(SHAPE, bool)
        pp.append(dict(split="val", cache_id=cid, B0_rad_dice=U.M.dice_count(B, gt),
                       B0_rad_subset="+".join(map(str, keep))))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    pd.DataFrame(pp).to_csv(tmp / "s3k" / "per_patient.csv", index=False)
    return [sys.executable, str(ROOT / "scripts" / "s3u_click_biomarkers.py"), "--cache_root", str(root),
            "--splits", str(tmp / "splits.csv"), "--s3k_out", str(tmp / "s3k"), "--val_cases", str(tmp / "vc"),
            "--out", str(tmp / "out")]


def test_end_to_end_masks(tmp_path):
    base = _project(tmp_path)
    r = subprocess.run(base + ["--step", "analyse"], capture_output=True, text=True)
    assert r.returncode != 0 and "run --step masks first" in r.stdout + r.stderr
    r = subprocess.run(base + ["--step", "masks", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PARITY OK" in r.stdout and "H used SAM): 1 / 3" in r.stdout
    out = tmp_path / "out"
    assert json.loads((out / "decision_rule.json").read_text())["features"] == U.FEATS
    pp = pd.read_csv(out / "per_patient.csv").set_index("cache_id")
    assert pp.E_dice.eq(1).all() and pp.loc["nact_000", "C_dice"] > 0.5 and pp.loc["nact_001", "C_dice"] == 0
    assert pp.loc["ispy2_002", "B_dice"] == 0 and len(pd.read_csv(out / "features.csv")) == 15
    sp = pd.read_csv(tmp_path / "s3k" / "per_patient.csv")
    sp.loc[0, "B0_rad_dice"] += 0.01
    sp.to_csv(tmp_path / "s3k" / "per_patient.csv", index=False)
    r = subprocess.run(base + ["--step", "masks", "--random_weights"], capture_output=True, text=True)
    assert r.returncode != 0 and "PARITY FAILED" in r.stdout + r.stderr
