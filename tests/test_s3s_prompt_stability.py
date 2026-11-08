"""Tests for scripts/s3s_prompt_stability.py (synthetic; CPU; the end-to-end test uses random SAM weights)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3s_prompt_stability as P  # noqa: E402

SHAPE = (70, 66, 48)


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def test_interior_points_deterministic_and_inside():
    comp = ball((40, 40, 40), (20, 20, 20), 8)
    p = P.interior_points(comp, "duke_001_3", 5)
    assert p[0] == (20, 20, 20) and len(p) == 5 and len(set(p)) == 5
    assert all(comp[q] for q in p) and p == P.interior_points(comp, "duke_001_3", 5)
    assert p != P.interior_points(comp, "duke_001_4", 5)
    tiny = np.zeros((10, 10, 10), bool); tiny[5, 5, 5] = True
    assert P.interior_points(tiny, "x", 5) == [(5, 5, 5)] * 5
    assert P.interior_points(np.zeros((5, 5, 5), bool), "x", 5) == []


def test_box_variants():
    v = P.box_variants(np.array([[10, 10, 10], [29, 29, 29]]))
    assert v.shape == (5, 2, 3)
    assert (v[0] == [[10] * 3, [29] * 3]).all() and (v[1] == [[13] * 3, [26] * 3]).all()
    assert (v[2] == [[7] * 3, [32] * 3]).all() and (v[3] == [[12] * 3, [31] * 3]).all()
    e = P.box_variants(np.array([[0, 0, 5], [3, 127, 5]]))                        # edge + 1-voxel-thin axis
    assert (e >= 0).all() and (e <= 127).all() and (e[:, 0] <= e[:, 1]).all()


def test_mask_stats():
    comp = ball((30, 30, 30), (15, 15, 15), 6)
    same = np.stack([comp] * 4)
    s = P.mask_stats(same, comp)
    assert s["stab"] == 1.0 and s["comp_dice"] == 1.0 and s["ml_cv"] == 0 and s["empty_frac"] == 0
    mixed = np.stack([comp, comp, np.zeros_like(comp), np.zeros_like(comp)])
    m = P.mask_stats(mixed, comp)
    assert np.isclose(m["stab"], 1 / 6) and m["empty_frac"] == 0.5 and np.isclose(m["comp_dice"], 0.5)
    far = np.stack([comp, ball((30, 30, 30), (5, 5, 5), 4)])
    assert P.mask_stats(far, comp)["stab"] == 0.0


def _project(tmp):
    """Synthetic cache, S3k tables (train OOF) and component maps for 3 train patients."""
    root = tmp / "cache"; (root / "train").mkdir(parents=True); (root / "val").mkdir()
    (tmp / "tc").mkdir(); (tmp / "s3k").mkdir()
    rng = np.random.default_rng(0)
    sp, tab, pp = [], [], []
    for i, cid in enumerate(["duke_000", "duke_001", "ispy1_002"]):
        img = (0.2 + 0.02 * rng.standard_normal((2,) + SHAPE)).astype(np.float32)
        gt = ball(SHAPE, (30, 30, 20), 6)
        img[1][gt] += 0.6
        lab = np.zeros(SHAPE, np.int32)
        lab[ball(SHAPE, (30, 30, 20), 5)] = 3
        lab[ball(SHAPE, (55, 50, 35), 3)] = 4
        np.savez(root / "train" / f"{cid}.npz", image=img.astype(np.float16), label=gt[None].astype(np.uint8))
        np.savez(tmp / "tc" / f"{cid}.npz", lab=lab)
        sp.append(dict(patient_id=cid.upper(), official="train", split="train", cache_id=cid,
                       collection=cid.split("_")[0].upper()))
        for k, on in ((3, 1), (4, 0)):
            tab.append(dict(cache_id=cid, k=k, on_tumour=on, size_vox=int((lab == k).sum()),
                            seg_prob_mean=0.9 if k == 4 else 0.6, seg_prob_max=0.95, agree=0.5))
        pp.append(dict(split="train_oof", cache_id=cid, collection=cid.split("_")[0].upper(),
                       B0_rad_dice=0.0 if i < 2 else 0.8))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    pd.DataFrame(tab).to_csv(tmp / "s3k" / "train_oof_table.csv", index=False)
    pd.DataFrame(pp).to_csv(tmp / "s3k" / "per_patient.csv", index=False)
    return [sys.executable, str(ROOT / "scripts" / "s3s_prompt_stability.py"), "--s3k_out", str(tmp / "s3k"),
            "--out", str(tmp / "out")], ["--cache_root", str(root), "--splits", str(tmp / "splits.csv"),
                                         "--train_cases", str(tmp / "tc")]


def test_end_to_end(tmp_path):
    base, data = _project(tmp_path)
    r = subprocess.run(base + ["--step", "gate"], capture_output=True, text=True)
    assert r.returncode != 0 and "run --step precheck first" in r.stdout + r.stderr
    r = subprocess.run(base + data + ["--step", "precheck", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PRECHECK DONE: 6 candidates" in r.stdout
    out = tmp_path / "out"
    assert json.loads((out / "decision_rule.json").read_text())["harm_max"] == 100
    st = pd.read_csv(out / "candidates_stability.csv")
    assert len(st) == 6 and st.turbo_stab.between(0, 1).all() and st.s2b_stab.between(0, 1).all()
    assert (st.n_turbo_points == 5).all() and not (out / "candidates_stability.partial.csv").exists()
    r = subprocess.run(base + ["--step", "gate"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    g = json.loads((out / "gate.json").read_text())
    assert g["n_wrong"] == 2 and g["n_ok"] == 1 and g["best_non_sam_hits"] == 2 and g["primary_turbo"]["needed"] == 7
    assert g["PASS"] is False and "VERDICT" in r.stdout
    t = pd.read_csv(out / "gate_table.csv")
    assert t[t.column == "seg_prob_mean"].wrong_hits.iloc[0] == 0 and t[t.column == "size_vox"].wrong_hits.iloc[0] == 2


def test_resume_and_test_refused(tmp_path):
    base, data = _project(tmp_path)
    r = subprocess.run(base + data + ["--step", "precheck", "--random_weights", "--limit", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "out"
    pd.read_csv(out / "candidates_stability.csv").to_csv(out / "candidates_stability.partial.csv", index=False)
    r = subprocess.run(base + data + ["--step", "precheck", "--random_weights"], capture_output=True, text=True)
    assert r.returncode == 0 and "(1 already done)" in r.stdout
    assert len(pd.read_csv(out / "candidates_stability.csv")) == 6
    r = subprocess.run(base + ["--step", "gate", "--memory_csv", str(tmp_path / "test_mem.csv")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "test files are refused" in r.stdout + r.stderr
