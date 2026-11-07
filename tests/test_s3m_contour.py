"""Tests for scripts/s3m_contour_study.py (synthetic volumes; CPU; the end-to-end test uses random SAM weights)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M  # noqa: E402

SHAPE = (70, 66, 48)


def test_points_and_fallback():
    comp = M.ball(SHAPE, (30, 30, 20), 6)
    assert M.geometric_point(comp) == (30, 30, 20)
    es = np.zeros(SHAPE, np.float32)
    es[33, 30, 20] = 5.0                                   # interior hotspot
    es[36, 30, 20] = 9.0                                   # brighter, but on the edge (< 2 mm inside)
    p, fb = M.enhancement_point(comp, es, 2.0)
    assert p == (33, 30, 20) and not fb
    slab = np.zeros(SHAPE, bool); slab[20:40, 20:40, 20:22] = True     # 2 voxels thick -> no interior
    p, fb = M.enhancement_point(slab, es, 2.0)
    assert fb and p == M.geometric_point(slab)


def test_coordinate_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p1 = rng.integers(0, 200, 3)
        origin = rng.integers(-60, 60, 3)
        back = M.crop_to_1mm(M.to_crop(p1, origin), origin)
        assert np.abs(back - p1).max() <= 0.75 + 1e-9      # within half a 1.5 mm voxel


def test_region_grow():
    blob = M.ball(SHAPE, (30, 30, 20), 7)
    es = ndimage.gaussian_filter(blob.astype(np.float32) * 2.0, 1.0)
    R = M.allowed_region(SHAPE, *M.bbox(blob), 10)
    m, s, weak = M.region_grow(es, (30, 30, 20), R, 0.5)
    assert not weak and s > 0 and M.dice_count(m, blob) > 0.85 and not (m & ~R).any()
    m, s, weak = M.region_grow(-es, (30, 30, 20), R, 0.5)
    assert weak and not m.any()


def test_keep_box_component():
    m = M.ball(SHAPE, (20, 20, 20), 4) | M.ball(SHAPE, (50, 50, 30), 6)
    k = M.keep_box_component(m, np.array([16, 16, 16]), np.array([24, 24, 24]))
    assert k[20, 20, 20] and not k[50, 50, 30]
    assert not M.keep_box_component(m, np.array([0, 60, 0]), np.array([3, 63, 3])).any()


class BallRunner:
    """Fake SAM: B = ball at the box centre (radius = half the box), C/D = ball at the given point; 'zero' -> empty."""

    def __init__(self, r15=4, zero=False):
        self.r, self.zero = r15, zero

    def __call__(self, crop, box, pts):
        n = crop.shape[0]
        box = np.asarray(box)
        c = (box[0] + box[1]) // 2
        r = max(1, int((box[1] - box[0]).max() // 2))
        out = [M.ball((n,) * 3, c, r)] + [M.ball((n,) * 3, p, self.r) for p in pts]
        return np.stack(out).astype(np.float32) * (0.0 if self.zero else 1.0)


def _case(hot=(33, 30, 20)):
    img = (0.2 + 0.02 * np.random.default_rng(0).standard_normal((2,) + SHAPE)).astype(np.float32)
    gt = M.ball(SHAPE, (30, 30, 20), 6)
    img[1][gt] += 0.6
    img[1][hot] += 2.0
    lab = np.zeros(SHAPE, np.int32)
    lab[M.ball(SHAPE, (30, 30, 20), 5)] = 3                 # candidate slightly smaller than the tumour
    lab[M.ball(SHAPE, (55, 50, 35), 3)] = 4                 # second candidate (not kept)
    return img, gt, lab


def test_patient_run_geometry():
    img, gt, lab = _case()
    r, rows = M.patient_run("duke_001", "train", img, gt, lab, [3], BallRunner())
    row = rows[0]
    assert r["A_fb_dice"] == M.dice_count(lab == 3, gt)
    assert r["B_raw_dice"] > 0.1 and r["C_raw_dice"] > 0.8  # box / point -> crop -> paste-back lands on the blob
    assert row["pC"] == [30, 30, 20] and row["pC_crop"] == [64, 64, 64]
    assert not row["B_fallback"] and r["B_fb_dice"] == r["B_raw_dice"]
    assert set(f"{a}_{v}_nsd" for a in M.ARMS for v in ("raw", "fb")) <= set(r)


def test_pasted_point_ball_is_centred():
    img, gt, lab = _case(hot=(30, 30, 20))
    import s3a_seg2sam as s3
    comp = lab == 3
    cc = s3.candidate_crops(img, lab, [(3, float(comp.sum()))])
    es = ndimage.gaussian_filter(img[1] - img[0], 2.0)
    out, _, _, row = M.candidate_contours(img, lab, 3, cc["crops"][0][0], cc["boxes"][0], cc["origins"][0],
                                          cc["shape15"], es, BallRunner(r15=3))
    for arm in "BD":
        cen = np.argwhere(out[arm]).mean(0)
        assert np.abs(cen - np.asarray(row["pD"])).max() <= 1.6     # box centre = point = blob centre here


def test_fallback_when_sam_empty():
    img, gt, lab = _case()
    r, rows = M.patient_run("duke_001", "train", img, gt, lab, [3], BallRunner(zero=True))
    assert rows[0]["B_fallback"] and rows[0]["C_fallback"] and rows[0]["D_fallback"]
    assert r["B_raw_empty"] and r["B_raw_dice"] == 0.0 and r["B_fb_dice"] == r["A_fb_dice"]
    assert r["E_raw_dice"] > 0                              # region growing does not depend on SAM


def test_empty_selection():
    img, gt, lab = _case()
    r, rows = M.patient_run("duke_002", "train", img, gt, lab, [], BallRunner())
    assert rows == [] and all(r[f"{a}_{v}_dice"] == 0 for a in M.ARMS for v in ("raw", "fb"))
    assert np.isnan(r["D_fb_hd95"]) and r["D_fb_nsd"] == 0


def test_rule_limits():
    base = dict(n=1080, lt05_A=137, mean_dnsd=0.012, dnsd_ci_lo=0.002, mean_ddice=-0.004, lt05_B=150, loss_ge_020=10)
    v = M.rule_check(base, dict(dnsd_ci_lo=0.001), dict(dnsd_ci_lo=-0.01))
    assert v["lt05_max"] == 150 and v["loss020_max"] == 10
    assert v["primary_D_beats_A"] and v["secondary_D_beats_C"] and not v["secondary_D_beats_E"]
    for k, bad in (("mean_dnsd", 0.009), ("dnsd_ci_lo", 0.0), ("mean_ddice", -0.006), ("lt05_B", 151),
                   ("loss_ge_020", 11)):
        assert not M.rule_check({**base, k: bad}, dict(dnsd_ci_lo=1), dict(dnsd_ci_lo=1))["primary_D_beats_A"]
    v = M.rule_check({**base, "mean_dnsd": 0}, dict(dnsd_ci_lo=1), dict(dnsd_ci_lo=1))
    assert not v["secondary_D_beats_C"] and not v["secondary_D_beats_E"]     # secondary only after primary


def test_choose_smoke():
    rows = [dict(cache_id=f"duke_{i:03d}", split="train", n_kept=[0, 1, 2, 3][i % 4], min_kept_vox=100 + i,
                 edge_mm=50 - i) for i in range(40)]
    rows += [dict(cache_id=f"nact_{i:03d}", split="val", n_kept=1, min_kept_vox=500, edge_mm=20) for i in range(5)]
    ch = M.choose_smoke(pd.DataFrame(rows), 20)
    reasons = pd.Series(ch).value_counts()
    assert len(ch) == 20 and reasons["empty_selection"] == 2 and reasons["multiple_kept"] == 3
    assert reasons["tiny_component"] == 3 and reasons["near_edge"] == 3 and reasons["val"] == 3


def _project(tmp):
    """Synthetic cache + S3k per_patient.csv + component maps for 4 train and 2 val patients."""
    root = tmp / "cache"
    for d in ("train", "val"):
        (root / d).mkdir(parents=True)
    (tmp / "tc").mkdir(); (tmp / "vc").mkdir()
    sp, pp = [], []
    for i, (cid, split, keep) in enumerate([("duke_000", "train", [3]), ("duke_001", "train", [3, 4]),
                                            ("duke_002", "train", []), ("ispy1_003", "train", [4]),
                                            ("nact_000", "val", [3]), ("nact_001", "val", [3])]):
        img, gt, lab = _case()
        np.savez(root / ("train" if split == "train" else "val") / f"{cid}.npz", image=img.astype(np.float16),
                 label=gt[None].astype(np.uint8))
        np.savez((tmp / ("tc" if split == "train" else "vc")) / f"{cid}.npz", lab=lab)
        m = np.isin(lab, keep) if keep else np.zeros(SHAPE, bool)
        sp.append(dict(patient_id=cid.upper(), official="train" if split == "train" else "val", split=split,
                       cache_id=cid, collection=cid.split("_")[0].upper()))
        pp.append(dict(split="train_oof" if split == "train" else "val", cache_id=cid,
                       B0_rad_dice=M.dice_count(m, gt), B0_rad_subset="+".join(map(str, keep))))
    pd.DataFrame(sp).to_csv(tmp / "splits.csv", index=False)
    (tmp / "s3k").mkdir()
    pd.DataFrame(pp).to_csv(tmp / "s3k" / "per_patient.csv", index=False)
    return [sys.executable, str(ROOT / "scripts" / "s3m_contour_study.py"), "--cache_root", str(root),
            "--splits", str(tmp / "splits.csv"), "--s3k_out", str(tmp / "s3k"), "--train_cases", str(tmp / "tc"),
            "--val_cases", str(tmp / "vc"), "--workers", "1"]


def test_end_to_end(tmp_path):
    base = _project(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(base + ["--step", "arms", "--random_weights", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode != 0 and "run --step parity first" in r.stdout + r.stderr
    r = subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PARITY OK" in r.stdout
    par = pd.read_csv(out / "parity.csv")
    assert len(par) == 6 and par.abs_diff.max() <= 1e-12 and (par.n_kept == 0).sum() == 1
    r = subprocess.run(base + ["--step", "arms", "--random_weights", "--smoke", "4", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "SMOKE RUN" in r.stdout
    df = pd.read_csv(out / "per_patient.csv")
    assert len(df) == 4 and df.smoke_reason.notna().all()
    cd = pd.read_csv(out / "candidates.csv")
    assert len(cd) == df.n_kept.sum() and len(list((out / "overlays").glob("*.png"))) == len(cd)
    assert json.loads((out / "decision_rule.json").read_text())["min_mean_dnsd"] == 0.010
    assert set(pd.read_csv(out / "paired.csv").kind) >= {"primary", "secondary_1", "secondary_2", "raw"}
    assert "same_sam_voxel_C_D" in json.loads((out / "technical.json").read_text())


def test_parity_failure_and_test_refused(tmp_path):
    base = _project(tmp_path)
    pp = pd.read_csv(tmp_path / "s3k" / "per_patient.csv")
    pp.loc[0, "B0_rad_dice"] += 1e-4
    pp.to_csv(tmp_path / "s3k" / "per_patient.csv", index=False)
    r = subprocess.run(base + ["--step", "parity", "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0 and "PARITY FAILED" in r.stdout + r.stderr
    r = subprocess.run(base + ["--step", "parity", "--eval_splits", "test", "--out", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "development study" in r.stdout + r.stderr


def test_shift_profile_finds_known_shift():
    gt = M.ball(SHAPE, (30, 30, 20), 7)
    assert M.shift_profile(gt, gt)[0] == (0, 0, 0)
    moved = np.roll(gt, (2, -1, 1), axis=(0, 1, 2))
    best, d0, bd = M.shift_profile(moved, gt, 3)
    assert best == (-2, 1, -1) and bd == 1.0 and d0 < 1.0


def test_shiftcheck_step(tmp_path):
    base = _project(tmp_path)
    out = tmp_path / "out"
    assert subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--step", "shiftcheck", "--random_weights", "--shift_n", "3", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    s = pd.read_csv(out / "shift_summary.csv", index_col=0)
    assert set(s.index) == set(M.ARMS) and s.loc["A", "mean_gain"] >= 0


def test_shift_tie_prefers_zero():
    gt = np.zeros(SHAPE, bool); gt[20:40, 20:40, 20:21] = True            # flat: z shifts of +-1 tie with 0? no
    m = gt.copy()
    assert M.shift_profile(m, gt, 2)[0] == (0, 0, 0)


def test_box_sam_mask_matches_arms_B():
    img, gt, lab = _case()
    r, rows, masks = M.patient_run("duke_001", "train", img, gt, lab, [3], BallRunner(), return_masks=True)
    b, t = M.box_sam_mask(img, lab, [3], BallRunner())
    assert np.array_equal(b, masks["B_raw"]) and t >= 0


def test_overlap_step(tmp_path):
    base = _project(tmp_path)
    out = tmp_path / "out"
    assert subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--step", "overlap", "--random_weights", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    o = pd.read_csv(out / "overlap.csv")
    assert len(o) == 6 and o.loc[o.n_kept == 0, "b_vox"].eq(0).all() and o.dice_ab.dropna().between(0, 1).all()
