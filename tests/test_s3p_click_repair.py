"""Tests for scripts/s3p_click_repair.py (synthetic volumes; the end-to-end test uses random SAM weights on CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))
import s3p_click_repair as P  # noqa: E402
from test_s3m_contour import SHAPE, _case, _project  # noqa: E402

B = P.ball if hasattr(P, "ball") else None


def ball(c, r):
    return P.M3.ball(SHAPE, c, r)


def test_next_click_sign_and_place():
    gt = ball((30, 30, 20), 6)
    m = ball((30, 30, 20), 3)
    p, sign, size = P.next_click(m, gt)
    assert sign == 1 and gt[p] and not m[p]
    m2 = gt | ball((55, 50, 35), 8)                       # a large wrongly included blob
    p, sign, size = P.next_click(m2, gt)
    assert sign == 0 and m2[p] and not gt[p] and np.linalg.norm(np.subtract(p, (55, 50, 35))) <= 1
    assert P.next_click(gt, gt) is None


def test_toggle_rule():
    lab = np.zeros(SHAPE, np.int32); lab[ball((30, 30, 20), 5)] = 3; lab[ball((55, 50, 35), 3)] = 4
    touches = {3: True, 4: False}
    assert P.toggle_action((30, 30, 20), 1, set(), {3, 4}, lab, touches) == ("toggle_on", 3)
    assert P.toggle_action((55, 50, 35), 0, {4}, {3, 4}, lab, touches) == ("toggle_off", 4)
    assert P.toggle_action((30, 30, 20), 0, {3}, {3, 4}, lab, touches) is None      # touches tumour -> not toggled
    assert P.toggle_action((5, 5, 5), 1, set(), {3, 4}, lab, touches) is None         # background


def test_brush():
    m = np.zeros(SHAPE, bool)
    a = P.brush(m, (30, 30, 20), 1, 5)
    assert a[30, 30, 20] and a[34, 30, 20] and not a[37, 30, 20]
    assert not P.brush(a, (30, 30, 20), 0, 5).any()


def test_mask_logit_roundtrip():
    mc = P.M3.ball((128,) * 3, (64, 64, 64), 20)
    rt = P.low_to_mask(P.mask_to_low(mc, 4.0))
    assert P.dice(rt, mc) > 0.9


def _editor(img, fn):
    e = P.ClickEditor(fn, 4.0)
    e.prepare(img)
    return e


def test_editor_adds_only_local_component():
    img, gt, lab = _case()
    m = (lab == 4)                                        # a wrong blob is kept; the tumour is missed
    calls = {}

    def fake(x, coords, labels, low):                      # SAM proposes a ball at the click + a far blob
        c = coords[0, -1].numpy().astype(int)
        calls["low"] = low
        out = P.M3.ball((128,) * 3, c, 4) | P.M3.ball((128,) * 3, (5, 5, 5), 3)
        return out.astype(np.float32)
    e = _editor(img, fake)
    new, info = e.propose(m, [((30, 30, 20), 1)])
    assert new[30, 30, 20] and (new & ~m).sum() == info["change_vox"] > 0
    assert (new & m).sum() == m.sum()                     # nothing removed by a positive click
    added = new & ~m
    assert np.abs(np.argwhere(added).mean(0) - (30, 30, 20)).max() <= 1.6    # only the component at the click
    assert info["click_back_err_mm"] <= 0.75 and calls["low"].shape == (1, 1, 32, 32, 32)


def test_editor_negative_removes_component_with_click():
    img, gt, lab = _case()
    m = (lab == 3) | (lab == 4)
    e = _editor(img, lambda x, c, l, low: np.zeros((128,) * 3, np.float32))
    new, info = e.propose(m, [((55, 50, 35), 0)])
    assert not new[55, 50, 35] and new[30, 30, 20]         # the clicked blob goes, the other stays


def test_run_arm_toggle_fixes_wrong_target():
    img, gt, lab = _case()
    A = lab == 4
    log = []
    touches = {3: True, 4: False}
    r = P.run_arm("T", "always", A, gt, lab, [4], {3, 4}, touches, None, 3, log, "x", ver="v1")
    assert r["dice_0"] == 0 and r["dice_3"] > 0.5
    assert "toggle_on" in [x["action"] for x in log]
    log2 = []
    r2 = P.run_arm("T", "always", A, gt, lab, [4], {3, 4}, touches, None, 3, log2, "x")          # v2 labels
    assert r2["dice_3"] == r["dice_3"] and "add_blob" in {x["action"] for x in log2}


def test_oracle_undoes_harmful_edit():
    img, gt, lab = _case()
    A = lab == 3
    log = []
    e = _editor(img, lambda x, c, l, low: np.zeros((128,) * 3, np.float32))   # SAM always proposes nothing
    r = P.run_arm("TS", "oracle", A, gt, lab, [3], {3, 4}, {3: True, 4: False}, e, 3, log, "x")
    assert all(r[f"dice_{s}"] >= r["dice_0"] - 1e-12 for s in range(4))


def _with_tables(tmp):
    base = _project(tmp)
    rows, rv = [], []
    pp = pd.read_csv(tmp / "s3k" / "per_patient.csv")
    for i, r in pp[pp.split == "train_oof"].iterrows():
        for k in (3, 4):
            rows.append(dict(cache_id=r.cache_id, k=k))
        rv.append(dict(cache_id=r.cache_id, risk_baseline=float(i), is_empty=int(str(r.B0_rad_subset) in ("", "nan")),
                       a_dice=r.B0_rad_dice, failure=int(r.B0_rad_dice < 0.5)))
    pd.DataFrame(rows).to_csv(tmp / "s3k" / "train_oof_table.csv", index=False)
    (tmp / "s3n").mkdir()
    pd.DataFrame(rv).to_csv(tmp / "s3n" / "train_oof_review.csv", index=False)
    keep = [x for x in base if x not in ("--workers", "1")]
    i = keep.index("--val_cases"); del keep[i:i + 2]
    return keep + ["--s3n_dir", str(tmp / "s3n"), "--budget_frac", "0.5"]


def test_end_to_end(tmp_path):
    base = [x.replace("s3m_contour_study.py", "s3p_click_repair.py") for x in _with_tables(tmp_path)]
    out = tmp_path / "out"
    r = subprocess.run(base + ["--step", "smoke", "--random_weights", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode != 0 and "parity first" in r.stdout + r.stderr
    r = subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PARITY OK" in r.stdout
    r = subprocess.run(base + ["--step", "full", "--random_weights", "--budget", "3", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "VERDICT" in r.stdout
    out = out / "_"                                                                       # results live in full_v2/
    out = out.parent
    df = pd.read_csv(out / "full_v2" / "per_patient.csv")
    assert set(df.arm) == set(P.ARMS) and set(df.variant) == set(P.VARIANTS)
    assert df.groupby(["arm", "variant"]).size().nunique() == 1
    a = df[df.arm == "TS"].set_index(["cache_id", "variant"]).dice_0
    assert (a == df[df.arm == "TB"].set_index(["cache_id", "variant"]).dice_0).all()   # same starting mask
    lg = pd.read_csv(out / "full_v2" / "clicks.csv")
    assert (lg.groupby(["cache_id", "arm", "variant"]).size() == 3).all()             # every click counted
    v = json.loads((out / "full_v2" / "decision_rule_check.json").read_text())
    assert set(v) >= {"mean_dnsd", "ci95", "harmed_TS", "harmed_TB", "advance_to_reader_study"}
    rule = json.loads((out / "decision_rule_v2.json").read_text())
    assert rule["min_mean_dnsd"] == 0.05 and rule["version"] == "v2"


def test_low_tau_keeps_small_blob():
    mc = P.M3.ball((128,) * 3, (64, 64, 64), 3)          # ~6 mm blob at 1.5 mm
    assert P.low_to_mask(P.mask_to_low(mc, 4.0, 0.5)).sum() <= P.low_to_mask(P.mask_to_low(mc, 4.0, 0.125)).sum()
    assert P.low_to_mask(P.mask_to_low(mc, 4.0, 0.125)).any()
    big = P.M3.ball((128,) * 3, (64, 64, 64), 20)
    assert np.allclose(P.mask_to_low(big, 4.0, 0.5).numpy(), 4.0 * (2 * __import__("torch").nn.functional.avg_pool3d(
        __import__("torch").from_numpy(big.astype(np.float32))[None, None], 4).numpy() - 1))


def test_calibrate_then_smoke(tmp_path):
    base = [x.replace("s3m_contour_study.py", "s3p_click_repair.py") for x in _with_tables(tmp_path)]
    out = tmp_path / "out"
    assert subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--step", "calibrate", "--random_weights", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    c = json.loads((out / "calibration.json").read_text())
    assert c["chosen_tau"] in c["tau_grid"] and c["chosen_L"] in c["L_grid"]
    par = pd.read_csv(out / "parity.csv")
    assert not set(c["patients_used"]) & set(par.cache_id)                          # outside the population
    r = subprocess.run(base + ["--step", "smoke", "--random_weights", "--budget", "2", "--smoke_n", "1", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "using calibration.json" in r.stdout
    rule = json.loads((out / "decision_rule_v2.json").read_text())
    assert rule["calibration"]["tau"] == c["chosen_tau"] and rule["config"]["mask_logit"] == c["chosen_L"]


def _zero_editor(img):
    return _editor(img, lambda x, c, l, low: np.zeros((128,) * 3, np.float32))


def test_v2_no_repeat_of_noop_click_and_every_attempt_counted():
    img, gt, lab = _case()
    A = lab == 3                                          # tumour partly missed -> positive clicks
    for ver, expect_repeat in (("v1", True), ("v2", False)):
        log = []
        P.run_arm("TS", "always", A, gt, lab, [3], {3, 4}, {3: True, 4: False}, _zero_editor(img), 3, log, "x", ver=ver)
        pts = [(tuple(r["p"]), r["sign"]) for r in log]
        assert len(log) == 3                               # every attempt counts
        assert (len(set(pts)) < len(pts)) == expect_repeat
    assert not any(r.get("duplicate_prompt") for r in log)


def test_v2_bounds_a_connected_leak():
    img, gt, lab = _case()
    e = _editor(img, lambda x, c, l, low: np.ones((128,) * 3, np.float32))       # SAM says 'tumour everywhere'
    m = lab == 3
    new_raw, info_raw = e.propose(m, [((45, 30, 20), 1)])
    new_b, info_b = e.propose(m, [((45, 30, 20), 1)], radius_mm=20)
    assert info_raw["raw_change_vox"] == info_raw["change_vox"] > info_b["change_vox"] == (new_b & ~m).sum()
    assert info_b["raw_change_vox"] == info_raw["raw_change_vox"]
    assert (new_b & ~m).sum() <= 4 / 3 * np.pi * 20 ** 3 + 1


def test_negative_click_on_partly_correct_blob_goes_to_editor():
    img, gt, lab = _case()
    lab2 = lab.copy()
    extra = P.M3.ball(SHAPE, (40, 30, 20), 5) & ~gt
    lab2[extra] = 3                                       # kept blob 3 = tumour part + an extra protrusion
    A = lab2 == 3
    p, sign, _ = P.next_click(A, gt)
    assert sign == 0 and extra[p]
    assert P.toggle_action(p, sign, {3}, {3, 4}, lab2, {3: True, 4: False}) is None   # not a whole-blob delete
    tb = P.brush(A, p, 0, 5)
    assert not tb[p] and tb[30, 30, 20]
    keep = A & ~extra
    e = _editor(img, lambda x, c, l, low: np.zeros((128,) * 3, np.float32))
    new, info = e.propose(A, [(p, 0)], radius_mm=20)
    assert not new[p] and info["change_vox"] > 0


def test_replay_reproduces_v1_and_probes_negatives(tmp_path):
    base = [x.replace("s3m_contour_study.py", "s3p_click_repair.py") for x in _with_tables(tmp_path)]
    out = tmp_path / "out"
    assert subprocess.run(base + ["--step", "parity", "--out", str(out)], capture_output=True, text=True).returncode == 0
    r = subprocess.run(base + ["--step", "smoke", "--version", "v1", "--random_weights", "--budget", "2", "--smoke_n", "2",
                               "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert (out / "smoke" / "clicks.csv").exists() and (out / "decision_rule_v1.json").exists()
    r = subprocess.run(base + ["--step", "replay", "--random_weights", "--budget", "2", "--smoke_n", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    import re
    m = re.search(r"reproduced: (\d+)/(\d+)", r.stdout)
    assert m and m.group(1) == m.group(2)
    assert (out / "replay" / "replay.csv").exists()
