"""Verification tests for scripts/s3i_subset_jury.py (synthetic data; CPU, seconds)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s3i_subset_jury as S  # noqa: E402

C = json.loads((ROOT / "configs" / "s3i_subset_jury.json").read_text())
C_FAST = {**C, "hgb_grid": C["hgb_grid"][:1], "inner_folds": 2, "outer_folds": 2}
COLS = C["finder_features"] + C["sam_features"]


def synth_table(n_pat=40, seed=0):
    """Synthetic candidate table with the S3f columns; components are disjoint by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        K = int(rng.integers(1, 6))
        vol = np.sort(rng.uniform(100, 20000, K))[::-1].round()
        G = float(rng.uniform(2000, 30000))
        for r in range(K):
            inter = float(min(vol[r], G) * rng.uniform(0, 1)) if rng.random() < 0.6 else 0.0
            rows.append(dict(cache_id=f"duke_{i:03d}", k=int(r + 3), rank=r, n_cand=K, seg_ml=vol[r] / 1000,
                             size_vox=vol[r], inter_vox=round(inter), gt_vox=G,
                             log_seg_ml=np.log(vol[r] / 1000), rel_size=vol[r] / vol[0], is_largest=int(r == 0),
                             seg_prob_mean=rng.uniform(.5, 1), seg_prob_max=rng.uniform(.8, 1),
                             agree=rng.uniform(0, 1), sam_ml=1.0, sam_soft_ml=1.0,
                             log_sam_ml=rng.normal(), log_sam_soft_ml=rng.normal(), log_sam_over_seg=rng.normal(),
                             on_tumour=int(inter > 0), rel20_inter=0, rel20_size=0))
    return pd.DataFrame(rows)


@pytest.fixture
def pats(tmp_path):
    p = tmp_path / "train_features.csv"
    synth_table().to_csv(p, index=False)
    return S.load_split(str(p), COLS, C["forbidden_features"])


def test_subset_counts():
    for K, n in [(1, 1), (2, 3), (3, 7), (5, 31)]:
        assert len(S.enumerate_subsets(K)) == n
    assert all(m.any() for m in S.enumerate_subsets(5))


def test_count_dice_equals_mask_dice():
    rng = np.random.default_rng(1)
    for _ in range(20):
        shape = (24, 24, 24)
        lab = np.zeros(shape, int)
        K = int(rng.integers(1, 6))
        for k in range(1, K + 1):                          # disjoint boxes as components
            x = 4 * k
            lab[x:x + 3, 2:10, 2:10] = k
        gt = rng.random(shape) < 0.3
        tgt = dict(I=np.array([(gt & (lab == k)).sum() for k in range(1, K + 1)], float),
                   V=np.array([(lab == k).sum() for k in range(1, K + 1)], float), G=float(gt.sum()))
        for m in S.enumerate_subsets(K):
            pred = np.isin(lab, np.flatnonzero(m) + 1)
            d_mask = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())
            assert abs(S.dice_counts(tgt, m) - d_mask) < 1e-12


def test_patient_weights_sum_to_one(pats):
    arm = S.SubsetArm("U0", False, C_FAST)
    tot = {}
    for p in pats:
        masks, _ = arm._sub(p)
        tot[p["cid"]] = sum(1.0 / len(masks) for _ in masks)
    assert all(abs(v - 1) < 1e-12 for v in tot.values())


def test_forbidden_columns_rejected():
    with pytest.raises(ValueError):
        S.check_whitelist(COLS + ["inter_vox"], C["forbidden_features"])
    for c in ["inter_vox", "gt_vox", "on_tumour", "rel20_inter"]:
        assert c not in COLS


def test_fold_separation(pats):
    folds = S.assign_folds([p["cid"] for p in pats], 5, 42)
    assert set(folds) == {p["cid"] for p in pats}
    for f in range(5):
        tr = {c for c, v in folds.items() if v != f}
        te = {c for c, v in folds.items() if v == f}
        assert not (tr & te) and tr | te == set(folds)


def _fit(pats, name):
    arm = {"B0": S.ThresholdArm("B0", "clf", C["finder_features"], C_FAST),
           "U1": S.SubsetArm("U1", True, C_FAST)}[name]
    arm.fit_locked(pats, C_FAST["inner_folds"], 0)
    return arm


def test_candidate_order_invariance(tmp_path):
    df = synth_table(seed=3)
    p1, p2 = tmp_path / "a_features.csv", tmp_path / "b_features.csv"
    df.to_csv(p1, index=False)
    df.sample(frac=1, random_state=7).to_csv(p2, index=False)       # shuffled row order
    A = S.load_split(str(p1), COLS, C["forbidden_features"])
    B = S.load_split(str(p2), COLS, C["forbidden_features"])
    arm = _fit(A, "U1")
    for pa, pb in zip(A, B):
        ma, _ = arm.decide(pa)
        mb, _ = arm.decide(copy.deepcopy(pb))
        assert set(pa["k"][ma]) == set(pb["k"][mb])


def test_gt_columns_cannot_change_inference(pats):
    arm_u, arm_b = _fit(pats, "U1"), _fit(pats, "B0")
    for p in pats:
        q = copy.deepcopy(p)
        q.pop("sub_sam", None)
        q["tgt"] = dict(I=np.zeros_like(p["tgt"]["I"]), V=p["tgt"]["V"] * 3, G=1.0)   # corrupt targets
        assert np.array_equal(arm_u.decide(p)[0], arm_u.decide(q)[0])
        assert np.array_equal(arm_b.decide(p)[0], arm_b.decide(q)[0])


def test_k1_and_k0_bypass(pats):
    arm = _fit(pats, "U1")
    p1 = next(p for p in pats if p["K"] == 1)
    m, st = arm.decide(p1)
    assert st == "single_candidate" and m.tolist() == [True]
    p0 = dict(cid="x", coll="X", K=0, k=np.array([], int), feats=pd.DataFrame(columns=COLS),
              tgt=dict(I=np.zeros(0), V=np.zeros(0), G=1.0))
    m, st = arm.decide(p0)
    assert st == "no_candidate" and m.size == 0 and S.dice_counts(p0["tgt"], m) == 0.0


def test_tie_prefers_reference(pats):
    class Flat:
        def predict(self, X):
            return np.zeros(len(X))
    arm = S.SubsetArm("U0", False, C_FAST)
    p = next(q for q in pats if q["K"] >= 3)
    ref = np.zeros(p["K"], bool)
    ref[[0, 2]] = True
    m, _ = arm.choose(Flat(), p, ref)
    assert np.array_equal(m, ref)
    m, _ = arm.choose(Flat(), p, None)
    assert m.sum() == 1 and m[0]                                     # fixed rule: first in enumeration
