"""S3n (Q1): can a box-only SAM second opinion help pick out failed B0_rad masks at a fixed review budget?

Exploratory, development data only (train OOF + val; test never read). The delivered mask stays A = B0_rad.
The question is which finished examinations a reader with a fixed budget should review.

Steps
  --step scores    CPU. Re-fit B0_rad exactly as S3k/S3l (same tables, features, outer/inner folds, seeds) and export
                   FOLD-CORRECT candidate scores: train = the outer-fold model that never saw the patient, val = the
                   model fitted on all training patients. Every patient's kept set and Dice must reproduce S3k
                   (per_patient.csv B0_rad_subset / B0_rad_dice) exactly, otherwise the step stops.
  --step evaluate  CPU. Review models and the frozen comparison (decision_rule.json is written before any
                   evaluation is computed).

Review scores (no expert mask used)
  baseline   logistic regression (standardised, L2, C = 1.0 fixed) on 7 patient features of the delivered mask:
             kept_prob_mean (size-weighted SegResNet mean probability of kept blobs), kept_prob_max, margin_kept_min
             (lowest kept jury score minus threshold), margin_drop_max (highest dropped score minus threshold;
             -1 if none dropped), n_cand, n_kept, log_kept_ml
  +SAM       the identical model plus sam_disagree = 1 - Dice(A, B), B = box-only SAM patient mask at 1 mm before
             fallback (S3m --step overlap)
  Train scores are out-of-fold (S3k outer folds); val scores come from the model fitted on all train patients.
  Patients with an empty delivered mask (no candidate) are reviewed first in BOTH methods and count in the budget.

Frozen rule (train OOF, 1,080 patients)
  failure = A Dice < 0.5; budget = round(10% x N) reviews (108); ties broken by cache_id
  SAM qualifies as a QC flag (exploratory) if  caught(+SAM) - caught(baseline) >= +10  AND the paired
  patient-bootstrap 95% CI lower bound of that difference > 0 (5000 resamples; the budget is recomputed as 10% of
  each resample; scores are fixed, so refitting variability is not included).
  Reported: caught / missed at 5, 10, 20%, AUROC, AUPRC, missed failures by type (wrong target: Dice < 0.1;
  poor boundary: 0.1 <= Dice < 0.5), SAM runtime. Evidence: the S2b decoder was trained on the 1,080 and
  checkpoint-selected on the 120, so a positive result needs independent confirmation.

Usage (server)
  OMP_NUM_THREADS=1 python scripts/s3n_qc_review.py --step scores --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3n_qc
  python scripts/s3n_qc_review.py --step evaluate --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --overlap ~/CIDS/outputs/s3m_overlap/overlap.csv --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3n_qc
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3i_subset_jury as S  # noqa: E402
import s3j_selection_tradeoff as J  # noqa: E402
import s3k_radiomics_matched as K  # noqa: E402
from cids.radiomics import RAD_ALL  # noqa: E402

BASE = ["kept_prob_mean", "kept_prob_max", "margin_kept_min", "margin_drop_max", "n_cand", "n_kept", "log_kept_ml"]
SAMF = ["sam_disagree"]
RULE = dict(failure="A Dice < 0.5", budget_frac=0.10, min_gain_caught=10, ci_lo_gt=0.0, bootstrap=5000, seed=0,
            model="LogisticRegression(C=1.0, L2) on standardised features", baseline_features=BASE,
            sam_features=BASE + SAMF, empty_first=True, tie_break="cache_id", secondary_budgets=[0.05, 0.20],
            evidence="exploratory; S2b decoder trained on train 1,080 and checkpoint-selected on val 120")


def kept_from_subset(s):
    if s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() in ("", "nan"):
        return set()
    return {int(float(x)) for x in str(s).split("+")}


# ---------------------------------------------------------------------------------------------- scores
def step_scores(a, C, out):
    src = Path(a.s3k_out).expanduser()
    FA = list(C["finder_features"]) + RAD_ALL + K.REL2
    train = J.load_patients(src / "train_oof_table.csv", FA, C["forbidden_features"])
    val = J.load_patients(src / "val_table.csv", FA, C["forbidden_features"])
    folds = S.assign_folds([p["cid"] for p in train], C["outer_folds"], C["seed"])
    rows = []

    def export(split, pats, arm, model, t, fold):
        for p in pats:
            if p["K"] == 0:
                continue
            sc = arm.scores(model, p)
            keep = J.decide_threshold(arm, model, p, t)
            for k, s_, kp in zip(p["k"], sc, keep):
                rows.append(dict(split=split, cache_id=p["cid"], k=int(k), score=float(s_), threshold=t,
                                 kept=bool(kp), outer_fold=fold))

    for f in range(C["outer_folds"]):
        tr = [p for p in train if folds[p["cid"]] != f]
        te = [p for p in train if folds[p["cid"]] == f]
        arm, model, t, _ = K.fit_arm("B0_rad", FA, tr, C, C["seed"] + 101 * (f + 1))
        export("train_oof", te, arm, model, t, f)
        print(f"outer fold {f}: t={t}", flush=True)
    arm, model, t, _ = K.fit_arm("B0_rad", FA, train, C, C["seed"])
    export("val", val, arm, model, t, -1)
    js = pd.DataFrame(rows)
    # reproduction check against S3k: identical kept sets and Dice for every patient
    pp = pd.read_csv(src / "per_patient.csv", dtype={"B0_rad_subset": str})
    bad = []
    for split, pats in (("train_oof", train), ("val", val)):
        ref = pp[pp.split == split].set_index("cache_id")
        g = js[js.split == split]
        for p in pats:
            mine = set(g[(g.cache_id == p["cid"]) & g.kept].k)
            theirs = kept_from_subset(ref.loc[p["cid"], "B0_rad_subset"])
            m = np.isin(p["k"], sorted(mine)) if p["K"] else np.zeros(0, bool)
            d = S.dice_counts(p["tgt"], m)
            if mine != theirs or abs(d - ref.loc[p["cid"], "B0_rad_dice"]) > 1e-9:
                bad.append((split, p["cid"], sorted(mine), sorted(theirs)))
    js.to_csv(out / "jury_scores.csv", index=False)
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)
    if bad:
        raise SystemExit(f"REPRODUCTION FAILED for {len(bad)} patients, e.g. {bad[:5]}")
    print(f"REPRODUCED: kept sets and Dice identical to S3k for all {len(train)} train OOF + {len(val)} val patients")
    print(f"-> {out / 'jury_scores.csv'}")


# ---------------------------------------------------------------------------------------------- features
def patient_features(table, scores, overlap, pp):
    """One row per patient: review features (no expert mask) + the failure label (evaluation only)."""
    rows = []
    sc = scores.set_index(["cache_id", "k"])
    ov = overlap.set_index("cache_id")
    lab = pp.set_index("cache_id")
    for cid, g in table.groupby("cache_id", sort=True):
        r = g[g.k >= 0]
        d = float(lab.loc[cid, "B0_rad_dice"])
        row = dict(cache_id=cid, a_dice=d, failure=int(d < 0.5), n_cand=len(r))
        if not len(r):
            row.update(is_empty=1, n_kept=0, **{c: 0.0 for c in BASE if c not in ("n_cand", "n_kept")}, sam_disagree=1.0)
            rows.append(row)
            continue
        s = np.array([sc.loc[(cid, int(k)), "score"] for k in r.k])
        t = float(sc.loc[(cid, int(r.k.iloc[0])), "threshold"])
        kept = np.array([bool(sc.loc[(cid, int(k)), "kept"]) for k in r.k])
        kk = r[kept]
        w = kk.size_vox.to_numpy(float)
        dab = ov.loc[cid, "dice_ab"]
        row.update(is_empty=0, n_kept=int(kept.sum()),
                   kept_prob_mean=float((kk.seg_prob_mean.to_numpy() * w).sum() / w.sum()),
                   kept_prob_max=float(kk.seg_prob_max.max()),
                   margin_kept_min=float((s[kept] - t).min()),
                   margin_drop_max=float((s[~kept] - t).max()) if (~kept).any() else -1.0,
                   log_kept_ml=float(np.log(w.sum() / 1000)),
                   sam_disagree=float(1 - dab) if not (isinstance(dab, float) and math.isnan(dab)) else 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def oof_risk(df, cols, folds):
    risk = pd.Series(np.nan, index=df.index)
    ne = df["is_empty"] == 0
    for f in sorted(set(folds.values())):
        te = ne & (df.cache_id.map(folds) == f)
        tr = ne & ~te
        m = _model().fit(df.loc[tr, cols], df.loc[tr, "failure"])
        risk[te] = m.predict_proba(df.loc[te, cols])[:, 1]
    risk[~ne] = np.inf                                         # empty masks are reviewed first
    return risk


def fit_apply(train_df, other, cols):
    ne_t = train_df["is_empty"] == 0
    m = _model().fit(train_df.loc[ne_t, cols], train_df.loc[ne_t, "failure"])
    r = pd.Series(np.inf, index=other.index)
    ne = other["is_empty"] == 0
    r[ne] = m.predict_proba(other.loc[ne, cols])[:, 1]
    return r


def review(risk, cid, frac):
    """Boolean mask of reviewed patients: top round(frac*N) by risk (empty = inf first), ties by cache_id."""
    n = int(round(frac * len(risk)))
    order = np.lexsort((np.asarray(cid), -np.asarray(risk, float)))
    sel = np.zeros(len(risk), bool)
    sel[order[:n]] = True
    return sel


def caught(fail, risk, cid, frac):
    return int((review(risk, cid, frac) & (fail == 1)).sum())


def boot_diff(fail, r_base, r_sam, cid, frac, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(fail)
    fail, rb, rs, cid = map(np.asarray, (fail, r_base, r_sam, cid))
    d = np.empty(n)
    for b in range(n):
        i = rng.integers(0, N, N)
        tie = np.arange(N)                                     # duplicates get distinct, fixed tie order
        d[b] = caught(fail[i], rs[i], tie, frac) - caught(fail[i], rb[i], tie, frac)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def metrics_block(split, df, rb, rs, frac_list):
    from sklearn.metrics import average_precision_score, roc_auc_score
    fail, cid = df.failure.to_numpy(), df.cache_id.to_numpy()
    out = []
    for frac in frac_list:
        for name, r in (("baseline", rb), ("+SAM", rs)):
            sel = review(r, cid, frac)
            miss = (~sel) & (fail == 1)
            out.append(dict(split=split, budget=frac, method=name, reviews=int(sel.sum()), failures=int(fail.sum()),
                            caught=int((sel & (fail == 1)).sum()), missed=int(miss.sum()),
                            missed_wrong_target=int((miss & (df.a_dice < 0.1)).sum()),
                            missed_poor_boundary=int((miss & (df.a_dice >= 0.1)).sum()),
                            empty_reviewed=int((sel & (df["is_empty"] == 1)).sum())))
    fin = np.isfinite(np.asarray(rb, float))
    auc = {}
    for name, r in (("baseline", rb), ("+SAM", rs)):
        rr = np.where(np.isfinite(r), r, 2.0)
        auc[name] = dict(auroc=float(roc_auc_score(fail, rr)) if 0 < fail.sum() < len(fail) else np.nan,
                         auprc=float(average_precision_score(fail, rr)) if fail.sum() else np.nan)
    return out, auc, int((~fin).sum())


def step_evaluate(a, C, out):
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))           # frozen before evaluation
    src = Path(a.s3k_out).expanduser()
    scores = pd.read_csv(out / "jury_scores.csv")
    folds = dict(pd.read_csv(out / "folds.csv").values)
    overlap = pd.read_csv(Path(a.overlap).expanduser())
    pp = pd.read_csv(src / "per_patient.csv", dtype={"B0_rad_subset": str})
    res, summ, verdict = {}, [], None
    for split, tab, osplit in (("train_oof", "train_oof_table.csv", "train"), ("val", "val_table.csv", "val")):
        t = pd.read_csv(src / tab)
        ov = overlap[overlap.split == osplit]
        if set(ov.cache_id) != set(t.cache_id):
            raise SystemExit(f"{split}: overlap.csv patients differ from the S3k table (run S3m --step overlap)")
        df = patient_features(t, scores[scores.split == split], ov, pp[pp.split == split])
        res[split] = df
    tr = res["train_oof"]
    rb = oof_risk(tr, BASE, folds); rs = oof_risk(tr, BASE + SAMF, folds)
    va = res["val"]
    vb = fit_apply(tr, va, BASE); vs = fit_apply(tr, va, BASE + SAMF)
    aucs = {}
    for split, df, b, s in (("train_oof", tr, rb, rs), ("val", va, vb, vs)):
        rows, auc, n_empty = metrics_block(split, df, b.to_numpy(), s.to_numpy(), [0.05, 0.10, 0.20])
        summ += rows; aucs[split] = auc
        df = df.assign(risk_baseline=b.to_numpy(), risk_sam=s.to_numpy())
        df.to_csv(out / f"{split}_review.csv", index=False)
    fail, cid = tr.failure.to_numpy(), tr.cache_id.to_numpy()
    cb, cs = caught(fail, rb.to_numpy(), cid, 0.10), caught(fail, rs.to_numpy(), cid, 0.10)
    lo, hi = boot_diff(fail, rb.to_numpy(), rs.to_numpy(), cid, 0.10, RULE["bootstrap"], RULE["seed"])
    verdict = dict(budget_reviews=int(round(0.10 * len(tr))), failures=int(fail.sum()), caught_baseline=cb,
                   caught_sam=cs, gain=cs - cb, gain_ci95=[lo, hi], min_gain=RULE["min_gain_caught"],
                   sam_qc_qualifies=bool(cs - cb >= RULE["min_gain_caught"] and lo > RULE["ci_lo_gt"]),
                   auc=aucs, sam_seconds_per_patient=float(overlap[overlap.split == "train"].sam_seconds.mean()))
    pd.DataFrame(summ).to_csv(out / "review_summary.csv", index=False)
    (out / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summ).to_string(index=False))
    print("\nfailure-detection AUROC / AUPRC:", json.dumps(aucs, indent=1))
    print("\nfrozen rule (train OOF, 10% budget):", json.dumps(verdict, indent=1, default=float))
    print("\nVERDICT:", "SAM QC FLAG QUALIFIES (exploratory; needs independent confirmation)" if verdict["sam_qc_qualifies"]
          else "SAM disagreement does not add enough failures caught at the 10% budget")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["scores", "evaluate"], required=True)
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlap", default="", help="S3m --step overlap output (overlap.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.loads(Path(a.config).read_text())
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "scores":
        step_scores(a, C, out)
    else:
        if not a.overlap:
            raise SystemExit("--overlap (S3m overlap.csv) is required")
        step_evaluate(a, C, out)


if __name__ == "__main__":
    main()
