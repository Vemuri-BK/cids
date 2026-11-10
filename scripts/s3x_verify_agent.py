"""S3x: verify-and-fallback for the one-click review agent (development data only; test refused).

The agent of the final review step (confirmed on test in S3t): QC triage -> one click -> stage-2 choice; the mask
delivered is the nearest SegResNet blob (Cn) or, when the agent picks SAM, F_max = SAM u SegResNet (R9).
S3x adds a VERIFIER after that choice: it predicts, without the expert mask, whether the delivered mask is still
bad (Dice < 0.5). Suspicious results are either switched to the other tool (fallback) or handed back to the
radiologist with a 'please check' flag.

Data: the S3z run (outputs/s3z_fusion: choices_<split>_deep.csv = S3v-OOF outcomes + S3z masks + agent choice).
Verifier: GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
  random_state=0) on VER features; target = delivered Dice < 0.5 if the patient were clicked; cross-fitted on the
  S3k outer folds over all 1,080 train patients (each prediction out-of-fold); val: fitted on all train.
VER features = S3v POST + S3z FUS + S3n BASE + QC risk + delivered-mask features (log volume, diameter, enhancement,
  which tool) + the alternative's log volume. No expert information.

Frozen rules (decision_rule.json written before the verifier is fitted; train OOF, deep click, 10% QC budget)
  R8 (primary)  among the QC-clicked patients, the verifier ranks still-failing results (Dice < 0.5) above the
                others: AUROC, patient-bootstrap 95% CI lower bound > 0.6
  R8b           fallback: the top 20% of clicked patients by verifier score switch to the other tool (Cn <-> F_max);
                mean Dice over all patients vs the agent without verification, paired bootstrap CI lower bound > 0
  reported      hand-back: of the still-failing clicked patients, how many fall in the flagged 20% (vs 20% random);
                AUROC with everyone clicked; val (12 clicked; 120 everyone clicked); oracle fallback (upper bound)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3n_qc_review as N  # noqa: E402
import s3v_review_agent as V  # noqa: E402
import s3z_fusion as Z  # noqa: E402

FLAG_FRAC, BUDGET, FAIL = 0.20, 0.10, 0.5
DELIV = ["d_log_ml", "d_diam", "d_enh", "d_is_fm", "alt_log_ml"]
VER = V.POST + Z.FUS + N.BASE + ["risk_baseline"] + DELIV
RULE = dict(
    data="S3z choices (S3v-OOF agent; delivered = F_max when the agent picks SAM, else nearest blob)",
    verifier=dict(model="GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, "
                        "subsample=0.8, random_state=0)", target=f"delivered Dice < {FAIL}", features=VER,
                  fit="cross-fitted on S3k outer folds over all train patients; val: fitted on all train"),
    R8=f"PRIMARY: train OOF, {BUDGET:.0%} QC-clicked patients: AUROC of the verifier for delivered Dice < {FAIL}, "
       "bootstrap CI lower bound > 0.6",
    R8b=f"top {FLAG_FRAC:.0%} of clicked by verifier switch to the other tool (Cn <-> F_max): mean Dice over all "
        "train patients vs no verification, paired bootstrap CI lower bound > 0",
    bootstrap=dict(n=5000, seed=0))


def gbc():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                      random_state=0)


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("cache_id").reset_index(drop=True).copy()
    fm = (d.tool2 == "S").to_numpy()
    d["deliv_dice"] = np.where(fm, d.FM_dice, d.Cn_dice)
    d["alt_dice"] = np.where(fm, d.Cn_dice, d.FM_dice)
    d["d_log_ml"] = np.log(np.where(fm, d.FM_ml, d.Cn_ml) + 1e-3)
    d["d_diam"] = np.where(fm, d.FM_diam, d.Cn_diam)
    d["d_enh"] = np.where(fm, d.FM_enh, d.Cn_enh)
    d["d_is_fm"] = fm.astype(int)
    d["alt_log_ml"] = np.log(np.where(fm, d.Cn_ml, d.FM_ml) + 1e-3)
    d["y"] = (d.deliv_dice < FAIL).astype(int)
    return d


def crossfit_proba(df, cols, folds):
    p = pd.Series(np.nan, index=df.index)
    f = df.cache_id.map(folds)
    for k in sorted(f.unique()):
        te = f == k
        m = gbc().fit(df.loc[~te, cols].fillna(-1), df.loc[~te, "y"])
        p[te] = m.predict_proba(df.loc[te, cols].fillna(-1))[:, 1]
    return p.to_numpy()


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot(fn, n_pat, n, seed):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n_pat, n_pat)) for _ in range(n)])
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2


def evaluate(d: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Rules on one split; d = prepare()d with a 'ver' column (verifier score)."""
    cid = d.cache_id.to_numpy()
    sel = V.select(d.qc.to_numpy(), cid, BUDGET)
    ci_idx = np.flatnonzero(sel)
    y, s = d.y.to_numpy(), d.ver.to_numpy()
    yc, sc = y[ci_idx], s[ci_idx]
    nflag = int(round(FLAG_FRAC * len(ci_idx)))
    flag_local = np.zeros(len(ci_idx), bool)
    flag_local[np.lexsort((cid[ci_idx], -sc))[:nflag]] = True
    flag = np.zeros(len(d), bool); flag[ci_idx[flag_local]] = True
    base = np.where(sel, d.deliv_dice, d.B_dice)
    fb = np.where(flag, d.alt_dice, base)
    oracle = np.where(sel, np.maximum(d.deliv_dice, d.alt_dice), d.B_dice)
    diff = fb - base
    caught = int((flag_local & (yc == 1)).sum())
    fails = int(yc.sum())
    r8 = auroc(yc, sc)
    lo8, hi8 = boot(lambda i: auroc(yc[i], sc[i]), len(ci_idx), n_boot, seed) if len(ci_idx) > 1 else [np.nan] * 2
    lo, hi = boot(lambda i: float(diff[i].mean()), len(d), n_boot, seed)
    hb = (flag_local & (yc == 1)).astype(float) - FLAG_FRAC * yc
    lo_h, hi_h = boot(lambda i: float(hb[i].sum()), len(ci_idx), n_boot, seed) if len(ci_idx) > 1 else [np.nan] * 2
    switched = flag & (d.alt_dice.to_numpy() != d.deliv_dice.to_numpy())
    return dict(
        n=int(len(d)), n_clicked=int(len(ci_idx)), still_failing_clicked=fails,
        R8=dict(auroc=r8, ci95=[lo8, hi8], passes=bool(lo8 > 0.6)),
        R8b=dict(n_flagged=nflag, dice_agent=float(base.mean()), dice_fallback=float(fb.mean()),
                 diff=float(diff.mean()), ci95=[lo, hi], better=int((diff > 1e-9).sum()),
                 worse=int((diff < -1e-9).sum()), passes=bool(lo > 0)),
        hand_back=dict(flagged=nflag, failures_caught=caught, failures_total=fails,
                       expected_random=FLAG_FRAC * fails, excess=caught - FLAG_FRAC * fails, excess_ci95=[lo_h, hi_h],
                       fails_01_after=int((base < 0.1).sum()), fails_05_after=int((base < 0.5).sum())),
        auroc_everyone_clicked=auroc(y, s),
        oracle_fallback_dice=float(oracle.mean()), n_switched_tool=int(switched.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3z_out", required=True, help="outputs/s3z_fusion (choices_<split>_deep.csv)")
    ap.add_argument("--folds", required=True, help="S3k folds.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = Path(a.s3z_out).expanduser()
    if "test" in z.name.lower():
        raise SystemExit("S3x is a development study: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any fit
    folds = dict(pd.read_csv(Path(a.folds).expanduser()).values)
    data = {}
    for split in ("train_oof", "val"):
        f = z / f"choices_{split}_deep.csv"
        if not f.exists():
            raise SystemExit(f"{f} missing: run s3z_fusion.py --step analyse first")
        d = pd.read_csv(f)
        miss = sorted(set(VER) - set(prepare(d).columns))
        if miss:
            raise SystemExit(f"{f}: missing columns {miss}")
        data[split] = prepare(d)
    tr, va = data["train_oof"], data["val"]
    if set(tr.cache_id) != set(folds):
        raise SystemExit("S3z train patients differ from the S3k folds")
    tr["ver"] = crossfit_proba(tr, VER, folds)
    va["ver"] = gbc().fit(tr[VER].fillna(-1), tr.y).predict_proba(va[VER].fillna(-1))[:, 1]
    n, seed = RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"]
    res = {s: evaluate(d, n, seed) for s, d in data.items()}
    imp = pd.Series(gbc().fit(tr[VER].fillna(-1), tr.y).feature_importances_, index=VER).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv", header=["importance"])
    for s, d in data.items():
        d[["cache_id", "qc", "tool2", "B_dice", "Cn_dice", "FM_dice", "deliv_dice", "alt_dice", "y", "ver"]].to_csv(
            out / f"verifier_{s}.csv", index=False)
    verdict = dict(R8_primary=res["train_oof"]["R8"]["passes"], R8b=res["train_oof"]["R8b"]["passes"], results=res)
    (out / "rules.json").write_text(json.dumps(verdict, indent=1, default=float))
    print("top verifier features:\n" + imp.head(8).round(3).to_string())
    print("\nRESULTS:", json.dumps(verdict, indent=1, default=float))


if __name__ == "__main__":
    main()
