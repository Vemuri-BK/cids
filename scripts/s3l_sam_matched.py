"""S3l: matched SAM comparison on top of the radiomics jury (development data only, CPU).

Question (frozen before fitting)
    Do the SAM-Med3D-turbo (S2b) checker features improve the binary radiomics jury B0_rad when everything
    else is identical: the same patients, candidate masks, radiomics, outer/inner folds, model grid and
    threshold rule?

Arms
    B0_rad      7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr          (S3k arm B, the reference)
    B0_rad_sam  the same 24 + 4 SAM checker features (agree, log_sam_ml, log_sam_soft_ml, log_sam_over_seg)
    SAM was run on these exact candidates (S3g out-of-fold candidates for train, S3f for val). The SAM-S2b
    decoder was fine-tuned on the training patients, so SAM itself is not out-of-fold (stated limitation).

Data
    The S3k tables (<s3k_out>/train_oof_table.csv, val_table.csv): candidate tables already merged with the
    radiomics, so both arms see byte-identical radiomics. Test never read.

Frozen decision rule (1,080 out-of-fold training patients; A = B0_rad, B = B0_rad_sam), same as S3k
    Route 1: CI lower bound of the paired mean Dice difference (B - A) > 0 and #Dice<0.1(B) <= #Dice<0.1(A)
    Route 2: #Dice<0.1(B) <= 0.75 * #Dice<0.1(A) and mean Dice(B) >= mean Dice(A) - 0.005
    Both:    #Dice<0.5(B) <= 1.10 * #Dice<0.5(A)
    SAM features qualify if either route passes; otherwise the jury stays B0_rad and only the measured AUC
    difference is reported (a ranking improvement only if its patient-bootstrap CI excludes 0).

Outputs (<out>): arms_summary.csv, paired_comparison.csv, auc.csv, failure_audit.csv, per_patient.csv,
    folds.csv, selected_thresholds.json, decision_rule.json, decision_rule_check.json, provenance.json

Usage (server)
    OMP_NUM_THREADS=1 python -u scripts/s3l_sam_matched.py --s3k_out ~/CIDS/outputs/s3k_radiomics \\
        --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3l_sam
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
import s3i_subset_jury as S  # noqa: E402
import s3j_selection_tradeoff as J  # noqa: E402
import s3k_radiomics_matched as K  # noqa: E402
from cids.radiomics import RAD_ALL  # noqa: E402

SAM4 = ["agree", "log_sam_ml", "log_sam_soft_ml", "log_sam_over_seg"]
RULE = {**K.RULE, "A": "B0_rad (7 finder + 15 radiomics + 2 relative)", "B": "B0_rad + 4 SAM features"}


def run(train, val, arms, C, out):
    """Fit both arms with identical folds/seeds/procedure; evaluate train OOF and val. Returns summaries."""
    folds = S.assign_folds([p["cid"] for p in train], C["outer_folds"], C["seed"])
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)
    names = list(arms)
    dec = {s: {n: {} for n in names} for s in ("train_oof", "val")}
    sco = {s: {n: {} for n in names} for s in ("train_oof", "val")}
    sel = []

    def apply(split, pats, fitted):
        for n, (arm, model, t) in fitted.items():
            for p in pats:
                dec[split][n][p["cid"]] = J.decide_threshold(arm, model, p, t)
                sco[split][n][p["cid"]] = arm.scores(model, p) if p["K"] > 0 else np.zeros(0)

    for f in range(C["outer_folds"]):
        tr = [p for p in train if folds[p["cid"]] != f]
        te = [p for p in train if folds[p["cid"]] == f]
        seed = C["seed"] + 101 * (f + 1)
        fitted, log = {}, {}
        for n in names:
            arm, model, t, lg = K.fit_arm(n, arms[n], tr, C, seed)
            fitted[n], log[n] = (arm, model, t), lg
        apply("train_oof", te, fitted)
        sel.append(dict(fold=f, **log))
        print(f"outer fold {f}: " + " | ".join(f"{n} t={v['threshold']}" for n, v in log.items()), flush=True)
    fitted, log = {}, {}
    for n in names:
        arm, model, t, lg = K.fit_arm(n, arms[n], train, C, C["seed"])
        fitted[n], log[n] = (arm, model, t), lg
    apply("val", val, fitted)
    sel.append(dict(fold="final_for_val", **log))
    (out / "selected_thresholds.json").write_text(json.dumps(sel, indent=1, default=float))

    a_, b_ = names
    summ, comp, aucs, aud, pp, verdict = [], [], [], [], [], None
    for split, pats in (("train_oof", train), ("val", val)):
        rows = {n: J.summarise(pats, dec[split][n], split, n) for n in names}
        summ += list(rows.values())
        d = {n: np.array([S.dice_counts(p["tgt"], dec[split][n][p["cid"]]) for p in pats]) for n in names}
        c = S.compare(b_, a_, d[b_], d[a_], C, "all")
        c["split"] = split
        comp.append(c)
        auc_a, auc_b = K.candidate_auc(pats, sco[split][a_]), K.candidate_auc(pats, sco[split][b_])
        lo, hi = K.auc_diff_boot(pats, sco[split][a_], sco[split][b_])
        aucs.append(dict(split=split, **{f"auc_{a_}": auc_a, f"auc_{b_}": auc_b}, diff=auc_b - auc_a,
                         ci95_lo=lo, ci95_hi=hi, ranking_improvement_supported=bool(lo > 0)))
        for i, p in enumerate(pats):
            da, db = d[a_][i], d[b_][i]
            pp.append(dict(split=split, cache_id=p["cid"], collection=p["coll"], K=p["K"],
                           outer_fold=folds.get(p["cid"], -1) if split == "train_oof" else -1,
                           **{f"{a_}_dice": da, f"{b_}_dice": db,
                              f"{a_}_subset": S.mask_str(p, dec[split][a_][p["cid"]]),
                              f"{b_}_subset": S.mask_str(p, dec[split][b_][p["cid"]])}))
            if (da < 0.1) != (db < 0.1):
                aud.append(dict(split=split, cache_id=p["cid"], K=p["K"], change="rescued" if db >= 0.1 else "harmed",
                                **{f"dice_{a_}": da, f"dice_{b_}": db},
                                reaches_0_5=bool(db >= 0.5) if db >= 0.1 else None))
        if split == "train_oof":
            verdict = K.rule_check(rows[a_], rows[b_], c["ci95_lo"])
    return summ, comp, aucs, aud, pp, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3k_out", required=True, help="S3k output folder (train_oof_table.csv, val_table.csv)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.loads(Path(a.config).read_text())
    src = Path(a.s3k_out).expanduser()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any fitting
    FA = list(C["finder_features"]) + RAD_ALL + K.REL2
    FB = FA + SAM4
    S.check_whitelist(FB, C["forbidden_features"])
    tr_csv, va_csv = src / "train_oof_table.csv", src / "val_table.csv"
    for f in (tr_csv, va_csv):
        if not f.exists():
            raise SystemExit(f"missing {f}: run S3k first")
    train = J.load_patients(tr_csv, FB, C["forbidden_features"])
    val = J.load_patients(va_csv, FB, C["forbidden_features"])
    if {p["cid"] for p in train} & {p["cid"] for p in val}:
        raise SystemExit("train and val share patients")
    nan_sam = sum(int(p["feats"][SAM4].isna().any(axis=1).sum()) for p in train + val if p["K"])
    print(f"S3l | train {len(train)} (K=0: {sum(p['K'] == 0 for p in train)}) | val {len(val)} | "
          f"A: {len(FA)} features | B: {len(FB)} features | candidates with a missing SAM value: {nan_sam}", flush=True)

    arms = {"B0_rad": FA, "B0_rad_sam": FB}
    summ, comp, aucs, aud, pp, verdict = run(train, val, arms, C, out)
    pd.DataFrame(summ).to_csv(out / "arms_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "paired_comparison.csv", index=False)
    pd.DataFrame(aucs).to_csv(out / "auc.csv", index=False)
    pd.DataFrame(aud).to_csv(out / "failure_audit.csv", index=False)
    pd.DataFrame(pp).to_csv(out / "per_patient.csv", index=False)
    (out / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    prov = dict(script="scripts/s3l_sam_matched.py", config=C, rule=RULE, features_A=FA, features_B=FB,
                inputs={k: dict(path=str(v), md5=K.md5(v)) for k, v in dict(train=tr_csv, val=va_csv).items()},
                note="Same tables as S3k (radiomics identical in both arms). SAM-S2b decoder saw the training "
                     "patients. Val is a reused development set. Test not read.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))

    pd.set_option("display.width", 200)
    cols = ["split", "policy", "dice", "dice_median", "dice_lt_0_1", "dice_lt_0_5", "kept_n", "harmful_removals"]
    print("\n" + pd.DataFrame(summ)[cols].round(4).to_string(index=False))
    print("\npaired B0_rad_sam - B0_rad (patient bootstrap):")
    print(pd.DataFrame(comp)[["split", "mean_diff", "ci95_lo", "ci95_hi", "better", "worse",
                              "new_fail_lt_0.1", "rescued_lt_0.1"]].round(4).to_string(index=False))
    print("\ncandidate AUC:")
    print(pd.DataFrame(aucs).round(4).to_string(index=False))
    print("\nfrozen rule (train OOF):", json.dumps(verdict, indent=1))
    print(f"\nVERDICT: {'SAM FEATURES QUALIFY' if verdict['radiomics_qualify'] else 'keep B0_rad (SAM features do not qualify)'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
