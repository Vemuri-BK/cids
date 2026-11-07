"""S3k: matched radiomics comparison on the main binary jury B0 (development data only, CPU).

Question (frozen before fitting)
    Do radiomics improve the binary jury B0 when everything else is identical: the same patients, the same
    candidate masks, the same outer/inner patient folds, the same model grid and the same threshold rule?

Arms
    B0      7 SegResNet features (config finder_features)
    B0_rad  the same 7 + 15 radiomics (shape, contrast, texture, distance to the anchor candidate)
            + 2 relative features that need no SAM / memory (rel_seg_prob, rel_ring_cnr)
    Both: HistGradientBoostingClassifier, config grid, threshold = best inner-fold mean Dice, highest-score
    fallback, K = 1 patients keep their candidate, K = 0 patients Dice 0 (real ids kept).

Data
    train  S3g out-of-fold candidate table + S3g component maps (cases/<cid>.npz, key 'lab')
    val    S3f val candidate table + S3f component maps (reused development set, reported separately)
    Radiomics are extracted here for BOTH splits with the same code (cids.radiomics, RAD_VERSION recorded),
    from the MRI cache inside each candidate's component (never from the ground truth). Test never read.

Frozen decision rule (1,080 out-of-fold training patients; A = B0, B = B0_rad)
    Route 1, better average segmentation:
        lower bound of the paired patient-bootstrap 95% CI of mean Dice (B - A) > 0
        and #Dice<0.1(B) <= #Dice<0.1(A)
    Route 2, fewer near-total failures:
        #Dice<0.1(B) <= 0.75 * #Dice<0.1(A)  and  mean Dice(B) >= mean Dice(A) - 0.005
    Both routes: #Dice<0.5(B) <= 1.10 * #Dice<0.5(A)
    Radiomics qualify if either route passes; otherwise B0 is kept and only the measured AUC difference
    is reported (a ranking improvement is claimed only if its patient-bootstrap CI excludes 0).

Outputs (<out>)
    {train,val}_radiomics.csv (+ manifest), arms_summary.csv, paired_comparison.csv, auc.csv,
    failure_audit.csv, per_patient.csv, selected_thresholds.json, decision_rule.json,
    decision_rule_check.json, provenance.json

Usage (server)
    OMP_NUM_THREADS=1 python -u scripts/s3k_radiomics_matched.py --cache_root ~/data \\
        --train ~/CIDS/outputs/s3g_oof_masks/train_features.csv --train_cases ~/CIDS/outputs/s3g_oof_masks/cases \\
        --val ~/CIDS/outputs/s3f_jury_masks/s3f_jury/val_features.csv --val_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases \\
        --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3k_radiomics --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3i_subset_jury as S  # noqa: E402
import s3j_selection_tradeoff as J  # noqa: E402
from cids.radiomics import RAD_ALL, RAD_EXTRACTED, RAD_VERSION, add_position, extract_case  # noqa: E402

REL2 = ["rel_seg_prob", "rel_ring_cnr"]
RULE = dict(route1="CI lower bound of mean Dice (B - A) > 0 and #Dice<0.1(B) <= #Dice<0.1(A)",
            route2="#Dice<0.1(B) <= 0.75 * #Dice<0.1(A) and mean Dice(B) >= mean Dice(A) - 0.005",
            both="#Dice<0.5(B) <= 1.10 * #Dice<0.5(A)", fail_cut=0.25, margin=0.005, mid_tol=0.10,
            evaluated_on="train out-of-fold (1,080 patients); val reported separately, no role in the decision")


# ----------------------------------------------------------------------------------------------
# radiomics extraction (same code for both splits) and table assembly
# ----------------------------------------------------------------------------------------------
def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def cases_dir(path: Path) -> Path:
    if any(path.glob("*.npz")):
        return path
    hit = next(path.rglob("*.npz"), None) if path.is_dir() else None
    if hit is None:
        raise SystemExit(f"no component maps (*.npz) under {path}")
    return hit.parent


def extract(split, df, cache_paths, cdir, out, workers, table_md5):
    f, mf = out / f"{split}_radiomics.csv", out / f"{split}_radiomics.manifest.json"
    manifest = dict(rad_version=RAD_VERSION, table_md5=table_md5, cases_dir=str(cdir))
    if f.exists() and mf.exists() and json.loads(mf.read_text()) == manifest:
        print(f"{split}: using cached {f.name}", flush=True)
        return pd.read_csv(f)
    jobs = []
    for cid, g in df[df.k >= 0].groupby("cache_id", sort=True):
        case = cdir / f"{cid}.npz"
        if not case.exists():
            raise SystemExit(f"{split}: missing component map {case}")
        jobs.append((cid, str(cache_paths[cid]), str(case), g.sort_values("rank").k.astype(int).tolist()))
    rows, t0 = [], time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(extract_case, jobs, chunksize=4)):
                rows.extend(r)
                if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                    print(f"{split}: {i + 1}/{len(jobs)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        for j in jobs:
            rows.extend(extract_case(j))
    rad = pd.DataFrame(rows, columns=["cache_id", "k"] + RAD_EXTRACTED)
    rad.to_csv(f, index=False)
    mf.write_text(json.dumps(manifest, indent=2))
    return rad


def add_rel2(df: pd.DataFrame) -> pd.DataFrame:
    """rel_seg_prob and rel_ring_cnr exactly as cids.radiomics.add_relative (the two clues needing no SAM)."""
    df = df.copy()
    for c in REL2:
        df[c] = 0.0
    r = (df.k >= 0).to_numpy()
    if not r.any():
        return df
    g = df[r].groupby("cache_id")
    df.loc[r, "rel_seg_prob"] = df.loc[r, "seg_prob_mean"].to_numpy() / np.maximum(
        g["seg_prob_mean"].transform("max").to_numpy(), 1e-6)
    df.loc[r, "rel_ring_cnr"] = df.loc[r, "rad_ring_cnr"].to_numpy() - g["rad_ring_cnr"].transform("max").to_numpy()
    return df


def assemble(df, rad):
    m = df.merge(rad, on=["cache_id", "k"], how="left", validate="one_to_one")
    real = m[m.k >= 0]
    if real[["cx", "cy", "cz"]].isna().all(axis=1).any():
        raise SystemExit("some candidates have no radiomics row")
    return add_rel2(add_position(m))


# ----------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------
def candidate_auc(pats, scores):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([(p["tgt"]["I"] > 0).astype(int) for p in pats if p["K"] > 0])
    s = np.concatenate([scores[p["cid"]] for p in pats if p["K"] > 0])
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auc_diff_boot(pats, sa, sb, n=2000, seed=0):
    """Patient-clustered bootstrap of AUC(B) - AUC(A) (all candidates of a patient resampled together)."""
    from sklearn.metrics import roc_auc_score
    P = [p for p in pats if p["K"] > 0]
    ys = [(p["tgt"]["I"] > 0).astype(int) for p in P]
    a_, b_ = [sa[p["cid"]] for p in P], [sb[p["cid"]] for p in P]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(P), len(P))
        y = np.concatenate([ys[i] for i in idx])
        if 0 < y.sum() < len(y):
            d.append(roc_auc_score(y, np.concatenate([b_[i] for i in idx])) -
                     roc_auc_score(y, np.concatenate([a_[i] for i in idx])))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def rule_check(sa, sb, ci_lo):
    """sa / sb: summary rows (dict) of A = B0 and B = B0_rad on train OOF."""
    mid_ok = sb["dice_lt_0_5"] <= (1 + RULE["mid_tol"]) * sa["dice_lt_0_5"]
    r1 = bool(ci_lo > 0 and sb["dice_lt_0_1"] <= sa["dice_lt_0_1"] and mid_ok)
    r2 = bool(sb["dice_lt_0_1"] <= (1 - RULE["fail_cut"]) * sa["dice_lt_0_1"]
              and sb["dice"] >= sa["dice"] - RULE["margin"] and mid_ok)
    return dict(route1_better_average=r1, route2_fewer_failures=r2, dice_lt_0_5_ok=bool(mid_ok),
                ci_lo_mean_dice_diff=float(ci_lo),
                dice_A=float(sa["dice"]), dice_B=float(sb["dice"]),
                fail01_A=int(sa["dice_lt_0_1"]), fail01_B=int(sb["dice_lt_0_1"]),
                fail01_route2_max=int(np.floor((1 - RULE["fail_cut"]) * sa["dice_lt_0_1"] + 1e-9)),
                fail05_A=int(sa["dice_lt_0_5"]), fail05_B=int(sb["dice_lt_0_5"]),
                fail05_max=int(np.floor((1 + RULE["mid_tol"]) * sa["dice_lt_0_5"] + 1e-9)),
                radiomics_qualify=bool(r1 or r2))


def fit_arm(name, cols, train_part, C, seed):
    arm = J.BinaryJury(name, cols, C, weighted=False)
    tab = arm.inner_table(train_part, C["inner_folds"], seed)
    best = J.pick_best(tab)
    model = arm._fit(train_part, C["hgb_grid"][int(best.config_index)], seed)
    log = dict(config_index=int(best.config_index), threshold=float(best.t), inner_dice=float(best.dice),
               inner_fail01=int(best.fail01))
    return arm, model, float(best.t), log


def main():
    from cids.data.cache import find_cache_root
    from cids.provenance import require_cases
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--train_cases", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_cases", required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for f in (a.train, a.val):
        if "test" in Path(f).name.lower():
            raise SystemExit("refusing to read a test file in development mode")
    C = json.loads(Path(a.config).read_text())
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any fitting
    FA = list(C["finder_features"])
    FB = FA + RAD_ALL + REL2
    S.check_whitelist(FB, C["forbidden_features"])

    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    tables = {}
    for split, csv, cd, name in (("train_oof", a.train, a.train_cases, "train"), ("val", a.val, a.val_cases, "val")):
        df = pd.read_csv(Path(csv).expanduser())
        paths = dict(require_cases(sp, root, name))
        missing = sorted(set(df.cache_id) - set(paths))
        if missing:
            raise SystemExit(f"{split}: {len(missing)} patients not in the '{name}' split (e.g. {missing[:3]})")
        rad = extract(split, df, paths, cases_dir(Path(cd).expanduser()), out, a.workers, md5(Path(csv).expanduser()))
        t = assemble(df, rad)
        t.to_csv(out / f"{split}_table.csv", index=False)
        tables[split] = J.load_patients(out / f"{split}_table.csv", FB, C["forbidden_features"])
    train, val = tables["train_oof"], tables["val"]
    if {p["cid"] for p in train} & {p["cid"] for p in val}:
        raise SystemExit("train and val share patients")
    print(f"S3k | train {len(train)} (K=0: {sum(p['K'] == 0 for p in train)}) | val {len(val)} | "
          f"A: {len(FA)} features | B: {len(FB)} features | RAD_VERSION {RAD_VERSION}", flush=True)

    folds = S.assign_folds([p["cid"] for p in train], C["outer_folds"], C["seed"])
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)
    arms = {"B0": FA, "B0_rad": FB}
    dec = {s: {n: {} for n in arms} for s in ("train_oof", "val")}
    sco = {s: {n: {} for n in arms} for s in ("train_oof", "val")}
    sel = []

    def apply(split, pats, fitted):
        for n, (arm, model, t) in fitted.items():
            for p in pats:
                dec[split][n][p["cid"]] = J.decide_threshold(arm, model, p, t)
                sco[split][n][p["cid"]] = arm.scores(model, p) if p["K"] > 0 else np.zeros(0)

    for f in range(C["outer_folds"]):
        tr = [p for p in train if folds[p["cid"]] != f]
        te = [p for p in train if folds[p["cid"]] == f]
        seed = C["seed"] + 101 * (f + 1)                                      # identical for both arms
        fitted, log = {}, {}
        for n, cols in arms.items():
            arm, model, t, lg = fit_arm(n, cols, tr, C, seed)
            fitted[n], log[n] = (arm, model, t), lg
        apply("train_oof", te, fitted)
        sel.append(dict(fold=f, **log))
        print(f"outer fold {f}: " + " | ".join(f"{n} t={v['threshold']}" for n, v in log.items()), flush=True)
    fitted, log = {}, {}
    for n, cols in arms.items():
        arm, model, t, lg = fit_arm(n, cols, train, C, C["seed"])
        fitted[n], log[n] = (arm, model, t), lg
    apply("val", val, fitted)
    sel.append(dict(fold="final_for_val", **log))
    (out / "selected_thresholds.json").write_text(json.dumps(sel, indent=1, default=float))

    summ, comp, aucs, aud, pp = [], [], [], [], []
    verdict = None
    for split, pats in (("train_oof", train), ("val", val)):
        rows = {n: J.summarise(pats, dec[split][n], split, n) for n in arms}
        summ += list(rows.values())
        d = {n: np.array([S.dice_counts(p["tgt"], dec[split][n][p["cid"]]) for p in pats]) for n in arms}
        c = S.compare("B0_rad", "B0", d["B0_rad"], d["B0"], C, "all")
        c["split"] = split
        comp.append(c)
        auc_a, auc_b = candidate_auc(pats, sco[split]["B0"]), candidate_auc(pats, sco[split]["B0_rad"])
        lo, hi = auc_diff_boot(pats, sco[split]["B0"], sco[split]["B0_rad"])
        aucs.append(dict(split=split, auc_B0=auc_a, auc_B0_rad=auc_b, diff=auc_b - auc_a, ci95_lo=lo, ci95_hi=hi,
                         ranking_improvement_supported=bool(lo > 0)))
        for i, p in enumerate(pats):
            da, db = d["B0"][i], d["B0_rad"][i]
            pp.append(dict(split=split, cache_id=p["cid"], collection=p["coll"], K=p["K"],
                           outer_fold=folds.get(p["cid"], -1) if split == "train_oof" else -1,
                           B0_dice=da, B0_rad_dice=db, B0_subset=S.mask_str(p, dec[split]["B0"][p["cid"]]),
                           B0_rad_subset=S.mask_str(p, dec[split]["B0_rad"][p["cid"]])))
            if (da < 0.1) != (db < 0.1):
                aud.append(dict(split=split, cache_id=p["cid"], K=p["K"],
                                change="rescued" if db >= 0.1 else "harmed", dice_B0=da, dice_B0_rad=db,
                                reaches_0_5=bool(db >= 0.5) if db >= 0.1 else None))
        if split == "train_oof":
            verdict = rule_check(rows["B0"], rows["B0_rad"], c["ci95_lo"])
    pd.DataFrame(summ).to_csv(out / "arms_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "paired_comparison.csv", index=False)
    pd.DataFrame(aucs).to_csv(out / "auc.csv", index=False)
    pd.DataFrame(aud).to_csv(out / "failure_audit.csv", index=False)
    pd.DataFrame(pp).to_csv(out / "per_patient.csv", index=False)
    (out / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    prov = dict(script="scripts/s3k_radiomics_matched.py", config=C, rule=RULE, rad_version=RAD_VERSION,
                features_A=FA, features_B=FB,
                inputs={k: dict(path=str(v), md5=md5(Path(v).expanduser())) for k, v in
                        dict(train=a.train, val=a.val).items()},
                note="Train: S3g out-of-fold candidates and component maps. Val: S3f (reused development set). "
                     "Radiomics extracted here for both with the same code. Test not read.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))

    pd.set_option("display.width", 200)
    cols = ["split", "policy", "dice", "dice_median", "dice_lt_0_1", "dice_lt_0_5", "kept_n", "harmful_removals"]
    print("\n" + pd.DataFrame(summ)[cols].round(4).to_string(index=False))
    print("\npaired B0_rad - B0 (patient bootstrap):")
    print(pd.DataFrame(comp)[["split", "mean_diff", "ci95_lo", "ci95_hi", "better", "worse",
                              "new_fail_lt_0.1", "rescued_lt_0.1"]].round(4).to_string(index=False))
    print("\ncandidate AUC:")
    print(pd.DataFrame(aucs).round(4).to_string(index=False))
    print("\nfrozen rule (train OOF):", json.dumps(verdict, indent=1))
    print(f"\nVERDICT: {'RADIOMICS QUALIFY' if verdict['radiomics_qualify'] else 'keep B0 (radiomics do not qualify)'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
