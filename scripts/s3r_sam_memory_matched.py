"""S3r: matched test of SAM's frozen-encoder memory as a second anchor for the radiomics jury (development, CPU).

Question (frozen before fitting)
    Pre-check (outputs/s3r_precheck): on the 46 train-OOF B0_rad wrong picks, the candidate ranked first by SAM's
    memory (mem_knn) is the tumour in 32, the best non-SAM pick (peak SegResNet probability / largest) in 22.
    Does adding SAM memory + a SAM anchor to B0_rad improve the jury, and is any gain due to SAM rather than to
    simply having a second anchor?

Arms (identical patients, candidates, radiomics, outer/inner folds, seeds, model grid, threshold rule)
    A  B0_rad        7 SegResNet + 15 radiomics + rel_seg_prob + rel_ring_cnr           (24, S3k reference)
    B  B0_rad_anchor A + non-SAM second anchor (control):                                (27)
                     rel_seg_prob_max, dist_to_peak_anchor_mm, peak_anchor_shift_mm
                     peak anchor = highest seg_prob_max (ties -> larger)
    C  B0_rad_mem    A + SAM memory + SAM anchor:                                         (30)
                     mem_knn, mem_probe, rel_mem_knn, rel_mem_probe, dist_to_sam_anchor_mm, sam_anchor_shift_mm
                     SAM anchor = highest mem_knn (ties -> larger); shift = distance finder anchor -> other anchor
                     (finder anchor = highest seg_prob_mean, ties -> larger, as rad_dist_to_dom_mm)
    rel_x = x - patient max. mem_knn chosen as the SAM anchor score after the pre-check (higher hit count and
    candidate AUC 0.80 vs 0.78 for mem_probe; S3e showed the same order).

Memory (no leakage)
    Descriptor: SAM-Med3D-turbo frozen image-encoder embedding, box-token mean || crop mean, 768-d (S3e/S3g).
    mem_knn: similarity-weighted share of real tumours among the 10 nearest memories (cosine, T 0.05);
    mem_probe: logistic regression (C 0.1, balanced). Both exactly as scripts/s3e_memory_jury.py.
    Outer fold f: held-out candidates are scored with a memory of the other folds' candidates only; training
    candidates get 5-fold patient-grouped cross-fitted scores inside the training part (never their own patient).
    Val: training candidates cross-fitted on all 1,080; val candidates scored with all training candidates.
    Limitations: the SAM encoder is frozen (not trained on MAMA-MIA) but its candidates' boxes come from the
    S3g/S3f pipeline; val is a reused development set.

Frozen decision rule (train OOF, 1,080 patients), K.rule_check (route 1 / route 2 / Dice<0.5 guard)
    PRIMARY   C vs A   -> SAM memory qualifies only if the rule passes
    ATTRIBUTE C vs B   -> the gain is credited to SAM only if the rule also passes against the control
    REPORTED  B vs A   (does a non-SAM second anchor already help?)
    Test never read. If C qualifies, any test evaluation is post-hoc (test already used once in S3o) and is a
    separate, explicitly labelled decision.

Parity: arm A must reproduce S3k's B0_rad per-patient train-OOF and val Dice exactly (checked when
    <s3k_out>/per_patient.csv exists).

Usage (server)
    OMP_NUM_THREADS=1 python -u scripts/s3r_sam_memory_matched.py --s3k_out ~/CIDS/outputs/s3k_radiomics \\
        --train_desc ~/CIDS/outputs/s3g_oof_masks/train_desc.npy \\
        --train_features ~/CIDS/outputs/s3g_oof_masks/train_features.csv \\
        --val_desc ~/CIDS/outputs/s3f_jury_masks/s3f_jury/val_desc.npy \\
        --val_features ~/CIDS/outputs/s3f_jury_masks/s3f_jury/val_features.csv \\
        --config configs/s3i_subset_jury.json --out ~/CIDS/outputs/s3r_sam_memory
"""
from __future__ import annotations

import argparse
import io
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

CENTRE = ["cx", "cy", "cz"]
CTRL3 = ["rel_seg_prob_max", "dist_to_peak_anchor_mm", "peak_anchor_shift_mm"]
MEM6 = ["mem_knn", "mem_probe", "rel_mem_knn", "rel_mem_probe", "dist_to_sam_anchor_mm", "sam_anchor_shift_mm"]
KNN_K, KNN_T, PROBE_C, CROSS_FOLDS = 10, 0.05, 0.1, 5
RULE = {**K.RULE, "A": "B0_rad (24)", "B": "B0_rad + non-SAM second anchor (27, control)",
        "C": "B0_rad + SAM memory + SAM anchor (30)",
        "primary": "C vs A: SAM memory qualifies if route 1 or route 2 passes (Dice<0.5 guard in both)",
        "attribution": "gain credited to SAM only if C also passes the same rule against B",
        "reported": "B vs A"}


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------
def knn_score(query, mem, mem_y, k=KNN_K):
    """Similarity-weighted share of real tumours among the k nearest memories (as scripts/s3e_memory_jury.py)."""
    sims = query.astype(np.float32) @ mem.astype(np.float32).T
    k = min(k, mem.shape[0])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / KNN_T)
    return (w * mem_y[idx]).sum(1) / w.sum(1)


def fit_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    if len(set(y)) < 2:
        return None
    return LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced").fit(X.astype(np.float32), y)


def score_memory(Dq, Dm, ym):
    pr = fit_probe(Dm, ym)
    probe = pr.predict_proba(Dq.astype(np.float32))[:, 1] if pr is not None else np.full(len(Dq), 0.5)
    return knn_score(Dq, Dm, ym), probe


def memory_columns(df_q, D_q, df_m=None, D_m=None):
    """mem_knn / mem_probe for the real rows of df_q. With a memory (df_m, D_m): scored against it.
    Without: patient-grouped cross-fitting inside df_q (never a candidate's own patient)."""
    from sklearn.model_selection import GroupKFold
    out = df_q.copy()
    out["mem_knn"], out["mem_probe"] = 0.0, 0.0
    rq = np.flatnonzero((df_q.k >= 0).to_numpy())
    if not len(rq):
        return out
    if df_m is not None:
        rm = np.flatnonzero((df_m.k >= 0).to_numpy())
        ym = df_m.on_tumour.to_numpy(float)[rm]
        knn, prb = score_memory(D_q[rq], D_m[rm], ym)
        out.iloc[rq, out.columns.get_loc("mem_knn")] = knn
        out.iloc[rq, out.columns.get_loc("mem_probe")] = prb
        return out
    y = df_q.on_tumour.to_numpy(float)
    groups = df_q.cache_id.to_numpy()[rq]
    nf = min(CROSS_FOLDS, len(set(groups)))
    if nf < 2:
        raise SystemExit("memory cross-fitting needs >= 2 patients with candidates")
    for fit, held in GroupKFold(nf).split(rq, groups=groups):
        a, b = rq[fit], rq[held]
        knn, prb = score_memory(D_q[b], D_q[a], y[a])
        out.iloc[b, out.columns.get_loc("mem_knn")] = knn
        out.iloc[b, out.columns.get_loc("mem_probe")] = prb
    return out


def _anchor_dist(real, score_col):
    """Per real row: distance (mm) to the patient's anchor (highest score_col, ties -> larger size_vox) and the
    shift between the finder anchor (highest seg_prob_mean, ties -> larger) and this anchor."""
    def anchor_of(col):
        o = real.sort_values(["cache_id", col, "size_vox"], ascending=[True, False, False])
        return o.groupby("cache_id")[CENTRE].first()
    a = anchor_of(score_col).reindex(real.cache_id).to_numpy()
    f = anchor_of("seg_prob_mean").reindex(real.cache_id).to_numpy()
    return np.linalg.norm(real[CENTRE].to_numpy() - a, axis=1), np.linalg.norm(f - a, axis=1)


def add_control(df):
    """Arm B features (label-free, no fitting)."""
    df = df.copy()
    for c in CTRL3:
        df[c] = 0.0
    r = (df.k >= 0).to_numpy()
    if not r.any():
        return df
    real = df[r]
    df.loc[r, "rel_seg_prob_max"] = (real.seg_prob_max - real.groupby("cache_id").seg_prob_max.transform("max")).to_numpy()
    d, s = _anchor_dist(real, "seg_prob_max")
    df.loc[r, "dist_to_peak_anchor_mm"], df.loc[r, "peak_anchor_shift_mm"] = d, s
    return df


def add_sam_derived(df):
    """Arm C derived features from mem_knn / mem_probe (already filled)."""
    df = df.copy()
    for c in MEM6[2:]:
        df[c] = 0.0
    r = (df.k >= 0).to_numpy()
    if not r.any():
        return df
    real = df[r]
    g = real.groupby("cache_id")
    df.loc[r, "rel_mem_knn"] = (real.mem_knn - g.mem_knn.transform("max")).to_numpy()
    df.loc[r, "rel_mem_probe"] = (real.mem_probe - g.mem_probe.transform("max")).to_numpy()
    d, s = _anchor_dist(real, "mem_knn")
    df.loc[r, "dist_to_sam_anchor_mm"], df.loc[r, "sam_anchor_shift_mm"] = d, s
    return df


def patients(df, cols, C):
    return J.load_patients(io.StringIO(df.to_csv(index=False)), cols, C["forbidden_features"])


def load_desc(table, desc_path, feat_path):
    D = np.load(Path(desc_path).expanduser()).astype(np.float32)
    if len(D) != len(table):
        raise SystemExit(f"{desc_path}: {len(D)} descriptor rows but {len(table)} table rows")
    if feat_path:
        f = pd.read_csv(Path(feat_path).expanduser(), usecols=["cache_id", "k"])
        if len(f) != len(table) or not (f.values == table[["cache_id", "k"]].values).all():
            raise SystemExit(f"{feat_path}: (cache_id, k) order differs from the S3k table -> descriptors misaligned")
    norms = np.linalg.norm(D[(table.k >= 0).to_numpy()], axis=1)
    if len(norms) and not np.allclose(norms, 1, atol=1e-2):
        raise SystemExit(f"{desc_path}: descriptors are not unit length (expected S3e/S3g descriptors)")
    return D


# ----------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------
def run(tr_df, Dtr, va_df, Dva, arms, C, out):
    names = list(arms)
    allc = sorted(set().union(*arms.values()))
    cids = sorted(tr_df.cache_id.unique())
    folds = S.assign_folds(cids, C["outer_folds"], C["seed"])
    pd.DataFrame(sorted(folds.items()), columns=["cache_id", "outer_fold"]).to_csv(out / "folds.csv", index=False)
    dec = {s: {n: {} for n in names} for s in ("train_oof", "val")}
    sco = {s: {n: {} for n in names} for s in ("train_oof", "val")}
    sel, mem_rows = [], []

    def apply(split, pats, fitted):
        for n, (arm, model, t) in fitted.items():
            for p in pats:
                dec[split][n][p["cid"]] = J.decide_threshold(arm, model, p, t)
                sco[split][n][p["cid"]] = arm.scores(model, p) if p["K"] > 0 else np.zeros(0)

    fold_of = tr_df.cache_id.map(folds).to_numpy()
    for f in range(C["outer_folds"]):
        itr, ite = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        a_df, b_df = tr_df.iloc[itr].reset_index(drop=True), tr_df.iloc[ite].reset_index(drop=True)
        a_df = add_sam_derived(memory_columns(a_df, Dtr[itr]))                          # cross-fitted inside
        b_df = add_sam_derived(memory_columns(b_df, Dtr[ite], a_df, Dtr[itr]))          # memory = other folds
        mem_rows.append(b_df.assign(outer_fold=f))
        tr, te = patients(a_df, allc, C), patients(b_df, allc, C)
        seed = C["seed"] + 101 * (f + 1)                                                 # identical for all arms
        fitted, log = {}, {}
        for n in names:
            arm, model, t, lg = K.fit_arm(n, arms[n], tr, C, seed)
            fitted[n], log[n] = (arm, model, t), lg
        apply("train_oof", te, fitted)
        sel.append(dict(fold=f, **log))
        print(f"outer fold {f}: " + " | ".join(f"{n} t={v['threshold']}" for n, v in log.items()), flush=True)
    full = add_sam_derived(memory_columns(tr_df, Dtr))
    vdf = add_sam_derived(memory_columns(va_df, Dva, tr_df, Dtr))
    train_all, val = patients(full, allc, C), patients(vdf, allc, C)
    fitted, log = {}, {}
    for n in names:
        arm, model, t, lg = K.fit_arm(n, arms[n], train_all, C, C["seed"])
        fitted[n], log[n] = (arm, model, t), lg
    apply("val", val, fitted)
    sel.append(dict(fold="final_for_val", **log))
    (out / "selected_thresholds.json").write_text(json.dumps(sel, indent=1, default=float))
    keep = ["cache_id", "k", "on_tumour"] + MEM6
    pd.concat(mem_rows)[keep + ["outer_fold"]].to_csv(out / "train_oof_memory_features.csv", index=False)
    vdf[keep].to_csv(out / "val_memory_features.csv", index=False)
    train_oof = patients(tr_df, list(arms[names[0]]), C)                                 # for metrics only
    return names, train_oof, val, folds, dec, sco, sel


def evaluate(names, train, val, folds, dec, sco, C):
    pairs = [("B0_rad_mem", "B0_rad"), ("B0_rad_mem", "B0_rad_anchor"), ("B0_rad_anchor", "B0_rad")]
    summ, comp, aucs, aud, pp, rules = [], [], [], [], [], {}
    for split, pats in (("train_oof", train), ("val", val)):
        rows = {n: J.summarise(pats, dec[split][n], split, n) for n in names}
        summ += list(rows.values())
        d = {n: np.array([S.dice_counts(p["tgt"], dec[split][n][p["cid"]]) for p in pats]) for n in names}
        auc = {n: K.candidate_auc(pats, sco[split][n]) for n in names}
        for b_, a_ in pairs:
            c = S.compare(b_, a_, d[b_], d[a_], C, "all")
            c["split"] = split
            comp.append(c)
            lo, hi = K.auc_diff_boot(pats, sco[split][a_], sco[split][b_])
            aucs.append(dict(split=split, comparison=f"{b_} - {a_}", auc_A=auc[a_], auc_B=auc[b_],
                             diff=auc[b_] - auc[a_], ci95_lo=lo, ci95_hi=hi, ranking_improvement_supported=bool(lo > 0)))
            if split == "train_oof":
                rules[f"{b_} vs {a_}"] = K.rule_check(rows[a_], rows[b_], c["ci95_lo"])
            for i, p in enumerate(pats):
                da, db = d[a_][i], d[b_][i]
                if (da < 0.1) != (db < 0.1):
                    aud.append(dict(split=split, comparison=f"{b_} vs {a_}", cache_id=p["cid"], K=p["K"],
                                    change="rescued" if db >= 0.1 else "harmed", dice_A=da, dice_B=db,
                                    reaches_0_5=bool(db >= 0.5) if db >= 0.1 else None))
        for i, p in enumerate(pats):
            r = dict(split=split, cache_id=p["cid"], collection=p["coll"], K=p["K"],
                     outer_fold=folds.get(p["cid"], -1) if split == "train_oof" else -1)
            for n in names:
                r[f"{n}_dice"] = d[n][i]
                r[f"{n}_subset"] = S.mask_str(p, dec[split][n][p["cid"]])
            pp.append(r)
    prim, attr = rules["B0_rad_mem vs B0_rad"], rules["B0_rad_mem vs B0_rad_anchor"]
    verdict = dict(primary_C_vs_A=prim, attribution_C_vs_B=attr, control_B_vs_A=rules["B0_rad_anchor vs B0_rad"],
                   sam_memory_qualifies=bool(prim["radiomics_qualify"]),
                   gain_attributable_to_sam=bool(prim["radiomics_qualify"] and attr["radiomics_qualify"]))
    return summ, comp, aucs, aud, pp, verdict


def parity(pp, s3k_pp_path):
    """Arm A must reproduce S3k's B0_rad per patient (train OOF and val)."""
    if not s3k_pp_path.exists():
        print("parity: S3k per_patient.csv not found -> skipped", flush=True)
        return dict(checked=False)
    ref = pd.read_csv(s3k_pp_path).set_index(["split", "cache_id"]).B0_rad_dice
    got = pd.DataFrame(pp).set_index(["split", "cache_id"]).B0_rad_dice
    common = got.index.intersection(ref.index)
    diff = float(np.abs(got[common] - ref[common]).max()) if len(common) else np.nan
    res = dict(checked=True, patients=int(len(common)), of=int(len(got)), max_abs_diff=diff)
    print(f"parity vs S3k B0_rad: {res}", flush=True)
    if len(common) != len(got) or not diff <= 1e-9:
        raise SystemExit(f"PARITY FAILED: arm A does not reproduce S3k B0_rad ({res})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3k_out", required=True, help="S3k output folder (train_oof_table.csv, val_table.csv)")
    ap.add_argument("--train_desc", required=True, help="S3g train_desc.npy (rows = S3g train_features.csv)")
    ap.add_argument("--val_desc", required=True, help="S3f val_desc.npy (rows = S3f val_features.csv)")
    ap.add_argument("--train_features", default="", help="S3g train_features.csv: row-order check (recommended)")
    ap.add_argument("--val_features", default="", help="S3f val_features.csv: row-order check (recommended)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.train_desc, a.val_desc, a.train_features, a.val_features):
        if "test" in Path(f).name.lower():
            raise SystemExit("refusing to read a test file in development mode")
    C = json.loads(Path(a.config).read_text())
    src = Path(a.s3k_out).expanduser()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))                   # frozen before fitting
    FA = list(C["finder_features"]) + RAD_ALL + K.REL2
    arms = {"B0_rad": FA, "B0_rad_anchor": FA + CTRL3, "B0_rad_mem": FA + MEM6}
    S.check_whitelist(FA + CTRL3 + MEM6, C["forbidden_features"])
    tr_csv, va_csv = src / "train_oof_table.csv", src / "val_table.csv"
    for f in (tr_csv, va_csv):
        if not f.exists():
            raise SystemExit(f"missing {f}: run S3k first")
    tr_df, va_df = pd.read_csv(tr_csv), pd.read_csv(va_csv)
    if set(tr_df.cache_id) & set(va_df.cache_id):
        raise SystemExit("train and val share patients")
    Dtr, Dva = load_desc(tr_df, a.train_desc, a.train_features), load_desc(va_df, a.val_desc, a.val_features)
    tr_df, va_df = add_control(tr_df), add_control(va_df)
    print(f"S3r | train {tr_df.cache_id.nunique()} patients / {(tr_df.k >= 0).sum()} candidates | "
          f"val {va_df.cache_id.nunique()} / {(va_df.k >= 0).sum()} | arms: "
          + ", ".join(f"{n} {len(c)}" for n, c in arms.items()), flush=True)

    names, train, val, folds, dec, sco, _ = run(tr_df, Dtr, va_df, Dva, arms, C, out)
    summ, comp, aucs, aud, pp, verdict = evaluate(names, train, val, folds, dec, sco, C)
    verdict["parity_A_vs_S3k"] = parity(pp, src / "per_patient.csv")
    pd.DataFrame(summ).to_csv(out / "arms_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "paired_comparison.csv", index=False)
    pd.DataFrame(aucs).to_csv(out / "auc.csv", index=False)
    pd.DataFrame(aud).to_csv(out / "failure_audit.csv", index=False)
    pd.DataFrame(pp).to_csv(out / "per_patient.csv", index=False)
    (out / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    ins = dict(train=tr_csv, val=va_csv, train_desc=Path(a.train_desc).expanduser(),
               val_desc=Path(a.val_desc).expanduser())
    prov = dict(script="scripts/s3r_sam_memory_matched.py", config=C, rule=RULE,
                features={n: c for n, c in arms.items()},
                memory=dict(knn_k=KNN_K, knn_T=KNN_T, probe_C=PROBE_C, cross_folds=CROSS_FOLDS),
                inputs={k: dict(path=str(v), md5=K.md5(v)) for k, v in ins.items()},
                note="Same S3k tables as S3k/S3l. Memory never contains the scored patient. Val reused "
                     "development set. Test not read.")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))

    pd.set_option("display.width", 200)
    cols = ["split", "policy", "dice", "dice_median", "dice_lt_0_1", "dice_lt_0_5", "kept_n", "harmful_removals"]
    print("\n" + pd.DataFrame(summ)[cols].round(4).to_string(index=False))
    print("\npaired differences (patient bootstrap):")
    print(pd.DataFrame(comp)[["split", "comparison", "mean_diff", "ci95_lo", "ci95_hi", "better", "worse",
                              "new_fail_lt_0.1", "rescued_lt_0.1"]].round(4).to_string(index=False))
    print("\ncandidate AUC:")
    print(pd.DataFrame(aucs)[["split", "comparison", "auc_A", "auc_B", "diff", "ci95_lo", "ci95_hi"]]
          .round(4).to_string(index=False))
    print("\nfrozen rule (train OOF):", json.dumps(verdict, indent=1))
    v = ("SAM MEMORY QUALIFIES AND BEATS THE NON-SAM CONTROL" if verdict["gain_attributable_to_sam"] else
         "SAM MEMORY QUALIFIES vs B0_rad BUT NOT vs THE NON-SAM CONTROL" if verdict["sam_memory_qualifies"] else
         "keep B0_rad (SAM memory does not qualify)")
    print(f"\nVERDICT: {v}\n-> {out}")


if __name__ == "__main__":
    main()
