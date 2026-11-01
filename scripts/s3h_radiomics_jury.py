"""S3h: radiomics clues for the memory jury (CPU only, no network training).

Reads the S3f outputs (the candidates of the 13-clue memory jury) and the MRI cache, then
  1. extracts radiomics clues for every candidate (cids.radiomics) -> <out>/<split>_radiomics.csv
     (+ manifest: re-used only for the same S3f tables, cache and RAD_VERSION); multi-core
  2. fits one jury per variant on train (cids.jury.CrossFitMemoryJury: the full jury + 5 out-of-fold juries,
     same gradient boosting as S3e/S3f) and chooses its cut-off on the TRAINING patients' out-of-fold scores
     (1080 patients) with the plateau-centre rule: the median of all cut-offs within 0.002 of the best
     out-of-fold Dice, rounded to 0.05 (the curve is flat, its exact maximum is noise -- run 2 picked 0.80)
  3. DEV mode (default): train out-of-fold + val only. The test tables are NOT loaded.
     FINAL mode (--final, run once): + test, paired statistics, and the export of the proposed jury and the
     jury13 baseline: model bundles (.pkl with cut-off, features, library versions, provenance), every
     candidate's score as CSV (version-independent record), final masks and full metrics (Dice, HD95, NSD,
     missed) on val and test, parity mask Dice == table Dice, reload check of the bundles.

Variants (features):
  finder_only            finder clues only (size, rank, SegResNet probability)
  jury11                 + SAM checker clues (= S3d)
  jury13                 + SAM memory clues (= S3e/S3f jury, refitted here under the same protocol)
  finder_rad             finder clues + radiomics + relative (no SAM, no memory)
  jury13_rel             jury13 + within-patient relative clues
  jury13_rad             jury13 + radiomics (shape, contrast, texture, position)
  jury13_rad_rel         jury13 + radiomics + relative = PROPOSED (fixed in advance)
  -<group>               proposed minus one group (shape / contrast / texture / position / relative)

Example (server)
  dev:    python scripts/s3h_radiomics_jury.py --s3f_dir ~/CIDS/outputs/s3f_jury_masks/s3f_jury \\
              --cache_root ~/data --out ~/CIDS/outputs/s3h_radiomics --workers 8
  final:  same command + --final   (once, after the dev results are agreed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3d_learned_jury as s3d  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import all_metrics  # noqa: E402
from cids.jury import JURY11, JURY13, CrossFitMemoryJury, keep_candidates  # noqa: E402
from cids.provenance import cases_digest, file_digest, require_cases, save_npz_atomic  # noqa: E402
from cids.radiomics import (GROUPS, RAD_ALL, RAD_EXTRACTED, RAD_VERSION, REL, add_position, add_relative,  # noqa: E402
                            extract_case, load_bundle, versions)

PROPOSED, BASELINE = "jury13_rad_rel", "jury13"
PLATEAU = 0.002                      # cut-off rule margin (fixed before the dev run)
FINDER = ["log_seg_ml", "rel_size", "is_largest", "rank", "n_cand", "seg_prob_mean", "seg_prob_max"]
assert set(FINDER) < set(JURY11)


def variants() -> dict:
    prop = JURY13 + RAD_ALL + REL
    v = {"finder_only": FINDER,
         "jury11": JURY11,
         "jury13": JURY13,
         "finder_rad": FINDER + RAD_ALL + ["rel_seg_prob", "rel_ring_cnr"],
         "jury13_rel": JURY13 + [c for c in REL if c != "rel_ring_cnr"],   # no radiomics-derived clue
         "jury13_rad": JURY13 + RAD_ALL,
         "jury13_rad_rel": prop}
    for g, cols in GROUPS.items():
        v[f"-{g}"] = [c for c in prop if c not in cols]
    return v


# ---------------------------------------------------------------------------- 1. extraction
def load_tables(s3f: Path, split: str, limit: int):
    df = pd.read_csv(s3f / f"{split}_features.csv")
    D = np.load(s3f / f"{split}_desc.npy")
    if len(df) != len(D):
        raise SystemExit(f"{split}: {len(df)} table rows != {len(D)} descriptors")
    if limit:
        keep_ids = list(dict.fromkeys(df.cache_id))[:limit]
        sel = df.cache_id.isin(keep_ids).to_numpy()
        df, D = df[sel].reset_index(drop=True), D[sel]
    return df, D


def find_cases_dir(s3f: Path) -> Path:
    """S3f's cases/<cid>.npz; Kaggle zips may unpack one level deeper (cases/cases/*.npz) -> follow it."""
    top = s3f / "cases"
    if any(top.glob("*.npz")):
        return top
    hit = next(top.rglob("*.npz"), None) if top.is_dir() else None
    if hit is None:
        raise SystemExit(f"no case files (*.npz) under {top}: S3f cases/ missing or still zipped")
    return hit.parent


def extract_split(split, df, cache_paths, cases_dir, out, workers, manifest):
    f, mf = out / f"{split}_radiomics.csv", out / f"{split}_radiomics.manifest.json"
    if s3d.cache_ok([f], manifest, mf):
        print(f"{split}: using cached {f.name} (same manifest)")
        return pd.read_csv(f)
    jobs = []
    for cid, g in df[df.k >= 0].groupby("cache_id", sort=False):
        ks = g.sort_values("rank").k.astype(int).tolist()          # rank 0 = largest = dominant
        case = cases_dir / f"{cid}.npz"
        if not case.exists():
            raise SystemExit(f"{split}: missing {case} (S3f cases/ incomplete)")
        jobs.append((cid, str(cache_paths[cid]), str(case), ks))
    rows, t0 = [], time.time()
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(extract_case, jobs, chunksize=4)):
                rows.extend(r)
                if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                    print(f"{split}: {i + 1}/{len(jobs)} cases | {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        for i, j in enumerate(jobs):
            rows.extend(extract_case(j))
    rad = pd.DataFrame(rows, columns=["cache_id", "k"] + RAD_EXTRACTED)
    rad.to_csv(f, index=False); mf.write_text(json.dumps(manifest, indent=2))
    print(f"{split}: {len(rad)} candidates from {len(jobs)} cases, {(time.time() - t0) / 60:.1f} min")
    return rad


def merge(df, rad):
    m = df.merge(rad, on=["cache_id", "k"], how="left", validate="one_to_one")
    real = (m.k >= 0).to_numpy()
    miss = m.loc[real, RAD_EXTRACTED].isna().all(axis=1).sum()
    if miss:
        raise SystemExit(f"{miss} real candidates without radiomics row (tables out of sync)")
    return add_position(m)


# ---------------------------------------------------------------------------- 2./3. juries + stats
def auc(df, p):
    from sklearn.metrics import roc_auc_score
    r = (df.k >= 0).to_numpy()
    y = df.on_tumour.to_numpy()[r]
    return float(roc_auc_score(y, p[r])) if 0 < y.sum() < len(y) else None


def paired(a: pd.Series, b: pd.Series, seed=0):
    from scipy.stats import wilcoxon
    d = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return dict(delta=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                wilcoxon_p=float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() > 0 else 1.0,
                better=int((d > 0.05).sum()), worse=int((d < -0.05).sum()))


def importance(jury, va, Dva, seed=0):
    from sklearn.inspection import permutation_importance
    rows = jury.memory_clues(va, Dva)
    r = (rows.k >= 0).to_numpy()
    X, y = rows.loc[r, jury.features].to_numpy(float), rows.on_tumour.to_numpy()[r]
    if len(set(y)) < 2:
        return pd.DataFrame(columns=["feature", "auc_drop", "sd"])
    res = permutation_importance(jury.trees, X, y, scoring="roc_auc", n_repeats=10, random_state=seed)
    return pd.DataFrame(dict(feature=jury.features, auc_drop=res.importances_mean, sd=res.importances_std)) \
        .sort_values("auc_drop", ascending=False)


def curve(df, p) -> pd.Series:
    """Mean Dice at every cut-off (index = threshold)."""
    m = s3d.rule_tables(df, p).mean()
    m = m[[c for c in m.index if c.startswith("learned_")]]
    return pd.Series(m.to_numpy(), index=[float(c.split("_t")[1]) for c in m.index])


def plateau_cut(c: pd.Series, margin: float = PLATEAU) -> float:
    """Centre of the cut-offs whose Dice is within `margin` of the best (median, rounded to 0.05)."""
    near = np.array(sorted(c.index[c >= c.max() - margin]))
    return round(float(np.round(np.median(near) / 0.05) * 0.05), 2)


def score_variant(name, n_feat, tr, ptr, va, pv, te=None, pt=None):
    """Main cut-off = plateau centre of the training patients' OUT-OF-FOLD Dice curve (1080 patients).
    Also reports the val-best cut-off (secondary). te / pt None in dev mode."""
    ctr = curve(tr, ptr) if ptr is not None else None
    cva = curve(va, pv)
    cte = curve(te, pt) if te is not None else None
    t_val = float(cva.idxmax())
    t_tr = plateau_cut(ctr) if ctr is not None else t_val
    tv = s3d.rule_tables(va, pv)
    row = dict(variant=name, n_features=n_feat, t_train=t_tr,
               train_oof_dice=float(ctr[np.isclose(ctr.index, t_tr)].iloc[0]) if ctr is not None else None,
               val_dice=float(tv[f"learned_t{t_tr:.2f}"].mean()), t_val=t_val, val_dice_best=float(cva.max()),
               auc_train_oof=auc(tr, ptr) if ptr is not None else None, auc_val=auc(va, pv))
    dt = None
    if te is not None:
        tt = s3d.rule_tables(te, pt)
        dt = tt[f"learned_t{t_tr:.2f}"]
        row.update(test_dice=float(dt.mean()), test_missed=int((dt < 0.1).sum()),
                   test_dice_valcut=float(tt[f"learned_t{t_val:.2f}"].mean()), auc_test=auc(te, pt))
    curves = [dict(variant=name, split=s, threshold=float(th), dice=float(v))
              for s, c in (("train_oof", ctr), ("val", cva), ("test", cte)) if c is not None for th, v in c.items()]
    return dict(row=row, test_cases=dt, curves=curves)


def published_s3f(s3f: Path, va, Dva, te, Dte):
    """The saved S3f jury (memory_jury.pkl) at its own cut-off; skipped with a message if it cannot be loaded
    here (it was saved with scikit-learn 1.6.1 on Kaggle and does not load with 1.9.1)."""
    import pickle
    f = s3f / "memory_jury.pkl"
    if not f.exists():
        print(f"[info] {f.name} not found -> no 's3f_published' row")
        return None
    try:
        b = pickle.load(open(f, "rb"))
        jury, t = b["jury"], float(b["threshold"])
        pv, pt = jury.proba(va, Dva), jury.proba(te, Dte)
    except Exception as e:
        print(f"[info] {f.name} cannot be used here ({type(e).__name__}: {e}) -> no 's3f_published' row")
        return None
    r = score_variant("s3f_published", len(jury.features), None, None, va, pv, te, pt)
    tt = s3d.rule_tables(te, pt)[f"learned_t{t:.2f}"]
    r["row"].update(t_train=None, t_val=t, val_dice=float(s3d.rule_tables(va, pv)[f"learned_t{t:.2f}"].mean()),
                    test_dice=float(tt.mean()), test_dice_valcut=float(tt.mean()), test_missed=int((tt < 0.1).sum()))
    r["test_cases"] = tt
    return r


def curve_wins(cv, base, lo=0.2, hi=0.8, splits=("test", "val")):
    """At how many cut-offs in [lo, hi] each version's Dice is above `base`."""
    out = {}
    sel = cv[(cv.threshold >= lo - 1e-9) & (cv.threshold <= hi + 1e-9)]
    for split in splits:
        piv = sel[sel.split == split].pivot(index="threshold", columns="variant", values="dice")
        if base not in piv:
            continue
        for n in piv.columns:
            if n != base:
                out.setdefault(n, dict(n=len(piv)))[split] = int((piv[n] > piv[base] + 1e-12).sum())
    return out


def print_row(r):
    tt = f"{r['t_train']:.2f}" if r["t_train"] is not None else "  - "
    s = (f"{r['variant']:16s} t_train={tt}  val {r['val_dice']:.4f}  AUC val {r['auc_val']:.3f}")
    if "test_dice" in r:
        s += f" | test {r['test_dice']:.4f}  AUC test {r['auc_test']:.3f} | t_val={r['t_val']:.2f} test {r['test_dice_valcut']:.4f}"
    print(s, flush=True)


# ---------------------------------------------------------------------------- 4. export (--final)
def _mask_job(args):
    cid, case_path, cache_path, ks = args
    lab = np.load(case_path)["lab"]
    _, lbl = load_case(cache_path)
    m = np.isin(lab, ks) if ks else np.zeros(lab.shape, bool)
    return cid, m, all_metrics(m, lbl > 0)


def export(name, jury, t, feats, splits_data, cases_dir, cache_paths, out, workers, provenance, coll):
    """Bundle + final masks + full metrics of one jury on val and test; parity and reload checks."""
    import pickle
    d = out / "final" / name
    (d / "masks").mkdir(parents=True, exist_ok=True)
    bundle = dict(variant=name, jury=jury, threshold=t, features=list(feats), rad_version=RAD_VERSION,
                  plateau_margin=PLATEAU, settings=dict(top_k=5, min_vox=100, prob=0.5, overlap=0.5),
                  versions=versions(), provenance=provenance)
    with open(d / f"{name}.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    re = load_bundle(d / f"{name}.pkl")                         # reload check: same scores, same kept candidates
    frames, parity = [], {}
    for split, (df, D) in splits_data.items():
        p = jury.proba(df, D)
        p2 = re["jury"].proba(df, D)
        assert np.array_equal(p, p2), f"{name} {split}: reloaded bundle scores differ"
        keep = keep_candidates(df, p, t)
        jobs = [(cid, str(cases_dir / f"{cid}.npz"), str(cache_paths[cid]), keep.get(cid, [])) for cid in keep]
        rows = []
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                res = list(pool.imap_unordered(_mask_job, jobs, chunksize=4))
        else:
            res = [_mask_job(j) for j in jobs]
        for cid, m, met in res:
            save_npz_atomic(d / "masks" / f"{cid}.npz", mask=m.astype(np.uint8), kept=np.array(keep[cid], np.int32))
            rows.append(dict(cache_id=cid, split=split, collection=coll.get(cid), **met))
        fr = pd.DataFrame(rows).sort_values("cache_id")
        tab = s3d.rule_tables(df, p)[f"learned_t{t:.2f}"]
        parity[split] = float(np.abs(fr.set_index("cache_id").dice - tab.reindex(fr.cache_id).to_numpy()).max())
        frames.append(fr)
    assert max(parity.values()) < 1e-6, f"{name}: mask Dice != table Dice {parity}"
    met = pd.concat(frames)
    met.to_csv(d / "metrics.csv", index=False)
    agg = met.groupby("split").agg(cases=("dice", "size"), dice=("dice", "mean"), dice_median=("dice", "median"),
                                   hd95_median=("hd95", "median"), hd95_mean=("hd95", "mean"), nsd=("nsd", "mean"),
                                   missed=("dice", lambda x: int((x < 0.1).sum())))
    per = met.pivot_table(index="collection", columns="split", values="dice").round(4)
    json.dump(dict(variant=name, threshold=t, metrics=agg.round(4).to_dict(orient="index"),
                   per_collection=per.to_dict(orient="index"), parity_max_abs_diff=parity, versions=bundle["versions"]),
              open(d / "summary.json", "w"), indent=2, default=float)
    print(f"\nFINAL {name} (cut-off {t:.2f}) -> {d}\n{agg.round(4).to_string()}\nper hospital Dice:\n{per.to_string()}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3f_dir", required=True, help="S3f output folder (train/val/test_features.csv, *_desc.npy, cases/)")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=10, help="memory neighbours (as S3f)")
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--final", action="store_true", help="ALSO evaluate test once + export (run only after dev)")
    a = ap.parse_args()
    t0 = time.time()
    s3f, out = Path(a.s3f_dir).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection))
    cases_dir = find_cases_dir(s3f)
    mode = "FINAL (train + val + test, export)" if a.final else "DEV (train out-of-fold + val; test NOT loaded)"
    print(f"S3f {s3f}\ncases {cases_dir} ({len(list(cases_dir.glob('*.npz')))} files)\ncache {root}\nout {out}\n"
          f"mode: {mode}\nproposed = {PROPOSED} (fixed) | cut-off = plateau centre (margin {PLATEAU}) of train out-of-fold Dice")

    data, cache_paths = {}, {}
    for split in (("train", "val", "test") if a.final else ("train", "val")):
        df, D = load_tables(s3f, split, a.limit)
        cases = [c for c in require_cases(sp, root, split) if c[0] in set(df.cache_id)]
        if len(cases) != df.cache_id.nunique():
            raise SystemExit(f"{split}: {df.cache_id.nunique()} cases in the table, {len(cases)} in the cache")
        manifest = dict(rad_version=RAD_VERSION, split=split, limit=a.limit,
                        table=file_digest(s3f / f"{split}_features.csv"), **cases_digest(cases))
        rad = extract_split(split, df, dict(cases), cases_dir, out, a.workers, manifest)
        data[split] = (merge(df, rad), D)
        cache_paths.update(dict(cases))
    (tr, Dtr), (va, Dva) = data["train"], data["val"]
    te, Dte = data.get("test", (None, None))

    res, per_case, curves, juries, scores = [], {}, [], {}, []
    for name, feats in variants().items():
        cross = CrossFitMemoryJury(tr, Dtr, k=a.k, features=feats, post=add_relative, allow_tiny=bool(a.limit))
        jury = cross.full
        ptr, pv = cross.proba(tr, Dtr), jury.proba(va, Dva)
        pt = jury.proba(te, Dte) if te is not None else None
        r = score_variant(name, len(feats), tr, ptr, va, pv, te, pt)
        res.append(r["row"]); curves.extend(r["curves"]); juries[name] = (jury, r["row"]["t_train"], feats)
        if r["test_cases"] is not None:
            per_case[name] = r["test_cases"]
        for split, df, p in (("train_oof", tr, ptr), ("val", va, pv), ("test", te, pt)):
            if df is not None:
                sel = (df.k >= 0).to_numpy()
                scores.append(pd.DataFrame(dict(variant=name, split=split, cache_id=df.cache_id[sel], k=df.k[sel],
                                                on_tumour=df.on_tumour[sel], p=p[sel])))
        print_row(r["row"])
    tag = "" if a.final else "dev_"
    table = pd.DataFrame(res)
    cv = pd.DataFrame(curves)
    cv.to_csv(out / f"{tag}curves.csv", index=False)
    table.to_csv(out / f"{tag}variants.csv", index=False)
    pd.concat(scores).to_csv(out / f"{tag}scores.csv", index=False)       # every candidate's score (any version)
    imp = importance(juries[PROPOSED][0], va, Dva)
    imp.to_csv(out / f"{tag}importance_val.csv", index=False)
    wins = {BASELINE: curve_wins(cv, BASELINE, splits=("test", "val") if a.final else ("val", "train_oof"))}
    summary = dict(mode="final" if a.final else "dev", proposed=PROPOSED, plateau_margin=PLATEAU,
                   variants=table.to_dict(orient="records"), curve_wins=wins,
                   cases={s: int(d[0].cache_id.nunique()) for s, d in data.items()}, limit=a.limit,
                   rad_version=RAD_VERSION, versions=versions())

    print("\nvariants (cut-off = plateau centre of train out-of-fold Dice):")
    cols = [c for c in ["variant", "n_features", "t_train", "train_oof_dice", "val_dice", "t_val", "val_dice_best",
                        "auc_train_oof", "auc_val", "test_dice", "test_missed", "test_dice_valcut", "auc_test"]
            if c in table]
    print(table[cols].round(4).to_string(index=False))
    for n, w in wins[BASELINE].items():
        print(f"  {n:16s} beats {BASELINE} at " + ", ".join(f"{s} {w.get(s, '-')}" for s in w if s != "n")
              + f" of {w['n']} cut-offs (0.20-0.80)")
    print("\ntop 10 clues of the proposed jury (val, AUC drop when shuffled):\n" + imp.head(10).round(4).to_string(index=False))

    if a.final:
        pub = published_s3f(s3f, va, Dva, te, Dte)
        if pub is not None:
            res.append(pub["row"]); per_case["s3f_published"] = pub["test_cases"]
        comp = {BASELINE: {n: paired(per_case[n], per_case[BASELINE]) for n in per_case if n != BASELINE}}
        pc = pd.DataFrame(per_case)
        pc["collection"] = pc.index.map(coll)
        per_coll = pc.groupby("collection").mean().round(4)
        pc.to_csv(out / "test_per_case.csv")
        ref = s3d.rule_tables(te).mean()
        summary.update(paired_test=comp, per_collection_test=per_coll.to_dict(orient="index"),
                       reference_rules_test={r: float(ref[r]) for r in ("seg_rel20", "segjury_0.7", "oracle") if r in ref})
        print(f"\nvs {BASELINE} (test, per case):")
        for n, c in comp[BASELINE].items():
            print(f"  {n:16s} delta {c['delta']:+.4f}  CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
                  f"p {c['wilcoxon_p']:.3g}  better/worse(>0.05) {c['better']}/{c['worse']}")
        print("\nper hospital (test Dice):\n" + per_coll.to_string())
        prov = dict(s3f_dir=str(s3f), tables={s: file_digest(s3f / f"{s}_features.csv") for s in data},
                    s3f_provenance=(json.loads((s3f / "summary.json").read_text()).get("provenance")
                                    if (s3f / "summary.json").exists() else None))
        fin = {}
        for name in (PROPOSED, BASELINE):
            jury, t, feats = juries[name]
            fin[name] = export(name, jury, t, feats, {"val": (va, Dva), "test": (te, Dte)}, cases_dir, cache_paths,
                               out, a.workers, prov, coll).round(4).to_dict(orient="index")
        summary["final_full_metrics"] = fin
    summary["minutes"] = (time.time() - t0) / 60
    json.dump(summary, open(out / f"{tag}summary.json", "w"), indent=2, default=float)
    print(f"\ntotal {summary['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
