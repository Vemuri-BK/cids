"""S3h: radiomics clues for the memory jury (CPU only, no network training).

Reads the S3f outputs (same candidates as the 13-clue jury, test 0.7495) and the MRI cache, then
  1. extracts radiomics clues for every candidate (cids.radiomics) -> <out>/<split>_radiomics.csv
     (+ manifest: re-used only for the same S3f tables, cache and RAD_VERSION); multi-core, resumable per split
  2. fits one jury per variant on train (cids.jury.CrossFitMemoryJury: the full jury + 5 out-of-fold juries, same
     gradient boosting as S3e/S3f), chooses its cut-off on the TRAINING patients' out-of-fold scores (1080
     patients; val's 120 were too few -- run 1), applies it ONCE to val and test; also reports the val-chosen
     cut-off, Dice-vs-cut-off curves (curves.csv) and the saved S3f jury itself (memory_jury.pkl, 0.7495)
  3. compares every variant with jury13 (paired per-case test Dice: mean delta, bootstrap CI, Wilcoxon,
     better / worse), per hospital, AUC, and permutation importance of the proposed jury on val

Variants (features):
  finder_only            finder clues only (size, rank, SegResNet probability)
  jury11                 + SAM checker clues (= S3d)
  jury13                 the S3f jury (reproduces its numbers; sanity check)
  finder_rad             finder clues + radiomics + relative (no SAM, no memory)  -> "radiomics instead of SAM?"
  jury13_rel             jury13 + within-patient relative clues
  jury13_rad             jury13 + radiomics (shape, contrast, texture, position)
  jury13_rad_rel         jury13 + radiomics + relative = PROPOSED
  -<group>               proposed minus one group (shape / contrast / texture / position / relative)
Dice here = exact table Dice (components are disjoint; == mask Dice, checked by S3f's parity test).
Full masks / HD95 / NSD for the winner: later (--write_masks is not part of this step).

Example (server)
  python scripts/s3h_radiomics_jury.py --s3f_dir ~/CIDS/outputs/s3f_jury_masks/s3f_jury \\
      --cache_root ~/data --out ~/CIDS/outputs/s3h_radiomics --workers 8
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
from cids.data.cache import find_cache_root  # noqa: E402
from cids.jury import JURY11, JURY13, CrossFitMemoryJury  # noqa: E402
from cids.provenance import cases_digest, file_digest, require_cases  # noqa: E402
from cids.radiomics import GROUPS, RAD_ALL, RAD_VERSION, REL, add_relative, extract_case  # noqa: E402

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
    rad = pd.DataFrame(rows, columns=["cache_id", "k"] + RAD_ALL)
    rad.to_csv(f, index=False); mf.write_text(json.dumps(manifest, indent=2))
    print(f"{split}: {len(rad)} candidates from {len(jobs)} cases, {(time.time() - t0) / 60:.1f} min")
    return rad


def merge(df, rad):
    m = df.merge(rad, on=["cache_id", "k"], how="left", validate="one_to_one")
    real = (m.k >= 0).to_numpy()
    miss = m.loc[real, RAD_ALL].isna().all(axis=1).sum()
    if miss:
        raise SystemExit(f"{miss} real candidates without radiomics row (tables out of sync)")
    return m


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


def score_variant(name, n_feat, tr, ptr, va, pv, te, pt):
    """Main cut-off = best mean Dice on the training patients' OUT-OF-FOLD scores (1080 patients);
    secondary = best on val (120). Test is evaluated at both, once."""
    ctr = curve(tr, ptr) if ptr is not None else None
    cva, cte = curve(va, pv), curve(te, pt)
    t_val = float(cva.idxmax())
    t_tr = float(ctr.idxmax()) if ctr is not None else t_val
    tv, tt = s3d.rule_tables(va, pv), s3d.rule_tables(te, pt)
    dt = tt[f"learned_t{t_tr:.2f}"]
    row = dict(variant=name, n_features=n_feat, t_train=t_tr,
               train_oof_dice=float(ctr.max()) if ctr is not None else None,
               val_dice=float(tv[f"learned_t{t_tr:.2f}"].mean()), test_dice=float(dt.mean()),
               test_missed=int((dt < 0.1).sum()), t_val=t_val,
               test_dice_valcut=float(tt[f"learned_t{t_val:.2f}"].mean()),
               auc_val=auc(va, pv), auc_test=auc(te, pt))
    curves = [dict(variant=name, split=s, threshold=float(th), dice=float(v))
              for s, c in (("train_oof", ctr), ("val", cva), ("test", cte)) if c is not None for th, v in c.items()]
    return dict(row=row, test_cases=dt, curves=curves)


def published_s3f(s3f: Path, va, Dva, te, Dte):
    """The saved S3f jury (memory_jury.pkl: the model behind the paper's 0.7495) at its own cut-off. Skipped with
    a message if the bundle is missing or cannot be loaded in this environment."""
    import pickle
    f = s3f / "memory_jury.pkl"
    if not f.exists():
        print(f"[info] {f.name} not found -> no 's3f_published' row")
        return None
    try:
        b = pickle.load(open(f, "rb"))
        jury, t = b["jury"], float(b["threshold"])
        pv, pt = jury.proba(va, Dva), jury.proba(te, Dte)
    except Exception as e:                                           # e.g. scikit-learn version mismatch
        print(f"[warn] could not use {f.name} ({type(e).__name__}: {e}) -> no 's3f_published' row")
        return None
    r = score_variant("s3f_published", len(jury.features), None, None, va, pv, te, pt)
    tt = s3d.rule_tables(te, pt)[f"learned_t{t:.2f}"]
    r["row"].update(t_train=None, t_val=t, val_dice=float(s3d.rule_tables(va, pv)[f"learned_t{t:.2f}"].mean()),
                    test_dice=float(tt.mean()), test_dice_valcut=float(tt.mean()), test_missed=int((tt < 0.1).sum()))
    r["test_cases"] = tt
    return r


def curve_wins(cv, base, lo=0.2, hi=0.8):
    """At how many cut-offs in [lo, hi] each version's Dice is above `base` (test and val curves)."""
    out = {}
    sel = cv[(cv.threshold >= lo - 1e-9) & (cv.threshold <= hi + 1e-9)]
    for split in ("test", "val"):
        piv = sel[sel.split == split].pivot(index="threshold", columns="variant", values="dice")
        if base not in piv:
            continue
        for n in piv.columns:
            if n != base:
                out.setdefault(n, dict(n=len(piv)))[split] = int((piv[n] > piv[base] + 1e-12).sum())
    return out


def print_row(r):
    tt = f"{r['t_train']:.2f}" if r["t_train"] is not None else "  - "
    print(f"{r['variant']:16s} t_train={tt}  test {r['test_dice']:.4f} | t_val={r['t_val']:.2f}  "
          f"test {r['test_dice_valcut']:.4f} | AUC val {r['auc_val']:.3f} test {r['auc_test']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3f_dir", required=True, help="S3f output folder (train/val/test_features.csv, *_desc.npy, cases/)")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=10, help="memory neighbours (as S3f)")
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    a = ap.parse_args()
    t0 = time.time()
    s3f, out = Path(a.s3f_dir).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection))
    cases_dir = find_cases_dir(s3f)
    print(f"S3f {s3f}\ncases {cases_dir} ({len(list(cases_dir.glob('*.npz')))} files)\ncache {root}\nout {out}")

    data = {}
    for split in ("train", "val", "test"):
        df, D = load_tables(s3f, split, a.limit)
        cases = [c for c in require_cases(sp, root, split) if c[0] in set(df.cache_id)]
        if len(cases) != df.cache_id.nunique():
            raise SystemExit(f"{split}: {df.cache_id.nunique()} cases in the table, {len(cases)} in the cache")
        manifest = dict(rad_version=RAD_VERSION, split=split, limit=a.limit,
                        table=file_digest(s3f / f"{split}_features.csv"), **cases_digest(cases))
        rad = extract_split(split, df, dict(cases), cases_dir, out, a.workers, manifest)
        data[split] = (merge(df, rad), D)
    (tr, Dtr), (va, Dva), (te, Dte) = data["train"], data["val"], data["test"]

    # juries: cut-off chosen on the 1080 TRAINING patients (out-of-fold scores, CrossFitMemoryJury), not on the
    # 120 val patients (too few: a tiny change in the scores moved jury13's val cut-off 0.40 -> 0.25 and its test
    # Dice by -0.02 in run 1). Val-chosen cut-offs are kept as a secondary column. Proposed = jury13_rad_rel (fixed).
    res, per_case, curves, juries = [], {}, [], {}
    for name, feats in variants().items():
        cross = CrossFitMemoryJury(tr, Dtr, k=a.k, features=feats, post=add_relative, allow_tiny=bool(a.limit))
        jury = cross.full
        ptr, pv, pt = cross.proba(tr, Dtr), jury.proba(va, Dva), jury.proba(te, Dte)
        r = score_variant(name, len(feats), tr, ptr, va, pv, te, pt)
        res.append(r["row"]); per_case[name] = r["test_cases"]; curves.extend(r["curves"]); juries[name] = jury
        print_row(r["row"])
    pub = published_s3f(s3f, va, Dva, te, Dte)               # the saved S3f jury (paper number 0.7495)
    if pub is not None:
        res.append(pub["row"]); per_case["s3f_published"] = pub["test_cases"]; curves.extend(pub["curves"])
        print_row(pub["row"])
    table = pd.DataFrame(res)
    cv = pd.DataFrame(curves)
    cv.to_csv(out / "curves.csv", index=False)

    # reference rows (same tables)
    ref = s3d.rule_tables(te).mean()
    refs = {r: float(ref[r]) for r in ("seg_rel20", "segjury_0.7", "oracle") if r in ref}

    # paired stats (test, per case, train-chosen cut-offs) vs jury13 and vs the published S3f jury
    bases = ["jury13"] + (["s3f_published"] if pub is not None else [])
    comp = {b: {n: paired(per_case[n], per_case[b]) for n in per_case if n != b} for b in bases}
    wins = {b: curve_wins(cv, b) for b in bases}
    pc = pd.DataFrame(per_case)
    pc["collection"] = pc.index.map(coll)
    per_coll = pc.groupby("collection").mean().round(4)
    pc.to_csv(out / "test_per_case.csv"); table.to_csv(out / "variants.csv", index=False)
    imp = importance(juries["jury13_rad_rel"], va, Dva)
    imp.to_csv(out / "importance_val.csv", index=False)

    s3f_test = None
    if (s3f / "summary.json").exists():
        j = json.loads((s3f / "summary.json").read_text())
        s3f_test = j.get("test", {}).get("metrics", {}).get("memory_jury", {}).get("dice")
    summary = dict(protocol="cut-off chosen on train out-of-fold scores; test applied once; proposed = jury13_rad_rel",
                   variants=table.to_dict(orient="records"), reference_rules_test=refs, paired_test=comp,
                   curve_wins=wins, per_collection_test=per_coll.to_dict(orient="index"),
                   s3f_reported_test_dice=s3f_test,
                   cases={s: int(d[0].cache_id.nunique()) for s, d in data.items()}, limit=a.limit,
                   rad_version=RAD_VERSION, minutes=(time.time() - t0) / 60)
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)

    print(f"\nreference rules (test): " + ", ".join(f"{k} {v:.4f}" for k, v in refs.items()))
    if pub is not None and s3f_test is not None:
        print(f"published S3f jury re-scored here: test {pub['row']['test_dice_valcut']:.4f} at its own cut-off "
              f"{pub['row']['t_val']:.2f} (summary.json: {s3f_test:.4f})")
    cols = ["variant", "n_features", "t_train", "train_oof_dice", "val_dice", "test_dice", "test_missed",
            "t_val", "test_dice_valcut", "auc_val", "auc_test"]
    print("\nvariants (main = cut-off from train out-of-fold):\n" + table[cols].round(4).to_string(index=False))
    for b in bases:
        print(f"\nvs {b} (test, per case, train-chosen cut-offs):")
        for n, c in comp[b].items():
            print(f"  {n:16s} delta {c['delta']:+.4f}  CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
                  f"p {c['wilcoxon_p']:.3g}  better/worse(>0.05) {c['better']}/{c['worse']}")
        print(f"  cut-offs 0.20-0.80 where each version beats {b} on test / val:")
        for n, w in wins[b].items():
            print(f"    {n:16s} test {w['test']}  val {w['val']}  (of {w['n']})")
    print("\nper hospital (test Dice):\n" + per_coll.to_string())
    print("\ntop 10 clues of the proposed jury (val, AUC drop when shuffled):\n" + imp.head(10).round(4).to_string(index=False))
    print(f"\ntotal {summary['minutes']:.1f} min -> {out}")

if __name__ == "__main__":
    main()
