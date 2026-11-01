"""S3h: radiomics clues for the memory jury (CPU only, no network training).

Reads the S3f outputs (same candidates as the 13-clue jury, test 0.7495) and the MRI cache, then
  1. extracts radiomics clues for every candidate (cids.radiomics) -> <out>/<split>_radiomics.csv
     (+ manifest: re-used only for the same S3f tables, cache and RAD_VERSION); multi-core, resumable per split
  2. fits one jury per variant on train (cids.jury.MemoryJury, same gradient boosting as S3e/S3f),
     chooses its cut-off on VAL (table Dice), applies it ONCE to test
  3. compares every variant with jury13 (paired per-case test Dice: mean delta, bootstrap CI, Wilcoxon,
     better / worse), per hospital, AUC, and permutation importance of the proposed jury on val

Variants (features):
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
from cids.jury import JURY11, JURY13, MemoryJury  # noqa: E402
from cids.provenance import cases_digest, file_digest, require_cases  # noqa: E402
from cids.radiomics import GROUPS, RAD_ALL, RAD_VERSION, REL, add_relative, extract_case  # noqa: E402

FINDER = ["log_seg_ml", "rel_size", "is_largest", "rank", "n_cand", "seg_prob_mean", "seg_prob_max"]
assert set(FINDER) < set(JURY11)


def variants() -> dict:
    prop = JURY13 + RAD_ALL + REL
    v = {"jury13": JURY13,
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


def extract_split(split, df, cache_paths, s3f, out, workers, manifest):
    f, mf = out / f"{split}_radiomics.csv", out / f"{split}_radiomics.manifest.json"
    if s3d.cache_ok([f], manifest, mf):
        print(f"{split}: using cached {f.name} (same manifest)")
        return pd.read_csv(f)
    jobs = []
    for cid, g in df[df.k >= 0].groupby("cache_id", sort=False):
        ks = g.sort_values("rank").k.astype(int).tolist()          # rank 0 = largest = dominant
        case = s3f / "cases" / f"{cid}.npz"
        if not case.exists():
            raise SystemExit(f"{split}: missing {case} (S3f cases/ incomplete or still zipped)")
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
def choose_threshold(va, p):
    t = s3d.rule_tables(va, p).mean()
    lt = t[[c for c in t.index if c.startswith("learned_")]]
    return float(lt.idxmax().split("_t")[1]), float(lt.max())


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
    print(f"S3f {s3f}\ncache {root}\nout {out}")

    data = {}
    for split in ("train", "val", "test"):
        df, D = load_tables(s3f, split, a.limit)
        cases = [c for c in require_cases(sp, root, split) if c[0] in set(df.cache_id)]
        if len(cases) != df.cache_id.nunique():
            raise SystemExit(f"{split}: {df.cache_id.nunique()} cases in the table, {len(cases)} in the cache")
        manifest = dict(rad_version=RAD_VERSION, split=split, limit=a.limit,
                        table=file_digest(s3f / f"{split}_features.csv"), **cases_digest(cases))
        rad = extract_split(split, df, dict(cases), s3f, out, a.workers, manifest)
        data[split] = (merge(df, rad), D)
    (tr, Dtr), (va, Dva), (te, Dte) = data["train"], data["val"], data["test"]

    # juries
    res, per_case, juries = [], {}, {}
    for name, feats in variants().items():
        jury = MemoryJury(k=a.k, features=feats, post=add_relative).fit(tr, Dtr)
        pv, pt = jury.proba(va, Dva), jury.proba(te, Dte)
        t, vd = choose_threshold(va, pv)
        dv = s3d.rule_tables(va, pv)[f"learned_t{t:.2f}"]
        dt = s3d.rule_tables(te, pt)[f"learned_t{t:.2f}"]
        per_case[name], juries[name] = dt, jury
        res.append(dict(variant=name, n_features=len(feats), threshold=t, val_dice=float(dv.mean()),
                        test_dice=float(dt.mean()), test_missed=int((dt < 0.1).sum()),
                        auc_val=auc(va, pv), auc_test=auc(te, pt)))
        print(f"{name:16s} t={t:.2f}  val {dv.mean():.4f}  test {dt.mean():.4f}  "
              f"AUC val {res[-1]['auc_val']:.3f} test {res[-1]['auc_test']:.3f}", flush=True)
    table = pd.DataFrame(res)
    base = table.set_index("variant").loc["jury13"]

    # reference rows (same tables)
    ref = s3d.rule_tables(te).mean()
    refs = {r: float(ref[r]) for r in ("seg_rel20", "segjury_0.7", "oracle") if r in ref}

    # paired stats vs jury13 (test, per case) + per hospital
    comp = {n: paired(per_case[n], per_case["jury13"]) for n in per_case if n != "jury13"}
    pc = pd.DataFrame(per_case)
    pc["collection"] = pc.index.map(coll)
    per_coll = pc.groupby("collection").mean().round(4)
    pc.to_csv(out / "test_per_case.csv"); table.to_csv(out / "variants.csv", index=False)
    imp = importance(juries["jury13_rad_rel"], va, Dva)
    imp.to_csv(out / "importance_val.csv", index=False)

    s3f_sum = s3f / "summary.json"
    s3f_test = None
    if s3f_sum.exists():
        j = json.loads(s3f_sum.read_text())
        s3f_test = j.get("test", {}).get("metrics", {}).get("memory_jury", {}).get("dice")
    summary = dict(variants=table.to_dict(orient="records"), reference_rules_test=refs,
                   vs_jury13_test=comp, per_collection_test=per_coll.to_dict(orient="index"),
                   s3f_reported_test_dice=s3f_test, reproduced_jury13_test=float(base.test_dice),
                   cases={s: int(d[0].cache_id.nunique()) for s, d in data.items()}, limit=a.limit,
                   rad_version=RAD_VERSION, minutes=(time.time() - t0) / 60)
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)

    print(f"\nreference rules (test): " + ", ".join(f"{k} {v:.4f}" for k, v in refs.items()))
    if s3f_test is not None:
        print(f"jury13 reproduced test {base.test_dice:.4f} vs S3f reported {s3f_test:.4f} "
              f"({'OK' if abs(base.test_dice - s3f_test) < 0.005 else 'CHECK: differs > 0.005'})")
    print("\nvariants:\n" + table.round(4).to_string(index=False))
    print("\nvs jury13 (test, per case):")
    for n, c in comp.items():
        print(f"  {n:16s} delta {c['delta']:+.4f}  CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"p {c['wilcoxon_p']:.3g}  better/worse(>0.05) {c['better']}/{c['worse']}")
    print("\nper hospital (test Dice):\n" + per_coll.to_string())
    print("\ntop 10 clues of the proposed jury (val, AUC drop when shuffled):\n" + imp.head(10).round(4).to_string(index=False))
    print(f"\ntotal {summary['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
