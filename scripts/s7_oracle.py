"""S7 oracle: how much can a boundary refiner of the memory-jury mask POSSIBLY gain? (no training, no GPU)

The refiner may only change voxels within R mm of the jury mask M0's boundary (the editable band A).
Diagnostic upper bound (uses the ground truth G -- analysis only, never a model input):
    M_oracle(R) = (M0 AND NOT A_R) OR (G AND A_R)          i.e. every editable voxel fixed perfectly
Reported per split (default val) for R = 4, 8, 12 mm:
  - mean Dice of M0 and M_oracle(R)  (+ NSD@2mm / HD95 for M0 and R = 8, the refiner's band)
  - error decomposition at each R (ml): fixable missed tumour (G & ~M0 & A), fixable extra (M0 & ~G & A),
    unreachable missed tumour (G & ~M0 & ~A), fixed extra core (M0 & ~G & ~A)
  - by collection and by tumour-size tertile; cases with an empty M0
Decision: if the 8 mm oracle gain on val is small, a boundary refiner cannot help much.
Example
  python scripts/s7_oracle.py --s3f_dir /kaggle/input --cache_root <cache> --out /kaggle/working/s7_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import all_metrics, dice  # noqa: E402
from cids.provenance import require_cases  # noqa: E402
from cids.refine.source import find_s3f  # noqa: E402

RADII = (4, 8, 12)
ML = 1e-3                                           # 1 voxel = 1 mm^3 = 0.001 ml


def band_distance(m0: np.ndarray, rmax: float):
    """|signed distance| to M0's boundary, computed in M0's bbox + rmax + 1 (inf elsewhere)."""
    d = np.full(m0.shape, np.inf, np.float32)
    if not m0.any():
        return d
    nz = np.argwhere(m0)
    lo = np.maximum(nz.min(0) - int(rmax) - 2, 0); hi = np.minimum(nz.max(0) + int(rmax) + 3, m0.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    m = m0[sl]
    d[sl] = np.where(m, ndimage.distance_transform_edt(m), ndimage.distance_transform_edt(~m))
    return d


def oracle_case(job):
    cid, cache_path, mask_path, split = job
    m0 = np.load(mask_path)["mask"].astype(bool)
    _, lbl = load_case(cache_path)
    g = lbl > 0
    dist = band_distance(m0, max(RADII))
    row = dict(cache_id=cid, split=split, gt_ml=g.sum() * ML, m0_ml=m0.sum() * ML, empty_m0=int(not m0.any()))
    m = all_metrics(m0, g)
    row.update(dice_m0=m["dice"], nsd_m0=m["nsd"], hd95_m0=m["hd95"])
    for r in RADII:
        a = dist <= r                                   # editable band (as in the refiner: |sdf| <= R)
        orc = (m0 & ~a) | (g & a)
        row[f"dice_oracle_{r}"] = dice(orc, g)
        row[f"fix_fn_ml_{r}"] = (g & ~m0 & a).sum() * ML
        row[f"fix_fp_ml_{r}"] = (m0 & ~g & a).sum() * ML
        row[f"unreach_fn_ml_{r}"] = (g & ~m0 & ~a).sum() * ML
        row[f"fixed_fp_ml_{r}"] = (m0 & ~g & ~a).sum() * ML
        if r == 8:
            mo = all_metrics(orc, g)
            row.update(nsd_oracle_8=mo["nsd"], hd95_oracle_8=mo["hd95"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3f_dir", required=True, help="notebook 12 output (summary.json + masks/ or masks.zip)")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--split", default="val", help="val (default; decision split) or train (out-of-fold masks)")
    ap.add_argument("--out", default="/kaggle/working/s7_oracle")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s3f = find_s3f(a.s3f_dir, out / "s3f_unpacked")
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection))
    rows = []
    for split in a.split.split(","):
        cases = require_cases(sp, root, split)[:a.limit or None]
        jobs = [(c, p, s3f / "masks" / f"{c}.npz", split) for c, p in cases]
        miss = [c for c, _, m, _ in jobs if not Path(m).exists()]
        if miss:
            raise SystemExit(f"{len(miss)} {split} cases have no jury mask (e.g. {miss[:3]})")
        with Pool(max(1, a.workers)) as pool:
            rows += list(tqdm(pool.imap(oracle_case, jobs), total=len(jobs), desc=f"oracle {split}",
                              file=sys.stdout, mininterval=30))
    df = pd.DataFrame(rows)
    df["collection"] = df.cache_id.map(coll)
    df["size"] = "all"
    for split, idx in df.groupby("split").groups.items():          # tumour-size tertiles within each split
        if len(idx) >= 3:
            df.loc[idx, "size"] = pd.qcut(df.loc[idx, "gt_ml"].rank(method="first"), 3,
                                          labels=["small", "medium", "large"]).astype(str)
    df.to_csv(out / "oracle_per_case.csv", index=False)

    summary = {}
    for split, d in df.groupby("split"):
        s = dict(cases=len(d), empty_m0=int(d.empty_m0.sum()), dice_m0=d.dice_m0.mean(),
                 nsd_m0=d.nsd_m0.mean(), hd95_m0_median=d.hd95_m0.median(),
                 nsd_oracle_8=d.nsd_oracle_8.mean(), hd95_oracle_8_median=d.hd95_oracle_8.median())
        for r in RADII:
            s[f"dice_oracle_{r}"] = d[f"dice_oracle_{r}"].mean()
            s[f"gain_{r}"] = s[f"dice_oracle_{r}"] - s["dice_m0"]
            s[f"cases_gain_gt_0.05_{r}"] = int((d[f"dice_oracle_{r}"] - d.dice_m0 > 0.05).sum())
            for k in ("fix_fn_ml", "fix_fp_ml", "unreach_fn_ml", "fixed_fp_ml"):
                s[f"{k}_{r}_total"] = d[f"{k}_{r}"].sum()
        s["by_collection"] = d.groupby("collection")[["dice_m0"] + [f"dice_oracle_{r}" for r in RADII]].mean().round(4).to_dict(orient="index")
        s["by_size"] = d.groupby("size", observed=True)[["gt_ml", "dice_m0"] + [f"dice_oracle_{r}" for r in RADII]].mean().round(4).to_dict(orient="index")
        lost = d[d.dice_m0 < 0.3]
        s["wrong_or_missed_cases(dice<0.3)"] = dict(n=len(lost), dice_m0=lost.dice_m0.mean() if len(lost) else None,
                                                   dice_oracle_8=lost.dice_oracle_8.mean() if len(lost) else None)
        summary[split] = s

        print(f"\n===== {split.upper()} ({len(d)} cases, {s['empty_m0']} with empty jury mask) =====")
        tab = pd.DataFrame({f"{r} mm": [s["dice_m0"], s[f"dice_oracle_{r}"], s[f"gain_{r}"], s[f"cases_gain_gt_0.05_{r}"],
                                        s[f"fix_fn_ml_{r}_total"], s[f"fix_fp_ml_{r}_total"],
                                        s[f"unreach_fn_ml_{r}_total"], s[f"fixed_fp_ml_{r}_total"]] for r in RADII},
                           index=["Dice jury mask (M0)", "Dice oracle (band fixed perfectly)", "possible gain",
                                  "cases gaining > 0.05", "fixable missed tumour (ml)", "fixable extra (ml)",
                                  "UNREACHABLE missed tumour (ml)", "fixed extra outside band (ml)"])
        print(tab.round(4).to_string())
        print(f"NSD@2mm: M0 {s['nsd_m0']:.4f} -> oracle 8 mm {s['nsd_oracle_8']:.4f} | "
              f"HD95 median: M0 {s['hd95_m0_median']:.2f} -> oracle 8 mm {s['hd95_oracle_8_median']:.2f} mm")
        print("\nby collection (Dice):\n" + pd.DataFrame(s["by_collection"]).T.round(4).to_string())
        print("\nby tumour size (tertiles of GT volume):\n" + pd.DataFrame(s["by_size"]).T.round(4).to_string())
        w = s["wrong_or_missed_cases(dice<0.3)"]
        print(f"\nwrong-blob / missed cases (M0 Dice < 0.3): {w['n']} -- a boundary refiner cannot rescue these "
              f"(oracle 8 mm Dice on them: {w['dice_oracle_8']})")
    summary["minutes"] = (time.time() - t0) / 60
    json.dump(summary, open(out / "oracle_summary.json", "w"), indent=2, default=float)
    print(f"\ntotal {summary['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
