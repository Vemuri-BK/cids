"""E0: recall-first ceiling check -- can a lower finder threshold bring missed tumours into the candidate list?

CPU only, read-only, no model is run or trained. Uses the SAVED SegResNet probability maps:
  train  out-of-fold maps (every patient predicted by the fold model that never saw it)
  val    SegResNet-100 maps (the pipeline's finder)
The test split is not used (decisions are made on train OOF + val only).

For every case and candidate setting it rebuilds the candidates exactly like the pipeline
(26-connected components of p > t, >= --min_vox voxels, the top-K by size) and measures
  found        a candidate touches the expert tumour (the tumour can reach SAM and the jury at all)
  coverage     share of the tumour inside all candidates
  oracle Dice  the best possible selection of these candidates (perfect jury) = the Dice ceiling
  n_cand / n_false   candidates per case / candidates not touching the tumour (extra work for the jury)
Two ways to lower the threshold:
  plain  candidates of p > t (low threshold blobs replace the 0.5 blobs; they can grow and merge)
  addon  the p > 0.5 candidates are kept unchanged; components of p > t that contain no p > 0.5 voxel are
         ADDED (largest first) up to K candidates in total -> only new blobs, existing ones do not change
The reference setting is plain t=0.5, K=5 (the locked pipeline).

Outputs (<out>): e0_per_case.csv (one row per case x setting), e0_summary.csv, and a printed table.
Example (server)
  python scripts/e0_recall_ceiling.py --cache_root ~/data \\
      --map_dirs ~/CIDS/outputs/seg_oof/fold0/seg_fold0,~/CIDS/outputs/seg_oof/fold1/seg_fold1,\\
~/CIDS/outputs/seg_oof/fold2/seg_fold2,~/CIDS/outputs/seg_oof/full/seg_full --out ~/CIDS/outputs/e0_recall
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STRUCT = np.ones((3, 3, 3))                         # 26-connectivity, as s3a_seg2sam.components


def components(mask, min_vox=100, top_k=5):
    """= s3a_seg2sam.components: labels + [(k, size)] of the top_k components >= min_vox (largest first);
    if none is large enough, the largest one is kept."""
    lab, n = ndimage.label(mask, structure=STRUCT)
    if n == 0:
        return lab, []
    size = np.bincount(lab.ravel().astype(np.int64), minlength=n + 1)[1:].astype(float)   # = ndimage.sum
    order = [int(i) + 1 for i in np.argsort(-size) if size[i] >= min_vox][:top_k]
    if not order:
        order = [int(np.argmax(size)) + 1]
    return lab, [(k, float(size[k - 1])) for k in order]


def oracle_dice(inter, size, g):
    """Best Dice over all subsets of disjoint candidates: the optimum keeps the candidates of highest
    precision inter/size (a prefix of that order), so trying every prefix is exact."""
    if g == 0:
        return np.nan
    if not len(inter):
        return 0.0
    o = np.argsort(-(np.asarray(inter) / np.maximum(np.asarray(size), 1)))
    ci, cs = np.cumsum(np.asarray(inter)[o]), np.cumsum(np.asarray(size)[o])
    return float(max(0.0, (2 * ci / (cs + g)).max()))


def candidate_stats(blobs, gt):
    """blobs: list of boolean masks (disjoint). -> dict of found / coverage / oracle / counts."""
    g = int(gt.sum())
    inter = [int((b & gt).sum()) for b in blobs]
    size = [int(b.sum()) for b in blobs]
    return dict(found=bool(any(i > 0 for i in inter)), coverage=float(sum(inter) / g) if g else np.nan,
                oracle=oracle_dice(inter, size, g), n_cand=len(blobs), n_false=int(sum(i == 0 for i in inter)),
                cand_ml=float(sum(size)) / 1000)


def settings_for(thresholds, ks):
    out = [("plain", 0.5, 5)]
    for t in thresholds:
        for k in ks:
            for mode in ("plain", "addon"):
                if (mode, t, k) == ("plain", 0.5, 5) or (mode == "addon" and t >= 0.5):
                    continue
                out.append((mode, t, k))
    return out


def case_rows(job):
    cid, split, map_path, case_path, settings, min_vox = job
    d = np.load(map_path)
    prob = (d["prob"] if "prob" in d else d["seg"]).astype(np.float32) / 255.0
    lbl = np.load(case_path)["label"]
    gt = (lbl[0] if lbl.ndim == 4 else lbl) > 0
    if prob.shape != gt.shape:
        return [dict(cache_id=cid, split=split, error=f"shape {prob.shape} vs {gt.shape}")]
    rows, cache = [], {}

    def comps(t, k):
        if (t, k) not in cache:
            lab, c = components(prob > t, min_vox, k)
            cache[(t, k)] = (lab, c)
        return cache[(t, k)]

    for mode, t, k in settings:
        if mode == "plain":
            lab, c = comps(t, k)
            blobs = [lab == kk for kk, _ in c]
        else:                                        # addon: keep the 0.5 / top-5 blobs, add new low-t blobs
            lab5, c5 = comps(0.5, 5)
            blobs = [lab5 == kk for kk, _ in c5]
            core = prob > 0.5
            labt, n = ndimage.label(prob > t, structure=STRUCT)
            if n and len(blobs) < k:
                flat = labt.ravel().astype(np.int64)            # plain counts (scipy's ndimage.sum on a bool
                size = np.bincount(flat, minlength=n + 1)[1:]   # input failed on the server: huge bincount)
                has_core = np.bincount(flat, weights=core.ravel().astype(np.float64), minlength=n + 1)[1:] > 0
                extra = [int(i) + 1 for i in np.argsort(-size) if size[i] >= min_vox and not has_core[i]]
                blobs += [labt == kk for kk in extra[:k - len(blobs)]]
        rows.append(dict(cache_id=cid, split=split, mode=mode, t=t, k=k, gt_ml=float(gt.sum()) / 1000,
                         **candidate_stats(blobs, gt)))
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[(df["mode"] == "plain") & (df.t == 0.5) & (df.k == 5)].set_index(["split", "cache_id"])
    out = []
    for (split, mode, t, k), g in df.groupby(["split", "mode", "t", "k"]):
        g = g.set_index(["split", "cache_id"])
        r = ref.loc[g.index]
        missed_ref = ~r.found.astype(bool)
        out.append(dict(split=split, mode=mode, t=t, k=k, cases=len(g),
                        oracle_dice=g.oracle.mean(), delta_vs_ref=g.oracle.mean() - r.oracle.mean(),
                        tumour_found=g.found.mean(), missed_at_ref=int(missed_ref.sum()),
                        recovered=int((missed_ref & g.found.astype(bool)).sum()),
                        lost_vs_ref=int((~missed_ref & ~g.found.astype(bool)).sum()),
                        cand_per_case=g.n_cand.mean(), false_per_case=g.n_false.mean(),
                        coverage=g.coverage.mean()))
    s = pd.DataFrame(out)
    s["order"] = (s["mode"] != "plain") * 1 + (s.t == 0.5) * -10
    return s.sort_values(["split", "order", "mode", "t", "k"], ascending=[True, True, True, False, True]) \
            .drop(columns="order").round(4)


def main():
    from cids.data.cache import find_cache_root
    from cids.provenance import require_cases
    from cids.refine.source import index_maps
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--map_dirs", required=True, help="comma-separated folders with heldout_pred/ val_pred/ ...")
    ap.add_argument("--eval_splits", default="train,val", help="train (OOF) and/or val; test is not allowed")
    ap.add_argument("--thresholds", default="0.5,0.4,0.3,0.2")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--min_vox", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N cases per split (smoke test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    splits = [s.strip() for s in a.eval_splits.split(",") if s.strip()]
    if "test" in splits:
        raise SystemExit("E0 is a development check: the test split is not used")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    maps = index_maps(a.map_dirs.split(","), unpack_to=out / "_unpacked")
    settings = settings_for([float(x) for x in a.thresholds.split(",")], [int(x) for x in a.ks.split(",")])
    jobs = []
    for split in splits:
        cases = require_cases(sp, root, split)
        missing = [c for c, _ in cases if c not in maps]
        if missing:
            raise SystemExit(f"{split}: {len(missing)} of {len(cases)} cases have no probability map "
                             f"(e.g. {missing[:5]}); check --map_dirs")
        cases = cases[:a.limit] if a.limit else cases
        jobs += [(c, split, maps[c], p, settings, a.min_vox) for c, p in cases]
    print(f"E0 recall ceiling | {len(jobs)} cases ({', '.join(splits)}) | {len(settings)} settings | "
          f"{a.jobs} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, r in enumerate(ex.map(case_rows, jobs, chunksize=4)):
            rows += r
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} cases | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    if "error" in df and df.error.notna().any():
        raise SystemExit(f"map/label mismatch: {df[df.error.notna()].head().to_dict('records')}")
    df.to_csv(out / "e0_per_case.csv", index=False)
    s = summarise(df)
    s.to_csv(out / "e0_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\noracle_dice = perfect-jury ceiling | tumour_found = tumour is among the candidates | "
          "recovered = missed at (plain, 0.5, 5) but found here\n")
    print(s.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
