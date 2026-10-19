"""S3e: memory clue for the learned jury (cache / memory of known candidates, idea from cache-augmented generation).

Question: does "what does this candidate look like compared with KNOWN real tumours and KNOWN false blobs"
add information beyond the 11 clues of the S3d jury (test Dice 0.735)?

Same fully automatic pipeline and candidates as S3d (SegResNet-100 -> components prob > 0.5, >= 0.1 ml, top 5
-> SAM-S2b crop + box each). For every candidate we additionally keep a descriptor of SAM's frozen embedding
(s3a.embedding_descriptor: mean over the box tokens | mean over the crop, 768-d, L2-normalised).

MEMORY = descriptors + labels (touches tumour yes/no) of all TRAINING candidates.
Memory score of a candidate = similarity-weighted share of real tumours among its k nearest memories (cosine).
  - val / test candidates: memory = all training candidates
  - training candidates: 5-fold by patient, memory = the other 4 folds (never its own patient -> no leakage)
Also a linear probe (logistic regression on the descriptor, same cross-fitting) as a second memory-style score.

Compared on val (choose model + cut-off) and test (once), all with the S3d decision rule:
  jury11            the S3d jury (reproduces 0.735 if the pipeline is unchanged)
  jury11+memory     + kNN memory score (12 clues)
  jury11+mem+probe  + kNN memory + probe score (13 clues)
Dice is computed exactly from per-candidate tables (no masks, no HD95) -> fast.

Feature tables + descriptors are cached per split (<out>/<split>_features.csv, <split>_desc.npy); re-runs skip
finished splits. Example
  python scripts/s3e_memory_jury.py --cache_root <cache> --seg_ckpt segresnet_img_e100_best.pt \\
      --sam_ckpt sam_s2b_best.pt --out /kaggle/working/s3e_memory
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3a_seg2sam as s3  # noqa: E402
import s3d_learned_jury as s3d  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.data.volumes import VolumeDataset, unpad  # noqa: E402
from cids.models.sam_med3d import download_turbo  # noqa: E402

THRESHOLDS = s3d.THRESHOLDS


# ---------------------------------------------------------------------------- pipeline + descriptors
def run_split(split, cases, seg, sam, box, amp, overlap, out, manifest=None):
    f, fd = out / f"{split}_features.csv", out / f"{split}_desc.npy"
    mf = out / f"{split}_features.manifest.json"
    manifest = dict(manifest if manifest is not None else s3d.split_manifest(split, cases, overlap, {}),
                    descriptor="box|crop mean, 768-d")
    if s3d.cache_ok([f, fd], manifest, mf):
        print(f"{split}: using cached {f.name} + {fd.name} (same manifest)")
        return pd.read_csv(f), np.load(fd)
    rows, descs, ds = [], [], VolumeDataset(cases, 96)
    for i in tqdm(range(len(ds)), desc=split, unit="case", file=sys.stdout, mininterval=20):
        item = ds[i]; cid = item["cid"]
        img, lbl = load_case(cases[i][1]); gt = lbl > 0
        sprob = unpad(s3.seg_prob(seg, item["image"], amp, overlap=overlap), item["pads"])
        lab, cands = s3.components(sprob > 0.5, top_k=5)
        if not cands:
            rows.extend(s3d.case_columns([dict(cache_id=cid, k=-1, size_vox=0, inter_vox=0, gt_vox=int(gt.sum()),
                                               on_tumour=0)], lab, gt))
            descs.append(np.zeros(768, np.float32))
            continue
        refined = s3.sam_refine(sam, box, img, lab, cands, amp, descriptors=True)
        by_k = {d["k"]: d["desc"] for d in refined}
        r = s3d.case_columns(s3d.candidate_rows(cid, sprob, lab, cands, refined, gt), lab, gt)
        rows.extend(r); descs.extend(by_k[x["k"]] for x in r)
    df, D = pd.DataFrame(rows), np.stack(descs).astype(np.float16)
    df.to_csv(f, index=False); np.save(fd, D); mf.write_text(json.dumps(manifest, indent=2))
    return df, D


# ---------------------------------------------------------------------------- memory scores
def knn_score(query, mem, mem_y, k=10):
    """Similarity-weighted share of real tumours among the k nearest memories (cosine; descriptors are unit)."""
    sims = query.astype(np.float32) @ mem.astype(np.float32).T
    k = min(k, mem.shape[0])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / 0.05)               # sharper weight for closer memories
    return (w * mem_y[idx]).sum(1) / w.sum(1)


def probe_model():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")


def memory_scores(tr, Dtr, others, k=10, n_folds=5):
    """Cross-fitted memory (kNN) and probe scores for training candidates; full-memory scores for the others."""
    from sklearn.model_selection import GroupKFold
    real = (tr.k >= 0).to_numpy()
    y = tr.on_tumour.to_numpy().astype(float)
    tr = tr.copy(); tr["mem_knn"] = 0.0; tr["mem_probe"] = 0.0
    idx = np.flatnonzero(real)
    groups = tr.cache_id.to_numpy()[idx]
    n_folds = min(n_folds, len(set(groups)))
    for fit, held in GroupKFold(n_folds).split(idx, groups=groups):
        a, b = idx[fit], idx[held]
        tr.loc[tr.index[b], "mem_knn"] = knn_score(Dtr[b], Dtr[a], y[a], k)
        pr = probe_model().fit(Dtr[a].astype(np.float32), y[a]) if len(set(y[a])) > 1 else None
        tr.loc[tr.index[b], "mem_probe"] = pr.predict_proba(Dtr[b].astype(np.float32))[:, 1] if pr else 0.5
    pr_full = probe_model().fit(Dtr[idx].astype(np.float32), y[idx])
    outs = []
    for df, D in others:
        df = df.copy(); r = (df.k >= 0).to_numpy()
        df["mem_knn"] = 0.0; df["mem_probe"] = 0.0
        if r.any():
            df.loc[r, "mem_knn"] = knn_score(D[r], Dtr[idx], y[idx], k)
            df.loc[r, "mem_probe"] = pr_full.predict_proba(D[r].astype(np.float32))[:, 1]
        outs.append(df)
    return tr, outs


# ---------------------------------------------------------------------------- jury variants
def fit_gb(df, feats):
    from sklearn.ensemble import HistGradientBoostingClassifier
    r = df[df.k >= 0]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0).fit(r[feats].to_numpy(float),
                                                                                      r.on_tumour.to_numpy())


def proba(model, df, feats):
    p = np.zeros(len(df)); r = (df.k >= 0).to_numpy()
    if r.any():
        p[r] = model.predict_proba(df.loc[r, feats].to_numpy(float))[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True, help="S2b best.pt")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s3e_memory")
    ap.add_argument("--k", type=int, default=10, help="nearest memories")
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    a = ap.parse_args()
    amp = s3.DEV == "cuda"
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    seg, sam, box = s3.load_models(a.seg_ckpt, a.sam_ckpt, turbo)
    print(f"cache {root}\nfinder {a.seg_ckpt}\nchecker {a.sam_ckpt}\n{s3.DEV}")
    sp = pd.read_csv(a.splits)

    def items(split):
        r = sp[sp.split == split]
        lst = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        return [x for x in lst if x[1].exists()][:a.limit or None]

    base = dict(seg=s3d.file_sig(a.seg_ckpt), sam=s3d.file_sig(a.sam_ckpt), turbo=s3d.file_sig(turbo),
                random_weights=a.random_weights, top_k=5)

    def run(split):
        c = items(split)
        return run_split(split, c, seg, sam, box, amp, 0.5, out, s3d.split_manifest(split, c, 0.5, base))

    tr, Dtr = run("train")
    va, Dva = run("val")
    te, Dte = run("test")
    n_true = int(tr[tr.k >= 0].on_tumour.sum()); n_all = int((tr.k >= 0).sum())
    print(f"memory: {n_all} training candidates ({n_true} real tumours, {n_all - n_true} false), "
          f"descriptor {Dtr.shape[1]}-d, k = {a.k}")

    tr, (va, te) = memory_scores(tr, Dtr, [(va, Dva), (te, Dte)], k=a.k)
    for name, df in (("train", tr), ("val", va), ("test", te)):
        df.to_csv(out / f"{name}_features_mem.csv", index=False)

    variants = {"jury11": list(s3d.FEATURES),
                "jury11+memory": list(s3d.FEATURES) + ["mem_knn"],
                "jury11+mem+probe": list(s3d.FEATURES) + ["mem_knn", "mem_probe"]}
    summary = {"k": a.k, "n_memory": n_all, "auc": {}, "variants": {}}
    for split, df in (("val", va), ("test", te)):
        r = df[df.k >= 0]
        for sc in ("mem_knn", "mem_probe", "sam_soft_ml", "agree"):
            summary["auc"][f"{split}_{sc}"] = s3d.auc(r[sc].to_numpy(), r.on_tumour.to_numpy())
    test_case = {}
    for name, feats in variants.items():
        m = fit_gb(tr, feats)
        pv, pt = proba(m, va, feats), proba(m, te, feats)
        vt = s3d.rule_tables(va, pv).mean()
        lt = vt[[c for c in vt.index if c.startswith("learned_")]]
        t = float(lt.idxmax().split("_t")[1])
        tt = s3d.rule_tables(te, pt)
        test_case[name] = tt[f"learned_t{t:.2f}"]
        summary["variants"][name] = dict(
            threshold=t, val_dice=round(float(lt.max()), 4), test_dice=round(float(test_case[name].mean()), 4),
            val_auc=round(s3d.auc(pv[(va.k >= 0).to_numpy()], va[va.k >= 0].on_tumour.to_numpy()), 4),
            test_auc=round(s3d.auc(pt[(te.k >= 0).to_numpy()], te[te.k >= 0].on_tumour.to_numpy()), 4))
        if name == "jury11":
            refs = ("seg_rel20", "segjury_0.7", "oracle", "oracle_touching")
            summary["reference"] = {c: round(float(tt[c].mean()), 4) for c in refs}
    from scipy.stats import wilcoxon
    base = test_case["jury11"]
    for name in list(variants)[1:]:
        d = test_case[name] - base
        summary["variants"][name]["test_minus_jury11"] = round(float(d.mean()), 4)
        summary["variants"][name]["wilcoxon_p"] = float(wilcoxon(test_case[name], base).pvalue) if (d != 0).any() else 1.0
        summary["variants"][name]["cases_better_worse"] = [int((d > 0.05).sum()), int((d < -0.05).sum())]
    coll = dict(zip(sp.cache_id, sp.collection))
    per = pd.DataFrame(test_case); per["collection"] = per.index.map(coll)
    summary["test_per_collection"] = per.groupby("collection").mean().round(4).to_dict(orient="index")
    summary["minutes"] = (time.time() - t0) / 60
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)

    print("\nAUC true vs false candidates:", {k: round(v, 3) for k, v in summary["auc"].items()})
    print("\nVAL (choose cut-off) and TEST (once):")
    print(pd.DataFrame(summary["variants"]).T.to_string())
    print("\nreference (test):", summary.get("reference"))
    print("\nper collection (test Dice):"); print(per.groupby("collection").mean().round(3).to_string())
    print(f"total {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
