"""S3f: the S3e memory jury as a saved model + its final masks for every case (input of the boundary refiner).

Same fully automatic pipeline as S3d/S3e (SegResNet-100 -> components prob > 0.5, >= 0.1 ml, top 5 -> SAM-S2b
crop + box each -> 13 clues -> keep / drop). This script
  1. runs the pipeline once on train / val / test and caches per split: candidate table + SAM descriptors
     (+ per case: SegResNet probability and component labels, so masks can be rebuilt without a GPU)
  2. fits the memory jury on train (cids.jury.MemoryJury) and a 5-fold-by-patient CrossFitMemoryJury
  3. chooses the cut-off on val (table Dice), applies it once to test
  4. writes the final mask of EVERY case: val / test from the deployable jury; TRAIN from the out-of-fold
     jury (a refiner trained on these masks sees mistakes as on new cases -- no stacking leak in the jury)
  5. full metrics (Dice, HD95, NSD, missed) on val / test for: memory jury (13 clues), S3d jury (11 clues),
     seg_rel20; parity check mask Dice == table Dice; paired Wilcoxon + bootstrap CI
  6. saves the bundle memory_jury.pkl (jury, threshold, settings, checkpoint hashes)

Outputs (<out>):
  <split>_features.csv / _desc.npy / .manifest.json   pipeline tables (re-used only with the same manifest)
  cases/<cid>.npz        seg (uint8 prob x255), lab (component labels, int32)          -- all cases
  masks/<cid>.npz        mask (uint8 0/1, 1 mm cache grid), kept (component labels)    -- all cases
  memory_jury.pkl        deployable jury bundle
  test_metrics.csv, val_metrics.csv, summary.json
Example
  python scripts/s3f_jury_masks.py --cache_root <cache> --seg_ckpt segresnet_img_e100_best.pt \\
      --sam_ckpt sam_s2b_best.pt --out /kaggle/working/s3f_jury
"""
from __future__ import annotations

import argparse
import json
import pickle
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
from cids.eval.metrics import all_metrics  # noqa: E402
from cids.eval.postproc import rel20  # noqa: E402
from cids.jury import JURY11, CrossFitMemoryJury, MemoryJury, keep_candidates  # noqa: E402
from cids.models.sam_med3d import download_turbo  # noqa: E402
from cids.provenance import file_digest, save_npz_atomic  # noqa: E402

THRESHOLDS = s3d.THRESHOLDS


# ---------------------------------------------------------------------------- 1. pipeline
def run_split(split, cases, seg, sam, box, amp, out, manifest):
    f, fd, mf = out / f"{split}_features.csv", out / f"{split}_desc.npy", out / f"{split}_features.manifest.json"
    cdir = out / "cases"; cdir.mkdir(exist_ok=True)
    have_cases = all((cdir / f"{c}.npz").exists() for c, _ in cases)
    if have_cases and s3d.cache_ok([f, fd], manifest, mf):
        print(f"{split}: using cached tables (same manifest)")
        return pd.read_csv(f), np.load(fd)
    rows, descs, ds = [], [], VolumeDataset(cases, 96)
    for i in tqdm(range(len(ds)), desc=split, unit="case", file=sys.stdout, mininterval=30):
        item = ds[i]; cid = item["cid"]
        img, lbl = load_case(cases[i][1]); gt = lbl > 0
        sprob = unpad(s3.seg_prob(seg, item["image"], amp, overlap=0.5), item["pads"])
        lab, cands = s3.components(sprob > 0.5, top_k=5)
        save_npz_atomic(cdir / f"{cid}.npz", seg=(np.clip(sprob, 0, 1) * 255).round().astype(np.uint8),
                        lab=lab.astype(np.int32))
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


# ---------------------------------------------------------------------------- helpers
def choose_threshold(va, p):
    t = s3d.rule_tables(va, p).mean()
    lt = t[[c for c in t.index if c.startswith("learned_")]]
    return float(lt.idxmax().split("_t")[1]), float(lt.max())


def trees11(tr):
    r = tr[tr.k >= 0]
    from cids.jury import _trees
    return _trees().fit(r[JURY11].to_numpy(float), r.on_tumour.to_numpy())


def proba11(m, df):
    p = np.zeros(len(df)); r = (df.k >= 0).to_numpy()
    if r.any():
        p[r] = m.predict_proba(df.loc[r, JURY11].to_numpy(float))[:, 1]
    return p


def write_masks(df, p, t, cases, out, gt_metrics=None, rules_extra=None):
    """Mask of every case from its kept components; optional full metrics vs GT (+ extra rules)."""
    keep = keep_candidates(df, p, t)
    mdir = out / "masks"; mdir.mkdir(exist_ok=True)
    rows = []
    for cid, path in tqdm(cases, desc="masks", unit="case", file=sys.stdout, mininterval=30):
        c = np.load(out / "cases" / f"{cid}.npz"); lab = c["lab"]
        ks = keep.get(cid, [])
        m = np.isin(lab, ks) if ks else np.zeros(lab.shape, bool)
        save_npz_atomic(mdir / f"{cid}.npz", mask=m.astype(np.uint8), kept=np.array(ks, np.int32))
        if gt_metrics is not None:
            _, lbl = load_case(path); gt = lbl > 0
            rows.append(dict(cache_id=cid, rule="memory_jury", **all_metrics(m, gt)))
            for name, ks2 in (rules_extra or {}).items():
                k2 = ks2.get(cid, [])
                m2 = np.isin(lab, k2) if k2 else np.zeros(lab.shape, bool)
                rows.append(dict(cache_id=cid, rule=name, **all_metrics(m2, gt)))
            rows.append(dict(cache_id=cid, rule="seg_rel20", **all_metrics(rel20(lab > 0), gt)))
    return pd.DataFrame(rows), keep


def stats(fr, coll):
    from scipy.stats import wilcoxon
    fr = fr.copy(); fr["collection"] = fr.cache_id.map(coll)
    agg = fr.groupby("rule").agg(dice=("dice", "mean"), dice_sd=("dice", "std"), hd95_median=("hd95", "median"),
                                 hd95_mean=("hd95", "mean"), nsd=("nsd", "mean"),
                                 missed=("dice", lambda x: int((x < 0.1).sum())), cases=("dice", "size"))
    piv = fr.pivot(index="cache_id", columns="rule", values="dice")
    comp, rng = {}, np.random.default_rng(0)
    for other in [c for c in piv.columns if c != "memory_jury"]:
        d = (piv["memory_jury"] - piv[other]).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        comp[other] = dict(delta=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                           wilcoxon_p=float(wilcoxon(piv["memory_jury"], piv[other]).pvalue) if (d != 0).any() else 1.0,
                           better=int((d > 0.05).sum()), worse=int((d < -0.05).sum()))
    per = fr.pivot_table(index="collection", columns="rule", values="dice")
    return agg, comp, per


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True, help="S2b best.pt")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s3f_jury")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--zip_outputs", action="store_true", help="pack masks/ and cases/ into zips at the end (Kaggle)")
    a = ap.parse_args()
    amp = s3.DEV == "cuda"
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    seg, sam, box = s3.load_models(a.seg_ckpt, a.sam_ckpt, turbo)
    print(f"cache {root}\nfinder {a.seg_ckpt}\nchecker {a.sam_ckpt}\n{s3.DEV}")
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection))
    base = dict(seg=file_digest(a.seg_ckpt), sam=file_digest(a.sam_ckpt), turbo=file_digest(turbo),
                random_weights=a.random_weights, top_k=5, descriptor="box|crop mean, 768-d")
    cases = {s: s3d.require_cases(sp, root, s)[:a.limit or None] for s in ("train", "val", "test")}
    if a.k < 1 or len(cases["train"]) < 2:
        raise SystemExit("need k >= 1 and >= 2 training patients")

    data = {s: run_split(s, c, seg, sam, box, amp, out, s3d.split_manifest(s, c, 0.5, base)) for s, c in cases.items()}
    (tr, Dtr), (va, Dva), (te, Dte) = data["train"], data["val"], data["test"]
    del seg, sam, box

    # 2. juries
    jury = MemoryJury(k=a.k).fit(tr, Dtr)
    cross = CrossFitMemoryJury(tr, Dtr, k=a.k)
    j11 = trees11(tr)
    # 3. cut-offs on val
    pv, pt, ptr = jury.proba(va, Dva), jury.proba(te, Dte), cross.proba(tr, Dtr)
    t, vdice = choose_threshold(va, pv)
    t11, vdice11 = choose_threshold(va, proba11(j11, va))
    jury.threshold = t
    print(f"val cut-off: memory jury t = {t:.2f} (val Dice {vdice:.4f}) | S3d jury t = {t11:.2f} ({vdice11:.4f})")
    # table Dice (exact, for the parity check)
    tab_te = s3d.rule_tables(te, pt)[f"learned_t{t:.2f}"]
    tab_va = s3d.rule_tables(va, pv)[f"learned_t{t:.2f}"]
    tab_tr = s3d.rule_tables(tr, ptr)[f"learned_t{t:.2f}"]

    # 4. masks (+ 5. metrics)
    k11_va = keep_candidates(va, proba11(j11, va), t11)
    k11_te = keep_candidates(te, proba11(j11, te), t11)
    fr_va, _ = write_masks(va, pv, t, cases["val"], out, True, {"s3d_jury11": k11_va})
    fr_te, _ = write_masks(te, pt, t, cases["test"], out, True, {"s3d_jury11": k11_te})
    write_masks(tr, ptr, t, cases["train"], out)                        # out-of-fold masks for the refiner
    fr_va.to_csv(out / "val_metrics.csv", index=False); fr_te.to_csv(out / "test_metrics.csv", index=False)

    parity = {}
    for name, fr, tab in (("val", fr_va, tab_va), ("test", fr_te, tab_te)):
        md = fr[fr.rule == "memory_jury"].set_index("cache_id").dice
        parity[name] = float(np.abs(md - tab.reindex(md.index)).max())
    assert max(parity.values()) < 1e-6, f"mask Dice != table Dice: {parity}"

    summary = dict(threshold=t, k=a.k, val_dice=vdice, threshold_s3d=t11, val_dice_s3d=vdice11,
                   parity_max_abs_diff=parity, train_oof_table_dice=float(tab_tr.mean()),
                   cases={s: len(c) for s, c in cases.items()}, limit=a.limit, provenance=base)
    from sklearn.metrics import roc_auc_score
    for name, df, p in (("train_oof", tr, ptr), ("val", va, pv), ("test", te, pt)):
        r = (df.k >= 0).to_numpy()
        y = df.on_tumour.to_numpy()[r]
        summary[f"auc_{name}"] = float(roc_auc_score(y, p[r])) if 0 < y.sum() < len(y) else None
    for name, fr in (("val", fr_va), ("test", fr_te)):
        agg, comp, per = stats(fr, coll)
        summary[name] = dict(metrics={r_: {k_: (round(float(v), 4) if pd.notna(v) else None) for k_, v in row.items()}
                                      for r_, row in agg.iterrows()},
                             memory_jury_vs=comp, per_collection=per.round(4).to_dict(orient="index"))
        print(f"\n{name.upper()} ({len(fr.cache_id.unique())} cases):\n{agg.round(4).to_string()}")
        print("memory jury vs:", json.dumps({k_: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                   for kk, vv in v.items()} for k_, v in comp.items()}))
        print(per.round(3).to_string())
    print(f"\nparity mask vs table Dice (max |diff|): {parity}")
    print(f"AUC: train out-of-fold {summary['auc_train_oof']}, val {summary['auc_val']}, test {summary['auc_test']}")
    print(f"train out-of-fold mask Dice (refiner input): {summary['train_oof_table_dice']:.4f}")

    pickle.dump(dict(jury=jury, threshold=t, settings=dict(top_k=5, min_vox=100, prob=0.5, overlap=0.5),
                     provenance=base), open(out / "memory_jury.pkl", "wb"))
    summary["minutes"] = (time.time() - t0) / 60
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    if a.zip_outputs:                     # Kaggle keeps at most 500 output files: ship masks/ and cases/ as zips
        import shutil
        import zipfile
        for d in ("masks", "cases"):
            shutil.make_archive(str(out / d), "zip", str(out), d)
            with zipfile.ZipFile(out / f"{d}.zip") as z:
                n = sum(f.endswith(".npz") for f in z.namelist())
            assert n == sum(len(c) for c in cases.values()), (d, n)
            shutil.rmtree(out / d)
            print(f"{d}/ -> {d}.zip ({n} files)")
    print(f"total {summary['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
