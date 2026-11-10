"""nnU-Net baseline on the CIDS split (same cache, same 3 channels, same metrics as the B0_rad pipeline).

Steps
  convert   cache (.npz, 1 mm) -> nnU-Net raw dataset:
              imagesTr/<cid>_0000|0001|0002.nii.gz = pre | post1 | post1 - pre  (as the SegResNet finder)
              labelsTr/<cid>.nii.gz              = expert mask
              train (1,080) + val (120) in imagesTr; test (306) in imagesTs
            + dataset.json, and splits_final.json (fold 0: train = the 1,080, val = our 120) to copy into
              nnUNet_preprocessed/<dataset>/ after planning. Resumable (existing files are kept).
  evaluate  nnU-Net predictions (.nii.gz) -> our metrics (Dice, HD95, NSD@2mm, Dice<0.1/<0.5, per hospital,
            bilateral subgroup), optional paired comparison with B0_rad per patient.
            --split test is allowed ONCE (done.json guard), like the B0_rad final test.

Usage (server)
  python scripts/nnunet_baseline.py --step convert --cache_root ~/data --nnunet_raw ~/nnunet/raw
  python scripts/nnunet_baseline.py --step evaluate --split val --cache_root ~/data \\
      --pred_dir <results>/fold_0/validation --compare ~/CIDS/outputs/s3k_radiomics/per_patient.csv \\
      --out ~/CIDS/outputs/nnunet_baseline/val
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
sys.path.insert(0, str(ROOT))

CHANNELS = {"0": "pre", "1": "post1", "2": "sub"}          # names are not CT -> nnU-Net z-score per channel
AFFINE = np.eye(4)                                         # 1 mm isotropic, same voxel grid as the cache


def dataset_name(i, name):
    return f"Dataset{int(i):03d}_{name}"


def save_nii(arr, path, dtype):
    import nibabel as nib
    img = nib.Nifti1Image(np.ascontiguousarray(arr).astype(dtype), AFFINE)
    img.header.set_zooms((1.0, 1.0, 1.0))
    tmp = path.with_name(path.name.replace(".nii.gz", ".partial.nii.gz"))
    nib.save(img, str(tmp))
    tmp.replace(path)


def load_nii(path):
    import nibabel as nib
    return np.asarray(nib.load(str(path)).dataobj)


def channels(img):
    pre, post = img[0].astype(np.float32), img[1].astype(np.float32)
    return [pre, post, post - pre]


def split_ids(sp, split_col="split"):
    ids = {s: sorted(sp.cache_id[sp[split_col] == s]) for s in ("train", "val", "test")}
    if any(len(set(ids[a]) & set(ids[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise SystemExit("splits overlap")
    return ids


def step_convert(a):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    sp = pd.read_csv(a.splits)
    ids = split_ids(sp)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    ds = Path(a.nnunet_raw).expanduser() / dataset_name(a.dataset_id, a.name)
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (ds / d).mkdir(parents=True, exist_ok=True)
        for f in (ds / d).glob("*.partial.nii.gz"):           # leftovers of an interrupted run
            f.unlink()
    jobs = []
    for split, folder in (("train", "imagesTr"), ("val", "imagesTr"), ("test", "imagesTs")):
        jobs += [(split, folder, cid) for cid in ids[split]]
    paths = {}
    for s_ in ("train", "val", "test"):                     # folder follows the official split (train/ or val/)
        paths.update(dict(require_cases(sp, root, s_)))
    if a.limit:
        keep = set()
        for s in ("train", "val", "test"):
            keep |= set(ids[s][:a.limit])
        jobs = [j for j in jobs if j[2] in keep]
    missing = [cid for _, _, cid in jobs if cid not in paths]
    if missing:
        raise SystemExit(f"{len(missing)} patients missing from the cache (e.g. {missing[:5]})")
    t0, done = time.time(), 0
    for i, (split, folder, cid) in enumerate(jobs):
        outs = [ds / folder / f"{cid}_{k:04d}.nii.gz" for k in range(3)]
        lab = ds / "labelsTr" / f"{cid}.nii.gz"
        if all(o.exists() for o in outs) and (folder == "imagesTs" or lab.exists()):
            continue
        img, gt = load_case(paths[cid])
        for o, ch in zip(outs, channels(img)):
            save_nii(ch, o, np.float32)
        if folder == "imagesTr":
            save_nii(gt > 0, lab, np.uint8)
        done += 1
        if (i + 1) % 100 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    tr = [c for s in ("train", "val") for c in ids[s] if any(j[2] == c for j in jobs)]
    dj = dict(channel_names=CHANNELS, labels=dict(background=0, tumour=1), numTraining=len(tr),
              file_ending=".nii.gz", name=a.name,
              description="MAMA-MIA, CIDS cache (1 mm), channels pre/post1/post1-pre, CIDS split")
    (ds / "dataset.json").write_text(json.dumps(dj, indent=1))
    keep = {j[2] for j in jobs}
    splits = [dict(train=[c for c in ids["train"] if c in keep], val=[c for c in ids["val"] if c in keep])]
    out = Path(a.out).expanduser() if a.out else ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits_final.json").write_text(json.dumps(splits, indent=1))
    (out / "test_ids.txt").write_text("\n".join(c for c in ids["test"] if c in keep) + "\n")
    print(f"CONVERT DONE: {dataset_name(a.dataset_id, a.name)} | imagesTr {len(tr)} "
          f"(train {len(splits[0]['train'])} + val {len(splits[0]['val'])}) | imagesTs "
          f"{sum(1 for j in jobs if j[1] == 'imagesTs')} | converted now {done}")
    print(f"splits_final.json -> {out}  (copy into nnUNet_preprocessed/{dataset_name(a.dataset_id, a.name)}/)")


def boot_mean_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def step_evaluate(a):
    from cids.data.cache import find_cache_root, load_case
    from cids.eval.metrics import all_metrics
    from cids.provenance import require_cases
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    guard = out / "done.json"
    if a.split == "test" and guard.exists():
        raise SystemExit(f"test already evaluated once ({guard}); refusing a second evaluation")
    sp = pd.read_csv(a.splits)
    ids = split_ids(sp)[a.split]
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(sp, root, a.split))
    pdir = Path(a.pred_dir).expanduser()
    if a.limit:
        ids = ids[:a.limit]
    missing = [c for c in ids if not (pdir / f"{c}.nii.gz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing in {pdir} (e.g. {missing[:5]})")
    rows = []
    for cid in ids:
        _, gt = load_case(paths[cid])
        pred = load_nii(pdir / f"{cid}.nii.gz") > 0
        if pred.shape != gt.shape:
            raise SystemExit(f"{cid}: prediction shape {pred.shape} != cache {gt.shape}")
        m = all_metrics(pred, gt > 0)
        rows.append(dict(cache_id=cid, collection=cid.split("_")[0].upper(), **m))
    pp = pd.DataFrame(rows)
    if a.clinical:
        p = Path(a.clinical).expanduser()
        cl = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
        bil = dict(zip(cl.patient_id.str.lower(), cl.bilateral_breast_cancer))
        pp["bilateral_cancer"] = pp.cache_id.map(bil).fillna(0).astype(int)
    pp.to_csv(out / "per_patient.csv", index=False)

    def summ(d, group):
        lo, hi = boot_mean_ci(d.dice)
        return dict(group=group, n=len(d), dice=d.dice.mean(), ci95_lo=lo, ci95_hi=hi, dice_median=d.dice.median(),
                    hd95_median=d.hd95.median(), hd95_n=int(d.hd95.notna().sum()), nsd=d.nsd.mean(),
                    dice_lt_0_1=int((d.dice < 0.1).sum()), dice_lt_0_5=int((d.dice < 0.5).sum()))
    s = [summ(pp, "all")] + [summ(g, c) for c, g in pp.groupby("collection")]
    if "bilateral_cancer" in pp:
        s += [summ(pp[pp.bilateral_cancer == 0], "excl_bilateral")]
        if (pp.bilateral_cancer == 1).any():
            s += [summ(pp[pp.bilateral_cancer == 1], "bilateral_only")]
    sm = pd.DataFrame(s)
    sm.to_csv(out / "summary.csv", index=False)
    res = dict(split=a.split, n=len(pp), pred_dir=str(pdir))
    if a.compare:
        ref = pd.read_csv(Path(a.compare).expanduser())
        want = {"val": "val", "test": "test"}[a.split]
        if "split" in ref:
            ref = ref[ref.split == want]
        col = a.compare_col
        r = pp.merge(ref[["cache_id", col]], on="cache_id", how="left")
        if r[col].isna().any():
            raise SystemExit(f"--compare has no {col} for {int(r[col].isna().sum())} patients")
        d = r.dice - r[col]
        lo, hi = boot_mean_ci(d)
        res["paired_nnunet_minus_ref"] = dict(ref=col, diff=float(d.mean()), ci95=[lo, hi],
                                              nnunet_better=int((d > 1e-9).sum()), ref_better=int((d < -1e-9).sum()),
                                              identical=int((d.abs() <= 1e-9).sum()),
                                              ref_dice=float(r[col].mean()))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    if a.split == "test":
        guard.write_text(json.dumps(dict(evaluated=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(pp)), indent=1))
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))
    print("\n" + json.dumps(res, indent=1))
    print(f"EVALUATE DONE ({a.split}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["convert", "evaluate"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--nnunet_raw", default="", help="nnUNet_raw folder (convert)")
    ap.add_argument("--dataset_id", type=int, default=501)
    ap.add_argument("--name", default="MAMAMIA3ch")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="evaluate")
    ap.add_argument("--pred_dir", default="", help="nnU-Net prediction folder (evaluate)")
    ap.add_argument("--compare", default="", help="per_patient.csv with the reference Dice (B0_rad)")
    ap.add_argument("--compare_col", default="B0_rad_dice")
    ap.add_argument("--clinical", default="", help="clinical_and_imaging_info.xlsx (bilateral subgroup)")
    ap.add_argument("--limit", type=int, default=0, help="tests only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.step == "convert":
        if not a.nnunet_raw:
            raise SystemExit("--nnunet_raw is required")
        step_convert(a)
    else:
        if not (a.pred_dir and a.out):
            raise SystemExit("--pred_dir and --out are required")
        step_evaluate(a)


if __name__ == "__main__":
    main()
