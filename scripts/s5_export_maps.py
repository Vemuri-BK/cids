"""S5: finder and checker probability maps for every case (inputs and teachers of the frequency council).

Fully automatic, no ground truth used:
  SegResNet-100 (finder) -> probability map -> candidates (components >= 0.1 ml, top 5)
  SAM-S2b (+ S4 frequency adapter, if given) on one crop + box per candidate (checker)
Saved per case at the 1 mm cache grid: <out>/<cache_id>.npz with
  seg  uint8  SegResNet probability x 255
  sam  uint8  max over candidates of SAM's probability x 255 (0 where no candidate);
             with --jury (S3d train_features.csv): each candidate's SAM map is multiplied by the learned
             jury's P(real tumour), so SegResNet's false blobs enter the council weak (no ground truth used;
             training cases get a cross-fitted jury that never saw them)
Train cases use sliding-window overlap 0.25 (faster), val/test 0.5. Existing files are skipped
(resumable). Prints a sanity table for val/test: Dice of finder (rel20) and checker (rel20).

Example
  python scripts/s5_export_maps.py --cache_root <cache> --seg_ckpt segresnet_img_e100_best.pt \\
      --sam_ckpt sam_s2b_best.pt --adapter_ckpt /kaggle/working/s4_adapter/best.pt --out /kaggle/working/maps
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3a_seg2sam as s3  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.data.volumes import VolumeDataset, unpad  # noqa: E402
from cids.eval.metrics import dice  # noqa: E402
from cids.eval.postproc import rel20  # noqa: E402
from cids.fcouncil.adapter import SamFreqAdapter  # noqa: E402
from cids.models.sam_med3d import download_turbo  # noqa: E402

DEV = s3.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True, help="S2b best.pt")
    ap.add_argument("--adapter_ckpt", default="", help="S4 best.pt (adapter + decoder state); '' = S2b alone")
    ap.add_argument("--jury", default="", help="S3d train_features.csv: weight each candidate's SAM map by the "
                    "learned jury's P(tumour); training cases use a cross-fitted jury that never saw them")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/maps")
    ap.add_argument("--only", default="train,val,test")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="per split (tests)")
    ap.add_argument("--time_budget_h", type=float, default=11.0)
    a = ap.parse_args()
    amp = DEV == "cuda"
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    seg, sam, box = s3.load_models(a.seg_ckpt, a.sam_ckpt, turbo)
    adapter = None
    if a.adapter_ckpt:
        st = torch.load(a.adapter_ckpt, map_location=DEV, weights_only=False)
        adapter = SamFreqAdapter().to(DEV).eval()
        adapter.load_state_dict(st["adapter"])
        sam.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    jury = None
    if a.jury:
        import s3d_learned_jury as s3d
        jury = s3d.CrossFitJury(a.jury)
    print(f"jury: {a.jury or 'none'}")
    print(f"cache {root}\nfinder {a.seg_ckpt}\nchecker {a.sam_ckpt} + adapter: {a.adapter_ckpt or 'none'}\n{DEV}")

    sp = pd.read_csv(a.splits)
    rows = []
    for split in a.only.split(","):
        r = sp[sp.split == split]
        cases = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        cases = [x for x in cases if x[1].exists()]
        if a.limit:
            cases = cases[:a.limit]
        todo = [x for x in cases if not (out / f"{x[0]}.npz").exists()]
        print(f"{split}: {len(cases)} cases, {len(cases) - len(todo)} already exported")
        ds = VolumeDataset(todo, 96)
        overlap = 0.25 if split == "train" else 0.5
        for i in tqdm(range(len(ds)), desc=split, unit="case", file=sys.stdout, mininterval=20):
            item = ds[i]; cid = item["cid"]
            img, lbl = load_case(todo[i][1])
            with torch.no_grad():
                sprob = unpad(s3.seg_prob(seg, item["image"], amp, overlap=overlap), item["pads"])
                lab, cands = s3.components(sprob > 0.5, top_k=a.top_k)
                refined = s3.sam_refine(sam, box, img, lab, cands, amp, adapter) if cands else []
            w = [1.0] * len(refined)
            if jury is not None and refined:
                feats = pd.DataFrame(s3d.candidate_rows(cid, sprob, lab, cands, refined, np.zeros_like(sprob, bool)))
                w = jury.predict(cid, feats)
                by_k = dict(zip(feats.k, w)); w = [by_k[d["k"]] for d in refined]
            sam_p = np.max([wi * d["prob"] for wi, d in zip(w, refined)], 0) if refined else np.zeros_like(sprob)
            np.savez_compressed(out / f"{cid}.npz", seg=(np.clip(sprob, 0, 1) * 255).round().astype(np.uint8),
                                sam=(np.clip(sam_p, 0, 1) * 255).round().astype(np.uint8))
            if split != "train":
                gt = lbl > 0
                rows.append(dict(cache_id=cid, split=split, collection=cid.split("_")[0].upper(),
                                 finder_rel20=dice(rel20(sprob > 0.5), gt), checker_rel20=dice(rel20(sam_p > 0.5), gt),
                                 n_cand=len(cands)))
            if (time.time() - t0) / 3600 > a.time_budget_h:
                print("time budget reached; re-run the same command to continue (finished cases are skipped)")
                break
    if rows:
        df = pd.DataFrame(rows); df.to_csv(out / "_export_check.csv", index=False)
        print("\nsanity check (cases exported in this run):")
        print(df.groupby(["split", "collection"])[["finder_rel20", "checker_rel20"]].mean().round(4).to_string())
        print(df.groupby("split")[["finder_rel20", "checker_rel20", "n_cand"]].mean().round(4).to_string())
    print(f"{len(list(out.glob('*.npz')))} map files in {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
