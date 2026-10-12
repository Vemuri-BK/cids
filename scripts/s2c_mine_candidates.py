"""S2c step 1: mine SegResNet's REAL candidates (true and false) and cache SAM embeddings for them.

Why: S2b learned to reject random boxes on tumour-free tissue (97%), but on SegResNet's real false
candidates (bright, tumour-like blobs) it rejected only 0.6% (S3b test). S2c trains the checker on
the finder's own mistakes. This script builds that training data.

Per case:
  SegResNet-100 -> probability (sliding window) -> components of prob > --thr (>= 0.1 ml, top --top_k)
  every candidate -> the SAME 128^3 SAM crop + box as the S3 pipeline (s3a.candidate_crops)
                  -> frozen SAM-Med3D-turbo encoder -> embedding (384,8,8,8) fp16
  label: true  = the candidate touches the tumour   -> SAM target = tumour inside the crop
         false = no overlap with the tumour          -> SAM target = EMPTY mask ("no tumour here")
Saved: <out>/<split>/<cache_id>.npz  emb (K,384,8,8,8) f16, box (K,2,3) i16, is_true (K,) bool,
       gt_bits (K, 128^3/8) packed tumour crop, seg_ml (K,), seg_prob_mean (K,)
       <out>/<split>_index.csv   one row per candidate.
GT is used only to LABEL candidates (true/false) for training and for measuring val/test, never for prompting.

Train cases: SegResNet saw them in training, so it makes fewer mistakes there than on new cases.
To still get enough hard negatives, train candidates use a lower threshold (--thr_train 0.3) and more
candidates (--top_k_train 8). Val/test use the pipeline setting (0.5, top 5) so they measure exactly
what S3 will see.

Resumable (existing case files are skipped). Example
  python scripts/s2c_mine_candidates.py --cache_root <cache> --seg_ckpt segresnet_img_e100_best.pt \\
      --out /kaggle/working/s2c_cands --only train,val,test
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
from cids.models.sam_med3d import download_turbo, load_sam_med3d  # noqa: E402
from cids.models.segresnet_text import TextSegResNet  # noqa: E402

DEV = s3.DEV


def mine_case(seg, sam, item, path, thr, top_k, amp, overlap):
    img, lbl = load_case(path)
    gt = lbl > 0
    sprob = unpad(s3.seg_prob(seg, item["image"], amp, overlap=overlap), item["pads"])
    lab, cands = s3.components(sprob > thr, top_k=top_k)
    if not cands:
        return None
    cc = s3.candidate_crops(img, lab, cands, lbl)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = torch.cat([sam.image_encoder(torch.from_numpy(cc["crops"][j:j + 4]).to(DEV))
                         for j in range(0, len(cands), 4)])
    is_true = np.array([bool(gt[lab == k].any()) for k, _ in cands])
    return dict(emb=emb.float().cpu().numpy().astype(np.float16),
                box=cc["boxes"].astype(np.int16), is_true=is_true,
                gt_bits=np.packbits(cc["gt"].reshape(len(cands), -1), axis=1),
                seg_ml=np.array([s / 1000 for _, s in cands], np.float32),
                seg_prob_mean=np.array([sprob[lab == k].mean() for k, _ in cands], np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--seg_ckpt", required=True, help="SegResNet-100 best.pt")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", default="/kaggle/working/s2c_cands")
    ap.add_argument("--only", default="train,val,test")
    ap.add_argument("--thr_train", type=float, default=0.3)
    ap.add_argument("--top_k_train", type=int, default=8)
    ap.add_argument("--thr_eval", type=float, default=0.5, help="val/test = the S3 pipeline setting")
    ap.add_argument("--top_k_eval", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--time_budget_h", type=float, default=11.0)
    a = ap.parse_args()
    amp = DEV == "cuda"
    t0 = time.time()
    out = Path(a.out)
    root = find_cache_root(a.cache_root)
    seg = TextSegResNet(text=False).to(DEV).eval()
    seg.load_state_dict(torch.load(a.seg_ckpt, map_location=DEV, weights_only=False))
    sam = load_sam_med3d(None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt")), DEV).eval()
    print(f"cache {root}\nfinder {a.seg_ckpt}\n{DEV}")

    sp = pd.read_csv(a.splits)
    stop = False
    for split in a.only.split(","):
        (out / split).mkdir(parents=True, exist_ok=True)
        r = sp[sp.split == split]
        cases = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
        cases = [x for x in cases if x[1].exists()][:a.limit or None]
        todo = [x for x in cases if not (out / split / f"{x[0]}.npz").exists()]
        thr, top_k = (a.thr_train, a.top_k_train) if split == "train" else (a.thr_eval, a.top_k_eval)
        print(f"{split}: {len(cases)} cases, {len(cases) - len(todo)} already mined | threshold {thr}, top {top_k}")
        ds = VolumeDataset(todo, 96)
        for i in tqdm(range(len(ds)), desc=split, unit="case", file=sys.stdout, mininterval=20):
            item = ds[i]
            d = mine_case(seg, sam, item, todo[i][1], thr, top_k, amp, 0.25 if split == "train" else 0.5)
            if d is None:                       # no candidate: empty marker so the case is not redone
                d = dict(emb=np.zeros((0, 384, 8, 8, 8), np.float16), box=np.zeros((0, 2, 3), np.int16),
                         is_true=np.zeros(0, bool), gt_bits=np.zeros((0, 128 ** 3 // 8), np.uint8),
                         seg_ml=np.zeros(0, np.float32), seg_prob_mean=np.zeros(0, np.float32))
            np.savez_compressed(out / split / f"{item['cid']}.npz", **d)
            if (time.time() - t0) / 3600 > a.time_budget_h:
                print("time budget reached; re-run the same command to continue"); stop = True; break
        rows = []
        for f in sorted((out / split).glob("*.npz")):
            d = np.load(f)
            for j in range(len(d["is_true"])):
                rows.append(dict(cache_id=f.stem, k=j, is_true=bool(d["is_true"][j]), seg_ml=float(d["seg_ml"][j]),
                                 seg_prob_mean=float(d["seg_prob_mean"][j])))
        idx = pd.DataFrame(rows, columns=["cache_id", "k", "is_true", "seg_ml", "seg_prob_mean"])
        idx.to_csv(out / f"{split}_index.csv", index=False)
        n_case = len(list((out / split).glob("*.npz")))
        print(f"{split}: {n_case} cases | {len(idx)} candidates = {int(idx.is_true.sum())} true + "
              f"{int((~idx.is_true).sum())} FALSE (hard negatives) | cases with >= 1 false candidate: "
              f"{idx[~idx.is_true].cache_id.nunique()}")
        if stop:
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min; size: "
          f"{sum(f.stat().st_size for f in out.rglob('*.npz')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
