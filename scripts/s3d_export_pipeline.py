"""Export SegResNet-100 + SAM-S2b + learned jury (S3d, test Dice 0.735) as ONE file: cids_s3d_pipeline.pt.

The jury is re-fitted from S3d's train_features.csv with the chosen settings (gradient boosting,
cut-off 0.60 chosen on val) -- deterministic, identical to the S3d run -- or loaded from jury_models.pkl.
Optionally checks the bundle on a few val cases (Dice must match S3d's pipeline).

Example
  python scripts/s3d_export_pipeline.py --seg_ckpt segresnet_img_e100_best.pt --sam_ckpt sam_s2b_best.pt \\
      --jury_features s3d_train_features.csv --out /kaggle/working/cids_s3d_pipeline.pt --check 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3d_learned_jury as s3d  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import dice  # noqa: E402
from cids.models.sam_med3d import download_turbo  # noqa: E402
from cids.pipeline import CIDSPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--jury_features", default="", help="S3d train_features.csv (re-fit, recommended)")
    ap.add_argument("--jury_pkl", default="", help="or S3d jury_models.pkl")
    ap.add_argument("--jury_model", default="gboost")
    ap.add_argument("--threshold", type=float, default=0.60, help="S3d cut-off chosen on val")
    ap.add_argument("--out", default="/kaggle/working/cids_s3d_pipeline.pt")
    ap.add_argument("--check", type=int, default=0, help="run the bundle on N val cases")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    a = ap.parse_args()
    if a.jury_features:
        jury = s3d.fit_models(pd.read_csv(a.jury_features).query("k >= 0"))[a.jury_model]
    else:
        jury = pickle.load(open(a.jury_pkl, "rb"))[a.jury_model]
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    pipe = CIDSPipeline.build(a.seg_ckpt, a.sam_ckpt, turbo, jury, a.threshold)
    pipe.save(a.out)
    print(f"saved {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB): SegResNet + SAM (full) + jury "
          f"({a.jury_model}, cut-off {a.threshold})")

    if a.check:
        p2 = CIDSPipeline.load(a.out)
        root = find_cache_root(a.cache_root)
        sp = pd.read_csv(a.splits)
        r = sp[sp.split == "val"].head(a.check)
        for cid, off in zip(r.cache_id, r.official):
            t0 = time.time()
            img, lbl = load_case(root / ("train" if off == "train" else "val") / f"{cid}.npz")
            out = p2.predict(img)
            print(f"{cid}: Dice {dice(out['mask'], lbl > 0):.3f} | candidates {len(out['candidates'])}, kept "
                  f"{int(out['candidates'].kept.sum()) if len(out['candidates']) else 0} | {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
