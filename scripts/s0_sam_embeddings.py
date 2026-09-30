"""S0: cache frozen SAM-Med3D-turbo embeddings + run the zero-shot click baseline.

For every case in the FADC 2-channel cache:
  1. resample 1 mm -> 1.5 mm (SAM-Med3D training spacing)
  2. build 128^3 crops: train cases 1 tumour-centred + N jittered crops; test cases 1 tumour-centred crop
  3. two SAM input modes: 'subtraction' (post1 - pre) and 'post1'; z-normalised per crop
  4. frozen image encoder -> embedding (K, 384, 8, 8, 8) fp16
  5. save <out>/<split>/<cache_id>.npz  (emb_subtraction, emb_post1, label, origin, center, shape15)

Test cases additionally get the SAM-Med3D zero-shot baseline (1..N GT-sampled clicks,
official 'random' click protocol) for both input modes -> zero_shot_clicks.csv.
This tells us (a) the "SAM-Med3D alone" row of the paper and (b) which input mode to use.

Resumable: cases whose .npz already exists are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import (SAM_MODES, crop_centers, find_cache_root, list_cases,  # noqa: E402
                             load_case, make_sam_crops, resample, seed_for)
from cids.models.sam_med3d import click_inference, dice, download_turbo, load_sam_med3d  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, cases, n_jitter, max_shift):
        self.cases, self.n_jitter, self.max_shift = cases, n_jitter, max_shift

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, split, path = self.cases[i]
        try:
            img, lbl = load_case(path)
            img15, lbl15 = resample(img, lbl)
            k = self.n_jitter if split == "train" else 0
            centers = crop_centers(lbl15, k, self.max_shift, seed_for(cid))
            crops = make_sam_crops(img15, lbl15, centers)
            tum_total = int(lbl15.sum())
            return dict(cid=cid, split=split, ok=True, centers=centers, shape15=np.array(lbl15.shape),
                        tumor_vox=tum_total, **crops)
        except Exception as e:  # keep going, report at the end
            return dict(cid=cid, split=split, ok=False, err=repr(e))


def _worker_init(_):
    torch.set_num_threads(1)  # avoid CPU oversubscription across loader workers


def _identity(x):
    return x  # keep numpy arrays (default_convert would turn them into tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--out", default="/kaggle/working/sam_emb")
    ap.add_argument("--ckpt", default="", help="local sam_med3d_turbo.pth (downloaded if empty)")
    ap.add_argument("--random_weights", action="store_true", help="for CPU tests only")
    ap.add_argument("--n_jitter", type=int, default=3)
    ap.add_argument("--max_shift", type=int, default=16, help="voxels at 1.5 mm (16 = 24 mm)")
    ap.add_argument("--clicks", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", choices=["all", "train", "test"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    root = find_cache_root(a.cache_root)
    out = Path(a.out)
    cases = list_cases(root)
    if a.only != "all":
        cases = [c for c in cases if c[1] == a.only]
    if a.limit:  # keep both splits in smoke tests
        tr = [c for c in cases if c[1] == "train"][: a.limit]
        te = [c for c in cases if c[1] == "test"][: a.limit]
        cases = tr + te
    todo = [c for c in cases if not (out / c[1] / f"{c[0]}.npz").exists()]
    print(f"cache: {root}\ncases: {len(cases)} | already done: {len(cases) - len(todo)} | to do: {len(todo)}")
    for s in ("train", "test"):
        (out / s).mkdir(parents=True, exist_ok=True)

    ckpt = None if a.random_weights else (a.ckpt or download_turbo("/tmp/sam_ckpt"))  # not in /kaggle/working
    model = load_sam_med3d(ckpt, a.device)
    use_amp = a.device.startswith("cuda")

    dl = DataLoader(CropDataset(todo, a.n_jitter, a.max_shift), batch_size=None, shuffle=False, collate_fn=_identity, worker_init_fn=_worker_init,
                    num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    idx_rows, zs_rows, errors = [], [], []
    zs_path, idx_path = out / "zero_shot_clicks.csv", out / "index.csv"
    t0 = time.time()
    bar = tqdm(total=len(cases), initial=len(cases) - len(todo), desc="S0", unit="case",
               file=sys.stdout, dynamic_ncols=True, mininterval=5)
    for item in dl:
        if not item["ok"]:
            errors.append((item["cid"], item["err"]))
            bar.update(1)
            continue
        cid, split = item["cid"], item["split"]
        embs = {}
        for m in SAM_MODES:
            x = torch.from_numpy(item[m]).to(a.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                embs[m] = model.image_encoder(x).float()
        lab = item["label"]
        # crop 0 is the tumour-centred crop: fraction of the tumour inside it
        in_crop = float(lab[0].sum()) / max(item["tumor_vox"], 1)
        if split == "test":
            gt = torch.from_numpy(lab[0] > 0)
            for m in SAM_MODES:
                preds = click_inference(model, embs[m][:1], gt, a.clicks, seed_for(cid, 233))
                for k, p in enumerate(preds, 1):
                    zs_rows.append(dict(cache_id=cid, mode=m, clicks=k, dice=dice(p, gt)))
        np.savez_compressed(out / split / f"{cid}.npz",
                            **{f"emb_{m}": embs[m].cpu().numpy().astype(np.float16) for m in SAM_MODES},
                            label=lab, origin=item["origin"], center=item["centers"], shape15=item["shape15"])
        idx_rows.append(dict(cache_id=cid, split=split, n_crops=len(lab), shape15="x".join(str(int(v)) for v in item["shape15"]),
                             tumor_vox_15mm=item["tumor_vox"], tumor_frac_in_crop0=round(in_crop, 4)))
        bar.update(1)
        if zs_rows:
            last = pd.DataFrame(zs_rows)
            last = last[last.clicks == 1].groupby("mode").dice.mean()
            bar.set_postfix({f"zs1_{k[:4]}": f"{v:.3f}" for k, v in last.items()})
        if len(idx_rows) % 50 == 0:  # checkpoint the tables
            _append(idx_path, idx_rows); _append(zs_path, zs_rows); idx_rows, zs_rows = [], []
    bar.close()
    _append(idx_path, idx_rows); _append(zs_path, zs_rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min | errors: {len(errors)}")
    for e in errors[:20]:
        print("  ", e)
    summarize(out)


def _append(path: Path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize(out: Path):
    zs = out / "zero_shot_clicks.csv"
    if zs.exists():
        d = pd.read_csv(zs).drop_duplicates(["cache_id", "mode", "clicks"], keep="last")
        d["dataset"] = d.cache_id.str.split("_").str[0].str.upper()
        print("\nSAM-Med3D-turbo zero-shot, test crops (mean Dice):")
        print(d.pivot_table(index="clicks", columns="mode", values="dice", aggfunc="mean").round(4).to_string())
        print("\n1 click, per collection:")
        print(d[d.clicks == 1].pivot_table(index="dataset", columns="mode", values="dice").round(4).to_string())
    ix = out / "index.csv"
    if ix.exists():
        i = pd.read_csv(ix).drop_duplicates("cache_id", keep="last")
        print(f"\nindexed cases: {len(i)} ({(i.split == 'train').sum()} train / {(i.split == 'test').sum()} test)")
        print(f"tumours fully inside the 128^3 (192 mm) crop: {(i.tumor_frac_in_crop0 >= 0.999).mean():.1%}; "
              f"median fraction {i.tumor_frac_in_crop0.median():.3f}")


if __name__ == "__main__":
    main()
