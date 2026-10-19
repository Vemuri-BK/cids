"""S7: boundary refinement of the memory-jury mask (frequency refiner vs spatial-only twin).

Input: notebook 12 output (s3f_jury_masks.py): masks/<cid>.npz = final mask M0 of every case (train masks from
an out-of-fold jury), summary.json. The refiner (cids/refine/model.py) only corrects M0 near its boundary;
before training its output is exactly M0 (checked on val at epoch 0).

  --mode freq      fixed 3D Haar bands (LLL context + 7 signed high-frequency bands) -- proposed
  --mode spatial   learned stride-2 conv in place of Haar, same architecture (spatial-only twin)

Steps
  1. prepare ROIs (bbox of M0 + 16 voxels, from M0 only) for train / val / test -> <work>/  (once, reused)
  2. train on 64^3 patches around M0's boundary; training masks randomly perturbed (grow / shrink / wobble)
  3. val (120) every --val_every epochs: exact full-volume Dice of refined vs M0 -> best.pt by val Dice
  4. test (306) once with best.pt: Dice, HD95, NSD, missed for refined and M0, Wilcoxon + bootstrap CI,
     per collection, changed voxels; refined test masks saved
Example
  python scripts/s7_refine.py --mode freq --s3f_dir /kaggle/input/.../s3f_jury --cache_root <cache> \\
      --out /kaggle/working/s7_freq
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import all_metrics  # noqa: E402
from cids.provenance import file_digest, require_cases, save_npz_atomic  # noqa: E402
from cids.refine.data import RefinePatchDataset, channels, pad_min, prepare_case  # noqa: E402
from cids.refine.model import BoundaryRefiner  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _identity(x):
    return x


def _prep(args):
    return prepare_case(*args)


def find_s3f(path: str, unpack_to: Path) -> Path:
    """Folder with summary.json + masks/ (or masks.zip: Kaggle keeps at most 500 output files, so notebook 12's
    masks are shipped as one zip, unpacked here once)."""
    p = Path(path)
    cands = [q.parent for q in p.rglob("summary.json")] if p.exists() else []
    for c in cands:
        if (c / "masks").is_dir() and any((c / "masks").glob("*.npz")):
            return c
    for c in cands:
        if (c / "masks.zip").exists():
            import shutil
            import zipfile
            dst = Path(unpack_to)
            if not (dst / "masks").is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(c / "masks.zip") as z:
                    z.extractall(dst)
                print(f"unpacked {c / 'masks.zip'} -> {dst / 'masks'} ({len(list((dst / 'masks').glob('*.npz')))} masks)")
            shutil.copy(c / "summary.json", dst / "summary.json")
            return dst
    raise SystemExit(f"no notebook 12 output (summary.json + masks/ or masks.zip) under {path}")


def dice_bce(z, y):
    p = torch.sigmoid(z)
    inter = (p * y).sum((1, 2, 3, 4)); den = p.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4))
    return (1 - (2 * inter + 1) / (den + 1)).mean() + F.binary_cross_entropy_with_logits(z, y)


@torch.no_grad()
def refine_case(model, work, meta, patch, amp):
    """Refined ROI mask (bool, ROI shape) or None for an empty M0."""
    if meta["empty"]:
        return None
    from monai.inferers import sliding_window_inference
    d = np.load(work / f"{meta['cid']}.npz")
    x, pads = pad_min(channels(d["img"], d["m0"].astype(bool)), patch)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        z = sliding_window_inference(torch.from_numpy(x)[None].to(DEV), (patch,) * 3, sw_batch_size=4,
                                     predictor=model, overlap=0.5, mode="gaussian")[0, 0].float().cpu().numpy()
    sl = tuple(slice(a, s - b) for (a, b), s in zip(pads, z.shape))
    return z[sl] > 0


def evaluate(model, work, metas, patch, amp):
    """Exact full-volume Dice of refined and M0 from ROI counts (M0 and the refined mask live inside the ROI)."""
    model.eval()
    rows = []
    for meta in metas:
        G = meta["gt_total"]
        if meta["empty"]:
            rows.append(dict(cache_id=meta["cid"], dice_m0=float(G == 0), dice_ref=float(G == 0), changed=0))
            continue
        d = np.load(work / f"{meta['cid']}.npz")
        gt, m0 = d["gt"].astype(bool), d["m0"].astype(bool)
        r = refine_case(model, work, meta, patch, amp)
        dm = lambda p: 2 * (p & gt).sum() / (p.sum() + G) if p.sum() + G else 1.0
        rows.append(dict(cache_id=meta["cid"], dice_m0=float(dm(m0)), dice_ref=float(dm(r)), changed=int((r ^ m0).sum()),
                         added=int((r & ~m0).sum()), removed=int((m0 & ~r).sum())))
    model.train()
    return pd.DataFrame(rows)


def lr_factor(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["freq", "spatial"], required=True)
    ap.add_argument("--s3f_dir", required=True, help="notebook 12 output (contains summary.json, masks/)")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--work", default="/kaggle/working/refine_rois", help="ROI crops (shared by both modes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val_every", type=int, default=3)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--n_patches", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()
    amp = DEV == "cuda" and not a.no_amp
    t0 = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    s3f = find_s3f(a.s3f_dir, Path(a.work).parent / "s3f_unpacked")
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    coll = dict(zip(sp.cache_id, sp.collection))
    cases = {s: require_cases(sp, root, s)[:a.limit or None] for s in ("train", "val", "test")}
    miss = [c for s in cases for c, _ in cases[s] if not (s3f / "masks" / f"{c}.npz").exists()]
    if miss:
        raise SystemExit(f"{len(miss)} cases have no S3f mask in {s3f} (e.g. {miss[:3]})")
    s3f_summary = json.loads((s3f / "summary.json").read_text())
    (out / "config.json").write_text(json.dumps(dict(vars(a), s3f=str(s3f), s3f_summary=file_digest(s3f / "summary.json"),
                                                     s3f_threshold=s3f_summary.get("threshold")), indent=2))
    print(f"S3f masks {s3f} (memory jury t = {s3f_summary.get('threshold')}) | cache {root} | {DEV} | mode {a.mode}")

    # 1. ROIs
    metas = {}
    for s, cs in cases.items():
        jobs = [(c, p, s3f / "masks" / f"{c}.npz", work) for c, p in cs]
        with Pool(max(1, a.workers)) as pool:
            metas[s] = list(tqdm(pool.imap(_prep, jobs), total=len(jobs), desc=f"ROIs {s}", file=sys.stdout, mininterval=30))
        print(f"{s}: {len(cs)} cases, {sum(m['empty'] for m in metas[s])} with empty M0 (kept as is)")
    train_ids = [m["cid"] for m in metas["train"] if not m["empty"]]

    # 2. model + identity check
    model = BoundaryRefiner(a.mode, a.width).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"refiner ({a.mode}): {n_par:,} parameters")
    v0 = evaluate(model, work, metas["val"], a.patch, amp)
    assert (v0.dice_ref == v0.dice_m0).all() and (v0.changed == 0).all(), "refiner at init must reproduce M0 exactly"
    print(f"epoch 0 (= M0): val Dice {v0.dice_m0.mean():.4f} (S3f val {s3f_summary.get('val_dice', float('nan')):.4f})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ds = RefinePatchDataset(train_ids, work, a.patch, a.n_patches, augment=True, seed=a.seed)
    total, warm = a.epochs * len(ds), 2 * len(ds)
    best, log, step, ep_t = float(v0.dice_m0.mean()), [dict(epoch=0, val_dice=float(v0.dice_m0.mean()))], 0, []
    torch.save(model.state_dict(), out / "best.pt")                    # M0 itself is the fallback

    for epoch in range(1, a.epochs + 1):
        te = time.time()
        ds.set_epoch(epoch)
        dl = DataLoader(ds, batch_size=None, shuffle=True, num_workers=a.workers, collate_fn=_identity,
                        prefetch_factor=2 if a.workers else None, generator=torch.Generator().manual_seed(a.seed + epoch))
        losses = []
        for item in tqdm(dl, desc=f"epoch {epoch}/{a.epochs}", unit="case", file=sys.stdout, mininterval=60):
            for g in opt.param_groups:
                g["lr"] = a.lr * lr_factor(step, total, warm)
            x, y = item["x"].to(DEV, non_blocking=True), item["y"].to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z = model(x)
            loss = dice_bce(z.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); step += 1
            losses.append(loss.item())
        rec = dict(epoch=epoch, loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"], minutes=(time.time() - te) / 60)
        msg = f"epoch {epoch}: loss {rec['loss']:.4f}"
        if epoch % a.val_every == 0 or epoch == a.epochs:
            v = evaluate(model, work, metas["val"], a.patch, amp)
            rec.update(val_dice=float(v.dice_ref.mean()), val_changed_mean=float(v.changed.mean()),
                       val_better=int((v.dice_ref - v.dice_m0 > 0.01).sum()), val_worse=int((v.dice_ref - v.dice_m0 < -0.01).sum()))
            if rec["val_dice"] > best:
                best = rec["val_dice"]; torch.save(model.state_dict(), out / "best.pt")
                v.to_csv(out / "val_best_per_case.csv", index=False)
            msg += (f" | val Dice {rec['val_dice']:.4f} (M0 {v.dice_m0.mean():.4f}) better/worse {rec['val_better']}/"
                    f"{rec['val_worse']} changed vox {rec['val_changed_mean']:.0f} | best {best:.4f}")
        print(msg, flush=True)
        log.append(rec); pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        ep_t.append(time.time() - te)
        if (time.time() - t0) / 3600 + 1.5 * np.mean(ep_t[-3:]) / 3600 > a.time_budget_h:
            print(f"time budget reached after epoch {epoch}; testing best.pt"); break

    # 4. test once
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV, weights_only=False))
    model.eval()
    rows, pred_dir = [], out / "test_refined_masks"; pred_dir.mkdir(exist_ok=True)
    paths = dict(cases["test"])
    for meta in tqdm(metas["test"], desc="test", unit="case", file=sys.stdout, mininterval=30):
        cid = meta["cid"]
        _, lbl = load_case(paths[cid]); gt = lbl > 0
        m0 = np.load(s3f / "masks" / f"{cid}.npz")["mask"].astype(bool)
        ref = m0.copy()
        r = refine_case(model, work, meta, a.patch, amp)
        if r is not None:
            lo, hi = meta["bbox"]
            sl = tuple(slice(p, q) for p, q in zip(lo, hi))
            ref[sl] = r
        save_npz_atomic(pred_dir / f"{cid}.npz", mask=ref.astype(np.uint8))
        for name, m in (("m0_memory_jury", m0), (f"refined_{a.mode}", ref)):
            rows.append(dict(cache_id=cid, rule=name, collection=coll.get(cid), **all_metrics(m, gt),
                             changed=int((ref ^ m0).sum())))
    fr = pd.DataFrame(rows); fr.to_csv(out / "test_metrics.csv", index=False)
    agg = fr.groupby("rule").agg(dice=("dice", "mean"), hd95_median=("hd95", "median"), hd95_mean=("hd95", "mean"),
                                 nsd=("nsd", "mean"), missed=("dice", lambda x: int((x < 0.1).sum())))
    piv = fr.pivot(index="cache_id", columns="rule", values="dice")
    d = (piv[f"refined_{a.mode}"] - piv["m0_memory_jury"]).to_numpy()
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    from scipy.stats import wilcoxon
    pn = fr.pivot(index="cache_id", columns="rule", values="nsd")
    comp = dict(delta_dice=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                wilcoxon_p=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0,
                better=int((d > 0.01).sum()), worse=int((d < -0.01).sum()),
                delta_nsd=float((pn[f"refined_{a.mode}"] - pn["m0_memory_jury"]).mean()))
    per = fr.pivot_table(index="collection", columns="rule", values="dice")
    print(f"\nTEST (306), refiner = {a.mode}, best val Dice {best:.4f} (M0 {v0.dice_m0.mean():.4f}):")
    print(agg.round(4).to_string())
    print("refined vs M0:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in comp.items()}))
    print(per.round(3).to_string())
    summary = dict(mode=a.mode, parameters=n_par, best_val_dice=best, val_dice_m0=float(v0.dice_m0.mean()),
                   test={r: {k: round(float(v), 4) for k, v in row.items()} for r, row in agg.iterrows()},
                   refined_vs_m0=comp, per_collection=per.round(4).to_dict(orient="index"),
                   cases={s: len(c) for s, c in cases.items()}, limit=a.limit, minutes=(time.time() - t0) / 60)
    json.dump(summary, open(out / "test_summary.json", "w"), indent=2)
    print(f"total {summary['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
