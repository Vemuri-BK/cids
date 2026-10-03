"""S1: train council member B = SegResNet, image-only (--text off) or + clinical instruction (--text on).

Split from configs/splits.csv (train 1080 / val 120 / test 306; or a loco_<C> column).
- train: random 96^3 patches (4 per patient per epoch), DiceCE loss, AdamW, cosine LR, AMP
- val:   whole-volume sliding-window Dice every --val_every epochs -> best.pt
- test:  best.pt on the official test set once, per-case Dice / HD95 / NSD -> test_metrics.csv
Everything is resumable from <out>/last.pt (or --resume_from); a time budget stops training
cleanly so the test evaluation always runs inside one Kaggle session.

Examples
  python scripts/s1_train_segresnet.py --text off --out /kaggle/working/s1_img
  python scripts/s1_train_segresnet.py --text on  --out /kaggle/working/s1_txt --prompts_csv .../cids_prompts.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import find_cache_root  # noqa: E402
from cids.data.volumes import PatchDataset, VolumeDataset, unpad  # noqa: E402
from cids.eval.metrics import all_metrics, dice  # noqa: E402
from cids.models.segresnet_text import TextSegResNet  # noqa: E402
from cids.prompts.sampling import pick_prompt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _identity(x):
    return x


def _worker_init(_):
    torch.set_num_threads(1)


def cases_for(split_df, split, root):
    out = []
    for cid, off in zip(split_df.cache_id, split_df.official):
        sub = "train" if off == "train" else "val"          # FADC cache folder names
        p = root / sub / f"{cid}.npz"
        if p.exists():
            out.append((cid, p))
    return out


@torch.no_grad()
def predict_volume(model, image, t, patch, amp, overlap=0.5):
    from monai.inferers import sliding_window_inference
    x = image[None].to(DEV, non_blocking=True)
    pred_fn = (lambda p: model(p, t.expand(p.shape[0], -1))) if t is not None else (lambda p: model(p))
    with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
        logits = sliding_window_inference(x, (patch,) * 3, sw_batch_size=4, predictor=pred_fn,
                                          overlap=overlap, mode="gaussian")
    return (torch.sigmoid(logits.float())[0, 0] > 0.5).cpu().numpy()


def evaluate(model, loader, prompts, args, text_enc, full_metrics, save_dir=None, column=None, desc="eval",
             overlap=0.5):
    model.eval()
    rows = []
    for item in tqdm(loader, desc=desc, unit="case", file=sys.stdout, dynamic_ncols=True, mininterval=10):
        cid = item["cid"]
        t = None
        if text_enc is not None:
            txt = pick_prompt(prompts[cid], "S", 0.0, column=column)
            t = text_enc.encode([txt])
        pred = unpad(predict_volume(model, item["image"], t, args.patch, args.amp, overlap), item["pads"])
        gt = item["label"].numpy() > 0
        m = all_metrics(pred, gt) if full_metrics else dict(dice=dice(pred, gt))
        rows.append(dict(cache_id=cid, **m))
        if save_dir is not None:
            np.savez_compressed(Path(save_dir) / f"{cid}.npz", mask=pred.astype(np.uint8))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--prompts_csv", default="")
    ap.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "configs" / "splits.csv"))
    ap.add_argument("--split_col", default="split", help="'split' or loco_DUKE / loco_ISPY1 / ...")
    ap.add_argument("--text", choices=["off", "on"], default="off")
    ap.add_argument("--prompt_mode", choices=["S", "N", "SN"], default="S")
    ap.add_argument("--field_dropout", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--n_patches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_overlap", type=float, default=0.25, help="sliding-window overlap for validation (test uses 0.5)")
    ap.add_argument("--time_budget_h", type=float, default=10.0, help="stop training after this; test still runs")
    ap.add_argument("--resume_from", default="", help="last.pt from a previous run (e.g. an attached Kaggle output)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()
    a.amp = (not a.no_amp) and DEV == "cuda"
    t_start = time.time()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    tr = cases_for(sp[sp[a.split_col] == "train"], "train", root)
    va = cases_for(sp[sp[a.split_col] == "val"], "val", root)
    te = cases_for(sp[sp[a.split_col] == "test"], "test", root)
    if a.limit_train:
        tr = tr[: a.limit_train]
    if a.limit_eval:
        va, te = va[: a.limit_eval], te[: a.limit_eval]
    print(f"cache {root}\ntrain {len(tr)} | val {len(va)} | test {len(te)} | text {a.text} ({a.prompt_mode})")
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    prompts, text_enc = {}, None
    if a.text == "on":
        if not a.prompts_csv:
            raise SystemExit("--text on needs --prompts_csv")
        pdf = pd.read_csv(a.prompts_csv, keep_default_na=False)
        prompts = {r["cache_id"]: r for r in pdf.to_dict("records")}
        missing = [c for c, _ in tr + va + te if c not in prompts]
        assert not missing, f"no prompt for {missing[:5]}"
        from cids.text.encoder import TextEncoder
        text_enc = TextEncoder(device=DEV)

    model = TextSegResNet(text=(a.text == "on"), text_dim=(text_enc.dim if text_enc else 768)).to(DEV)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    from monai.losses import DiceCELoss
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / a.warmup if e < a.warmup else
        0.5 * (1 + math.cos(math.pi * (e - a.warmup) / max(1, a.epochs - a.warmup))))
    scaler = torch.amp.GradScaler(DEV, enabled=a.amp)

    start_epoch, best, log = 0, -1.0, []
    ck = out / "last.pt"
    if a.resume_from and not ck.exists():
        shutil.copy(a.resume_from, ck)
        bp = Path(a.resume_from).with_name("best.pt")
        if bp.exists():
            shutil.copy(bp, out / "best.pt")
    if ck.exists():
        s = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start_epoch, best, log = s["epoch"] + 1, s["best"], s["log"]
        print(f"resumed at epoch {start_epoch}, best val Dice {best:.4f}")

    vol_loader = lambda cs: DataLoader(VolumeDataset(cs, a.patch), batch_size=None, num_workers=2,
                                       collate_fn=_identity, worker_init_fn=_worker_init)

    if not a.eval_only:
        ds = PatchDataset(tr, a.patch, a.n_patches, seed=a.seed)
        rng = random.Random(a.seed)
        epoch_times = []
        for epoch in range(start_epoch, a.epochs):
            t0 = time.time()
            ds.set_epoch(epoch)
            dl = DataLoader(ds, batch_size=None, shuffle=True, num_workers=a.workers, collate_fn=_identity,
                            worker_init_fn=_worker_init, prefetch_factor=2 if a.workers else None,
                            generator=torch.Generator().manual_seed(a.seed + epoch))
            model.train()
            losses = []
            bar = tqdm(dl, desc=f"epoch {epoch + 1}/{a.epochs}", unit="pt", file=sys.stdout,
                       dynamic_ncols=True, mininterval=10)
            for item in bar:
                x = item["image"].to(DEV, non_blocking=True)
                y = item["label"].to(DEV, non_blocking=True)
                t = None
                if text_enc is not None:
                    txt = pick_prompt(prompts[item["cid"]], a.prompt_mode, a.field_dropout, rng)
                    t = text_enc.encode([txt]).expand(x.shape[0], -1)
                with torch.autocast(DEV, dtype=torch.float16, enabled=a.amp):
                    logits = model(x, t) if t is not None else model(x)
                    loss = loss_fn(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(opt); scaler.update()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")
            sched.step()
            rec = dict(epoch=epoch + 1, loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"],
                       train_min=(time.time() - t0) / 60)
            if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
                v = evaluate(model, vol_loader(va), prompts, a, text_enc, False, desc="val", overlap=a.val_overlap)
                rec["val_dice"] = float(v.dice.mean())
                if rec["val_dice"] > best:
                    best = rec["val_dice"]
                    torch.save(model.state_dict(), out / "best.pt")
                    v.to_csv(out / "val_best_per_case.csv", index=False)
                print(f"epoch {epoch + 1}: loss {rec['loss']:.4f} | val Dice {rec['val_dice']:.4f} | best {best:.4f}")
            log.append(rec)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, best=best, log=log), ck)
            epoch_times.append(time.time() - t0)
            elapsed_h = (time.time() - t_start) / 3600
            if elapsed_h + 1.5 * np.mean(epoch_times[-3:]) / 3600 > a.time_budget_h:
                print(f"time budget reached after epoch {epoch + 1} ({elapsed_h:.1f} h) -> stopping; "
                      f"re-run with --resume_from {ck} to continue")
                break
        if not (out / "best.pt").exists():      # no validation happened yet
            torch.save(model.state_dict(), out / "best.pt")

    if a.skip_test:
        return
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV, weights_only=False))
    pred_dir = out / "test_pred"; pred_dir.mkdir(exist_ok=True)
    tm = evaluate(model, vol_loader(te), prompts, a, text_enc, True, save_dir=pred_dir, desc="test")
    tm["collection"] = tm.cache_id.str.split("_").str[0].str.upper()
    tm.to_csv(out / "test_metrics.csv", index=False)
    summ = {"cases": len(tm), "dice_mean": tm.dice.mean(), "dice_median": tm.dice.median(),
            "hd95_mean": tm.hd95.mean(), "nsd_mean": tm.nsd.mean(), "missed(<0.1)": (tm.dice < 0.1).mean(),
            "best_val_dice": best}
    print("\nTEST (official 306, evaluated once):")
    print(json.dumps({k: round(float(v), 4) for k, v in summ.items()}, indent=2))
    print(tm.groupby("collection")[["dice", "hd95", "nsd"]].mean().round(4).to_string())
    (out / "test_summary.json").write_text(json.dumps({k: float(v) for k, v in summ.items()}, indent=2))


if __name__ == "__main__":
    main()
