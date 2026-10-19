"""S6: train and evaluate the frequency council (⑥, no metadata yet).

Input per voxel: image (pre, post1, post1-pre) + finder map (SegResNet-100) + checker map (SAM-S2b + adapter),
from scripts/s5_export_maps.py. 4 independent band experts (cids/fcouncil/experts.py) each predict their own
tumour mask from their Haar band group + the two maps; a gate combines them into the final mask; the spread
of the 4 expert probabilities is the uncertainty.
Loss (cids/fcouncil/losses.py): GT on the final mask and on each expert + agreement-gated distillation from
the finder and the checker (--no_teachers = ablation without dual supervision) + gate balance.

Validation every --val_every epochs (120 cases) -> best.pt by council Dice after rel20.
Test (306, once): council vs finder vs checker (rel20; Dice/HD95/NSD for council and finder), Wilcoxon
council vs finder, each expert alone, gate weight per expert inside the tumour, and how well the
uncertainty predicts failure (Spearman between mean uncertainty and Dice).

Example
  python scripts/s6_freq_council.py --cache_root <cache> --maps_dir /kaggle/working/maps --out /kaggle/working/s6_council
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cids.data.cache import find_cache_root  # noqa: E402
from cids.data.volumes import unpad  # noqa: E402
from cids.eval.metrics import all_metrics, dice  # noqa: E402
from cids.eval.postproc import rel20  # noqa: E402
from cids.fcouncil import GROUP_NAMES  # noqa: E402
from cids.fcouncil.data import CouncilPatchDataset, CouncilVolumeDataset  # noqa: E402
from cids.fcouncil.losses import council_loss  # noqa: E402
from cids.fcouncil.model import FreqCouncil  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = len(GROUP_NAMES)


def _identity(x):
    return x


def _worker_init(_):
    torch.set_num_threads(1)


def cases_for(sp, split, root):
    r = sp[sp.split == split]
    lst = [(c, root / ("train" if o == "train" else "val") / f"{c}.npz") for c, o in zip(r.cache_id, r.official)]
    return [x for x in lst if x[1].exists()]


@torch.no_grad()
def predict_volume(model, x5, patch, amp, overlap):
    from monai.inferers import sliding_window_inference

    def fn(p):
        o = model(p)
        return torch.cat([o["final"], o["experts"], o["weights"], o["uncertainty"]], 1).float()

    with torch.autocast(DEV, dtype=torch.float16, enabled=amp):
        y = sliding_window_inference(x5[None].to(DEV), (patch,) * 3, sw_batch_size=4, predictor=fn,
                                     overlap=overlap, mode="gaussian")[0].float()
    return dict(final=torch.sigmoid(y[0]).cpu().numpy(), experts=torch.sigmoid(y[1:1 + K]).cpu().numpy(),
                weights=y[1 + K:1 + 2 * K].cpu().numpy(), unc=y[1 + 2 * K].cpu().numpy())


def evaluate(model, cases, a, full, desc, save_dir=None):
    model.eval()
    rows = []
    dl = DataLoader(CouncilVolumeDataset(cases, a.maps_dir, a.patch), batch_size=None, num_workers=2,
                    collate_fn=_identity, worker_init_fn=_worker_init)
    for item in tqdm(dl, desc=desc, unit="case", file=sys.stdout, dynamic_ncols=True, mininterval=10):
        cid, pads, x5 = item["cid"], item["pads"], item["image"]
        p = predict_volume(model, x5, a.patch, a.amp, 0.5 if full else a.val_overlap)
        gt = item["label"].numpy() > 0
        fin = unpad(p["final"], pads) > 0.5
        fin20 = rel20(fin)
        finder20 = rel20(unpad(x5[3].numpy(), pads) > 0.5)
        checker20 = rel20(unpad(x5[4].numpy(), pads) > 0.5)
        region = fin20 if fin20.any() else np.ones_like(fin20)
        w, unc = unpad(p["weights"], pads), unpad(p["unc"], pads)
        r = dict(cache_id=cid, collection=cid.split("_")[0].upper(), council_raw=dice(fin, gt),
                 checker_rel20_dice=dice(checker20, gt), uncertainty=float(unc[region].mean()))
        for k, n in enumerate(GROUP_NAMES):
            r[f"{n}_alone"] = dice(unpad(p["experts"][k], pads) > 0.5, gt)
            r[f"w_{n}"] = float(w[k][region].mean())
        if full:
            for name, m in (("council_rel20", fin20), ("finder_rel20", finder20)):
                for key, v in all_metrics(m, gt).items():
                    r[f"{name}_{key}"] = v
        else:
            r["council_rel20_dice"], r["finder_rel20_dice"] = dice(fin20, gt), dice(finder20, gt)
        rows.append(r)
        if save_dir is not None:
            np.savez_compressed(Path(save_dir) / f"{cid}.npz", mask=fin20.astype(np.uint8),
                                uncertainty=(np.clip(unc, 0, 1) * 255).astype(np.uint8))
    return pd.DataFrame(rows)


def lr_factor(e, a):
    if e < a.warmup:
        return (e + 1) / a.warmup
    return 0.5 * (1 + math.cos(math.pi * (e - a.warmup) / max(1, a.epochs - a.warmup)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--maps_dir", required=True, help="output of s5_export_maps.py")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--n_patches", type=int, default=4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lam_exp", type=float, default=0.5)
    ap.add_argument("--lam_kd", type=float, default=0.3)
    ap.add_argument("--lam_bal", type=float, default=0.01)
    ap.add_argument("--no_teachers", action="store_true", help="ablation: ground truth only (no dual supervision)")
    ap.add_argument("--no_map_aug", action="store_true")
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_overlap", type=float, default=0.25)
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--resume_from", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()
    a.amp = (not a.no_amp) and DEV == "cuda"
    t_start = time.time()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    tr, va, te = cases_for(sp, "train", root), cases_for(sp, "val", root), cases_for(sp, "test", root)
    if a.limit_train:
        tr = tr[:a.limit_train]
    if a.limit_eval:
        va, te = va[:a.limit_eval], te[:a.limit_eval]
    missing = [c for c, _ in tr + va + te if not (Path(a.maps_dir) / f"{c}.npz").exists()]
    assert not missing, f"{len(missing)} cases have no maps (e.g. {missing[:3]}); run s5_export_maps.py first"
    print(f"cache {root} | maps {a.maps_dir}\ntrain {len(tr)} | val {len(va)} | test {len(te)} | "
          f"dual supervision: {'off' if a.no_teachers else 'on'}")

    model = FreqCouncil(width=a.width).to(DEV)
    print(f"council parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(per expert {sum(p.numel() for p in model.experts[0].parameters()):,})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_factor(e, a))
    scaler = torch.amp.GradScaler(DEV, enabled=a.amp)
    start, best, log = 0, -1.0, []
    ck = Path(a.resume_from) if a.resume_from else out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, best, log = s["epoch"] + 1, s["best"], s["log"]
        for g, b in zip(opt.param_groups, sched.base_lrs):
            g["lr"] = b * lr_factor(start, a)
        print(f"resumed at epoch {start}, best {best:.4f}")
        if ck.parent.resolve() != out.resolve():                    # resuming into a NEW folder: bring the
            for f in ("best.pt", "val_best_per_case.csv"):          # old best along, or it would be lost
                if (ck.parent / f).exists() and not (out / f).exists():
                    shutil.copy(ck.parent / f, out / f)
            if best > -1 and not (out / "best.pt").exists():
                raise SystemExit(f"{ck} says best {best:.4f}, but there is no best.pt next to it; attach the "
                                 f"whole previous output folder")
    ck = out / "last.pt"

    ds = CouncilPatchDataset(tr, a.maps_dir, a.patch, a.n_patches, map_aug=not a.no_map_aug, seed=a.seed)
    ep_times = []
    for epoch in range(start, a.epochs):
        t0 = time.time()
        ds.set_epoch(epoch)
        dl = DataLoader(ds, batch_size=None, shuffle=True, num_workers=a.workers, collate_fn=_identity,
                        worker_init_fn=_worker_init, prefetch_factor=2 if a.workers else None,
                        generator=torch.Generator().manual_seed(a.seed + epoch))
        model.train()
        parts = []
        for item in tqdm(dl, desc=f"epoch {epoch + 1}/{a.epochs}", unit="pt", file=sys.stdout, dynamic_ncols=True,
                         mininterval=10):
            x = item["image"].to(DEV, non_blocking=True)
            y = item["label"].to(DEV, non_blocking=True)
            with torch.autocast(DEV, dtype=torch.float16, enabled=a.amp):
                o = model(x)
            tch = item["teacher"].to(DEV, non_blocking=True)        # CLEAN maps (no map augmentation)
            L = council_loss(o, y, tch[:, 0:1].float(), tch[:, 1:2].float(), a.lam_exp, a.lam_kd, a.lam_bal,
                             use_teachers=not a.no_teachers)
            opt.zero_grad(set_to_none=True)
            scaler.scale(L["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt); scaler.update()
            parts.append({k: float(v) for k, v in L.items()})
        sched.step()
        rec = {k: float(np.mean([q[k] for q in parts])) for k in parts[0]} if parts else {}
        rec.update(epoch=epoch + 1, lr=opt.param_groups[0]["lr"], train_min=(time.time() - t0) / 60)
        msg = (f"epoch {epoch + 1}: loss {rec.get('total', float('nan')):.4f} (final {rec.get('main', 0):.4f} "
               f"experts {rec.get('experts', 0):.4f} kd {rec.get('kd', 0):.4f})")
        if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
            v = evaluate(model, va, a, False, "val")
            rec.update(val_council=v.council_rel20_dice.mean(), val_finder=v.finder_rel20_dice.mean(),
                       val_checker=v.checker_rel20_dice.mean(),
                       **{f"val_{n}": v[f"{n}_alone"].mean() for n in GROUP_NAMES},
                       **{f"val_w_{n}": v[f"w_{n}"].mean() for n in GROUP_NAMES})
            if rec["val_council"] > best:
                best = rec["val_council"]
                torch.save(model.state_dict(), out / "best.pt")
                v.to_csv(out / "val_best_per_case.csv", index=False)
            msg += (f" | val council {rec['val_council']:.4f} finder {rec['val_finder']:.4f} checker "
                    f"{rec['val_checker']:.4f} | experts " + " ".join(f"{rec[f'val_{n}']:.3f}" for n in GROUP_NAMES)
                    + " | gate " + " ".join(f"{rec[f'val_w_{n}']:.2f}" for n in GROUP_NAMES) + f" | best {best:.4f}")
        print(msg)
        log.append(rec)
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best=best, log=log), ck)
        ep_times.append(time.time() - t0)
        if (time.time() - t_start) / 3600 + 1.5 * np.mean(ep_times[-3:]) / 3600 > a.time_budget_h:
            print(f"time budget reached after epoch {epoch + 1}; re-run with --resume_from {ck} to continue")
            break
    if not (out / "best.pt").exists():
        torch.save(model.state_dict(), out / "best.pt")
    if a.skip_test:
        return

    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV, weights_only=False))
    pred_dir = out / "test_pred"; pred_dir.mkdir(exist_ok=True)
    tm = evaluate(model, te, a, True, "test", save_dir=pred_dir)
    tm.to_csv(out / "test_metrics.csv", index=False)
    from scipy.stats import spearmanr, wilcoxon
    d = tm.council_rel20_dice - tm.finder_rel20_dice
    p = float(wilcoxon(tm.council_rel20_dice, tm.finder_rel20_dice).pvalue) if (d != 0).any() else 1.0

    def block(pref):
        return dict(dice=tm[f"{pref}_dice"].mean(), dice_median=tm[f"{pref}_dice"].median(),
                    hd95_median=tm[f"{pref}_hd95"].median(), nsd=tm[f"{pref}_nsd"].mean(),
                    missed=int((tm[f"{pref}_dice"] < 0.1).sum()))

    summ = dict(cases=len(tm), best_val_council=best, council_rel20=block("council_rel20"),
                finder_rel20=block("finder_rel20"), checker_rel20_dice=tm.checker_rel20_dice.mean(),
                council_minus_finder=dict(mean=float(d.mean()), wilcoxon_p=p),
                experts_alone={n: tm[f"{n}_alone"].mean() for n in GROUP_NAMES},
                gate_weight_in_tumour={n: tm[f"w_{n}"].mean() for n in GROUP_NAMES},
                uncertainty_vs_dice_spearman=float(spearmanr(tm.uncertainty, tm.council_rel20_dice).correlation),
                minutes=(time.time() - t_start) / 60)
    print("\nTEST (306, once)"); print(json.dumps(summ, indent=2, default=float))
    print("\nper collection (Dice mean):")
    print(tm.groupby("collection")[["council_rel20_dice", "finder_rel20_dice", "checker_rel20_dice"]
                                   + [f"{n}_alone" for n in GROUP_NAMES]].mean().round(4).to_string())
    print("\ngate weight inside the predicted tumour, per collection:")
    print(tm.groupby("collection")[[f"w_{n}" for n in GROUP_NAMES]].mean().round(3).to_string())
    (out / "test_summary.json").write_text(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
