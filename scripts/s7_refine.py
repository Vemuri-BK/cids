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
from cids.refine.source import find_s3f, index_maps  # noqa: E402
from cids.refine.data import R_ALLOW, RefinePatchDataset, pad_min, prepare_case, roi_channels  # noqa: E402
from cids.refine.model import BoundaryRefiner  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _identity(x):
    return x


def _prep(args):
    return prepare_case(*args)


def dice_bce(z, y, allow):
    """Dice + BCE over the voxels the refiner can change (allow band) only. Outside the band the output is
    fixed at M0 (p = 0.25 / 0.75): counting those voxels flooded the Dice denominator (a perfect boundary
    still scored loss ~0.97) and diluted BCE ~8x -- review REFINEMENT_BUGS_AND_FIXES.txt, bugs 1-3."""
    m = allow > 0.5
    if not m.any():
        return z.sum() * 0.0
    za, ya = z[m], y[m]
    pa = torch.sigmoid(za)
    dice = 1 - (2 * (pa * ya).sum() + 1) / (pa.sum() + ya.sum() + 1)
    return dice + F.binary_cross_entropy_with_logits(za, ya)


def balanced_loss(z, y, x, w_fix=1.0, w_keep=1.0):
    """Masked Dice + group-balanced BCE over the allow band (training labels only). Groups:
    correctable missed tumour (GT & ~M0), correctable extra (M0 & ~GT), already-correct band voxels.
    Each non-empty group's BCE is its own mean, so the few error voxels are not outvoted by the many
    correct ones; the preservation group stays in so the model does not edit everywhere."""
    band, m0 = x[:, 5:6] > 0.5, x[:, 3:4] > 0.5
    if not band.any():
        return z.sum() * 0.0
    yb = y > 0.5
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    groups = [(band & yb & ~m0, w_fix), (band & ~yb & m0, w_fix), (band & (yb == m0), w_keep)]
    num, den = 0.0, 0.0
    for g, w in groups:
        if g.any():
            num = num + w * bce[g].mean(); den += w
    za, ya = z[band], y[band]
    pa = torch.sigmoid(za)
    dice = 1 - (2 * (pa * ya).sum() + 1) / (pa.sum() + ya.sum() + 1)
    return dice + num / den


def refine_loss(name, z, y, x, w_keep=1.0):
    return balanced_loss(z, y, x, w_keep=w_keep) if name == "balanced" else dice_bce(z, y, x[:, 5:6])


@torch.no_grad()
def refine_case(model, work, meta, patch, amp):
    """Refined ROI mask (bool, ROI shape) or None for an empty M0."""
    if meta["empty"]:
        return None
    from monai.inferers import sliding_window_inference
    d = np.load(work / f"{meta['cid']}.npz")
    x, pads = pad_min(roi_channels(d), patch)
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
        add, rem = r & ~m0, m0 & ~r
        rows.append(dict(cache_id=meta["cid"], dice_m0=float(dm(m0)), dice_ref=float(dm(r)), changed=int((r ^ m0).sum()),
                         added=int(add.sum()), removed=int(rem.sum()),
                         add_helpful=int((add & gt).sum()), add_harmful=int((add & ~gt).sum()),     # diagnostics only:
                         rem_helpful=int((rem & ~gt).sum()), rem_harmful=int((rem & gt).sum()),     # GT never enters
                         vol_m0=int(m0.sum()), vol_ref=int(r.sum()), gt_vol=int(G)))                # the model
    model.train()
    return pd.DataFrame(rows)


def lr_factor(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["freq", "freq_lll", "spatial"], default=None, help="required unless --test_only")
    ap.add_argument("--s3f_dir", default=None, help="notebook 12 output (contains summary.json, masks/); required unless --test_only")
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
    ap.add_argument("--p_perturb", type=float, default=0.8, help="share of training masks randomly roughened (v1 0.8)")
    ap.add_argument("--perturb_scale", type=float, default=1.0, help="roughening strength (v1 1.0)")
    ap.add_argument("--min_m0_dice", type=float, default=0.0,
                    help="train only on cases whose jury mask already overlaps the tumour (Dice >= this): wrong-blob "
                         "cases cannot be fixed by a boundary refiner and only add noise (v1 0)")
    ap.add_argument("--weight_hard", action="store_true",
                    help="sample training cases in proportion to their boundary error (1 - Dice of the jury mask)")
    ap.add_argument("--error_sampling", action="store_true",
                    help="patch centres 25%% correctable missed / 25%% correctable extra / 50%% boundary (training GT only)")
    ap.add_argument("--skip_test", action="store_true", help="pilot runs: no test-set evaluation")
    ap.add_argument("--loss", choices=["dice_bce", "balanced"], default="dice_bce",
                    help="balanced = masked Dice + BCE averaged per group (missed / extra / already correct)")
    ap.add_argument("--w_keep", type=float, default=1.0, help="balanced loss: weight of the already-correct group")
    ap.add_argument("--no_gate", action="store_true", help="ablation: z = z0 + allow * delta (no output gate)")
    ap.add_argument("--prob_dirs", default="",
                    help="refiner v2: comma-separated SegResNet map folders (out-of-fold seg_fold0..2 for training "
                         "patients + seg_foldfull for val/test); adds the probability map as an input channel")
    ap.add_argument("--train_check", type=int, default=120, help="fixed training cases evaluated (unaugmented) at each val")
    ap.add_argument("--time_budget_h", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="cases per split (tests)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--test_only", action="store_true",
                    help="no training: evaluate the saved <out>/best.pt on test once, with the run's own settings "
                         "(mode, masks, ROIs, probability maps, width, gate, patch) read from <out>/config.json")
    a = ap.parse_args()
    if a.test_only:
        cfg_f = Path(a.out) / "config.json"
        if not (Path(a.out) / "best.pt").exists() or not cfg_f.exists():
            raise SystemExit(f"--test_only needs {a.out}/best.pt and {a.out}/config.json from the training run")
        if a.skip_test:
            raise SystemExit("--test_only and --skip_test exclude each other")
        cfg = json.loads(cfg_f.read_text())
        for k in ("mode", "s3f_dir", "cache_root", "splits", "work", "width", "patch", "no_gate", "prob_dirs", "limit"):
            if k in cfg:
                setattr(a, k, cfg[k])
        print(f"TEST ONLY: settings of the training run ({cfg_f}): mode {a.mode}, masks {a.s3f_dir}, ROIs {a.work}, "
              f"probability maps {'yes' if a.prob_dirs else 'no'}")
    elif a.mode is None or a.s3f_dir is None:
        ap.error("--mode and --s3f_dir are required (unless --test_only)")
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
    # ROI cache identity: the crops in --work must come from exactly these masks and settings
    import hashlib
    h = hashlib.md5()
    for c, _ in sorted(x for s_ in cases.values() for x in s_):
        h.update(c.encode()); h.update((s3f / "masks" / f"{c}.npz").read_bytes())
    maps = {}
    if a.prob_dirs:
        maps = index_maps(a.prob_dirs.split(","), unpack_to=Path(a.work).parent / "prob_unpacked")
        miss = [c for s_ in cases.values() for c, _ in s_ if c not in maps]
        if miss:
            raise SystemExit(f"{len(miss)} cases have no SegResNet probability map (e.g. {miss[:5]})")
        hp = hashlib.md5()
        for c, _ in sorted(x for s_ in cases.values() for x in s_):
            hp.update(c.encode()); hp.update(Path(maps[c]).read_bytes())
    roi_manifest = dict(version=3, masks_md5=h.hexdigest(), n_cases=sum(len(v) for v in cases.values()), margin=16,
                        r_allow=R_ALLOW, s3f_threshold=s3f_summary.get("threshold"),
                        prob_md5=hp.hexdigest() if maps else None)
    mf = work / "rois_manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != roi_manifest:
        raise SystemExit(f"{work} holds ROIs from other masks/settings ({mf}); use a new --work")
    if not mf.exists() and any(work.glob("*.npz")):
        raise SystemExit(f"{work} holds ROIs without a manifest; use a new --work")
    mf.write_text(json.dumps(roi_manifest, indent=2))
    (out / ("config_test_only.json" if a.test_only else "config.json")).write_text(json.dumps(dict(vars(a), s3f=str(s3f), s3f_summary=file_digest(s3f / "summary.json"),
                                                     s3f_threshold=s3f_summary.get("threshold")), indent=2))
    print(f"S3f masks {s3f} (memory jury t = {s3f_summary.get('threshold')}) | cache {root} | {DEV} | mode {a.mode}")

    # 1. ROIs
    metas = {}
    for s, cs in cases.items():
        jobs = [(c, p, s3f / "masks" / f"{c}.npz", work, 16, maps.get(c)) for c, p in cs]
        with Pool(max(1, a.workers)) as pool:
            metas[s] = list(tqdm(pool.imap(_prep, jobs), total=len(jobs), desc=f"ROIs {s}", file=sys.stdout, mininterval=30))
        print(f"{s}: {len(cs)} cases, {sum(m['empty'] for m in metas[s])} with empty M0 (kept as is)")
    def m0_dice(meta):                                   # older ROI folders have no m0_dice in the meta
        if "m0_dice" not in meta:
            d = np.load(work / f"{meta['cid']}.npz"); mm, gg = d["m0"].astype(bool), d["gt"].astype(bool)
            meta["m0_dice"] = float(2 * (mm & gg).sum() / max(mm.sum() + meta["gt_total"], 1))
        return meta["m0_dice"]
    train_metas = [m for m in metas["train"] if not m["empty"] and m0_dice(m) >= a.min_m0_dice]
    train_ids = [m["cid"] for m in train_metas]
    td = np.array([m["m0_dice"] for m in train_metas])
    print(f"training cases: {len(train_ids)} (jury-mask Dice >= {a.min_m0_dice}; mean {td.mean():.3f}) | "
          f"roughening p={a.p_perturb} scale={a.perturb_scale} | hard-case weighting {'on' if a.weight_hard else 'off'}")
    sample_w = np.clip(1 - td, 0.05, 0.5) if a.weight_hard else None

    # 2. model + identity check
    model = BoundaryRefiner(a.mode, a.width, gate=not a.no_gate, extra_ch=1 if maps else 0).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"refiner ({a.mode}, output gate {'off' if a.no_gate else 'on'}, loss {a.loss}, inputs: MRI + M0"
          f"{' + SegResNet probability map (v2)' if maps else ' (v1)'}): {n_par:,} parameters")
    v0 = evaluate(model, work, metas["val"], a.patch, amp)
    assert (v0.dice_ref == v0.dice_m0).all() and (v0.changed == 0).all(), "refiner at init must reproduce M0 exactly"
    vm = s3f / "val_metrics.csv"                         # per-case parity with notebook 12 (not just the mean)
    if vm.exists() and not a.limit:
        ref = pd.read_csv(vm).query("rule == 'memory_jury'").set_index("cache_id").dice
        diff = (v0.set_index("cache_id").dice_m0 - ref.reindex(v0.cache_id).values).abs().max()
        if not diff < 1e-6:
            raise SystemExit(f"epoch-0 per-case val Dice differs from notebook 12 (max |diff| {diff:.2e}): wrong masks?")
        print(f"per-case parity with notebook 12 val_metrics.csv: max |diff| {diff:.1e}")
    print(f"epoch 0 (= M0): val Dice {v0.dice_m0.mean():.4f} (S3f val {s3f_summary.get('val_dice', float('nan')):.4f})")
    rng_c = np.random.default_rng(0)
    check_metas = [train_metas[i] for i in sorted(rng_c.choice(len(train_metas), min(a.train_check, len(train_metas)),
                                                                replace=False))]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ds = RefinePatchDataset(train_ids, work, a.patch, a.n_patches, augment=True, seed=a.seed,
                            p_perturb=a.p_perturb, perturb_scale=a.perturb_scale, error_sampling=a.error_sampling)
    print(f"patch sampling: {'error-aware (25/25/50)' if a.error_sampling else 'boundary 75% / random 25%'} | "
          f"test: {'SKIPPED (pilot)' if a.skip_test else 'once, at the end'}")
    total, warm = a.epochs * len(ds), 2 * len(ds)
    best, log, step, ep_t = float(v0.dice_m0.mean()), [dict(epoch=0, val_dice=float(v0.dice_m0.mean()))], 0, []
    if a.test_only:                                     # keep the trained best.pt; report its logged val Dice
        lg = out / "train_log.csv"
        if lg.exists() and "val_dice" in pd.read_csv(lg):
            best = float(pd.read_csv(lg).val_dice.max())
    else:
        torch.save(model.state_dict(), out / "best.pt")                    # M0 itself is the fallback

    for epoch in range(1, (0 if a.test_only else a.epochs) + 1):
        te = time.time()
        ds.set_epoch(epoch)
        g = torch.Generator().manual_seed(a.seed + epoch)
        sampler = (torch.utils.data.WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(ds),
                                                          replacement=True, generator=g) if sample_w is not None else None)
        dl = DataLoader(ds, batch_size=None, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                        collate_fn=_identity, prefetch_factor=2 if a.workers else None, generator=g)
        losses = []
        for item in tqdm(dl, desc=f"epoch {epoch}/{a.epochs}", unit="case", file=sys.stdout, mininterval=60):
            for g in opt.param_groups:
                g["lr"] = a.lr * lr_factor(step, total, warm)
            x, y = item["x"].to(DEV, non_blocking=True), item["y"].to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z = model(x)
            loss = refine_loss(a.loss, z.float(), y, x, a.w_keep)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); step += 1
            losses.append(loss.item())
        rec = dict(epoch=epoch, loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"], minutes=(time.time() - te) / 60)
        msg = f"epoch {epoch}: loss {rec['loss']:.4f}"
        if epoch % a.val_every == 0 or epoch == a.epochs:
            v = evaluate(model, work, metas["val"], a.patch, amp)
            v.to_csv(out / f"val_epoch{epoch:03d}.csv", index=False)
            tc = evaluate(model, work, check_metas, a.patch, amp)          # unaugmented training cases
            torch.save(model.state_dict(), out / "last_model.pt")         # trained state, even if it loses to M0
            rec.update(val_dice=float(v.dice_ref.mean()), val_changed_mean=float(v.changed.mean()),
                       val_better=int((v.dice_ref - v.dice_m0 > 0.01).sum()), val_worse=int((v.dice_ref - v.dice_m0 < -0.01).sum()),
                       val_add_helpful=int(v.add_helpful.sum()), val_add_harmful=int(v.add_harmful.sum()),
                       val_rem_helpful=int(v.rem_helpful.sum()), val_rem_harmful=int(v.rem_harmful.sum()),
                       train_check_m0=float(tc.dice_m0.mean()), train_check_ref=float(tc.dice_ref.mean()))
            if rec["val_dice"] > best:
                best = rec["val_dice"]; torch.save(model.state_dict(), out / "best.pt")
                v.to_csv(out / "val_best_per_case.csv", index=False)
            msg += (f" | val Dice {rec['val_dice']:.4f} (M0 {v.dice_m0.mean():.4f}) better/worse {rec['val_better']}/"
                    f"{rec['val_worse']} changed vox {rec['val_changed_mean']:.0f} | best {best:.4f}"
                    f"\n   val edits (voxels): added {rec['val_add_helpful']} helpful / {rec['val_add_harmful']} harmful, "
                    f"removed {rec['val_rem_helpful']} helpful / {rec['val_rem_harmful']} harmful"
                    f" | train-check ({len(check_metas)} cases, unaugmented): {rec['train_check_m0']:.4f} -> "
                    f"{rec['train_check_ref']:.4f}")
        print(msg, flush=True)
        log.append(rec); pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        ep_t.append(time.time() - te)
        if (time.time() - t0) / 3600 + 1.5 * np.mean(ep_t[-3:]) / 3600 > a.time_budget_h:
            print(f"time budget reached after epoch {epoch}; testing best.pt"); break

    if a.skip_test:                                     # boundary metrics of the best model on val (full volumes)
        model.load_state_dict(torch.load(out / "best.pt", map_location=DEV, weights_only=False))
        vp, vrows = dict(cases["val"]), []
        for meta in metas["val"]:
            cid = meta["cid"]
            _, lbl = load_case(vp[cid]); gt = lbl > 0
            m0 = np.load(s3f / "masks" / f"{cid}.npz")["mask"].astype(bool)
            ref = m0.copy(); r = refine_case(model, work, meta, a.patch, amp)
            if r is not None:
                ref[tuple(slice(p_, q_) for p_, q_ in zip(*meta["bbox"]))] = r
            for name, m in (("m0", m0), ("refined", ref)):
                vrows.append(dict(cache_id=cid, rule=name, **all_metrics(m, gt)))
        vb = pd.DataFrame(vrows); vb.to_csv(out / "val_best_metrics.csv", index=False)
        vbs = vb.groupby("rule").agg(dice=("dice", "mean"), nsd=("nsd", "mean"), hd95_median=("hd95", "median"))
        print("val, best model (full volumes):\n" + vbs.round(4).to_string())
        json.dump(dict(mode=a.mode, best_val_dice=best, val_dice_m0=float(v0.dice_m0.mean()), log=log,
                       val_boundary=vbs.round(4).to_dict(orient="index"),
                       config=vars(a), minutes=(time.time() - t0) / 60), open(out / "pilot_summary.json", "w"), indent=2)
        print(f"pilot done (test skipped): best val Dice {best:.4f} vs M0 {v0.dice_m0.mean():.4f} -> {out}")
        return

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
