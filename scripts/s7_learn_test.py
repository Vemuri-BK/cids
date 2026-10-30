"""S7 learning test: can the refiner learn to fix M0 on a FEW real training cases? (run before any full run)

Picks --n_cases training cases whose jury mask M0 has both correctable missed tumour and correctable extra
voxels inside the 8 mm band (fixable by a boundary refiner), cuts --n_patches fixed error-centred patches per
case (no augmentation, same patches every step) and trains each configuration from scratch on them.
Every --eval_every steps it reports
  patches   : band Dice of M0 vs refined on the SAME training patches, share of missed / extra voxels fixed,
              share of correct voxels kept, mean residual (z - z0) pushed towards the truth on error voxels
              (|residual| must exceed log 3 = 1.10 to flip a voxel)
  full ROI  : Dice of M0 vs refined on the whole ROI of the same cases (sliding windows, as at inference)
  unseen    : the same on --n_unseen other training cases (never trained on: does anything transfer?)
Reading
  patches do not improve           -> the model / loss cannot fit even fixed examples (optimisation problem)
  patches improve, full ROI not    -> inference problem (patch context, sliding-window blending)
  fitted cases improve, unseen not -> it memorises; corrections do not transfer from the image alone
Configurations: --configs "mode:loss:gate,..." with mode freq|freq_lll|spatial, loss dice_bce|balanced,
gate gate|nogate. Training labels only; val and test are never touched.
Example
  python scripts/s7_learn_test.py --s3f_dir /kaggle/input/... --cache_root <cache> --out /kaggle/working/s7_learn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cids.data.cache import find_cache_root  # noqa: E402
from cids.provenance import require_cases  # noqa: E402
from cids.refine.data import R_ALLOW, RefinePatchDataset, signed_distance  # noqa: E402
from cids.refine.model import LOG3, BoundaryRefiner  # noqa: E402
from cids.refine.source import find_s3f, index_maps  # noqa: E402
from s7_refine import DEV, _prep, evaluate, refine_loss  # noqa: E402


def case_errors(work, meta):
    """Correctable missed / extra voxels of M0 inside the allow band (training labels)."""
    d = np.load(work / f"{meta['cid']}.npz")
    m0, gt = d["m0"].astype(bool), d["gt"].astype(bool)
    band = np.abs(signed_distance(m0)) <= R_ALLOW
    return int((gt & ~m0 & band).sum()), int((m0 & ~gt & band).sum())


@torch.no_grad()
def patch_eval(model, X, Y, amp, bs=8):
    """Metrics on the fixed training patches (hard decisions, allow band only)."""
    model.eval()
    acc = dict(d_m0=[], d_ref=[], fn=[0, 0], fp=[0, 0], keep=[0, 0], r_fn=[], r_fp=[])
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(DEV), Y[i:i + bs].to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            z = model(x).float()
        band, m0, g, p = x[:, 5:6] > 0.5, x[:, 3:4] > 0.5, y > 0.5, z > 0
        r = z - (2 * m0.float() - 1) * LOG3                                  # residual added to M0's logit
        for k in range(len(x)):
            b = band[k]
            for key, pr in (("d_m0", m0[k]), ("d_ref", p[k])):
                s = (pr & b).sum() + (g[k] & b).sum()
                acc[key].append(float(2 * (pr & g[k] & b).sum() / s) if s else 1.0)
        fn, fp, keep = band & g & ~m0, band & ~g & m0, band & (g == m0)
        acc["fn"][0] += int((p & fn).sum()); acc["fn"][1] += int(fn.sum())
        acc["fp"][0] += int((~p & fp).sum()); acc["fp"][1] += int(fp.sum())
        acc["keep"][0] += int((p == m0)[keep].sum()); acc["keep"][1] += int(keep.sum())
        if fn.any():
            acc["r_fn"].append(float(r[fn].mean()))
        if fp.any():
            acc["r_fp"].append(float(-r[fp].mean()))
    model.train()
    fr = lambda t: t[0] / t[1] if t[1] else float("nan")
    return dict(patch_dice_m0=float(np.mean(acc["d_m0"])), patch_dice_ref=float(np.mean(acc["d_ref"])),
                fixed_missed=fr(acc["fn"]), fixed_extra=fr(acc["fp"]), kept_correct=fr(acc["keep"]),
                resid_missed=float(np.mean(acc["r_fn"])) if acc["r_fn"] else float("nan"),
                resid_extra=float(np.mean(acc["r_fp"])) if acc["r_fp"] else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3f_dir", required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--work", default="/kaggle/working/learn_rois")
    ap.add_argument("--out", default="/kaggle/working/s7_learn")
    ap.add_argument("--configs", default="freq:dice_bce:gate,freq:balanced:gate,freq:dice_bce:nogate,freq:balanced:nogate")
    ap.add_argument("--n_cases", type=int, default=16)
    ap.add_argument("--n_unseen", type=int, default=16)
    ap.add_argument("--pool", type=int, default=200, help="training cases screened for fixable errors")
    ap.add_argument("--min_err", type=int, default=300, help="min correctable missed AND extra voxels per case")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--n_patches", type=int, default=4, help="fixed patches per case")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--w_keep", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prob_dirs", default="", help="refiner v2: SegResNet map folders (adds the probability channel)")
    a = ap.parse_args()
    t0 = time.time()
    amp = DEV == "cuda"
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    s3f = find_s3f(a.s3f_dir, work.parent / "s3f_unpacked_learn")
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    train = require_cases(sp, root, "train")
    rng = np.random.default_rng(a.seed)
    pool = [train[i] for i in rng.permutation(len(train))[:a.pool]]
    maps = index_maps(a.prob_dirs.split(","), unpack_to=work.parent / "prob_unpacked") if a.prob_dirs else {}
    if maps and any(c not in maps for c, _ in pool):
        raise SystemExit("some training cases have no SegResNet probability map")
    with Pool(max(1, a.workers)) as p:
        metas = list(p.imap(_prep, [(c, q, s3f / "masks" / f"{c}.npz", work, 16, maps.get(c)) for c, q in pool]))
    if maps:                                             # ROIs cached by an earlier v1 run have no probability
        stale = [m["cid"] for m in metas if not m["empty"] and "prob" not in np.load(work / f"{m['cid']}.npz").files]
        if stale:
            raise SystemExit(f"{work} holds ROIs without probability maps (e.g. {stale[:3]}); use a new --work")
    rows = []
    for m in metas:
        if m["empty"]:
            continue
        fn, fp = case_errors(work, m)
        rows.append(dict(cid=m["cid"], m0_dice=m["m0_dice"], fix_missed=fn, fix_extra=fp, meta=m))
    cand = pd.DataFrame(rows)
    ok = cand[(cand.fix_missed >= a.min_err) & (cand.fix_extra >= a.min_err) & cand.m0_dice.between(0.4, 0.95)]
    ok = ok.assign(score=np.minimum(ok.fix_missed, ok.fix_extra)).sort_values("score", ascending=False)
    if len(ok) < a.n_cases + a.n_unseen:
        raise SystemExit(f"only {len(ok)} eligible cases in the pool of {a.pool}; raise --pool or lower --min_err")
    fit, unseen = ok.iloc[:a.n_cases], ok.iloc[a.n_cases:a.n_cases + a.n_unseen]
    fit_m, uns_m = list(fit.meta), list(unseen.meta)
    print(f"fitted cases: {len(fit)} (M0 Dice {fit.m0_dice.mean():.3f}, fixable missed {fit.fix_missed.median():.0f} / "
          f"extra {fit.fix_extra.median():.0f} voxels median) | unseen: {len(unseen)} (M0 Dice {unseen.m0_dice.mean():.3f})")
    fit.drop(columns="meta").assign(role="fit").pipe(
        lambda d: pd.concat([d, unseen.drop(columns="meta").assign(role="unseen")])).to_csv(out / "cases.csv", index=False)

    ds = RefinePatchDataset(list(fit.cid), work, a.patch, a.n_patches, augment=False, seed=a.seed, error_sampling=True)
    items = [ds[i] for i in range(len(ds))]
    X = torch.cat([it["x"] for it in items]); Y = torch.cat([it["y"] for it in items])
    print(f"fixed patches: {len(X)} x {a.patch}^3 (error-centred, no augmentation)")

    log, summ = [], {}
    for cfg in a.configs.split(","):
        mode, loss, gate = cfg.split(":")
        torch.manual_seed(a.seed)
        model = BoundaryRefiner(mode, a.width, gate=(gate == "gate"), extra_ch=1 if maps else 0).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        g = torch.Generator().manual_seed(a.seed)
        print(f"\n=== {cfg} ({sum(p.numel() for p in model.parameters()):,} parameters) ===", flush=True)
        losses = []
        for step in range(1, a.steps + 1):
            idx = torch.randint(len(X), (a.batch,), generator=g)
            x, y = X[idx].to(DEV), Y[idx].to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z = model(x)
            l = refine_loss(loss, z.float(), y, x, a.w_keep)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); losses.append(l.item())
            if step % a.eval_every == 0 or step == a.steps:
                pe = patch_eval(model, X, Y, amp)
                vf = evaluate(model, work, fit_m, a.patch, amp)
                vu = evaluate(model, work, uns_m, a.patch, amp)
                rec = dict(config=cfg, step=step, loss=float(np.mean(losses[-a.eval_every:])), **pe,
                           roi_fit_m0=float(vf.dice_m0.mean()), roi_fit_ref=float(vf.dice_ref.mean()),
                           roi_fit_add=f"{int(vf.add_helpful.sum())}/{int(vf.add_harmful.sum())}",
                           roi_fit_rem=f"{int(vf.rem_helpful.sum())}/{int(vf.rem_harmful.sum())}",
                           unseen_m0=float(vu.dice_m0.mean()), unseen_ref=float(vu.dice_ref.mean()),
                           unseen_better_worse=f"{int((vu.dice_ref - vu.dice_m0 > 0.01).sum())}/"
                                               f"{int((vu.dice_ref - vu.dice_m0 < -0.01).sum())}")
                log.append(rec); pd.DataFrame(log).to_csv(out / "learn_log.csv", index=False)
                print(f"step {step}: loss {rec['loss']:.3f} | patches band Dice {pe['patch_dice_m0']:.3f} -> "
                      f"{pe['patch_dice_ref']:.3f}, fixed missed {pe['fixed_missed']:.2f} extra {pe['fixed_extra']:.2f}, "
                      f"kept {pe['kept_correct']:.3f}, residual missed {pe['resid_missed']:+.2f} extra "
                      f"{pe['resid_extra']:+.2f}\n   full ROI (fitted) {rec['roi_fit_m0']:.3f} -> {rec['roi_fit_ref']:.3f} "
                      f"[added {rec['roi_fit_add']}, removed {rec['roi_fit_rem']} helpful/harmful] | unseen "
                      f"{rec['unseen_m0']:.3f} -> {rec['unseen_ref']:.3f} better/worse {rec['unseen_better_worse']}",
                      flush=True)
        r = log[-1]
        summ[cfg] = dict(patch_gain=r["patch_dice_ref"] - r["patch_dice_m0"], roi_gain=r["roi_fit_ref"] - r["roi_fit_m0"],
                         unseen_gain=r["unseen_ref"] - r["unseen_m0"], fixed_missed=r["fixed_missed"],
                         fixed_extra=r["fixed_extra"], kept_correct=r["kept_correct"])

    s = pd.DataFrame(summ).T
    print("\n===== SUMMARY (last step) =====\n" + s.round(4).to_string())
    print("\nreading: patch_gain ~0 -> cannot fit fixed examples | patch_gain > 0 but roi_gain ~0 -> inference/context |"
          " roi_gain > 0 but unseen_gain <= 0 -> memorises, no transfer")
    json.dump(dict(summary=s.round(5).to_dict(orient="index"), config=vars(a), minutes=(time.time() - t0) / 60),
              open(out / "learn_summary.json", "w"), indent=2)
    print(f"total {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
