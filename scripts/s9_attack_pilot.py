"""S9 pilot: image-space adversarial attack on the finder, run end to end through the LOCKED pipeline.

Question: when a small perturbation of the MRI fools SegResNet, does the jury still reject the result?
Read-only: no model is trained or changed; the jury bundles of the final S3h run are loaded as they are.
Development set only (val, by default 30 cases); the test set is not touched.

Threat model (non-adaptive, white-box on the finder only):
  the attacker adds delta to the pre- and post-contrast phases, |delta| <= eps per voxel (intensity scale [0, 1]),
  only inside the sliding windows that touch the target and never on padding voxels (exact zeros stay zero); the third finder channel
  (post - pre) is recomputed from the perturbed phases. PGD with sign steps on the EXACT sliding-window output of
  SegResNet (same windows, Gaussian blending, padding as the pipeline; checked per case: sw_check).
  decoy  push SegResNet's probability UP in a ball (radius --radius mm) in tumour-free tissue >= --min_dist mm
         from the expert tumour -> does a false candidate appear, and is it kept in the final mask?
  hide   push the probability DOWN inside the expert tumour -> is the tumour lost?

For every case and eps the whole pipeline is re-run on the attacked scan: finder (sliding window) ->
candidates (p > 0.5, >= 0.1 ml, top 5) -> SAM-S2b crop + box + descriptor -> memory -> radiomics (+ position)
-> each jury at its OWN locked cut-off (bundle threshold: jury13 0.70, jury13_rad_rel 0.40). SegResNet + rel20
is the finder-only reference.

Outputs (<out>):  runs.csv (one row per case x mode x eps), candidates.csv (every candidate with both jury
scores), summary.json, clean_parity.json (clean re-run vs the locked scores, if --ref_scores is given).
Resumable: finished cases are stored in <out>/cases/<cid>.json and skipped.

Example (server, GPU 0)
  CUDA_VISIBLE_DEVICES=0 python scripts/s9_attack_pilot.py --cache_root ~/data \\
      --seg_ckpt ~/CIDS/checkpoints/segresnet_img_e100_best.pt --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt \\
      --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth \\
      --bundles ~/CIDS/outputs/s3h_radiomics/final --ref_scores ~/CIDS/outputs/s3h_radiomics/scores.csv \\
      --out ~/CIDS/outputs/s9_attack_pilot --n_cases 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cids.data.cache import find_cache_root, load_case, seed_for  # noqa: E402
from cids.data.volumes import pad_to, three_channel, unpad  # noqa: E402
from cids.eval.metrics import dice  # noqa: E402
from cids.eval.postproc import rel20  # noqa: E402
from cids.jury import keep_candidates  # noqa: E402
from cids.provenance import require_cases  # noqa: E402
from cids.radiomics import RAD_EXTRACTED, add_position, case_features, load_bundle  # noqa: E402

JURIES = ("jury13", "jury13_rad_rel")          # locked bundles: <bundles>/<name>/<name>.pkl


# ---------------------------------------------------------------------------- geometry
def ball(shape, centre, r) -> np.ndarray:
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((g[i] - centre[i]) ** 2 for i in range(3)) <= r ** 2


def choose_decoy(img, sprob, gt, rng, min_dist=30.0, max_prob=0.1, r=5):
    """Centre of the decoy ball: non-padding tissue at or above the median post-contrast intensity (not air),
    clean finder probability < max_prob, >= min_dist mm from the expert tumour, ball inside the volume.
    Returns None if no voxel qualifies."""
    pre, post = img[0], img[1]
    tissue = (pre != 0) | (post != 0)
    if not tissue.any():
        return None
    ok = tissue & (post >= np.median(post[tissue])) & (sprob < max_prob)
    if gt.any():
        ok &= ndimage.distance_transform_edt(~gt) >= min_dist
    inner = np.zeros_like(ok)
    inner[tuple(slice(r, s - r) for s in ok.shape)] = True
    idx = np.argwhere(ok & inner)
    if not len(idx):
        return None
    return tuple(int(v) for v in idx[rng.integers(len(idx))])


ROI, OVERLAP, SIGMA = 96, 0.5, 0.125         # = s3a.seg_prob: sliding window 96^3, overlap 0.5, Gaussian blending


def sw_windows(shape, roi=ROI, overlap=OVERLAP):
    """The exact window slices MONAI's sliding_window_inference uses on a volume of this (padded) shape."""
    from monai.data.utils import dense_patch_slices
    try:
        from monai.inferers.utils import _get_scan_interval
        interval = _get_scan_interval(tuple(shape), (roi,) * 3, 3, (overlap,) * 3)
    except ImportError:                                                  # same rule as MONAI 1.x
        interval = tuple(roi if roi == s else max(int(roi * (1 - overlap)), 1) for s in shape)
    return [tuple(w) for w in dense_patch_slices(tuple(shape), (roi,) * 3, interval)]


def restrict_target(target, centre, max_windows, shape):
    """Hide mode, large tumours: if more than max_windows sliding windows touch the target, keep only the target
    voxels whose output comes entirely from the windows containing the tumour centre (<= 8, up to 27 at
    clipped borders), so the attack stays affordable and its gradient stays exact. Returns (target, kept fraction)."""
    wins = sw_windows(shape)
    if len([w for w in wins if target[w].any()]) <= max_windows:
        return target, 1.0
    sel = [w for w in wins if all(w[d].start <= centre[d] < w[d].stop for d in range(3))]
    n_all, n_sel = np.zeros(shape, np.int16), np.zeros(shape, np.int16)
    for w in wins:
        n_all[w] += 1
    for w in sel:
        n_sel[w] += 1
    t = target & (n_sel == n_all)
    return t, float(t.sum() / max(int(target.sum()), 1))


# ---------------------------------------------------------------------------- attack
def pgd_sw(seg, img, target, mode, eps, steps, alpha, amp, batch=4):
    """PGD on EXACTLY what the pipeline computes: the Gaussian-blended sliding-window logits of SegResNet
    (same windows, weights and padding as s3a.seg_prob), restricted to the windows touching the target.
    Exact gradient in two passes per step: (1) blended logits without graph -> dL/dZ, (2) per window batch,
    backward of sum(z * w / W * dL/dZ) (the blend is linear in the window logits). Memory = one batch.
    img (2,X,Y,Z) float32, target bool (X,Y,Z). mode 'decoy' raises, 'hide' lowers the target logits.
    Returns (adv image, info) with info['p_clean_check'] = max |sigmoid(Z_clean) - ...| computed by the caller."""
    from monai.data.utils import compute_importance_map
    dev = next(seg.parameters()).device
    xp, pads = pad_to(img, (ROI,) * 3)
    tp, _ = pad_to(target, (ROI,) * 3)
    wins = [w for w in sw_windows(xp.shape[1:]) if tp[w].any()]
    if not wins:
        return img.copy(), dict(loss0=np.nan, loss=np.nan, linf=0.0, n_windows=0, z_clean=None)
    lo = [min(w[d].start for w in wins) for d in range(3)]
    hi = [max(w[d].stop for w in wins) for d in range(3)]
    R = tuple(slice(a, b) for a, b in zip(lo, hi))
    loc = [tuple(slice(w[d].start - lo[d], w[d].stop - lo[d]) for d in range(3)) for w in wins]
    x0 = torch.from_numpy(np.ascontiguousarray(xp[(slice(None),) + R])).to(dev)
    valid = ((x0[0] != 0) | (x0[1] != 0)).float()[None]                 # padding stays exactly zero
    tm = torch.from_numpy(np.ascontiguousarray(tp[R])).to(dev)
    imp = compute_importance_map((ROI,) * 3, mode="gaussian", sigma_scale=SIGMA, device=dev,
                                 dtype=torch.float32).reshape((ROI,) * 3)
    den = torch.zeros(tm.shape, device=dev)
    for w in loc:
        den[w] += imp
    for prm in seg.parameters():
        prm.requires_grad_(False)

    def three(xa):
        return torch.stack([xa[0], xa[1], xa[1] - xa[0]])

    def logits(x3, ws):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return seg(torch.stack([x3[(slice(None),) + w] for w in ws]))[:, 0].float()

    def lossf(z):
        return F.softplus(-z).mean() if mode == "decoy" else F.softplus(z).mean()   # minimised

    delta = torch.zeros_like(x0)
    loss0 = z_clean = None
    for step in range(steps + 1):
        with torch.no_grad():                                             # pass 1: blended logits
            x3 = three(x0 + delta * valid)
            num = torch.zeros(tm.shape, device=dev)
            for i in range(0, len(loc), batch):
                z = logits(x3, loc[i:i + batch])
                for j, w in enumerate(loc[i:i + batch]):
                    num[w] += imp * z[j]
            Z = num / den.clamp_min(1e-12)
        Zt = Z[tm].clone().requires_grad_(True)
        loss = lossf(Zt)
        if step == 0:
            loss0, z_clean = float(loss.detach()), Z.detach()
        if step == steps:
            break
        gZ = torch.zeros(tm.shape, device=dev)
        gZ[tm] = torch.autograd.grad(loss, Zt)[0]
        coef = gZ / den.clamp_min(1e-12)
        dvar = delta.clone().requires_grad_(True)                         # pass 2: exact gradient
        x3g = three(x0 + dvar * valid)
        for i in range(0, len(loc), batch):
            ws = loc[i:i + batch]
            z = logits(x3g, ws)
            sur = sum((z[j] * imp * coef[w]).sum() for j, w in enumerate(ws))
            sur.backward(retain_graph=i + batch < len(loc))
        with torch.no_grad():
            delta -= alpha * dvar.grad.sign()
            delta.clamp_(-eps, eps)
    advp = xp.copy()
    advp[(slice(None),) + R] = (x0 + delta * valid).cpu().numpy()
    adv = unpad(advp, pads)
    zc = np.full(tp.shape, np.nan, np.float32)
    zc[R] = z_clean.cpu().numpy()
    return adv, dict(loss0=loss0, loss=float(loss.detach()), linf=float(np.abs(adv - img).max()), n_windows=len(wins),
                     z_clean=unpad(zc, pads))


# ---------------------------------------------------------------------------- pipeline (as S3f + S3h)
def run_pipeline(seg, sam, box, img, gt, cid, amp):
    """Candidate table (S3f columns + radiomics + position), descriptors, finder probability, components."""
    import s3a_seg2sam as s3
    import s3d_learned_jury as s3d
    x, pads = pad_to(three_channel(img), (96, 96, 96))
    sprob = unpad(s3.seg_prob(seg, torch.from_numpy(x), amp, overlap=0.5), pads)
    lab, cands = s3.components(sprob > 0.5, top_k=5)
    if not cands:
        rows = s3d.case_columns([dict(cache_id=cid, k=-1, size_vox=0, inter_vox=0, gt_vox=int(gt.sum()),
                                      on_tumour=0)], lab, gt)
        df = pd.DataFrame(rows)
        for c in RAD_EXTRACTED:
            df[c] = np.nan
        return add_position(df), np.zeros((1, 768), np.float32), sprob, lab
    refined = s3.sam_refine(sam, box, img, lab, cands, amp, descriptors=True)
    by_k = {d["k"]: d["desc"] for d in refined}
    rows = s3d.case_columns(s3d.candidate_rows(cid, sprob, lab, cands, refined, gt), lab, gt)
    df = pd.DataFrame(rows)
    rad = pd.DataFrame(case_features(cid, img, lab, [r["k"] for r in rows]), columns=["cache_id", "k"] + RAD_EXTRACTED)
    df = add_position(df.merge(rad, on=["cache_id", "k"], how="left", validate="one_to_one"))
    D = np.stack([by_k[r["k"]] for r in rows]).astype(np.float16)      # S3f stores descriptors as float16
    return df, D, sprob, lab


def judge(bundles, df, D, lab, gt, target, cid):
    """Scores, kept candidates and outcome flags of every jury + the rel20 reference on one (attacked) scan."""
    real = df[df.k >= 0]
    hits = [int(k) for k in real.k if (lab == k)[target].any()]         # candidates touching the target region
    out, scores = {}, pd.DataFrame(dict(cache_id=real.cache_id, k=real.k.astype(int), size_vox=real.size_vox,
                                        on_target=real.k.isin(hits).to_numpy()))
    m = rel20(lab > 0)
    out["rel20"] = dict(dice=dice(m, gt), target_kept=bool(m[target].any()))
    for name, b in bundles.items():
        p = b["jury"].proba(df, D)
        keep = keep_candidates(df, p, b["threshold"]).get(cid, [])
        mask = np.isin(lab, keep) if keep else np.zeros(lab.shape, bool)
        out[name] = dict(dice=dice(mask, gt), target_kept=bool(set(keep) & set(hits)),
                         p_target=float(p[(df.k >= 0).to_numpy()][scores.on_target.to_numpy()].max())
                         if hits else np.nan)
        scores[f"p_{name}"] = p[(df.k >= 0).to_numpy()]
    return out, hits, scores


# ---------------------------------------------------------------------------- main
def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["mode", "eps255"])
    agg = g.agg(cases=("cache_id", "nunique"), finder_hit=("finder_hit", "mean"),
                rel20_kept=("rel20_target_kept", "mean"), jury13_kept=("jury13_target_kept", "mean"),
                proposed_kept=("jury13_rad_rel_target_kept", "mean"),
                dice_rel20=("rel20_dice", "mean"), dice_jury13=("jury13_dice", "mean"),
                dice_proposed=("jury13_rad_rel_dice", "mean"))
    return agg.round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--split", default="val", choices=["val"], help="pilot = development set only")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--turbo_ckpt", required=True)
    ap.add_argument("--bundles", required=True, help="s3h final/ folder with jury13/ and jury13_rad_rel/")
    ap.add_argument("--ref_scores", default="", help="s3h scores.csv: clean re-run parity check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cases", type=int, default=30)
    ap.add_argument("--eps", default="2,4,8", help="L-inf budgets in units of 1/255")
    ap.add_argument("--modes", default="decoy", help="decoy and/or hide (comma separated)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.1, help="PGD step = alpha * eps")
    ap.add_argument("--max_windows", type=int, default=27, help="hide mode: shrink large tumours to <= this many windows")
    ap.add_argument("--radius", type=float, default=5.0, help="decoy ball radius (mm = voxels)")
    ap.add_argument("--min_dist", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import s3a_seg2sam as s3
    t_start = time.time()
    out = Path(a.out).expanduser(); (out / "cases").mkdir(parents=True, exist_ok=True)
    amp = torch.cuda.is_available()
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    cases = require_cases(sp, root, a.split)
    rng = np.random.default_rng(a.seed)
    cases = [cases[i] for i in sorted(rng.choice(len(cases), min(a.n_cases, len(cases)), replace=False))]
    bdir = Path(a.bundles).expanduser()
    bundles = {n: load_bundle(bdir / n / f"{n}.pkl") for n in JURIES}
    seg, sam, box = s3.load_models(str(Path(a.seg_ckpt).expanduser()), str(Path(a.sam_ckpt).expanduser()),
                                   str(Path(a.turbo_ckpt).expanduser()))
    eps_list = [float(e) for e in a.eps.split(",")]
    modes = a.modes.split(",")
    cfg = dict(vars(a), cases=[c for c, _ in cases], thresholds={n: b["threshold"] for n, b in bundles.items()},
               versions=bundles["jury13_rad_rel"]["versions"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"S9 attack pilot | {len(cases)} {a.split} cases | eps {eps_list}/255 | modes {modes} | "
          f"steps {a.steps} | cut-offs {cfg['thresholds']} | device {s3.DEV}", flush=True)

    for i, (cid, path) in enumerate(cases):
        cf = out / "cases" / f"{cid}.json"
        if cf.exists():
            continue
        t0 = time.time()
        img, lbl = load_case(path); gt = lbl > 0
        crng = np.random.default_rng(seed_for(cid, a.seed))
        df0, D0, sprob0, lab0 = run_pipeline(seg, sam, box, img, gt, cid, amp)
        runs, cand_frames = [], []
        clean_scores = {n: b["jury"].proba(df0, D0)[(df0.k >= 0).to_numpy()].tolist() for n, b in bundles.items()}
        decoy_c = choose_decoy(img, sprob0, gt, crng, a.min_dist, 0.1, int(round(a.radius)))
        targets = {}
        if "decoy" in modes and decoy_c is not None:
            targets["decoy"] = (ball(gt.shape, decoy_c, a.radius), decoy_c)
        if "hide" in modes and gt.any():
            idx = np.argwhere(gt)
            targets["hide"] = (gt, tuple(int(v) for v in (idx.min(0) + idx.max(0)) // 2))
        for mode, (target, centre) in targets.items():
            target, frac = (restrict_target(target, centre, a.max_windows, gt.shape) if mode == "hide"
                            else (target, 1.0))
            check = np.nan
            for e in [0.0] + eps_list:
                if e == 0:
                    adv, info, (df, D, sprob, lab) = img, dict(loss0=np.nan, loss=np.nan, linf=0.0), (df0, D0, sprob0, lab0)
                else:
                    eps = e / 255.0
                    adv, info = pgd_sw(seg, img, target, mode, eps, a.steps, a.alpha * eps, amp)
                    zc = info.pop("z_clean")
                    if zc is not None and np.isnan(check):                # attack model == pipeline finder?
                        check = float(np.nanmax(np.abs(1 / (1 + np.exp(-zc[target])) - sprob0[target])))
                    df, D, sprob, lab = run_pipeline(seg, sam, box, adv, gt, cid, amp)
                res, hits, sc = judge(bundles, df, D, lab, gt, gt if mode == "hide" else target, cid)   # hide: whole tumour
                row = dict(cache_id=cid, mode=mode, eps255=e, finder_hit=bool(hits), n_cand=int((df.k >= 0).sum()),
                           target_prob=float(sprob[target].mean()), target_ml=float(target.sum()) / 1000,
                           target_frac=frac, sw_check=check, **info)
                for name, r in res.items():
                    row.update({f"{name}_{key}": v for key, v in r.items()})
                runs.append(row)
                cand_frames.append(sc.assign(mode=mode, eps255=e).to_dict(orient="records"))
        cf.write_text(json.dumps(dict(runs=runs, candidates=[c for f in cand_frames for c in f],
                                      clean_scores=clean_scores, decoy_centre=decoy_c), default=float))
        last = {r["mode"]: r for r in runs if r["eps255"] == max(eps_list)}
        msg = " | ".join(f"{m} eps{int(max(eps_list))}: finder {'hit' if r['finder_hit'] else 'miss'}, "
                         f"rel20 {int(r['rel20_target_kept'])} j13 {int(r['jury13_target_kept'])} "
                         f"prop {int(r['jury13_rad_rel_target_kept'])}, p_target {r['target_prob']:.2f}, check {r['sw_check']:.1e}"
                         for m, r in last.items())
        print(f"[{i + 1}/{len(cases)}] {cid} {time.time() - t0:.0f}s | {msg}", flush=True)

    allc = [json.loads(p.read_text()) for p in sorted((out / "cases").glob("*.json"))]
    runs = pd.DataFrame([r for c in allc for r in c["runs"]])
    cands = pd.DataFrame([r for c in allc for r in c["candidates"]])
    runs.to_csv(out / "runs.csv", index=False); cands.to_csv(out / "candidates.csv", index=False)
    agg = summarise(runs)
    parity = {}
    if a.ref_scores:
        ref = pd.read_csv(Path(a.ref_scores).expanduser())
        ref = ref[ref.split == a.split]
        for name in JURIES:
            clean = cands[(cands.eps255 == 0)].drop_duplicates(["cache_id", "k"])
            m = clean.merge(ref[ref.variant == name][["cache_id", "k", "p"]], on=["cache_id", "k"], how="inner")
            parity[name] = dict(matched=int(len(m)), of=int(len(clean)),
                                max_abs_diff=float(np.abs(m[f"p_{name}"] - m.p).max()) if len(m) else None)
        (out / "clean_parity.json").write_text(json.dumps(parity, indent=2))
    json.dump(dict(summary=agg.reset_index().to_dict(orient="records"), parity=parity,
                   cases=int(runs.cache_id.nunique())), open(out / "summary.json", "w"), indent=2, default=float)
    print("\nshare of cases where the attack target reaches the final mask (decoy: kept; hide: still found)")
    print("finder_hit = target region is a candidate after the attack (decoy: fooled; hide: NOT hidden)")
    print(agg.to_string())
    if parity:
        print(f"\nclean re-run vs locked scores: {parity}")
    print(f"total {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
