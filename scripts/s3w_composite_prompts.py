"""S3w: composite, SegResNet-guided prompts for SAM in the one-click review (idea adapted from RADAR,
arXiv 2610.14944: point + negative point + mask prompts instead of a single prompt).

Same patients, same simulated clicks and same out-of-fold SAM decoders as S3v (outputs/s3v_review_agent_oof);
only how SAM is prompted changes. No expert information is used to build prompts (the click itself is the
simulated clinician click, as in S3v). Test refused.

Prompt variants (all from one encoder pass on the S3v crop: 128^3 post1 at 1.5 mm centred on the click)
  S1  one positive point at the click (= S3v; parity check against S3v's S_dice)
  SP  points only: the click + up to 2 extra positives + up to 3 negatives
  SC  SP + a mask prompt (PRIMARY)
Built from SegResNet's (out-of-fold) probability map P and its candidate blobs, never from the expert mask:
  region R   26-connected component of P > 0.2 containing the click, else the nearest one within 10 mm
  positives  the click + up to 2 voxels of R with the highest P (P >= 0.3), >= 5 mm from the click and each other
  negatives  deepest point of each SegResNet candidate that does not touch R, within 60 mm of the click,
             nearest first, up to 3 (competing structures: vessels, other lesions)
  mask       f = P inside R, resampled to the crop, 4^3 block mean -> logits L (f - tau) / max(tau, 1 - tau),
             tau 0.25, L 1 (the S3p mask-prompt calibration)
SAM output: logit > 0, pasted back to 1 mm, the component containing the click (as S3u / S3v).

Frozen rules (decision_rule.json written before any SAM call)
  R5a (primary)  on the train OOF patients the S3v (OOF) stage-2 agent sent to SAM (deep click), mean
                 Dice(SC) - Dice(S1) has a patient-bootstrap 95% CI lower bound > 0
  R5b            re-running the S3v policy with SC as the SAM tool (--step policy of s3v on <out>/policy_SC):
                 R4 still passes and the 10%-budget gain over always-blob is not smaller than with S1
  reported       SP vs S1; SC vs S1 on all clicked patients and on val; sensitivity with the offset click.

Steps
  --step masks    GPU: S1 / SP / SC for every patient and both S3v clicks -> detail.csv, policy_SC/outcomes.csv
  --step analyse  CPU: R5a + summaries (run s3v --step policy --out <out>/policy_SC for R5b)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M  # noqa: E402
import s3u_click_biomarkers as U  # noqa: E402
import s3v_review_agent as V  # noqa: E402

CFG = dict(region_t=0.2, region_max_mm=10.0, pos_extra=2, pos_min_p=0.3, pos_min_sep_mm=5.0,
           neg_max=3, neg_within_mm=60.0, mask_tau=0.25, mask_L=1.0)
STRUCT = np.ones((3, 3, 3), bool)
VARIANTS = ["S1", "SP", "SC"]
RULE = dict(
    cohort="S3v patients and clicks (train OOF 1080 with out-of-fold SAM decoders, val 120); test refused",
    variants=dict(S1="one click (S3v)", SP="click + extra positives + negatives", SC="SP + mask prompt (PRIMARY)"),
    prompts_from="SegResNet out-of-fold probability map and candidate blobs; never the expert mask", config=CFG,
    R5a="train OOF patients sent to SAM by the S3v-OOF stage-2 agent: mean Dice(SC) - Dice(S1), bootstrap CI "
        "lower bound > 0",
    R5b="s3v policy re-run with SC as the SAM tool: R4 passes and 10%-budget gain >= the S1 run's gain",
    bootstrap=dict(n=5000, seed=0))


# --------------------------------------------------------------------------------------------- prompts
def region_at(prob, p, t=CFG["region_t"], max_mm=CFG["region_max_mm"]):
    """Component of prob > t containing p, else the nearest one within max_mm (voxels = mm); empty if none."""
    lab, n = ndimage.label(prob > t, structure=STRUCT)
    if n == 0:
        return np.zeros(prob.shape, bool)
    k = lab[tuple(p)]
    if k == 0:
        idx = np.argwhere(lab > 0)
        d = np.sqrt(((idx - np.asarray(p)) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] > max_mm:
            return np.zeros(prob.shape, bool)
        k = lab[tuple(idx[j])]
    return lab == k


def extra_positives(prob, region, p, n=CFG["pos_extra"], min_p=CFG["pos_min_p"], sep=CFG["pos_min_sep_mm"]):
    """Up to n voxels of the region with the highest probability, >= sep mm from the click and each other."""
    idx = np.argwhere(region & (prob >= min_p))
    if not len(idx) or n <= 0:
        return []
    order = np.argsort(-prob[tuple(idx.T)], kind="stable")[:5000]
    chosen = [np.asarray(p)]
    out = []
    for q in idx[order]:
        if all(np.sqrt(((q - c) ** 2).sum()) >= sep for c in chosen):
            out.append(tuple(int(v) for v in q))
            chosen.append(q)
            if len(out) == n:
                break
    return out


def deepest_point(blob, sl):
    """M.geometric_point(blob) computed on the blob's bounding box only. Identical result: geometric_point pads
    with background (as the full-volume call does at the volume edge), the voxels around a tight bounding box
    are background anyway, and the centroid tie-break / lexicographic order are translation invariant."""
    q = M.geometric_point(blob[sl])
    return tuple(int(a + s.start) for a, s in zip(q, sl))


def negatives(lab, region, p, n=CFG["neg_max"], within=CFG["neg_within_mm"]):
    """Deepest point of each candidate blob not touching the region, within `within` mm, nearest first."""
    pts = []
    boxes = ndimage.find_objects(lab)
    for k in np.unique(lab[lab > 0]):
        sl = boxes[k - 1]
        blob = np.zeros(lab.shape, bool)
        blob[sl] = lab[sl] == k
        if (blob[sl] & region[sl]).any():
            continue
        q = deepest_point(blob, sl)
        d = float(np.sqrt(((np.asarray(q) - np.asarray(p)) ** 2).sum()))
        if d <= within:
            pts.append((d, tuple(int(v) for v in q)))
    return [q for _, q in sorted(pts)[:n]]


def build_prompts(prob, lab, p):
    region = region_at(prob, p)
    return dict(region=region, pos=[tuple(int(v) for v in p)] + extra_positives(prob, region, p),
                neg=negatives(lab, region, p))


# --------------------------------------------------------------------------------------------- SAM
def sam_variants(img, prob, p, pr, rn):
    """S1 / SP / SC masks (1 mm) from one encoder pass; pr = build_prompts output; rn = SAM runner."""
    import torch
    import torch.nn.functional as F
    import s3a_seg2sam as s3
    from cids.models.sam_lora import decode, upsample
    img15, _ = s3.resample(img, np.zeros(img.shape[1:], np.uint8))
    shape15 = img15.shape[1:]
    to15 = lambda q: np.clip(M.to_crop(q, np.zeros(3, int)), 0, np.asarray(shape15) - 1)  # noqa: E731
    c15 = to15(p)
    post, fg = img15[1], (img15[1] > 0).astype(np.uint8)
    vc, origin = s3.crop_pad(post, c15)
    fc, _ = s3.crop_pad(fg, c15)
    crop = s3.znorm(vc, fc)

    def in_crop(q):
        c = to15(q) - origin
        return tuple(int(v) for v in c) if ((c >= 0) & (c < s3.SAM_SIZE)).all() else None
    pos = [c for c in (in_crop(q) for q in pr["pos"]) if c is not None]
    neg = [c for c in (in_crop(q) for q in pr["neg"]) if c is not None]
    f = torch.from_numpy((prob * pr["region"]).astype(np.float32))[None, None]
    f15 = F.interpolate(f, size=tuple(shape15), mode="trilinear", align_corners=False)[0, 0].numpy()
    fcrop, _ = s3.crop_pad(f15, c15)
    fb = F.avg_pool3d(torch.from_numpy(fcrop.astype(np.float32))[None, None], 4)
    tau, L = CFG["mask_tau"], CFG["mask_L"]
    low_mask = (L * (fb - tau) / max(tau, 1 - tau)).to(rn.dev)
    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=rn.amp):
        x = torch.from_numpy(np.ascontiguousarray(crop[None, None])).float().to(rn.dev)
        emb = rn.sam.image_encoder(x).float()
        for v in VARIANTS:
            pts = pos[:1] if v == "S1" else pos + neg
            lbl = [1] if v == "S1" else [1] * len(pos) + [0] * len(neg)
            co = torch.tensor(np.asarray(pts, np.float32)[None], device=rn.dev)
            lb = torch.tensor([lbl], dtype=torch.long, device=rn.dev)
            low = decode(rn.sam, rn.box, emb, co, lb, None, low_mask if v == "SC" else None)
            pr_ = torch.sigmoid(upsample(low.float()))[0, 0].cpu().numpy()
            up = M.paste_back(pr_, origin, shape15, img.shape[1:])
            out[v] = U.component_at(up > 0.5, p)
    return out, len(pos), len(neg)


# --------------------------------------------------------------------------------------------- steps
def step_masks(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any SAM call
    s3v = Path(a.s3v_out).expanduser()
    oc = pd.read_csv(s3v / "outcomes.csv")
    if "sam_decoder" not in oc or not oc[oc.split == "train_oof"].sam_decoder.astype(str).str.startswith("fold").all():
        raise SystemExit("--s3v_out must be the OUT-OF-FOLD S3v run (outputs/s3v_review_agent_oof)")
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    paths, cdirs = {}, {}
    for split, cache_split, cdir in (("train_oof", "train", a.train_cases), ("val", "val", a.val_cases)):
        d = Path(cdir).expanduser()
        d = d if any(d.glob("*.npz")) else next(d.rglob("*.npz")).parent
        for cid, path in require_cases(sp, root, cache_split):
            paths[(split, cid)], cdirs[(split, cid)] = path, d
    if a.random_weights:
        import torch
        torch.manual_seed(0)
    runner = M.load_runner(a.sam_ckpt, a.turbo_ckpt, a.random_weights)
    fold_of = dict(pd.read_csv(Path(a.folds).expanduser()).values)
    fold_runner = {k: M.load_runner(str(Path(a.sam_fold_ckpt.format(fold=k)).expanduser()), a.turbo_ckpt,
                                    a.random_weights) for k in sorted(set(fold_of.values()))}
    pats = oc[["split", "cache_id"]].drop_duplicates().values.tolist()
    if a.limit:
        pats = [x for s in ("train_oof", "val") for x in [q for q in pats if q[0] == s][:a.limit]]
    part = out / "detail.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(r["split"], r["cache_id"]) for r in rows}
    todo = [q for q in pats if tuple(q) not in done]
    print(f"S3w | {len(pats)} patients ({len(pats) - len(todo)} already done) | variants {VARIANTS}", flush=True)
    t0 = time.time()
    for i, (split, cid) in enumerate(todo):
        img, gt = load_case(paths[(split, cid)])
        gt = gt > 0
        z = np.load(cdirs[(split, cid)] / f"{cid}.npz")
        prob, lab = z["seg"].astype(np.float32) / 255.0, z["lab"]
        rn = fold_runner[fold_of[cid]] if split == "train_oof" else runner
        for _, r in oc[(oc.split == split) & (oc.cache_id == cid)].iterrows():
            p = tuple(int(v) for v in json.loads(r.click_xyz))
            pr = build_prompts(prob, lab, p)
            masks, n_pos, n_neg = sam_variants(img, prob, p, pr, rn)
            Cn, _ = U.nearest_blob(lab, p)
            row = dict(split=split, cache_id=cid, click=r.click, n_pos=n_pos, n_neg=n_neg,
                       region_ml=float(pr["region"].sum()) / 1000, S3v_S_dice=r.S_dice)
            for v, m in masks.items():
                row[f"{v}_dice"] = M.dice_count(m, gt)
                for q, val in V.measurements(img, m).items():
                    row[f"{v}_{q}"] = val
                row[f"{v}_log_ml"] = float(np.log(m.sum() / 1000 + 1e-3))
                row[f"{v}_dice_cn"] = M.dice_count(m, Cn)
            rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  {i + 1}/{len(todo)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    det = pd.DataFrame(rows)
    det.to_csv(out / "detail.csv", index=False)
    if part.exists():
        part.unlink()
    diff = (det.S1_dice - det.S3v_S_dice).abs()
    print(f"parity S1 vs S3v S_dice: mean |diff| {diff.mean():.5f}, max {diff.max():.4f}, "
          f"> 0.01 in {int((diff > 0.01).sum())} of {len(det)}", flush=True)
    if diff.mean() > 0.005:
        raise SystemExit("PARITY FAILED: S1 does not reproduce S3v's one-click SAM")
    pol = out / "policy_SC"
    pol.mkdir(exist_ok=True)
    m = oc.merge(det[["split", "cache_id", "click", "SC_dice", "SC_ml", "SC_diam", "SC_enh", "SC_log_ml", "SC_dice_cn"]],
                 on=["split", "cache_id", "click"], how="inner", validate="one_to_one")
    m["S_dice"], m["S_ml"], m["S_diam"], m["S_enh"] = m.SC_dice, m.SC_ml, m.SC_diam, m.SC_enh
    m["s_log_ml"], m["dice_s_cn"] = m.SC_log_ml, m.SC_dice_cn
    m.drop(columns=[c for c in m.columns if c.startswith("SC_")]).to_csv(pol / "outcomes.csv", index=False)
    print(det.groupby(["split", "click"])[[f"{v}_dice" for v in VARIANTS]].mean().round(4).to_string())
    print(f"MASKS DONE -> --step analyse; then s3v --step policy --out {pol}")


def boot(x, n, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def step_analyse(a, out):
    det = pd.read_csv(out / "detail.csv")
    s3v = Path(a.s3v_out).expanduser()
    n, seed = RULE["bootstrap"]["n"], RULE["bootstrap"]["seed"]
    res, rows = {}, []
    for split in ("train_oof", "val"):
        ch = pd.read_csv(s3v / f"choices_{split}_deep.csv")[["cache_id", "tool2", "qc", "Cn_dice", "B_dice"]]
        d = det[(det.split == split) & (det.click == "deep")].merge(ch, on="cache_id", how="left")
        sel = V.select(d.qc.to_numpy(), d.cache_id.to_numpy(), 0.10)
        for name, g in (("sent_to_SAM", d[d.tool2 == "S"]), ("QC_10pct_clicked", d[sel]), ("all", d)):
            r = dict(split=split, population=name, n=len(g))
            for v in VARIANTS:
                r[f"{v}_dice"] = float(g[f"{v}_dice"].mean()) if len(g) else np.nan
            r["Cn_dice"] = float(g.Cn_dice.mean()) if len(g) else np.nan
            for v in ("SP", "SC"):
                dd = g[f"{v}_dice"] - g.S1_dice
                r[f"{v}_minus_S1"] = float(dd.mean()) if len(g) else np.nan
                lo, hi = boot(dd, n, seed) if len(g) > 1 else (np.nan, np.nan)
                r[f"{v}_minus_S1_ci95"] = [lo, hi]
                r[f"{v}_better_worse"] = [int((dd > 1e-9).sum()), int((dd < -1e-9).sum())]
            rows.append(r)
    t = pd.DataFrame(rows)
    p = t[(t.split == "train_oof") & (t.population == "sent_to_SAM")].iloc[0]
    res["R5a"] = dict(n=int(p.n), dice_S1=p.S1_dice, dice_SC=p.SC_dice, diff=p.SC_minus_S1,
                      ci95=p.SC_minus_S1_ci95, passes=bool(p.n > 1 and p.SC_minus_S1_ci95[0] > 0))
    off = det[det.click == "offset"].groupby("split")[[f"{v}_dice" for v in VARIANTS]].mean().round(4)
    t.to_csv(out / "summary.csv", index=False)
    (out / "rules.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(t[["split", "population", "n", "S1_dice", "SP_dice", "SC_dice", "Cn_dice", "SC_minus_S1",
             "SC_minus_S1_ci95", "SC_better_worse"]].round(4).to_string(index=False))
    print("\noffset click (3-5 mm), all patients:\n" + off.to_string())
    print("\nR5a:", json.dumps(res["R5a"], indent=1, default=float))
    print(f"\nnext (R5b): python scripts/s3v_review_agent.py --step policy --s3k_out <s3k> --s3n_out <s3n> "
          f"--out {out / 'policy_SC'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["masks", "analyse"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3v_out", required=True, help="the OUT-OF-FOLD S3v run (outcomes.csv, choices_*.csv)")
    ap.add_argument("--train_cases", default="", help="S3g OOF maps (seg, lab)")
    ap.add_argument("--val_cases", default="", help="S3f val maps (seg, lab)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--sam_fold_ckpt", default="", help=".../s2b_folds/f{fold}/best.pt")
    ap.add_argument("--folds", default="", help="S3k folds.csv")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.train_cases, a.val_cases, a.s3v_out):
        if "test" in Path(f).name.lower():
            raise SystemExit("S3w is a development study: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "masks":
        if not (a.train_cases and a.val_cases and a.folds and a.sam_fold_ckpt):
            raise SystemExit("--train_cases, --val_cases, --folds and --sam_fold_ckpt are required")
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt and --turbo_ckpt are required")
        step_masks(a, out)
    else:
        if not (out / "detail.csv").exists():
            raise SystemExit("run --step masks first")
        step_analyse(a, out)


if __name__ == "__main__":
    main()
