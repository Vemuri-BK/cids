"""S3m: contour study -- does SAM-Med3D-turbo (S2b) draw a better boundary for the blobs B0_rad already selected?

Development experiment (train OOF + val; the test split is refused). Selection is FROZEN: every arm receives
exactly the candidate IDs that B0_rad kept in S3k (per_patient.csv, column B0_rad_subset). The patient mask of
an arm is the union of that arm's contours over the kept candidates; patients with wrong selections, empty
selections and no candidates stay in the evaluation.

Arms (per kept candidate k)
  A  original SegResNet component (lab == k)                                        = B0_rad mask (S3k)
  B  SAM, box only                     decode(box, no point, empty mask prompt)    = S2b step 0 / S3 pipeline
  C  SAM, box + 1 positive point at the deepest interior voxel (geometric placement), previous logits = B's
  D  SAM, box + 1 positive point at the enhancement hotspot (enhancement placement), previous logits = B's
  E  region growing from the SAME seed as D on the SAME enhancement map in the SAME allowed region
  C and D follow the S2b training sequence (box-only first pass, then box + corrective point with the previous
  low-res logits as the dense prompt); one automatic positive point each. The point placements are new.

Frozen details
  enhancement map   E = post1 - pre of the 2-channel cache at 1 mm (the finder's 3rd channel), Gaussian
                    sigma 2 mm. APPROXIMATE enhancement contrast: the cache scales each phase separately.
  allowed region R  candidate bounding box (1 mm) dilated by 10 mm, clipped to the volume; the SAM output (crop
                    pasted back) and region growing are both restricted to R.
  geometric point   argmax of the Euclidean distance to the outside of the blob (1 mm grid); ties -> nearest to
                    the blob centroid, then lexicographic.
  enhancement point argmax of smoothed E over the blob interior (distance to outside >= 2 mm), same tie-break;
                    if the interior is empty (blob thinner than ~4 mm) the geometric point is used (counted).
  1 mm -> SAM grid  j = floor((i + 0.5) / 1.5) (1.5 mm voxel containing the 1 mm voxel centre); crop = j - origin.
  SAM contour       sigmoid at 1.5 mm, pasted into the 1.5 mm volume, trilinear to 1 mm (as s3a.sam_refine),
                    > 0.5, & R, largest 26-connected component overlapping the candidate's bounding box.
  region growing    seed value s = mean smoothed E in a 3 mm ball around the seed; voxels with smoothed E >= tau*s
                    inside R, 26-connected to the seed, 1-voxel closing, hole filling, & R. tau = 0.5 FROZEN
                    in advance (no tuning). Weak seed (s <= 0): no region -> empty raw output (counted).
  fallback          raw contour < 0.1 ml -> the candidate's A contour (the same rule for B-E; A's components
                    are >= 0.1 ml by construction). Both RAW and FALLBACK-ASSISTED results are reported;
                    the decision uses fallback-assisted (the deployable pipeline).
Metrics (1 mm, every patient): Dice (counts), NSD at 2 mm (primary), HD95 (NaN for an empty-vs-nonempty pair,
  reported with the number of jointly non-empty pairs), predicted volume, empty outputs, fallbacks, runtime.

Frozen decision rule (train OOF, fallback-assisted, written to decision_rule.json before any SAM call)
  Primary D vs A: mean dNSD >= +0.010 AND paired patient-bootstrap 95% CI lower bound > 0 AND mean dDice >= -0.005
    AND #patients Dice<0.5 (D) <= floor(1.10 * #(A)) AND #patients losing >= 0.20 Dice <= floor(0.01 * N).
  Secondary, fixed order, only if the primary passes: D vs C (dNSD CI lower bound > 0), then D vs E (same).
  Anything else (B or C vs A, raw variants) is exploratory.
Evidence labels: train 1,080 = SegResNet out-of-fold, but the S2b decoder was TRAINED on them (feasibility);
  val 120 = used for S2b checkpoint selection (development). A failed development run stops GPU spending on this
  implementation; a passed run earns a locked evaluation on patients excluded from task-specific fitting and
  checkpoint selection (the 306-patient benchmark has been looked at by earlier pipeline versions).

Steps
  --step parity   CPU: rebuild A from B0_rad_subset + component maps and match every patient's saved S3k Dice
                  (|diff| <= 1e-6); writes parity.csv (also the smoke-selection attributes). Stops on mismatch.
  --step arms     GPU: all arms. --smoke 20 = technical run on 20 chosen patients (empty selection, several kept
                  components, tiny components, blobs near the image edge, val, random) + overlays.
Example (server)
  python scripts/s3m_contour_study.py --step parity --cache_root ~/data --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --train_cases ~/CIDS/outputs/s3g_oof_masks/cases --val_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases \\
      --out ~/CIDS/outputs/s3m_contour
  python scripts/s3m_contour_study.py --step arms --smoke 20 ... --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt \\
      --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth --out ~/CIDS/outputs/s3m_contour_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import hd95, nsd  # noqa: E402
from cids.provenance import require_cases  # noqa: E402

STRUCT = np.ones((3, 3, 3), bool)
ARMS = ("A", "B", "C", "D", "E")
CFG = dict(sigma_mm=2.0, interior_mm=2.0, margin_mm=10, tau=0.5, seed_ball_mm=3, reject_ml=0.1, nsd_tol_mm=2.0,
           sam_spacing=1.5, sam_size=128)
RULE = dict(
    primary="D vs A (fallback-assisted), train OOF",
    min_mean_dnsd=0.010, ci_lo_dnsd_gt=0.0, min_mean_ddice=-0.005, lt05_factor=1.10, loss020_frac=0.01,
    secondary_fixed_order=["D vs C: dNSD CI lower bound > 0", "D vs E: dNSD CI lower bound > 0"],
    secondary_only_if_primary=True, bootstrap=5000, seed=0, endpoint="NSD at 2 mm", variant="fallback-assisted",
    exploratory=["B vs A", "C vs A", "E vs A", "raw variants"], config=CFG,
    evidence="train = SegResNet OOF but S2b-trained (feasibility); val = S2b checkpoint selection (development)")


# --------------------------------------------------------------------------------------------- geometry
def kept_ids(s) -> list:
    if s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() in ("", "nan"):
        return []
    return [int(float(x)) for x in str(s).split("+")]


def dice_count(m, g) -> float:
    s = int(m.sum()) + int(g.sum())
    return 0.0 if s == 0 else 2.0 * int((m & g).sum()) / s


def bbox(mask):
    idx = np.argwhere(mask)
    return idx.min(0), idx.max(0)


def _tie_break(cands, comp):
    """cands (N,3) -> the one nearest to the blob centroid, then lexicographic."""
    c = np.argwhere(comp).mean(0)
    d = ((cands - c) ** 2).sum(1)
    best = cands[d == d.min()]
    return tuple(int(v) for v in best[np.lexsort(best.T[::-1])][0])


def inside_distance(comp):
    """Euclidean distance (mm, 1 mm grid) of every blob voxel to the nearest voxel outside the blob."""
    return ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1, 1:-1]


def geometric_point(comp, dt=None):
    dt = inside_distance(comp) if dt is None else dt
    return _tie_break(np.argwhere(dt == dt.max()), comp)


def enhancement_point(comp, es, interior_mm=2.0, dt=None):
    """(point, used_geometric_fallback)."""
    dt = inside_distance(comp) if dt is None else dt
    inner = dt >= interior_mm
    if not inner.any():
        return geometric_point(comp, dt), True
    v = np.where(inner, es, -np.inf)
    return _tie_break(np.argwhere(v == v.max()), comp), False


def to_crop(p1, origin, sp=1.5):
    j = np.floor((np.asarray(p1, float) + 0.5) / sp).astype(int)
    return j - np.asarray(origin, int)


def crop_to_1mm(pc, origin, sp=1.5):
    return (np.asarray(pc, float) + np.asarray(origin, float) + 0.5) * sp - 0.5


def allowed_region(shape, lo, hi, margin):
    r = np.zeros(shape, bool)
    a = np.maximum(np.asarray(lo) - margin, 0)
    b = np.minimum(np.asarray(hi) + margin + 1, shape)
    r[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
    return r


def keep_box_component(mask, lo, hi):
    """Largest 26-connected component of mask that overlaps the box [lo, hi] (inclusive); empty if none."""
    lab = np.zeros(mask.shape, np.int32)
    ndimage.label(mask, structure=STRUCT, output=lab)
    sub = lab[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    ids = np.unique(sub[sub > 0])
    if not len(ids):
        return np.zeros(mask.shape, bool)
    size = np.bincount(lab.ravel())
    return lab == ids[np.argmax(size[ids])]


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r ** 2


def region_grow(es, seed, region, tau=0.5, ball_mm=3):
    """(mask, seed_value, weak). Seed value = mean smoothed E in a ball around the seed."""
    lo = np.maximum(np.asarray(seed) - ball_mm, 0); hi = np.minimum(np.asarray(seed) + ball_mm + 1, es.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    b = ball(es[sl].shape, np.asarray(seed) - lo, ball_mm)
    s = float(es[sl][b].mean())
    if not s > 0:
        return np.zeros(es.shape, bool), s, True
    cand = (es >= tau * s) & region
    cand[tuple(seed)] = True
    lab = np.zeros(es.shape, np.int32)
    ndimage.label(cand, structure=STRUCT, output=lab)
    m = lab == lab[tuple(seed)]
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=STRUCT)) & region
    return m, s, False


def paste_back(p_crop, origin, shape15, shape1):
    """SAM crop probability (128^3 at 1.5 mm) -> 1 mm volume, exactly as s3a.sam_refine."""
    import torch
    import torch.nn.functional as F
    vol = torch.zeros(tuple(int(s) for s in shape15))
    o = np.asarray(origin); n = p_crop.shape[0]
    lo_ = np.maximum(o, 0); hi_ = np.minimum(o + n, shape15)
    a, b = lo_ - o, hi_ - o
    vol[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]] = torch.as_tensor(p_crop)[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    return F.interpolate(vol[None, None], size=tuple(shape1), mode="trilinear", align_corners=False)[0, 0].numpy()


# --------------------------------------------------------------------------------------------- SAM
class SamRunner:
    """B (box), C and D (box + 1 positive point, B's low-res logits as the dense prompt) for one crop."""

    def __init__(self, sam, box_mod, dev, amp):
        self.sam, self.box, self.dev, self.amp = sam, box_mod, dev, amp

    def __call__(self, crop, box, pts):
        import torch
        from cids.models.sam_lora import decode, upsample
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            x = torch.from_numpy(np.ascontiguousarray(crop[None, None])).float().to(self.dev)   # (1,1,128^3)
            emb = self.sam.image_encoder(x).float()
            bx = torch.from_numpy(np.asarray(box)[None]).float().to(self.dev)
            low_b = decode(self.sam, self.box, emb, None, None, bx, None)
            n = len(pts)
            coords = torch.from_numpy(np.asarray(pts, np.float32)[:, None]).to(self.dev)     # (n,1,3)
            labels = torch.ones(n, 1, dtype=torch.long, device=self.dev)
            low_p = decode(self.sam, self.box, emb.expand(n, -1, -1, -1, -1), coords, labels,
                           bx.expand(n, -1, -1), low_b.expand(n, -1, -1, -1, -1))
            prob = torch.sigmoid(upsample(torch.cat([low_b, low_p]).float()))[:, 0].cpu().numpy()
        return prob                                                     # (1 + n, 128, 128, 128)


def load_runner(sam_ckpt, turbo, random_weights):
    import torch
    import s3a_seg2sam as s3
    sam = s3.load_sam_med3d(None if random_weights else turbo, s3.DEV)
    s3.add_lora(sam.mask_decoder, ("q_proj", "v_proj"), 8, 16.0)
    box = s3.BoxCorners(sam.prompt_encoder)
    if not random_weights:
        st = torch.load(sam_ckpt, map_location="cpu", weights_only=False)
        sam.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    sam.to(s3.DEV).eval(); box.to(s3.DEV).eval()
    return SamRunner(sam, box, s3.DEV, s3.DEV == "cuda")


# --------------------------------------------------------------------------------------------- one patient
def surface_metrics(m, g):
    return dict(dice=dice_count(m, g), nsd=nsd(m, g, CFG["nsd_tol_mm"]), hd95=hd95(m, g),
                ml=float(m.sum()) / 1000, empty=bool(not m.any()))


def candidate_contours(img, lab, k, crop, box15, origin, shape15, es, runner):
    """Raw contours (1 mm bool) of B, C, D, E for one kept candidate + a log row."""
    comp = lab == k
    lo, hi = bbox(comp)
    R = allowed_region(lab.shape, lo, hi, CFG["margin_mm"])
    dt = inside_distance(comp)
    pc = geometric_point(comp, dt)
    pd_, d_fb = enhancement_point(comp, es, CFG["interior_mm"], dt)
    cc, cd = to_crop(pc, origin), to_crop(pd_, origin)
    n = CFG["sam_size"]
    cc_c, cd_c = np.clip(cc, 0, n - 1), np.clip(cd, 0, n - 1)
    t0 = time.time()
    prob = runner(crop, box15, [cc_c, cd_c])
    t_sam = time.time() - t0
    out = {}
    for arm, p in zip("BCD", prob):
        up = paste_back(p, origin, shape15, lab.shape)
        out[arm] = keep_box_component((up > 0.5) & R, lo, hi)
    t0 = time.time()
    out["E"], seed_val, weak = region_grow(es, pd_, R, CFG["tau"], CFG["seed_ball_mm"])
    t_e = time.time() - t0
    edge = int(min(min(lo), min(np.asarray(lab.shape) - 1 - hi)))
    row = dict(k=k, comp_vox=int(comp.sum()), lo=lo.tolist(), hi=hi.tolist(), origin15=np.asarray(origin).tolist(),
               box15=np.asarray(box15).tolist(), edge_mm=edge,
               crop_outside_volume=bool((np.asarray(origin) < 0).any() or (np.asarray(origin) + n > shape15).any()),
               pC=list(pc), pD=list(pd_), pC_crop=cc.tolist(), pD_crop=cd.tolist(),
               point_clipped=bool((cc != cc_c).any() or (cd != cd_c).any()),
               same_sam_voxel=bool((cc == cd).all()), cd_dist_mm=float(np.linalg.norm(np.subtract(pc, pd_))),
               d_geometric_fallback=d_fb, seed_value=seed_val, e_weak_seed=weak,
               dt_max_mm=float(dt.max()), t_sam_s=t_sam, t_e_s=t_e)
    for arm in "BCDE":
        row[f"{arm}_raw_ml"] = float(out[arm].sum()) / 1000
        row[f"{arm}_fallback"] = bool(out[arm].sum() < CFG["reject_ml"] * 1000)
    return out, comp, R, row


def patient_run(cid, split, img, gt, lab, ks, runner, overlay_dir=None, return_masks=False):
    import s3a_seg2sam as s3
    t0 = time.time()
    masks = {f"{a}_{v}": np.zeros(gt.shape, bool) for a in ARMS for v in ("raw", "fb")}
    rows = []
    if ks:
        es = ndimage.gaussian_filter(img[1] - img[0], CFG["sigma_mm"])
        sizes = [(k, float((lab == k).sum())) for k in ks]
        cc = s3.candidate_crops(img, lab, sizes)
        for j, k in enumerate(ks):
            out, comp, R, row = candidate_contours(img, lab, k, cc["crops"][j][0], cc["boxes"][j], cc["origins"][j],
                                                   cc["shape15"], es, runner)
            row.update(cache_id=cid, split=split)
            rows.append(row)
            for v in ("raw", "fb"):
                masks[f"A_{v}"] |= comp
            for arm in "BCDE":
                masks[f"{arm}_raw"] |= out[arm]
                masks[f"{arm}_fb"] |= comp if row[f"{arm}_fallback"] else out[arm]
            if overlay_dir is not None:
                overlay(overlay_dir / f"{cid}_k{k}.png", img, gt, comp, out, row, R)
    res = dict(cache_id=cid, split=split, n_kept=len(ks))
    done = {}
    for name, m in masks.items():
        a = name.split("_")[0]
        twin = "A_raw" if name == "A_fb" else (f"{a}_raw" if name.endswith("_fb") else None)
        if twin and twin in done and np.array_equal(m, masks[twin]):
            met = done[twin]
        else:
            met = surface_metrics(m, gt)
        done[name] = met
        for q, v in met.items():
            res[f"{name}_{q}"] = v
    res["seconds"] = time.time() - t0
    return (res, rows, masks) if return_masks else (res, rows)


# --------------------------------------------------------------------------------------------- overlap (for Q1)
def box_sam_mask(img, lab, ks, runner):
    """Patient-level box-only SAM mask B (raw, before fallback), exactly the arms-step B; and SAM seconds."""
    import s3a_seg2sam as s3
    b = np.zeros(lab.shape, bool)
    if not ks:
        return b, 0.0
    cc = s3.candidate_crops(img, lab, [(k, float((lab == k).sum())) for k in ks])
    t = 0.0
    for j, k in enumerate(ks):
        comp = lab == k
        lo, hi = bbox(comp)
        R = allowed_region(lab.shape, lo, hi, CFG["margin_mm"])
        c = np.clip(to_crop(geometric_point(comp), cc["origins"][j]), 0, CFG["sam_size"] - 1)
        t0 = time.time()
        p = runner(cc["crops"][j][0], cc["boxes"][j], [c])[0]          # [0] = box-only logits (B)
        t += time.time() - t0
        up = paste_back(p, cc["origins"][j], cc["shape15"], lab.shape)
        b |= keep_box_component((up > 0.5) & R, lo, hi)
    return b, t


def step_overlap(a, root, sp, pp, splits, out):
    """Exact patient-level prediction-to-prediction overlap Dice(A, B_raw) at 1 mm (no expert mask used for it)."""
    arms_pp = Path(a.arms_dir).expanduser() / "per_patient.csv" if a.arms_dir else None
    ref = pd.read_csv(arms_pp).set_index("cache_id") if arms_pp and arms_pp.exists() else None
    jobs = jobs_for(a, root, sp, pp, splits)
    jobs = jobs[:a.limit] if a.limit else jobs
    runner = load_runner(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights)
    print(f"S3m overlap | {len(jobs)} patients | reference arms run: {arms_pp if ref is not None else 'none'}", flush=True)
    rows, t0 = [], time.time()
    for i, (cid, split, cpath, lpath, subset, saved) in enumerate(jobs):
        img, lbl = load_case(cpath)
        gt = lbl > 0
        lab = np.load(lpath)["lab"]
        ks = kept_ids(subset)
        A = np.isin(lab, ks) if ks else np.zeros(gt.shape, bool)
        B, ts = box_sam_mask(img, lab, ks, runner)
        rows.append(dict(cache_id=cid, split=split, n_kept=len(ks), a_vox=int(A.sum()), b_vox=int(B.sum()),
                         inter_ab=int((A & B).sum()), dice_ab=dice_count(A, B) if (A.any() or B.any()) else np.nan,
                         a_dice_gt=dice_count(A, gt), b_raw_dice_gt=dice_count(B, gt), sam_seconds=ts))
        if abs(rows[-1]["a_dice_gt"] - saved) > 1e-6:
            raise SystemExit(f"{cid}: A Dice differs from S3k")
        if (i + 1) % 100 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    if ref is not None:
        d = (df.set_index("cache_id").b_raw_dice_gt - ref.reindex(df.cache_id).B_raw_dice.to_numpy()).abs()
        df["b_parity_absdiff"] = d.to_numpy()
        print(f"B parity vs arms run: max |dDice| {d.max():.2e}, patients > 1e-3: {int((d > 1e-3).sum())}")
    df.to_csv(out / "overlap.csv", index=False)
    print(df.groupby("split")[["dice_ab", "b_raw_dice_gt", "sam_seconds"]].mean().round(4).to_string())
    print(f"-> {out / 'overlap.csv'}")


# --------------------------------------------------------------------------------------------- shift check
def shift_profile(m, gt, max_shift=3):
    """Dice of m shifted by every integer (dx,dy,dz) in [-max_shift, max_shift]^3 against gt.
    Returns (best_shift, dice_at_0, best_dice). A correctly aligned contour peaks at (0,0,0) on average."""
    if not m.any() or not gt.any():
        return (0, 0, 0), dice_count(m, gt), dice_count(m, gt)
    idx = np.argwhere(m | gt)
    lo = np.maximum(idx.min(0) - max_shift - 1, 0); hi = np.minimum(idx.max(0) + max_shift + 2, m.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    mm = np.pad(m[sl], max_shift); gg = gt[sl]
    best, d0, bd = (0, 0, 0), None, -1.0
    sm, sg = int(m.sum()), int(gt.sum())
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            for dz in range(-max_shift, max_shift + 1):
                w = mm[max_shift - dx:max_shift - dx + gg.shape[0], max_shift - dy:max_shift - dy + gg.shape[1],
                       max_shift - dz:max_shift - dz + gg.shape[2]]
                d = 2.0 * int((w & gg).sum()) / (sm + sg)
                if (dx, dy, dz) == (0, 0, 0):
                    d0 = d
                if d > bd + 1e-12 or (abs(d - bd) <= 1e-12 and abs(dx) + abs(dy) + abs(dz) < sum(map(abs, best))):
                    best, bd = (dx, dy, dz), d             # ties -> the smallest shift (zero first)
    return best, d0, bd


def step_shiftcheck(a, root, sp, pp, splits, out):
    """Geometry diagnostic only (no decision): are the SAM contours aligned with the image as well as A is?"""
    par = pd.read_csv(Path(a.parity_csv).expanduser() if a.parity_csv else out / "parity.csv")
    pool = par[(par.n_kept == 1) & (par.dice >= 0.6) & par.split.isin(splits)]
    pick = set(pool.sample(min(a.shift_n, len(pool)), random_state=a.seed).cache_id)
    jobs = [j for j in jobs_for(a, root, sp, pp, splits) if j[0] in pick]
    runner = load_runner(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights)
    print(f"S3m shift check | {len(jobs)} patients (one kept blob, A Dice >= 0.6)", flush=True)
    rows = []
    for cid, split, cpath, lpath, subset, saved in jobs:
        img, lbl = load_case(cpath)
        gt = lbl > 0
        lab = np.load(lpath)["lab"]
        _, _, masks = patient_run(cid, split, img, gt, lab, kept_ids(subset), runner, None, return_masks=True)
        for arm in ("A", "B", "C", "D", "E"):
            m = masks[f"{arm}_raw"]
            (dx, dy, dz), d0, bd = shift_profile(m, gt, 3)
            rows.append(dict(cache_id=cid, split=split, arm=arm, dx=dx, dy=dy, dz=dz, dice_0=d0, dice_best=bd,
                             gain=bd - d0, centroid_offset=(np.argwhere(m).mean(0) - np.argwhere(gt).mean(0)).round(2).tolist()
                             if m.any() else None))
    df = pd.DataFrame(rows)
    df.to_csv(out / "shift_check.csv", index=False)
    s = df.groupby("arm").agg(n=("dx", "size"), mean_dx=("dx", "mean"), mean_dy=("dy", "mean"), mean_dz=("dz", "mean"),
                              zero_shift_share=("gain", lambda g: float((g <= 1e-12).mean())),
                              mean_dice_0=("dice_0", "mean"), mean_dice_best=("dice_best", "mean"),
                              mean_gain=("gain", "mean"))
    for arm, g in df.groupby("arm"):
        for ax in ("dx", "dy", "dz"):
            s.loc[arm, f"{ax}_pos_neg"] = f"{int((g[ax] > 0).sum())}/{int((g[ax] < 0).sum())}"
    s.to_csv(out / "shift_summary.csv")
    pd.set_option("display.width", 220)
    print("\nbest integer shift (mm) of each arm's contour against the expert mask (A = reference alignment):")
    print(s.round(3).to_string())
    print("\nRead: B/C/D aligned like A -> mean shifts near 0 and +/- counts balanced. A consistent same-sign shift in"
          " B/C/D (not in A) = paste-back error.")
    print(f"-> {out}")


# --------------------------------------------------------------------------------------------- overlays
def overlay(png, img, gt, comp, out, row, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.argwhere(R).min(0), np.argwhere(R).max(0)
    zs = sorted({row["pD"][2], row["pC"][2]})
    fig, axes = plt.subplots(len(zs), 2, figsize=(10, 5 * len(zs)), squeeze=False)
    o = np.asarray(row["origin15"]); b15 = np.asarray(row["box15"])
    blo, bhi = crop_to_1mm(b15[0], o) - 0.75, crop_to_1mm(b15[1], o) + 0.75     # outer edges of the 1.5 mm voxels
    colours = dict(GT="yellow", A="lime", B="deepskyblue", C="cyan", D="red", E="magenta")
    for r, z in enumerate(zs):
        sl = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), z)
        for c, show in enumerate((("GT", "A"), ("A", "B", "C", "D", "E"))):
            ax = axes[r, c]
            ax.imshow(img[1][sl].T, cmap="gray", origin="lower", extent=(lo[0] - .5, hi[0] + .5, lo[1] - .5, hi[1] + .5))
            for name in show:
                m = gt if name == "GT" else comp if name == "A" else out[name]
                s = m[sl].T.astype(float)
                if s.any():
                    ax.contour(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), s, [0.5],
                               colors=colours[name], linewidths=1.2)
                ax.plot([], [], color=colours[name], label=name)
            ax.add_patch(plt.Rectangle((blo[0], blo[1]), bhi[0] - blo[0], bhi[1] - blo[1], fill=False, ls="--",
                                       ec="orange", label="SAM box"))
            for key, mk, col in (("pC", "x", "cyan"), ("pD", "+", "red")):
                p = row[key]; pb = crop_to_1mm(row[f"{key}_crop"], o)
                ax.plot(p[0], p[1], mk, color=col, ms=12, mew=2, label=f"{key[1]} point (z={p[2]})")
                ax.plot(pb[0], pb[1], "o", mfc="none", color=col, ms=8, label=f"{key[1]} as given to SAM")
            ax.set_title(f"{row['cache_id']} k={row['k']} z={z} | same SAM voxel: {row['same_sam_voxel']}", fontsize=9)
            ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------------------------- parity (step 0)
def parity_job(job):
    cid, split, cpath, lpath, subset, saved = job
    try:
        _, lbl = load_case(cpath)
        gt = lbl > 0
        lab = np.load(lpath)["lab"]
        if lab.shape != gt.shape:
            return dict(cache_id=cid, split=split, error=f"lab {lab.shape} vs label {gt.shape}")
        ks = kept_ids(subset)
        present = set(np.unique(lab).tolist())
        miss = [k for k in ks if k not in present]
        if miss:
            return dict(cache_id=cid, split=split, error=f"kept ids {miss} not in the component map")
        m = np.isin(lab, ks) if ks else np.zeros(gt.shape, bool)
        d = dice_count(m, gt)
        sizes, edges = [], []
        for k in ks:
            c = lab == k
            lo, hi = bbox(c)
            sizes.append(int(c.sum())); edges.append(int(min(min(lo), min(np.asarray(lab.shape) - 1 - hi))))
        return dict(cache_id=cid, split=split, subset="+".join(map(str, ks)), n_kept=len(ks), dice=d, saved_dice=saved,
                    abs_diff=abs(d - saved), min_kept_vox=min(sizes) if sizes else np.nan,
                    edge_mm=min(edges) if edges else np.nan, gt_vox=int(gt.sum()), error=None)
    except Exception as e:                                       # reported, never silently dropped
        return dict(cache_id=cid, split=split, error=f"{type(e).__name__}: {e}")


def load_selection(s3k_out):
    pp = pd.read_csv(Path(s3k_out) / "per_patient.csv", dtype={"B0_rad_subset": str})
    need = {"split", "cache_id", "B0_rad_dice", "B0_rad_subset"}
    if need - set(pp.columns):
        raise SystemExit(f"per_patient.csv lacks {sorted(need - set(pp.columns))}: is this the S3k output?")
    return pp


def jobs_for(a, root, sp, pp, splits):
    sel = {("train" if s == "train_oof" else s, c): (sub, d)
           for s, c, sub, d in zip(pp.split, pp.cache_id, pp.B0_rad_subset, pp.B0_rad_dice)}
    cdirs = dict(train=Path(a.train_cases).expanduser(), val=Path(a.val_cases).expanduser())
    jobs = []
    for split in splits:
        cases = require_cases(sp, root, split)
        miss = [c for c, _ in cases if (split, c) not in sel]
        if miss:
            raise SystemExit(f"{split}: {len(miss)} patients have no B0_rad selection (e.g. {miss[:5]})")
        for c, p in cases:
            lp = cdirs[split] / f"{c}.npz"
            if not lp.exists():
                raise SystemExit(f"missing component map {lp}")
            jobs.append((c, split, p, lp) + sel[(split, c)])
    return jobs


def step_parity(a, root, sp, pp, splits, out):
    jobs = jobs_for(a, root, sp, pp, splits)
    print(f"S3m step 0 (parity) | {len(jobs)} patients | {a.workers} workers", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(parity_job, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "parity.csv", index=False)
    bad = df[df.error.notna() | (df.abs_diff.fillna(1) > 1e-6)]
    for split, g in df.groupby("split"):
        print(f"  {split}: n={len(g)} rebuilt mean Dice {g.dice.mean():.6f} | saved {g.saved_dice.mean():.6f} | "
              f"max |diff| {g.abs_diff.max():.2e} | kept 0/1/2+: {(g.n_kept == 0).sum()}/{(g.n_kept == 1).sum()}/"
              f"{(g.n_kept >= 2).sum()}")
    if len(bad):
        print(bad.head(10).to_string(index=False))
        raise SystemExit(f"PARITY FAILED for {len(bad)} patient(s): see parity.csv; do not run the arms")
    print("PARITY OK: every patient's A mask reproduces its S3k B0_rad Dice within 1e-6")
    return df


def choose_smoke(par, n=20, seed=0):
    chosen = {}

    def add(df, reason, m):
        got = 0
        for cid in df.cache_id:
            if got >= m or len(chosen) >= n:
                break
            if cid not in chosen:
                chosen[cid] = reason; got += 1
    k = par[par.n_kept > 0]
    add(par[par.n_kept == 0].sort_values("cache_id"), "empty_selection", 2)
    add(k[k.n_kept >= 2].sort_values(["n_kept", "cache_id"], ascending=[False, True]), "multiple_kept", 3)
    add(k.sort_values(["min_kept_vox", "cache_id"]), "tiny_component", 3)
    add(k.sort_values(["edge_mm", "cache_id"]), "near_edge", 3)
    add(k[k.split == "val"].sample(frac=1, random_state=seed), "val", 3)
    add(par.sample(frac=1, random_state=seed), "random", n)
    return chosen


# --------------------------------------------------------------------------------------------- evaluation
def boot_ci(d, n=5000, seed=0):
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(df, b, a, label):
    db, da = df[f"{b}_dice"].to_numpy(), df[f"{a}_dice"].to_numpy()
    nb, na = df[f"{b}_nsd"].to_numpy(), df[f"{a}_nsd"].to_numpy()
    lo, hi = boot_ci(nb - na, RULE["bootstrap"], RULE["seed"])
    dlo, dhi = boot_ci(db - da, RULE["bootstrap"], RULE["seed"])
    both = df[f"{b}_hd95"].notna() & df[f"{a}_hd95"].notna()
    return dict(comparison=label, B=b, A=a, n=len(df), mean_dnsd=float((nb - na).mean()), dnsd_ci_lo=lo,
                dnsd_ci_hi=hi, mean_ddice=float((db - da).mean()), ddice_ci_lo=dlo, ddice_ci_hi=dhi,
                lt05_A=int((da < 0.5).sum()), lt05_B=int((db < 0.5).sum()),
                lt01_A=int((da < 0.1).sum()), lt01_B=int((db < 0.1).sum()),
                new_lt01=int(((db < 0.1) & (da >= 0.1)).sum()), rescued_lt01=int(((da < 0.1) & (db >= 0.1)).sum()),
                new_lt05=int(((db < 0.5) & (da >= 0.5)).sum()), rescued_lt05=int(((da < 0.5) & (db >= 0.5)).sum()),
                loss_ge_020=int((da - db >= 0.20).sum()), gain_ge_020=int((db - da >= 0.20).sum()),
                hd95_joint_nonempty=int(both.sum()),
                hd95_median_A=float(df.loc[both, f"{a}_hd95"].median()) if both.any() else np.nan,
                hd95_median_B=float(df.loc[both, f"{b}_hd95"].median()) if both.any() else np.nan)


COMPARISONS = [("primary", "D_fb", "A_fb"), ("secondary_1", "D_fb", "C_fb"), ("secondary_2", "D_fb", "E_fb"),
               ("exploratory", "B_fb", "A_fb"), ("exploratory", "C_fb", "A_fb"), ("exploratory", "E_fb", "A_fb"),
               ("raw", "D_raw", "A_raw"), ("raw", "D_raw", "C_raw"), ("raw", "D_raw", "E_raw"),
               ("raw", "B_raw", "A_raw"), ("raw", "C_raw", "A_raw"), ("raw", "E_raw", "A_raw")]


def rule_check(p, s1, s2):
    n = p["n"]
    lim05 = math.floor(RULE["lt05_factor"] * p["lt05_A"])
    lim02 = math.floor(RULE["loss020_frac"] * n)
    c = dict(mean_dnsd_ok=p["mean_dnsd"] >= RULE["min_mean_dnsd"], ci_lo_ok=p["dnsd_ci_lo"] > RULE["ci_lo_dnsd_gt"],
             mean_ddice_ok=p["mean_ddice"] >= RULE["min_mean_ddice"], lt05_ok=p["lt05_B"] <= lim05,
             loss020_ok=p["loss_ge_020"] <= lim02, lt05_max=lim05, loss020_max=lim02)
    c["primary_D_beats_A"] = all(c[k] for k in ("mean_dnsd_ok", "ci_lo_ok", "mean_ddice_ok", "lt05_ok", "loss020_ok"))
    c["secondary_D_beats_C"] = bool(c["primary_D_beats_A"] and s1["dnsd_ci_lo"] > 0)
    c["secondary_D_beats_E"] = bool(c["secondary_D_beats_C"] and s2["dnsd_ci_lo"] > 0)
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in c.items()}


def summarise(df, split):
    rows = []
    for a in ARMS:
        for v in ("fb", "raw"):
            n = f"{a}_{v}"
            ne = df[f"{n}_hd95"].notna()
            rows.append(dict(split=split, arm=a, variant=v, n=len(df), dice=df[f"{n}_dice"].mean(),
                             dice_median=df[f"{n}_dice"].median(), nsd=df[f"{n}_nsd"].mean(),
                             hd95_median=df.loc[ne, f"{n}_hd95"].median(), hd95_n=int(ne.sum()),
                             empty_outputs=int(df[f"{n}_empty"].sum()), lt01=int((df[f"{n}_dice"] < 0.1).sum()),
                             lt05=int((df[f"{n}_dice"] < 0.5).sum())))
    return rows


def md5(p):
    if not p or not Path(p).exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def step_arms(a, root, sp, pp, splits, out):
    par_csv = Path(a.parity_csv).expanduser() if a.parity_csv else out / "parity.csv"
    if not par_csv.exists():
        raise SystemExit(f"{par_csv} not found: run --step parity first (A must reproduce S3k before SAM runs)")
    par = pd.read_csv(par_csv)
    if par.error.notna().any() or (par.abs_diff > 1e-6).any():
        raise SystemExit("parity.csv contains failures: fix parity first")
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any SAM call
    jobs = [j for j in jobs_for(a, root, sp, pp, splits)]
    smoke = {}
    if a.smoke:
        smoke = choose_smoke(par[par.split.isin(splits)], a.smoke, a.seed)
        jobs = [j for j in jobs if j[0] in smoke]
    elif a.limit:
        jobs = jobs[:a.limit]
    od = out / "overlays" if a.smoke else None
    if od is not None:
        od.mkdir(exist_ok=True)
    turbo = None if a.random_weights else a.turbo_ckpt
    runner = load_runner(a.sam_ckpt, turbo, a.random_weights)
    print(f"S3m arms | {len(jobs)} patients ({'smoke' if a.smoke else 'full'}) | splits {splits}", flush=True)
    res, cands, t0 = [], [], time.time()
    for i, (cid, split, cpath, lpath, subset, saved) in enumerate(jobs):
        img, lbl = load_case(cpath)
        gt = lbl > 0
        lab = np.load(lpath)["lab"]
        r, rows = patient_run(cid, split, img, gt, lab, kept_ids(subset), runner, od)
        if abs(r["A_fb_dice"] - saved) > 1e-6:
            raise SystemExit(f"{cid}: A Dice {r['A_fb_dice']} != S3k {saved} (parity broke during the arms step)")
        r["smoke_reason"] = smoke.get(cid)
        res.append(r); cands += rows
        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    df, cd = pd.DataFrame(res), pd.DataFrame(cands)
    df.to_csv(out / "per_patient.csv", index=False); cd.to_csv(out / "candidates.csv", index=False)
    summ, comp, verdict = [], [], None
    for split, g in df.groupby("split", sort=False):
        summ += summarise(g, split)
        cs = {}
        for kind, b, a_ in COMPARISONS:
            c = compare(g, b, a_, f"{b} vs {a_}"); c.update(split=split, kind=kind); comp.append(c); cs[kind, b, a_] = c
        if split == "train" and not a.smoke:
            verdict = rule_check(cs["primary", "D_fb", "A_fb"], cs["secondary_1", "D_fb", "C_fb"],
                                 cs["secondary_2", "D_fb", "E_fb"])
    pd.DataFrame(summ).to_csv(out / "arms_summary.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "paired.csv", index=False)
    tech = {}
    if len(cd):
        tech = dict(candidates=len(cd), same_sam_voxel_C_D=int(cd.same_sam_voxel.sum()),
                    cd_dist_mm_median=float(cd.cd_dist_mm.median()), d_geometric_fallback=int(cd.d_geometric_fallback.sum()),
                    e_weak_seed=int(cd.e_weak_seed.sum()), point_clipped=int(cd.point_clipped.sum()),
                    crop_outside_volume=int(cd.crop_outside_volume.sum()),
                    **{f"{x}_fallback": int(cd[f"{x}_fallback"].sum()) for x in "BCDE"},
                    sam_seconds_per_candidate=float(cd.t_sam_s.mean()), e_seconds_per_candidate=float(cd.t_e_s.mean()),
                    seconds_per_patient=float(df.seconds.mean()))
    (out / "technical.json").write_text(json.dumps(tech, indent=1))
    (out / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    prov = dict(script="scripts/s3m_contour_study.py", step="arms", smoke=a.smoke, splits=splits, rule=RULE,
                inputs=dict(per_patient=md5(Path(a.s3k_out).expanduser() / "per_patient.csv"), parity=md5(par_csv),
                            sam_ckpt=md5(a.sam_ckpt), turbo=md5(turbo), random_weights=a.random_weights),
                note="selection frozen from S3k B0_rad; test not read")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))

    pd.set_option("display.width", 220)
    s = pd.DataFrame(summ)
    print("\n" + s[s.variant == "fb"].round(4).to_string(index=False))
    print("\nraw (no fallback):\n" + s[s.variant == "raw"].round(4).to_string(index=False))
    cols = ["split", "kind", "comparison", "mean_dnsd", "dnsd_ci_lo", "dnsd_ci_hi", "mean_ddice", "lt05_A", "lt05_B",
            "loss_ge_020", "new_lt01", "rescued_lt01", "new_lt05", "rescued_lt05"]
    print("\npaired (patient bootstrap):\n" + pd.DataFrame(comp)[cols].round(4).to_string(index=False))
    print("\ntechnical:", json.dumps(tech, indent=1))
    if a.smoke:
        print("\nSMOKE RUN: technical check only, the decision rule is not evaluated. Check overlays/ and candidates.csv")
    elif verdict is not None:
        print("\nfrozen rule (train OOF):", json.dumps(verdict, indent=1))
        print("\nVERDICT:", "D BEATS A (SAM contour refinement passes; held-out evaluation next)"
              if verdict["primary_D_beats_A"] else "keep A (SAM contours do not pass the primary rule)")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["parity", "arms", "shiftcheck", "overlap"], required=True)
    ap.add_argument("--shift_n", type=int, default=30, help="patients for --step shiftcheck")
    ap.add_argument("--arms_dir", default="", help="--step overlap: full arms output, for the B parity check")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--eval_splits", default="train,val")
    ap.add_argument("--s3k_out", required=True, help="S3k output folder (per_patient.csv with B0_rad_subset)")
    ap.add_argument("--train_cases", required=True, help="S3g cases/ (OOF component maps, key lab)")
    ap.add_argument("--val_cases", required=True, help="S3f cases/ (component maps, key lab)")
    ap.add_argument("--parity_csv", default="", help="parity.csv of step 0 (default <out>/parity.csv)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--smoke", type=int, default=0, help="technical run on N chosen patients (+ overlays)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    splits = [s.strip() for s in a.eval_splits.split(",") if s.strip()]
    if "test" in splits or not set(splits) <= {"train", "val"}:
        raise SystemExit("S3m is a development study: only train (OOF) and val are allowed")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    pp = load_selection(Path(a.s3k_out).expanduser())
    if a.step == "parity":
        step_parity(a, root, sp, pp, splits, out)
    else:
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt (S2b best.pt) and --turbo_ckpt are required")
        dict(arms=step_arms, shiftcheck=step_shiftcheck, overlap=step_overlap)[a.step](a, root, sp, pp, splits, out)


if __name__ == "__main__":
    main()
