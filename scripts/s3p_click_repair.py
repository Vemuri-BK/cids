"""S3p: simulated few-click repair of B0_rad masks -- does SAM add value over the same clicks spent on simple editing?

Simulation pilot on a DEVELOPMENT cohort (train OOF; the test split is not used). Every arm starts from the same
frozen B0_rad mask (S3k) and receives the same click budget, the same click-placement policy and the same blob-
toggle rule. After the masks diverge, each arm's next click comes from ITS OWN remaining error. Rejected and
ineffective clicks count against the budget.

Arms
  T    blob toggle only: a positive click inside a not-kept candidate adds that candidate; a negative click inside a
       kept candidate that does not touch the tumour removes it; any other click is ineffective
  TB   blob toggle + 5 mm brush: clicks the toggle rule does not handle add (positive) or remove (negative) a 5 mm ball
  TS   blob toggle + SAM: clicks the toggle rule does not handle go to SAM-Med3D-turbo (S2b decoder). SAM gets a 128^3
       crop at 1.5 mm CENTRED ON THE CLICK (so a missed tumour outside the original box can be recovered), all of this
       arm's earlier SAM clicks inside the crop, and the CURRENT mask as its low-res mask prompt. Only the connected
       part of the proposed change that contains the click is applied (additions for a positive click, removals for
       a negative click); the mask outside it is preserved.
Simulated reader (sees the expert mask; separate from the real-click editor)
  next click = deepest interior point (largest distance to the region's edge) of the largest connected error region
  of the arm's current mask: missed tumour -> positive click, wrongly included tissue -> negative click.
Acceptance
  primary      always-accept (every proposed edit is applied)
  sensitivity  oracle: an edit is undone if it lowers Dice against the expert mask (the click still counts)
Mask prompt: the current mask is resampled to 1.5 mm, cropped, averaged over 4^3 blocks -> 32^3 fraction f, logits
  = 4 * (2f - 1). The smoke test reports the round-trip agreement of this conversion and the coordinate mapping.

Population (frozen): the 108-patient S3n review queue (top 10% by the baseline review score, empty masks first,
  ties by cache_id; 65 failures + 43 masks with Dice >= 0.5) = PRIMARY development cohort. The 137 train-OOF failures
  (A Dice < 0.5) = SECONDARY, overlapping analysis. Evidence: the S2b decoder was trained on these patients and the
  clicks are simulated with the expert mask, so this is a feasibility pilot.
Frozen pilot advancement rule (queue, always-accept, after 3 clicks, TS vs TB): advance to a reader study if
  mean paired dNSD@2mm >= 0.05 AND patient-bootstrap 95% CI lower bound > 0 AND #patients whose Dice ends below their
  own starting Dice is not higher with TS than with TB. Estimates and harm counts are reported whatever the outcome.
Reported: Dice and NSD@2mm after 0, 1, 3, 5 clicks, rescues (<0.5 -> >=0.5), newly harmed (>=0.5 -> <0.5), patients
  ending below their starting Dice, patients never reaching Dice 0.8 by click 5, click actions, runtime.

Steps: --step parity (CPU) -> --step calibrate (GPU; mask-prompt tau / L on 20 train patients OUTSIDE the
  population, by round-trip agreement and click compliance only) -> --step smoke (GPU, 12 chosen patients + overlays)
  -> --step full (GPU). smoke / full read <out>/calibration.json and record it in decision_rule.json.
Usage (server)
  python scripts/s3p_click_repair.py --step parity --cache_root ~/data --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --train_cases ~/CIDS/outputs/s3g_oof_masks/cases --s3n_dir ~/CIDS/outputs/s3n_qc --out ~/CIDS/outputs/s3p_click
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M3  # noqa: E402
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.eval.metrics import nsd  # noqa: E402
from cids.provenance import require_cases  # noqa: E402

STRUCT = np.ones((3, 3, 3), bool)
ARMS_V1 = ("T", "TB", "TS")
ARMS = ("T", "TB", "TS", "TS_hist", "TS_hybrid")
# v2: TS (PRIMARY SAM arm) = current click only + current mask; TS_hist (exploratory) = de-duplicated in-crop click
# history + current mask; TS_hybrid (exploratory) = SAM (current click) for positive clicks, brush for negative clicks
SAM_ARMS = {"TS": "current", "TS_hist": "history", "TS_hybrid": "current"}
VARIANTS = ("always", "oracle")
CFG = dict(budget=5, brush_mm=5, edit_radius_mm=20, mask_logit=4.0, mask_tau=0.5, sam_spacing=1.5, sam_size=128, report_steps=[0, 1, 3, 5],
           budget_frac=0.10)
RULE = dict(population="S3n 108-patient review queue (development cohort)", acceptance="always-accept",
            comparison="TS (toggle + SAM) vs TB (toggle + 5 mm brush) after 3 clicks", min_mean_dnsd=0.05,
            ci_lo_gt=0.0, harm="#patients with Dice(3 clicks) < own starting Dice: TS <= TB", bootstrap=5000, seed=0,
            secondary=["137 train-OOF failures", "oracle acceptance (sensitivity)", "blob toggle alone (T)"],
            evidence="simulated clicks from the expert mask; S2b decoder trained on these patients", config=CFG)


# ----------------------------------------------------------------------------------------------- basics
def dice(m, g):
    s = int(m.sum()) + int(g.sum())
    return 0.0 if s == 0 else 2.0 * int((m & g).sum()) / s


def label26(mask):
    lab = np.zeros(mask.shape, np.int32)
    ndimage.label(mask, structure=STRUCT, output=lab)
    return lab


def deepest_point(comp):
    lo, hi = M3.bbox(comp)
    sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
    p = M3.geometric_point(comp[sl])
    return tuple(int(a + b) for a, b in zip(p, lo))


def ball_slice(shape, p, r):
    lo = np.maximum(np.asarray(p) - r, 0); hi = np.minimum(np.asarray(p) + r + 1, shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    b = M3.ball(tuple(hi - lo), np.asarray(p) - lo, r)
    return sl, b


# ----------------------------------------------------------------------------------------------- simulated reader
def ordered_points(comp):
    """All voxels of a region, deepest first (distance to the region's edge), ties -> nearest to the centroid,
    then lexicographic. The first point equals deepest_point(comp)."""
    lo, hi = M3.bbox(comp)
    sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
    c = comp[sl]
    dt = M3.inside_distance(c)
    idx = np.argwhere(c)
    d = dt[c]
    d2 = ((idx - np.argwhere(comp).mean(0) + lo) ** 2).sum(1)
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0], d2, -d))
    return idx[order] + lo


def next_click(m, gt, exclude=None):
    """Simulated reader (uses the expert mask): (point, sign, region voxels) or None if the mask is perfect.
    Largest connected error region (missed tumour or wrongly included tissue; ties -> missed tumour first), clicked at
    its deepest interior point. sign 1 = positive (add), 0 = negative (remove).
    v2: exclude = {(point, sign)} already attempted on the UNCHANGED mask -> the next eligible voxel is used
    (deeper-first within the region, then the next largest region)."""
    regions = []
    for order_, (sign, err) in enumerate(((1, gt & ~m), (0, m & ~gt))):
        if not err.any():
            continue
        lab = label26(err)
        size = np.bincount(lab.ravel())[1:]
        for k in np.argsort(-size, kind="stable")[: (1 if not exclude else len(size))]:
            regions.append((-int(size[k]), order_, sign, lab, int(k) + 1))
    if not regions:
        return None
    regions.sort(key=lambda r: (r[0], r[1]))
    for negsize, _, sign, lab, k in regions:
        comp = lab == k
        if not exclude:
            return deepest_point(comp), sign, -negsize
        for q in ordered_points(comp):
            q = tuple(int(v) for v in q)
            if (q, sign) not in exclude:
                return q, sign, -negsize
    return None


# ----------------------------------------------------------------------------------------------- editors (no GT)
def toggle_action(p, sign, kept, cand_ids, lab, cand_touches_gt):
    """Shared blob-toggle rule. cand_touches_gt is the simulated reader's judgement 'this blob is not tumour'."""
    c = int(lab[tuple(p)])
    if c in cand_ids:
        if sign == 1 and c not in kept:
            return "toggle_on", c
        if sign == 0 and c in kept and not cand_touches_gt[c]:
            return "toggle_off", c
    return None


def reader_action(p, sign, kept, lab, cand_ids, gt):
    """SIMULATED READER (uses the expert mask, like click placement): an explicit 'delete this whole blob' action when
    a negative click falls in a kept candidate that the reader judges not to be tumour (no overlap with the expert
    mask). Counted as one click in every arm."""
    c = int(lab[tuple(p)])
    if sign == 0 and c in cand_ids and c in kept and not (gt & (lab == c)).any():
        return "delete_blob", c
    return None


def editor_blob_rule(p, sign, kept, lab, cand_ids):
    """EDITOR (no expert mask): a positive click inside a candidate blob that is not kept adds that whole blob."""
    c = int(lab[tuple(p)])
    if sign == 1 and c in cand_ids and c not in kept:
        return "add_blob", c
    return None


def brush(m, p, sign, r):
    out = m.copy()
    sl, b = ball_slice(m.shape, p, r)
    if sign == 1:
        out[sl] |= b
    else:
        out[sl] &= ~b
    return out


def mask_to_low(mc, L=4.0, tau=0.5):
    """128^3 bool crop -> (1,1,32,32,32) logits L*(f - tau)/max(tau, 1 - tau), f = tumour fraction of each 4^3
    block (tau = 0.5 gives L*(2f - 1)); a block counts as tumour when f > tau."""
    import torch
    import torch.nn.functional as F
    f = F.avg_pool3d(torch.from_numpy(mc.astype(np.float32))[None, None], 4)
    return L * (f - tau) / max(tau, 1 - tau)


def low_to_mask(low):
    from cids.models.sam_lora import upsample
    return (upsample(low.float())[0, 0] > 0).numpy()


class ClickEditor:
    """REAL-click SAM inference: takes click coordinates and the current mask, never the expert mask.
    predictor(x (1,1,128^3), coords (1,n,3) | None, labels (1,n) | None, low (1,1,32^3)) -> prob (128^3) numpy."""

    def __init__(self, predictor, L=4.0, tau=0.5):
        self.predict, self.L, self.tau = predictor, L, tau

    def prepare(self, img):
        from cids.data.cache import resample
        self.shape1 = img.shape[1:]
        img15, _ = resample(img, np.zeros(self.shape1, np.uint8))
        self.post, self.shape15 = img15[1], img15.shape[1:]
        self.fg = (self.post > 0).astype(np.uint8)

    def _mask15(self, m):
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        return (F.interpolate(t, size=tuple(self.shape15), mode="trilinear", align_corners=False)[0, 0] > 0.5).numpy()

    def propose(self, m, clicks, at=None, sign=None, radius_mm=None, zero_prompt=False):
        """clicks = [(point_1mm, sign), ...] given to SAM (the last one is the new click). Returns (edited mask, info).
        at / sign override the edit location and polarity (replay probes); radius_mm bounds the applied change to a
        ball around the click (v2); zero_prompt replaces the encoded mask prompt by zeros (diagnostic)."""
        import torch
        from cids.data.cache import crop_pad, znorm
        p = tuple(at) if at is not None else tuple(clicks[-1][0])
        s = sign if sign is not None else clicks[-1][1]
        n = CFG["sam_size"]
        c15 = np.floor((np.asarray(p, float) + 0.5) / CFG["sam_spacing"]).astype(int)
        vc, origin = crop_pad(self.post, c15)
        fc, _ = crop_pad(self.fg, c15)
        x = torch.from_numpy(znorm(vc, fc)[None, None])
        mc, _ = crop_pad(self._mask15(m).astype(np.uint8), c15)
        low = mask_to_low(mc > 0, self.L, self.tau)
        if zero_prompt:
            low = torch.zeros_like(low)
        pts, labs = [], []
        for q, sg in clicks:
            cq = M3.to_crop(q, origin)
            if (cq >= 0).all() and (cq < n).all():
                pts.append(cq); labs.append(sg)
        coords = torch.tensor(np.asarray(pts, np.float32)[None]) if pts else None
        labels = torch.tensor(np.asarray(labs)[None], dtype=torch.long) if pts else None
        prob = self.predict(x, coords, labels, low)
        up = M3.paste_back(prob, origin, self.shape15, self.shape1)
        S = up > 0.5
        # crop extent at 1 mm (only there can SAM change the mask)
        a = np.maximum(np.floor(np.asarray(origin) * 1.5).astype(int), 0)
        b = np.minimum(np.ceil((np.asarray(origin) + n) * 1.5).astype(int), self.shape1)
        ext = tuple(slice(i, j) for i, j in zip(a, b))
        rt = M3.paste_back(low_to_mask(low).astype(np.float32), origin, self.shape15, self.shape1) > 0.5
        info = dict(roundtrip_dice=dice(rt[ext], m[ext]) if (rt[ext].any() or m[ext].any()) else np.nan,
                    p_at_click=float(up[tuple(p)]), click_back_err_mm=float(np.abs(
                        M3.crop_to_1mm(M3.to_crop(p, origin), origin) - np.asarray(p)).max()),
                    n_points=len(pts), origin15=np.asarray(origin).tolist(), mask_in_crop_vox=int(m[ext].sum()))
        Dx = (S[ext] & ~m[ext]) if s == 1 else (m[ext] & ~S[ext])
        pl = tuple(np.asarray(p) - a)
        if not Dx[pl]:
            info.update(change_vox=0, raw_change_vox=0)
            return m.copy(), info
        lab = label26(Dx)
        info["raw_change_vox"] = int((lab == lab[pl]).sum())
        if radius_mm:
            g = np.ogrid[tuple(slice(0, n_) for n_ in Dx.shape)]
            Dx = Dx & (sum((g[i] - pl[i]) ** 2 for i in range(3)) <= radius_mm ** 2)
            lab = label26(Dx)
        comp = np.zeros(m.shape, bool)
        comp[ext] = lab == lab[pl]
        info["change_vox"] = int(comp.sum())
        info["max_edit_dist_mm"] = float(np.sqrt(((np.argwhere(lab == lab[pl]) - np.asarray(pl)) ** 2).sum(1).max()))
        return (m | comp) if s == 1 else (m & ~comp), info


def sam_predictor(sam, box_mod, dev, amp):
    import torch
    from cids.models.sam_lora import decode, upsample

    def f(x, coords, labels, low):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            emb = sam.image_encoder(x.float().to(dev)).float()
            c_ = None if coords is None else coords.to(dev)
            l_ = None if labels is None else labels.to(dev)
            lo = decode(sam, box_mod, emb, c_, l_, None, low.to(dev).float())
            return torch.sigmoid(upsample(lo.float()))[0, 0].cpu().numpy()
    return f


def load_predictor(sam_ckpt, turbo, random_weights):
    if random_weights:                       # tests: identical random weights in every process (smoke vs replay)
        import torch
        torch.manual_seed(0)
    r = M3.load_runner(sam_ckpt, turbo, random_weights)
    return sam_predictor(r.sam, r.box, r.dev, r.amp)


# ----------------------------------------------------------------------------------------------- one patient, one arm
def run_arm(arm, variant, A, gt, lab, kept0, cand_ids, touches, editor, budget, log, cid, overlay=None, img=None,
            ver="v2", probe=None):
    """ver v1 = the original simulator (kept for the record and the replay). ver v2: an attempted point/sign is not
    retried while the mask is unchanged (every attempt still counts), SAM's prompt history has no duplicates, a SAM
    edit is bounded to CFG['edit_radius_mm'] around the click, and whole-blob removal is an explicit simulated-reader
    action (reader_delete_blob; it uses the reader's knowledge of the truth, like click placement, for every arm)."""
    m, kept, sam_clicks = A.copy(), set(kept0), []
    exclude = set()
    dices = [dice(m, gt)]
    masks = {0: m.copy()}
    for step in range(1, budget + 1):
        t0 = time.time()
        c = next_click(m, gt, exclude if ver == "v2" else None)
        row = dict(cache_id=cid, arm=arm, variant=variant, step=step, version=ver)
        if c is None:
            row.update(action="no_error", effective=False)
            log.append(row); dices.append(dices[-1]); masks[step] = m.copy()
            continue
        p, sign, size = c
        row.update(p=list(p), sign=sign, region_vox=size, region_type="missed" if sign else "extra")
        if ver == "v1":
            act = toggle_action(p, sign, kept, cand_ids, lab, touches)
            act = None if act is None else ({"toggle_on": "toggle_on", "toggle_off": "toggle_off"}[act[0]], act[1])
        else:
            act = reader_action(p, sign, kept, lab, cand_ids, gt) or editor_blob_rule(p, sign, kept, lab, cand_ids)
        info, new_kept, appended = {}, set(kept), False
        if act is not None:
            kind, k = act
            on = kind in ("toggle_on", "add_blob")
            new = (m | (lab == k)) if on else (m & ~(lab == k))
            (new_kept.add if on else new_kept.discard)(k)
            action = {"toggle_on": "toggle_on", "toggle_off": "toggle_off", "add_blob": "add_blob",
                      "delete_blob": "reader_delete_blob"}[kind]
        elif arm == "T":
            new, action = m.copy(), "ineffective"
        elif arm == "TB" or (arm == "TS_hybrid" and sign == 0):
            new, action = brush(m, p, sign, CFG["brush_mm"]), "brush" if arm == "TB" else "hybrid_brush_negative"
        else:
            mode = "history" if (ver == "v1" or SAM_ARMS.get(arm) == "history") else "current"
            if mode == "current":
                prompts = [(tuple(p), sign)]
                hist_before = []
            else:
                hist_before = list(sam_clicks)
                if ver == "v2" and (tuple(p), sign) in {(tuple(q), sg) for q, sg in sam_clicks}:
                    appended = False                   # no duplicate prompt tokens
                else:
                    sam_clicks.append((tuple(p), sign)); appended = True
                prompts = sam_clicks
            if probe is not None:
                probe(m, hist_before, p, sign, step)
            new, info = editor.propose(m, prompts, at=p, sign=sign,
                                       radius_mm=CFG["edit_radius_mm"] if ver == "v2" else None)
            info["duplicate_prompt"] = bool(mode == "history" and not appended)
            info["history_mode"] = mode
            action = {"TS": "sam", "TS_hist": "sam_hist", "TS_hybrid": "hybrid_sam_positive"}[arm]
            if overlay is not None and step <= 3:
                overlay(cid, arm, variant, step, img, gt, m, new, p, sign)
        accepted = True
        if variant == "oracle" and dice(new, gt) < dice(m, gt) - 1e-12:
            accepted = False
            if appended:
                sam_clicks.pop()                       # an undone click is not kept as a prompt
        if accepted:
            effective = not np.array_equal(new, m)
            m, kept = new, new_kept
        else:
            effective = False
        if effective:
            exclude.clear()
        else:
            exclude.add((tuple(int(v) for v in p), sign))
        row.update(action=action, accepted=accepted, effective=effective, seconds=time.time() - t0,
                   **{k: v for k, v in info.items() if k != "origin15"})
        log.append(row)
        dices.append(dice(m, gt)); masks[step] = m.copy()
    out = {f"dice_{s}": dices[s] for s in range(budget + 1)}
    for s in CFG["report_steps"]:
        if s <= budget:
            out[f"nsd_{s}"] = nsd(masks[s], gt, 2.0)
    return out


# ----------------------------------------------------------------------------------------------- data
def queue_from_s3n(s3n_dir, frac):
    rv = pd.read_csv(Path(s3n_dir).expanduser() / "train_oof_review.csv")
    n = int(round(frac * len(rv)))
    order = np.lexsort((rv.cache_id.to_numpy(), -rv.risk_baseline.to_numpy(float)))
    q = rv.iloc[order[:n]].copy()
    return q, rv


def load_patient(cid, cache_path, case_path, subset, table):
    img, lbl = load_case(cache_path)
    gt = lbl > 0
    lab = np.load(case_path)["lab"]
    kept = M3.kept_ids(subset)
    cand = [int(k) for k in table.loc[table.cache_id == cid, "k"] if k >= 0]
    touches = {k: bool((gt & (lab == k)).any()) for k in cand}
    A = np.isin(lab, kept) if kept else np.zeros(gt.shape, bool)
    return img, gt, lab, kept, set(cand), touches, A


def population(a):
    pp = pd.read_csv(Path(a.s3k_out).expanduser() / "per_patient.csv", dtype={"B0_rad_subset": str})
    pp = pp[pp.split == "train_oof"].set_index("cache_id")
    q, rv = queue_from_s3n(a.s3n_dir, a.budget_frac)
    fails = set(pp.index[pp.B0_rad_dice < 0.5])
    pop = sorted(set(q.cache_id) | fails)
    return pp, q, fails, pop


# ----------------------------------------------------------------------------------------------- calibration
CALIB = dict(n_patients=20, seed=0, tau_grid=[0.5, 0.25, 0.125], L_grid=[4.0, 2.0, 1.0],
             tau_criterion="highest mean round-trip agreement (mask vs its converted prompt) in click-centred crops; "
                           "ties -> larger tau",
             L_criterion="highest click compliance = mean(share of positive clicks with SAM p > 0.5, share of "
                         "negative clicks with SAM p < 0.5); ties -> larger L",
             patients="train OOF patients OUTSIDE the 180-patient S3p population, with a non-empty starting mask",
             note="technical calibration only: no criterion uses Dice against the expert mask")


def largest_region_click(err):
    if not err.any():
        return None
    lab = label26(err)
    size = np.bincount(lab.ravel())[1:]
    return deepest_point(lab == int(np.argmax(size)) + 1)


def roundtrip_in_crop(editor, m, p, tau):
    from cids.data.cache import crop_pad
    c15 = np.floor((np.asarray(p, float) + 0.5) / CFG["sam_spacing"]).astype(int)
    mc, origin = crop_pad(editor._mask15(m).astype(np.uint8), c15)
    low = mask_to_low(mc > 0, 1.0, tau)
    rt = M3.paste_back(low_to_mask(low).astype(np.float32), origin, editor.shape15, editor.shape1) > 0.5
    n = CFG["sam_size"]
    a = np.maximum(np.floor(np.asarray(origin) * 1.5).astype(int), 0)
    b = np.minimum(np.ceil((np.asarray(origin) + n) * 1.5).astype(int), editor.shape1)
    ext = tuple(slice(i, j) for i, j in zip(a, b))
    return dice(rt[ext], m[ext]) if m[ext].any() else np.nan


def step_calibrate(a, root, sp, out):
    pp, q, fails, pop = population(a)
    table = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    pool = [c for c in pp.index if c not in set(pop) and M3.kept_ids(pp.loc[c, "B0_rad_subset"])]
    rng = np.random.default_rng(CALIB["seed"])
    ids = sorted(rng.choice(pool, min(CALIB["n_patients"], len(pool)), replace=False).tolist()) if pool else []
    if not ids:
        raise SystemExit("no calibration patients outside the population")
    paths = dict(require_cases(sp, root, "train"))
    cdir = Path(a.train_cases).expanduser()
    pred = load_predictor(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights)
    rows = []
    for cid in ids:
        img, gt, lab, kept, cand, touches, A = load_patient(cid, paths[cid], cdir / f"{cid}.npz",
                                                            pp.loc[cid, "B0_rad_subset"], table)
        ed = ClickEditor(pred); ed.prepare(img)
        clicks = [(sg, largest_region_click(err)) for sg, err in ((1, gt & ~A), (0, A & ~gt))]
        for sg, p in clicks:
            if p is None:
                continue
            for tau in CALIB["tau_grid"]:
                rows.append(dict(cache_id=cid, sign=sg, kind="roundtrip", tau=tau, value=roundtrip_in_crop(ed, A, p, tau)))
        rows.append(dict(cache_id=cid, kind="_clicks", value=json.dumps([[sg, list(p)] for sg, p in clicks if p])))
        ed._clicks = [(sg, p) for sg, p in clicks if p is not None]
        ed._A = A
        rows[-1]["_editor"] = ed
    rt = pd.DataFrame([r for r in rows if r.get("kind") == "roundtrip"])
    tau_tab = rt.groupby("tau").value.mean().sort_index(ascending=False)
    tau = float(tau_tab.idxmax())                                   # first max in descending order = larger tau
    comp = []
    for r in [r for r in rows if r.get("kind") == "_clicks"]:
        ed = r["_editor"]
        for L in CALIB["L_grid"]:
            ed.L, ed.tau = L, tau
            for sg, p in ed._clicks:
                _, info = ed.propose(ed._A, [(p, sg)])
                comp.append(dict(cache_id=r["cache_id"], L=L, sign=sg, p_at_click=info["p_at_click"],
                                 comply=bool(info["p_at_click"] > 0.5) if sg == 1 else bool(info["p_at_click"] < 0.5),
                                 change_vox=info.get("change_vox", 0)))
    cp = pd.DataFrame(comp)
    by = cp.groupby(["L", "sign"]).comply.mean().unstack()
    score = by.mean(axis=1).sort_index(ascending=False)
    L = float(score.idxmax())
    res = dict(**CALIB, patients_used=ids, chosen_tau=tau, chosen_L=L, roundtrip_by_tau=tau_tab.round(4).to_dict(),
               compliance_by_L={str(k): {"positive": float(by.loc[k].get(1, np.nan)), "negative": float(by.loc[k].get(0, np.nan)),
                                         "score": float(score.loc[k])} for k in score.index},
               n_positive_clicks=int((cp.sign == 1).sum() / len(CALIB["L_grid"])),
               n_negative_clicks=int((cp.sign == 0).sum() / len(CALIB["L_grid"])))
    rt.to_csv(out / "calibration_roundtrip.csv", index=False); cp.to_csv(out / "calibration_clicks.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("patients_used",)}, indent=1, default=float))
    print(f"CALIBRATED: tau = {tau}, L = {L}  -> {out / 'calibration.json'} (used by --step smoke / full)")


def apply_calibration(out):
    f = out / "calibration.json"
    if not f.exists():
        print("no calibration.json: using the default mask prompt (tau 0.5, L 4)")
        return None
    c = json.loads(f.read_text())
    CFG["mask_tau"], CFG["mask_logit"] = float(c["chosen_tau"]), float(c["chosen_L"])
    RULE["calibration"] = dict(file=str(f), tau=CFG["mask_tau"], L=CFG["mask_logit"], patients=len(c["patients_used"]),
                               note=c["note"])
    print(f"using calibration.json: tau = {CFG['mask_tau']}, L = {CFG['mask_logit']}")
    return c


# ----------------------------------------------------------------------------------------------- steps
def step_parity(a, root, sp, out):
    pp, q, fails, pop = population(a)
    table = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    paths = dict(require_cases(sp, root, "train"))
    cdir = Path(a.train_cases).expanduser()
    rows = []
    for cid in pop:
        img, gt, lab, kept, cand, touches, A = load_patient(cid, paths[cid], cdir / f"{cid}.npz",
                                                            pp.loc[cid, "B0_rad_subset"], table)
        d = dice(A, gt)
        rows.append(dict(cache_id=cid, in_queue=cid in set(q.cache_id), failure=cid in fails, start_dice=d,
                         saved=float(pp.loc[cid, "B0_rad_dice"]), abs_diff=abs(d - pp.loc[cid, "B0_rad_dice"]),
                         n_cand=len(cand), n_kept=len(kept), start_empty=not A.any()))
    df = pd.DataFrame(rows)
    df.to_csv(out / "parity.csv", index=False)
    qq = df[df.in_queue]
    print(f"S3p parity | population {len(df)} = queue {len(qq)} U failures {int(df.failure.sum())} | "
          f"queue: failures {int(qq.failure.sum())}, Dice>=0.5 {int((~qq.failure).sum())}, empty {int(qq.start_empty.sum())}")
    print(f"  max |A Dice - S3k| = {df.abs_diff.max():.2e} | queue mean start Dice {qq.start_dice.mean():.4f}")
    chk = Path(a.s3n_dir).expanduser() / "decision_rule_check.json"
    if chk.exists() and abs(a.budget_frac - 0.10) < 1e-12:
        v = json.loads(chk.read_text())
        if int(qq.failure.sum()) != v["caught_baseline"] or len(qq) != v["budget_reviews"]:
            raise SystemExit(f"PARITY FAILED: queue does not reproduce S3n ({len(qq)} reviews, {int(qq.failure.sum())} "
                             f"failures vs {v['budget_reviews']} / {v['caught_baseline']})")
        print(f"  queue reproduces S3n: {v['budget_reviews']} reviews, {v['caught_baseline']} failures caught")
    if df.abs_diff.max() > 1e-6:
        raise SystemExit("PARITY FAILED: A does not reproduce S3k")
    print("PARITY OK")


def overlay_fn(od):
    def f(cid, arm, variant, step, img, gt, before, after, p, sign):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = p[2]
        r = 40
        lo = np.maximum(np.asarray(p[:2]) - r, 0); hi = np.minimum(np.asarray(p[:2]) + r, gt.shape[:2])
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), z)
        add, rem = after & ~before, before & ~after
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        ax.imshow(img[1][sl].T, cmap="gray", origin="lower", extent=(lo[0] - .5, hi[0] - .5, lo[1] - .5, hi[1] - .5))
        xs, ys = np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1])
        for m, col, name in ((gt, "yellow", "expert"), (before, "lime", "mask before"), (add, "deepskyblue", "SAM adds"),
                             (rem, "red", "SAM removes")):
            s = m[sl].T.astype(float)
            if s.any():
                ax.contour(xs, ys, s, [0.5], colors=col, linewidths=1.2)
            ax.plot([], [], color=col, label=name)
        ax.plot(p[0], p[1], "+" if sign else "x", color="white", ms=14, mew=2.5, label="positive" if sign else "negative")
        ax.set_title(f"{cid} {arm} {variant} click {step} (z={z})", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(od / f"{cid}_{variant}_c{step}.png", dpi=110); plt.close(fig)
    return f


def choose_smoke(par, n=12):
    q = par[par.in_queue].sort_values("cache_id")
    pick = []
    for cond, m in ((q.start_empty, 3), ((q.start_dice < 0.1) & ~q.start_empty, 3), ((q.start_dice >= 0.1) & (q.start_dice < 0.5), 3),
                    (q.start_dice >= 0.5, 3)):
        pick += [c for c in q[cond].cache_id if c not in pick][:m]
    pick += [c for c in q.cache_id if c not in pick][:max(0, n - len(pick))]
    return pick[:n]


def step_run(a, root, sp, out, smoke):
    par_f = out / "parity.csv"
    if not par_f.exists():
        raise SystemExit("run --step parity first")
    par = pd.read_csv(par_f)
    if par.abs_diff.max() > 1e-6:
        raise SystemExit("parity.csv shows a mismatch")
    apply_calibration(out)
    ver = a.version
    RULE["version"] = ver
    if ver == "v2":
        RULE["v2_changes"] = ["no retry of the same point/sign while the mask is unchanged (every attempt counts); "
                              "SAM prompt history without duplicates (TS_hist)",
                              f"SAM edit bounded to a {CFG['edit_radius_mm']} mm ball around the click before choosing the "
                              "connected change at the click (raw and applied volumes logged)",
                              "whole-blob removal = explicit simulated-reader action (reader_delete_blob, uses the expert "
                              "mask like click placement), same for every arm; the editor (add_blob rule, brush, SAM) never "
                              "sees the expert mask",
                              "PRIMARY SAM arm TS = current click only + current mask; exploratory TS_hist = de-duplicated "
                              "in-crop click history + current mask; exploratory TS_hybrid = SAM for positive, brush for "
                              "negative clicks. Same checkpoint, tau, L, 20 mm bound, blob rules and click budget",
                              "On replay of previously failed negative clicks, removing point history restored "
                              "below-threshold tumour probability in 15/18 always-accept attempts and 6/6 oracle attempts. "
                              "These were repeated attempts on a small smoke sample; the replay identifies an effect of "
                              "including history, not which earlier clicks caused it or whether the edits improve Dice/NSD",
                              "revision prompted by the v1 smoke results; v1 code (tag s3p-v1) and outputs preserved",
                              "sensitivity: the primary comparison repeated excluding the 12 smoke patients"]
    (out / f"decision_rule_{ver}.json").write_text(json.dumps(RULE, indent=1))   # frozen before any SAM call
    pp, q, fails, pop = population(a)
    ids = choose_smoke(par, a.smoke_n) if smoke else pop
    table = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    paths = dict(require_cases(sp, root, "train"))
    cdir = Path(a.train_cases).expanduser()
    editor = ClickEditor(load_predictor(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights),
                         CFG["mask_logit"], CFG["mask_tau"])
    rd = out / (("smoke" if smoke else "full") + ("" if ver == "v1" else "_v2")); rd.mkdir(exist_ok=True)
    arms = ARMS_V1 if ver == "v1" else ARMS
    od = rd / "overlays" if smoke else None
    if od is not None:
        od.mkdir(exist_ok=True)
    res, log, t0 = [], [], time.time()
    qset = set(q.cache_id)
    for i, cid in enumerate(ids):
        img, gt, lab, kept, cand, touches, A = load_patient(cid, paths[cid], cdir / f"{cid}.npz",
                                                            pp.loc[cid, "B0_rad_subset"], table)
        editor.prepare(img)
        for arm in arms:
            for variant in VARIANTS:
                ov = overlay_fn(od) if (od is not None and arm == "TS") else None
                r = run_arm(arm, variant, A, gt, lab, kept, cand, touches, editor, a.budget, log, cid, ov, img, ver=ver)
                res.append(dict(cache_id=cid, arm=arm, variant=variant, in_queue=cid in qset, failure=cid in fails,
                                start_type="empty" if not A.any() else ("wrong_target" if r["dice_0"] < 0.1 else
                                                                         "poor_boundary" if r["dice_0"] < 0.5 else "ok"),
                                **r))
        if (i + 1) % 20 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    df, lg = pd.DataFrame(res), pd.DataFrame(log)
    df.to_csv(rd / "per_patient.csv", index=False); lg.to_csv(rd / "clicks.csv", index=False)
    report(df, lg, rd, a.budget, smoke, set(choose_smoke(par, 12)) if ver == "v2" else set())


def step_replay(a, root, sp, out):
    """Diagnostic (no decision): replay the v1 smoke trajectories of the SAM arm and probe every negative SAM click on
    its identical pre-edit state: no new click / negative with history / same-location positive with history /
    negative only / positive only. For positive clicks in an empty crop, compare the zero mask prompt with the encoded
    empty prompt. Reports probability at the clicked voxel, raw and 20 mm-bounded edit volume, duplicate status."""
    apply_calibration(out)
    par = pd.read_csv(out / "parity.csv")
    ids = choose_smoke(par, a.smoke_n)
    pp, q, fails, pop = population(a)
    table = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    paths = dict(require_cases(sp, root, "train"))
    cdir = Path(a.train_cases).expanduser()
    ed = ClickEditor(load_predictor(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights),
                     CFG["mask_logit"], CFG["mask_tau"])
    ref = None
    if (out / "smoke" / "clicks.csv").exists():
        ref = pd.read_csv(out / "smoke" / "clicks.csv")
        ref = ref[ref.arm == "TS"]
    rows, match = [], []
    R = CFG["edit_radius_mm"]
    for cid in ids:
        img, gt, lab, kept, cand, touches, A = load_patient(cid, paths[cid], cdir / f"{cid}.npz",
                                                            pp.loc[cid, "B0_rad_subset"], table)
        ed.prepare(img)
        for variant in VARIANTS:
            states, log = [], []
            run_arm("TS", variant, A, gt, lab, kept, cand, touches, ed, a.budget, log, cid, ver="v1",
                    probe=lambda m, hist, p, sg, st: states.append((m.copy(), hist, tuple(p), sg, st)))
            sam_rows = [r for r in log if r.get("action") == "sam"]
            if ref is not None:
                rr = ref[(ref.cache_id == cid) & (ref.variant == variant) & (ref.action == "sam")]
                match += [int(x["change_vox"]) == int(y) for x, y in zip(sam_rows, rr.change_vox)]
            for (m, hist, p, sg, st), r in zip(states, sam_rows):
                dup = (p, sg) in {(tuple(x), y) for x, y in hist}
                base = dict(cache_id=cid, variant=variant, step=st, sign=sg, p=list(p), duplicate=dup,
                            v1_change_vox=int(r["change_vox"]), v1_failed=int(r["change_vox"]) == 0)
                if sg == 0:
                    conds = dict(no_new_click=(hist, 0), neg_with_history=(hist + [(p, 0)], 0),
                                 pos_with_history=(hist + [(p, 1)], 1), neg_only=([(p, 0)], 0), pos_only=([(p, 1)], 1))
                    for name, (cl, s_) in conds.items():
                        _, info = ed.propose(m, cl, at=p, sign=0)
                        _, inb = ed.propose(m, cl, at=p, sign=0, radius_mm=R)
                        rows.append(dict(base, condition=name, p_at_click=info["p_at_click"],
                                         raw_removal_vox=info["raw_change_vox"], bounded_removal_vox=inb["change_vox"],
                                         n_points=info["n_points"]))
                else:
                    _, info = ed.propose(m, hist + [(p, 1)], at=p, sign=1)
                    if info["mask_in_crop_vox"] == 0:
                        _, iz = ed.propose(m, hist + [(p, 1)], at=p, sign=1, zero_prompt=True)
                        for name, inf in (("empty_crop_encoded_prompt", info), ("empty_crop_zero_prompt", iz)):
                            rows.append(dict(base, condition=name, p_at_click=inf["p_at_click"],
                                             raw_addition_vox=inf["raw_change_vox"]))
    df = pd.DataFrame(rows)
    rd = out / "replay"; rd.mkdir(exist_ok=True)
    df.to_csv(rd / "replay.csv", index=False)
    pd.set_option("display.width", 220)
    if ref is not None:
        print(f"v1 trajectory reproduced: {sum(match)}/{len(match)} SAM edits have the same change volume as smoke/clicks.csv")
    if not len(df):
        print("no SAM clicks to probe in these trajectories")
        print(f"-> {rd}")
        return
    neg = df[df.sign == 0]
    if len(neg):
        g = neg.groupby(["variant", "v1_failed", "condition"]).agg(states=("p_at_click", "size"),
                                                                    p_median=("p_at_click", "median"),
                                                                    p_below_05=("p_at_click", lambda x: float((x < 0.5).mean())),
                                                                    raw_removal_ml=("raw_removal_vox", lambda x: x.median() / 1000),
                                                                    bounded_removal_ml=("bounded_removal_vox", lambda x: x.median() / 1000))
        print("\nnegative SAM clicks, probability at the clicked voxel under each condition:\n" + g.round(3).to_string())
        nf = neg[(neg.condition == "neg_with_history") & neg.v1_failed]
        print(f"\nfailed negative clicks: {len(nf)} states | unique (patient, variant, point): "
              f"{nf.assign(pk=nf.p.astype(str)).drop_duplicates(['cache_id', 'variant', 'pk']).shape[0] if len(nf) else 0}"
              f" | duplicates among them: {int(nf.duplicate.sum())}")
    emp = df[df.condition.astype(str).str.startswith("empty_crop")]
    if len(emp):
        print("\nempty crop, positive click:\n" + emp.groupby("condition").agg(
            n=("p_at_click", "size"), p_median=("p_at_click", "median"),
            raw_addition_ml=("raw_addition_vox", lambda x: x.median() / 1000)).round(3).to_string())
    print(f"-> {rd}")


def boot_ci(d, n=5000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df, label, budget):
    rows = []
    for (arm, variant), g in df.groupby(["arm", "variant"]):
        for s in CFG["report_steps"]:
            if s > budget:
                continue
            d0, ds = g.dice_0, g[f"dice_{s}"]
            rows.append(dict(population=label, arm=arm, variant=variant, clicks=s, n=len(g), dice=ds.mean(),
                             nsd=g[f"nsd_{s}"].mean(), rescued_05=int(((d0 < 0.5) & (ds >= 0.5)).sum()),
                             newly_harmed_05=int(((d0 >= 0.5) & (ds < 0.5)).sum()),
                             below_start=int((ds < d0 - 1e-9).sum()), above_start=int((ds > d0 + 1e-9).sum())))
        rows[-1]["never_reach_08_by_5"] = int((g[[f"dice_{s}" for s in range(budget + 1)]].max(axis=1) < 0.8).sum())
    return rows


def primary_verdict(df, exclude=()):
    g = df[df.in_queue & (df.variant == "always") & ~df.cache_id.isin(set(exclude))]
    ts = g[g.arm == "TS"].set_index("cache_id"); tb = g[g.arm == "TB"].set_index("cache_id").loc[ts.index]
    d = ts.nsd_3 - tb.nsd_3
    lo, hi = boot_ci(d, RULE["bootstrap"], RULE["seed"])
    h_ts = int((ts.dice_3 < ts.dice_0 - 1e-9).sum()); h_tb = int((tb.dice_3 < tb.dice_0 - 1e-9).sum())
    v = dict(n=len(ts), mean_dnsd=float(d.mean()), ci95=[lo, hi], harmed_TS=h_ts, harmed_TB=h_tb,
             mean_dnsd_ok=bool(d.mean() >= RULE["min_mean_dnsd"]), ci_ok=bool(lo > RULE["ci_lo_gt"]),
             harm_ok=bool(h_ts <= h_tb))
    v["advance_to_reader_study"] = bool(v["mean_dnsd_ok"] and v["ci_ok"] and v["harm_ok"])
    return v


def implementation_checks(df, lg):
    """v2 checks: current-click SAM arms send one point; no retry of an unchanged-mask attempt; edits within the bound;
    every arm starts from the same mask; the editor functions take no expert mask."""
    import inspect
    sam = lg[lg.action.isin(["sam", "sam_hist", "hybrid_sam_positive"])]
    cur = sam[sam.history_mode == "current"] if "history_mode" in sam else sam.iloc[0:0]
    retries = 0
    for _, g in lg.dropna(subset=["p"]).groupby(["cache_id", "arm", "variant"]):
        prev = None
        for _, r in g.sort_values("step").iterrows():
            key = (str(r.p), r.sign)
            if prev is not None and key == prev[0] and not prev[1]:
                retries += 1
            prev = (key, bool(r.effective))
    edited = sam[sam.change_vox > 0] if len(sam) else sam
    no_gt = all("gt" not in inspect.signature(f).parameters
                for f in (editor_blob_rule, brush, ClickEditor.propose, mask_to_low))
    c = dict(current_click_arms_send_one_point=bool((cur.n_points == 1).all()) if len(cur) else None,
             retries_of_unchanged_attempts=int(retries),
             max_edit_dist_mm=float(edited.max_edit_dist_mm.max()) if len(edited) and "max_edit_dist_mm" in edited else None,
             edits_within_bound=bool((edited.max_edit_dist_mm <= CFG["edit_radius_mm"] + 1e-6).all())
             if len(edited) and "max_edit_dist_mm" in edited else None,
             same_start_mask_all_arms=bool((df.groupby(["cache_id", "variant"]).dice_0.nunique() == 1).all()),
             editor_functions_take_no_expert_mask=no_gt)
    c["all_pass"] = bool(c["current_click_arms_send_one_point"] is not False and c["retries_of_unchanged_attempts"] == 0
                         and c["edits_within_bound"] is not False and c["same_start_mask_all_arms"] and no_gt)
    return c


def report(df, lg, rd, budget, smoke, smoke_ids=frozenset()):
    summ = summarise(df[df.in_queue], "queue_108", budget) + summarise(df[df.failure], "failures_137", budget)
    for t, g in df[df.in_queue].groupby("start_type"):
        summ += summarise(g, f"queue_{t}", budget)
    s = pd.DataFrame(summ)
    s.to_csv(rd / "summary.csv", index=False)
    act = lg.groupby(["arm", "variant", "action"]).size().unstack(fill_value=0)
    act.to_csv(rd / "click_actions.csv")
    def tech_of(sm):
        if not len(sm):
            return {}
        return dict(sam_edits=len(sm), roundtrip_dice_median=float(sm.roundtrip_dice.median()),
                    roundtrip_dice_min=float(sm.roundtrip_dice.min()), click_back_err_mm_max=float(sm.click_back_err_mm.max()),
                    p_at_positive_click_median=float(sm[sm.sign == 1].p_at_click.median()) if (sm.sign == 1).any() else None,
                    p_at_negative_click_median=float(sm[sm.sign == 0].p_at_click.median()) if (sm.sign == 0).any() else None,
                    sam_no_change=int((sm.change_vox == 0).sum()), seconds_per_sam_edit=float(sm.seconds.mean()),
                    negative_compliance=float((sm[sm.sign == 0].p_at_click < 0.5).mean()) if (sm.sign == 0).any() else None,
                    positive_compliance=float((sm[sm.sign == 1].p_at_click > 0.5).mean()) if (sm.sign == 1).any() else None,
                    no_change_negative=int(((sm.sign == 0) & (sm.change_vox == 0)).sum()),
                    no_change_positive=int(((sm.sign == 1) & (sm.change_vox == 0)).sum()),
                    max_addition_ml=float(sm[sm.sign == 1].change_vox.max() / 1000) if (sm.sign == 1).any() else None,
                    max_raw_addition_ml=float(sm[sm.sign == 1].raw_change_vox.max() / 1000)
                    if "raw_change_vox" in sm and (sm.sign == 1).any() else None,
                    duplicate_prompts=int(sm.duplicate_prompt.sum()) if "duplicate_prompt" in sm else None,
                    mask_tau=CFG["mask_tau"], mask_logit=CFG["mask_logit"])
    tech = {}
    for arm, act_name in (("TS", "sam"), ("TS_hist", "sam_hist"), ("TS_hybrid", "hybrid_sam_positive")):
        sm = lg[(lg.arm == arm) & (lg.action == act_name)]
        if len(sm):
            tech[arm] = {"pooled": tech_of(sm), **{v: tech_of(sm[sm.variant == v]) for v in VARIANTS}}
    rep_pts = lg.dropna(subset=["p"]).assign(pkey=lambda d: d.p.astype(str) + "|" + d.sign.astype(str))
    tech["repeated_point_sign_attempts"] = int(rep_pts.duplicated(["cache_id", "arm", "variant", "pkey"]).sum())
    if "version" in lg and (lg.version == "v2").any():
        tech["implementation_checks"] = implementation_checks(df, lg)
    (rd / "technical.json").write_text(json.dumps(tech, indent=1))
    verdict = None
    if budget >= 3 and not smoke:
        verdict = primary_verdict(df)
        if smoke_ids:
            verdict["sensitivity_excluding_smoke_patients"] = primary_verdict(df, smoke_ids)
        for arm in ("TS_hist", "TS_hybrid"):                                   # exploratory, same contrast vs TB
            if arm in set(df.arm):
                verdict[f"exploratory_{arm}_vs_TB"] = primary_verdict(df.assign(arm=df.arm.replace(
                    {"TS": "_TS_primary", arm: "TS"})))
    (rd / "decision_rule_check.json").write_text(json.dumps(verdict, indent=1))
    pd.set_option("display.width", 230)
    cols = ["population", "arm", "variant", "clicks", "n", "dice", "nsd", "rescued_05", "newly_harmed_05", "below_start",
            "never_reach_08_by_5"]
    main = s[s.population.isin(["queue_108", "failures_137"])]
    print("\n" + main[cols].round(4).to_string(index=False))
    print("\nby starting type (queue):\n" + s[~s.population.isin(["queue_108", "failures_137"]) & (s.variant == "always")]
          [cols].round(4).to_string(index=False))
    print("\nclick actions:\n" + act.to_string())
    print("\ntechnical:", json.dumps(tech, indent=1))
    if smoke:
        ic = tech.get("implementation_checks")
        if ic is not None:
            print("\nIMPLEMENTATION CHECKS:", "PASS" if ic["all_pass"] else "FAIL", json.dumps(ic))
        print("\nSMOKE RUN: technical check only; the advancement rule is not evaluated. Check overlays/")
    elif verdict:
        print("\nfrozen pilot rule (queue 108, always-accept, 3 clicks, TS vs TB; sensitivity and exploratory arms "
              "reported, not decisive):", json.dumps(verdict, indent=1))
        print("\nVERDICT:", "ADVANCE TO A READER STUDY" if verdict["advance_to_reader_study"] else
              "do not advance (SAM does not add enough over toggle + brush)")
    print(f"-> {rd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["parity", "calibrate", "replay", "smoke", "full"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--train_cases", required=True, help="S3g cases/ (OOF component maps)")
    ap.add_argument("--s3n_dir", required=True, help="S3n output (train_oof_review.csv)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--budget", type=int, default=CFG["budget"])
    ap.add_argument("--budget_frac", type=float, default=CFG["budget_frac"], help="review queue share (frozen 0.10)")
    ap.add_argument("--smoke_n", type=int, default=12)
    ap.add_argument("--version", choices=["v1", "v2"], default="v2", help="v1 = original simulator (record only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    if a.step == "parity":
        step_parity(a, root, sp, out)
    elif a.step in ("calibrate", "replay"):
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt and --turbo_ckpt are required")
        (step_calibrate if a.step == "calibrate" else step_replay)(a, root, sp, out)
    else:
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt and --turbo_ckpt are required")
        step_run(a, root, sp, out, a.step == "smoke")


if __name__ == "__main__":
    main()
