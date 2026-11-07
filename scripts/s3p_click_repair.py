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

Steps: --step parity (CPU) -> --step smoke (GPU, 12 chosen patients + overlays) -> --step full (GPU)
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
ARMS = ("T", "TB", "TS")
VARIANTS = ("always", "oracle")
CFG = dict(budget=5, brush_mm=5, mask_logit=4.0, sam_spacing=1.5, sam_size=128, report_steps=[0, 1, 3, 5],
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
def next_click(m, gt):
    """Simulated reader (uses the expert mask): (point, sign, region voxels) or None if the mask is perfect.
    Largest connected error region (missed tumour or wrongly included tissue; ties -> missed tumour first),
    clicked at its deepest interior point. sign 1 = positive (add), 0 = negative (remove)."""
    best = None
    for sign, err in ((1, gt & ~m), (0, m & ~gt)):
        if not err.any():
            continue
        lab = label26(err)
        size = np.bincount(lab.ravel())[1:]
        k = int(np.argmax(size)) + 1
        if best is None or size[k - 1] > best[0]:
            best = (int(size[k - 1]), sign, lab == k)
    if best is None:
        return None
    size, sign, comp = best
    return deepest_point(comp), sign, size


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


def brush(m, p, sign, r):
    out = m.copy()
    sl, b = ball_slice(m.shape, p, r)
    if sign == 1:
        out[sl] |= b
    else:
        out[sl] &= ~b
    return out


def mask_to_low(mc, L=4.0):
    """128^3 bool crop -> (1,1,32,32,32) logits L*(2f-1), f = fraction of tumour voxels in each 4^3 block."""
    import torch
    import torch.nn.functional as F
    f = F.avg_pool3d(torch.from_numpy(mc.astype(np.float32))[None, None], 4)
    return L * (2 * f - 1)


def low_to_mask(low):
    from cids.models.sam_lora import upsample
    return (upsample(low.float())[0, 0] > 0).numpy()


class ClickEditor:
    """REAL-click SAM inference: takes click coordinates and the current mask, never the expert mask.
    predictor(x (1,1,128^3), coords (1,n,3) | None, labels (1,n) | None, low (1,1,32^3)) -> prob (128^3) numpy."""

    def __init__(self, predictor, L=4.0):
        self.predict, self.L = predictor, L

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

    def propose(self, m, clicks):
        """clicks = [(point_1mm, sign), ...]; the last one is new. Returns (edited mask, info)."""
        import torch
        from cids.data.cache import crop_pad, znorm
        p, s = clicks[-1]
        n = CFG["sam_size"]
        c15 = np.floor((np.asarray(p, float) + 0.5) / CFG["sam_spacing"]).astype(int)
        vc, origin = crop_pad(self.post, c15)
        fc, _ = crop_pad(self.fg, c15)
        x = torch.from_numpy(znorm(vc, fc)[None, None])
        mc, _ = crop_pad(self._mask15(m).astype(np.uint8), c15)
        low = mask_to_low(mc > 0, self.L)
        pts, labs = [], []
        for q, sg in clicks:
            cq = M3.to_crop(q, origin)
            if (cq >= 0).all() and (cq < n).all():
                pts.append(cq); labs.append(sg)
        coords = torch.tensor(np.asarray(pts, np.float32)[None])
        labels = torch.tensor(np.asarray(labs)[None], dtype=torch.long)
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
                    n_points=len(pts), origin15=np.asarray(origin).tolist())
        D = np.zeros(m.shape, bool)
        D[ext] = (S[ext] & ~m[ext]) if s == 1 else (m[ext] & ~S[ext])
        if not D[tuple(p)]:
            info["change_vox"] = 0
            return m.copy(), info
        lab = label26(D[ext])
        comp = np.zeros(m.shape, bool)
        comp[ext] = lab == lab[tuple(np.asarray(p) - a)]
        info["change_vox"] = int(comp.sum())
        return (m | comp) if s == 1 else (m & ~comp), info


def sam_predictor(sam, box_mod, dev, amp):
    import torch
    from cids.models.sam_lora import decode, upsample

    def f(x, coords, labels, low):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            emb = sam.image_encoder(x.float().to(dev)).float()
            lo = decode(sam, box_mod, emb, coords.to(dev), labels.to(dev), None, low.to(dev).float())
            return torch.sigmoid(upsample(lo.float()))[0, 0].cpu().numpy()
    return f


def load_predictor(sam_ckpt, turbo, random_weights):
    r = M3.load_runner(sam_ckpt, turbo, random_weights)
    return sam_predictor(r.sam, r.box, r.dev, r.amp)


# ----------------------------------------------------------------------------------------------- one patient, one arm
def run_arm(arm, variant, A, gt, lab, kept0, cand_ids, touches, editor, budget, log, cid, overlay=None, img=None):
    m, kept, sam_clicks = A.copy(), set(kept0), []
    dices = [dice(m, gt)]
    masks = {0: m.copy()}
    for step in range(1, budget + 1):
        t0 = time.time()
        c = next_click(m, gt)
        row = dict(cache_id=cid, arm=arm, variant=variant, step=step)
        if c is None:
            row.update(action="no_error", effective=False)
            log.append(row); dices.append(dices[-1]); masks[step] = m.copy()
            continue
        p, sign, size = c
        row.update(p=list(p), sign=sign, region_vox=size, region_type="missed" if sign else "extra")
        act = toggle_action(p, sign, kept, cand_ids, lab, touches)
        info, new_kept = {}, set(kept)
        if act is not None:
            kind, k = act
            new = (m | (lab == k)) if kind == "toggle_on" else (m & ~(lab == k))
            (new_kept.add if kind == "toggle_on" else new_kept.discard)(k)
            action = kind
        elif arm == "T":
            new, action = m.copy(), "ineffective"
        elif arm == "TB":
            new, action = brush(m, p, sign, CFG["brush_mm"]), "brush"
        else:
            sam_clicks.append((p, sign))
            new, info = editor.propose(m, sam_clicks)
            action = "sam"
            if overlay is not None and step <= 3:
                overlay(cid, arm, variant, step, img, gt, m, new, p, sign)
        accepted = True
        if variant == "oracle" and dice(new, gt) < dice(m, gt) - 1e-12:
            accepted = False
            if action == "sam":
                sam_clicks.pop()                       # an undone click is not kept as a prompt
        if accepted:
            effective = not np.array_equal(new, m)
            m, kept = new, new_kept
        else:
            effective = False
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
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any SAM call
    pp, q, fails, pop = population(a)
    ids = choose_smoke(par, a.smoke_n) if smoke else pop
    table = pd.read_csv(Path(a.s3k_out).expanduser() / "train_oof_table.csv")
    paths = dict(require_cases(sp, root, "train"))
    cdir = Path(a.train_cases).expanduser()
    editor = ClickEditor(load_predictor(a.sam_ckpt, None if a.random_weights else a.turbo_ckpt, a.random_weights),
                         CFG["mask_logit"])
    rd = out / ("smoke" if smoke else "full"); rd.mkdir(exist_ok=True)
    od = rd / "overlays" if smoke else None
    if od is not None:
        od.mkdir(exist_ok=True)
    res, log, t0 = [], [], time.time()
    qset = set(q.cache_id)
    for i, cid in enumerate(ids):
        img, gt, lab, kept, cand, touches, A = load_patient(cid, paths[cid], cdir / f"{cid}.npz",
                                                            pp.loc[cid, "B0_rad_subset"], table)
        editor.prepare(img)
        for arm in ARMS:
            for variant in VARIANTS:
                ov = overlay_fn(od) if (od is not None and arm == "TS") else None
                r = run_arm(arm, variant, A, gt, lab, kept, cand, touches, editor, a.budget, log, cid, ov, img)
                res.append(dict(cache_id=cid, arm=arm, variant=variant, in_queue=cid in qset, failure=cid in fails,
                                start_type="empty" if not A.any() else ("wrong_target" if r["dice_0"] < 0.1 else
                                                                         "poor_boundary" if r["dice_0"] < 0.5 else "ok"),
                                **r))
        if (i + 1) % 20 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    df, lg = pd.DataFrame(res), pd.DataFrame(log)
    df.to_csv(rd / "per_patient.csv", index=False); lg.to_csv(rd / "clicks.csv", index=False)
    report(df, lg, rd, a.budget, smoke)


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


def report(df, lg, rd, budget, smoke):
    summ = summarise(df[df.in_queue], "queue_108", budget) + summarise(df[df.failure], "failures_137", budget)
    for t, g in df[df.in_queue].groupby("start_type"):
        summ += summarise(g, f"queue_{t}", budget)
    s = pd.DataFrame(summ)
    s.to_csv(rd / "summary.csv", index=False)
    act = lg.groupby(["arm", "variant", "action"]).size().unstack(fill_value=0)
    act.to_csv(rd / "click_actions.csv")
    tech = {}
    sm = lg[lg.action == "sam"]
    if len(sm):
        tech = dict(sam_edits=len(sm), roundtrip_dice_median=float(sm.roundtrip_dice.median()),
                    roundtrip_dice_min=float(sm.roundtrip_dice.min()), click_back_err_mm_max=float(sm.click_back_err_mm.max()),
                    p_at_positive_click_median=float(sm[sm.sign == 1].p_at_click.median()) if (sm.sign == 1).any() else None,
                    p_at_negative_click_median=float(sm[sm.sign == 0].p_at_click.median()) if (sm.sign == 0).any() else None,
                    sam_no_change=int((sm.change_vox == 0).sum()), seconds_per_sam_edit=float(sm.seconds.mean()))
    (rd / "technical.json").write_text(json.dumps(tech, indent=1))
    verdict = None
    if budget >= 3 and not smoke:
        g = df[df.in_queue & (df.variant == "always")]
        ts = g[g.arm == "TS"].set_index("cache_id"); tb = g[g.arm == "TB"].set_index("cache_id").loc[ts.index]
        d = ts.nsd_3 - tb.nsd_3
        lo, hi = boot_ci(d, RULE["bootstrap"], RULE["seed"])
        h_ts = int((ts.dice_3 < ts.dice_0 - 1e-9).sum()); h_tb = int((tb.dice_3 < tb.dice_0 - 1e-9).sum())
        verdict = dict(n=len(ts), mean_dnsd=float(d.mean()), ci95=[lo, hi], harmed_TS=h_ts, harmed_TB=h_tb,
                       mean_dnsd_ok=bool(d.mean() >= RULE["min_mean_dnsd"]), ci_ok=bool(lo > RULE["ci_lo_gt"]),
                       harm_ok=bool(h_ts <= h_tb))
        verdict["advance_to_reader_study"] = bool(verdict["mean_dnsd_ok"] and verdict["ci_ok"] and verdict["harm_ok"])
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
        print("\nSMOKE RUN: technical check only; the advancement rule is not evaluated. Check overlays/")
    elif verdict:
        print("\nfrozen pilot rule (queue 108, always-accept, 3 clicks, TS vs TB):", json.dumps(verdict, indent=1))
        print("\nVERDICT:", "ADVANCE TO A READER STUDY" if verdict["advance_to_reader_study"] else
              "do not advance (SAM does not add enough over toggle + brush)")
    print(f"-> {rd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["parity", "smoke", "full"], required=True)
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    if a.step == "parity":
        step_parity(a, root, sp, out)
    else:
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt and --turbo_ckpt are required")
        step_run(a, root, sp, out, a.step == "smoke")


if __name__ == "__main__":
    main()
