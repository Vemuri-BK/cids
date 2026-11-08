"""S3u: one-click tumour masks for radiomic biomarkers -- does a single click (with or without SAM) give tumour
measurements as close to the expert's as the automatic pipeline, and do they predict pCR as well?

Development cohort: the 120 VALIDATION patients only (SAM-S2b was never trained on them; test refused).

Mask sources per patient (1 mm, whole-patient mask)
  E  expert mask                                                     (reference)
  B  B0_rad automatic mask (kept S3f candidates, S3k selection)      (automatic pipeline)
  C  click -> the SegResNet candidate under the click, no SAM        (non-SAM interactive control)
  S  SAM-S2b, one positive click, zero mask prompt                   (SAM alone)
  H  click inside a SegResNet candidate -> that candidate; otherwise -> S   (one-click tool)
  Click = simulated clinician click: the deepest interior voxel of the largest expert component (ties -> nearest
  to its centroid, then lexicographic; s3m.geometric_point). The click is the only use of the expert mask.
  SAM: 128^3 post1 crop at 1.5 mm centred on the click (pipeline preprocessing), point at the click, decoder
  output (logit > 0 after trilinear upsampling, pasted back to 1 mm, > 0.5), the 26-connected component that
  contains the click (else the component nearest to it).

Features (the same code for every source, on the whole patient mask): volume_ml, diameter_mm (largest distance
  between two mask voxels), enh_mean (mean post1 - pre inside), and cids.radiomics shape / contrast / texture
  (RAD_SHAPE + RAD_CONTRAST + RAD_TEXTURE) with the mask as the only component. 17 features.

Frozen analysis (decision_rule.json is written before any mask is built)
  Agreement  Lin's CCC of each feature with E, on patients where every source is non-empty; summary = median
             CCC over the 17 features; patient bootstrap (2,000) CIs; paired differences S-B, S-C, H-C, C-B.
  Failures   gross failures = Dice < 0.1 vs E (empty mask counts), all 120 patients.
  pCR        EXPLORATORY (~36 responders): logistic regression on each source's features, repeated stratified
             5-fold CV (10 repeats), AUC mean / sd; empty-mask features imputed with the training-fold median.
  Claims     "one click (SAM) removes gross failures" if failures(S) < failures(B);
             "SAM preserves biomarkers better than the automatic pipeline" if CI of median-CCC(S - B) > 0;
             "SAM beats the non-SAM click control" only if CI of median-CCC(S - C) > 0.
             H differs from C only where the click lands outside every candidate (expected rare on val: E0 found
             every val tumour among the candidates); the number of such patients is reported.

Steps
  --step masks    GPU: build E/B/C/S/H, Dice vs E, features -> per_patient.csv, features.csv
  --step analyse  CPU: agreement, failures, pCR (needs --clinical) -> agreement_*.csv, paired.csv, pcr_auc.csv
Example (server)
  python -u scripts/s3u_click_biomarkers.py --step masks --cache_root ~/data \\
      --s3k_out ~/CIDS/outputs/s3k_radiomics --val_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases \\
      --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth \\
      --out ~/CIDS/outputs/s3u_click_biomarkers
  python scripts/s3u_click_biomarkers.py --step analyse --clinical <clinical_and_imaging_info.xlsx> \\
      --s3k_out ~/CIDS/outputs/s3k_radiomics --out ~/CIDS/outputs/s3u_click_biomarkers
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
from cids.radiomics import RAD_CONTRAST, RAD_SHAPE, RAD_TEXTURE, candidate_features, lins_ccc  # noqa: E402

SOURCES = ["E", "B", "C", "S", "H"]
FEATS = ["volume_ml", "diameter_mm", "enh_mean"] + RAD_SHAPE + RAD_CONTRAST + RAD_TEXTURE
STRUCT = np.ones((3, 3, 3), bool)
RULE = dict(
    cohort="120 validation patients (SAM-S2b not trained on them); test refused",
    sources=dict(E="expert", B="B0_rad automatic", C="click -> SegResNet candidate (no SAM)",
                 S="SAM-S2b one click", H="click in candidate -> candidate, else SAM"),
    click="deepest interior voxel of the largest expert component (simulated clinician click)",
    features=FEATS, agreement="Lin's CCC vs E per feature on patients where all sources are non-empty; "
    "summary = median over features; patient bootstrap 2000",
    failures="Dice < 0.1 vs E, empty counts, all patients",
    pcr="exploratory: logistic regression, repeated stratified 5-fold x 10, AUC",
    claims=["one click (SAM) removes gross failures: failures(S) < failures(B)",
            "SAM preserves biomarkers better than automatic: CI lower bound of median-CCC(S - B) > 0",
            "SAM beats the non-SAM click control: CI lower bound of median-CCC(S - C) > 0"],
    bootstrap=dict(n=2000, seed=0))


# --------------------------------------------------------------------------------------------- masks
def click_point(gt):
    """Deepest interior voxel of the largest 26-connected expert component; None if the mask is empty."""
    if not gt.any():
        return None
    lab, n = ndimage.label(gt, structure=STRUCT)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return M.geometric_point(big)


def component_at(mask, p):
    """26-connected component of mask containing p; else the component nearest to p; empty if mask empty."""
    if not mask.any():
        return np.zeros_like(mask)
    lab, n = ndimage.label(mask, structure=STRUCT)
    k = lab[tuple(p)]
    if k == 0:
        idx = np.argwhere(lab > 0)
        k = lab[tuple(idx[np.argmin(((idx - np.asarray(p)) ** 2).sum(1))])]
    return lab == k


def sam_one_click(img, p, sam, box_mod, dev, amp):
    """SAM-S2b with one positive click at p (1 mm voxel) -> 1 mm bool mask (component at the click)."""
    import torch
    import s3a_seg2sam as s3
    from cids.models.sam_lora import decode, upsample
    img15, _ = s3.resample(img, np.zeros(img.shape[1:], np.uint8))
    shape15 = img15.shape[1:]
    c15 = np.clip(M.to_crop(p, np.zeros(3, int)), 0, np.asarray(shape15) - 1)
    post, fg = img15[1], (img15[1] > 0).astype(np.uint8)
    vc, origin = s3.crop_pad(post, c15)
    fc, _ = s3.crop_pad(fg, c15)
    crop = s3.znorm(vc, fc)
    pc = np.clip(c15 - origin, 0, s3.SAM_SIZE - 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        x = torch.from_numpy(np.ascontiguousarray(crop[None, None])).float().to(dev)
        emb = sam.image_encoder(x).float()
        co = torch.tensor(np.asarray([pc], np.float32)[:, None], device=dev)
        lb = torch.ones(1, 1, dtype=torch.long, device=dev)
        low = decode(sam, box_mod, emb, co, lb, None, None)
        prob = torch.sigmoid(upsample(low.float()))[0, 0].cpu().numpy()
    up = M.paste_back(prob, origin, shape15, img.shape[1:])
    return component_at(up > 0.5, p)


def build_masks(img, gt, lab, kept, sam_fn):
    """All five masks for one patient + bookkeeping."""
    p = click_point(gt)
    B = np.isin(lab, kept) if kept else np.zeros(gt.shape, bool)
    if p is None:
        z = np.zeros(gt.shape, bool)
        return dict(E=gt.astype(bool), B=B, C=z, S=z, H=z), dict(click=None, k_click=0, h_used_sam=False)
    k = int(lab[p])
    C = lab == k if k > 0 else np.zeros(gt.shape, bool)
    S = sam_fn(img, p)
    H = C if C.any() else S
    return dict(E=gt.astype(bool), B=B, C=C, S=S, H=H), dict(click=list(p), k_click=k, h_used_sam=not C.any())


def diameter_mm(mask):
    pts = np.argwhere(mask).astype(np.float64)
    if len(pts) < 2:
        return 0.0 if len(pts) else np.nan
    try:
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    except Exception:                                                  # flat / degenerate: all points
        pts = pts if len(pts) <= 3000 else pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    return float(d.max())


def mask_features(img, mask):
    if not mask.any():
        return dict.fromkeys(FEATS, np.nan)
    out = dict(volume_ml=float(mask.sum()) / 1000, diameter_mm=diameter_mm(mask),
               enh_mean=float((img[1] - img[0])[mask].mean()))
    out.update(candidate_features(img, mask.astype(np.int32), 1))
    return {f: out[f] for f in FEATS}


def step_masks(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    (out / "decision_rule.json").write_text(json.dumps(RULE, indent=1))          # frozen before any mask
    pp = M.load_selection(Path(a.s3k_out).expanduser())
    pp = pp[pp.split == "val"].set_index("cache_id")
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    cases = require_cases(pd.read_csv(a.splits), root, "val")
    cdir = Path(a.val_cases).expanduser()
    cdir = cdir if any(cdir.glob("*.npz")) else next(cdir.rglob("*.npz")).parent
    if a.limit:
        cases = cases[:a.limit]
    runner = M.load_runner(a.sam_ckpt, a.turbo_ckpt, a.random_weights)
    if a.random_weights:
        import torch
        torch.manual_seed(0)

    def sam_fn(img, p):
        return sam_one_click(img, p, runner.sam, runner.box, runner.dev, runner.amp)

    rows, feats, t0 = [], [], time.time()
    for i, (cid, path) in enumerate(cases):
        if cid not in pp.index:
            raise SystemExit(f"{cid}: no B0_rad selection in S3k per_patient.csv")
        img, gt = load_case(path)
        gt = gt > 0
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        t1 = time.time()
        masks, info = build_masks(img, gt, lab, M.kept_ids(pp.loc[cid, "B0_rad_subset"]), sam_fn)
        d_b = M.dice_count(masks["B"], gt)
        if abs(d_b - float(pp.loc[cid, "B0_rad_dice"])) > 1e-6:
            raise SystemExit(f"PARITY FAILED {cid}: rebuilt B0_rad Dice {d_b} vs S3k {pp.loc[cid, 'B0_rad_dice']}")
        r = dict(cache_id=cid, collection=cid.split("_")[0].upper(), seconds=time.time() - t1, **info)
        for s in SOURCES:
            r[f"{s}_dice"] = M.dice_count(masks[s], gt)
            r[f"{s}_ml"] = float(masks[s].sum()) / 1000
            feats.append(dict(cache_id=cid, source=s, **mask_features(img, masks[s])))
        rows.append(r)
        if (i + 1) % 20 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)} patients | {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_patient.csv", index=False)
    pd.DataFrame(feats).to_csv(out / "features.csv", index=False)
    print("PARITY OK: B rebuilt = S3k B0_rad on every patient")
    print(df[[f"{s}_dice" for s in SOURCES]].describe().loc[["mean", "50%"]].round(4).to_string())
    print(f"click outside every candidate (H used SAM): {int(df.h_used_sam.sum())} / {len(df)}")
    print("MASKS DONE -> run --step analyse")


# --------------------------------------------------------------------------------------------- analysis
def agreement(F, pats, sources=None):
    """Per-feature CCC of every source vs E on the patient list pats -> {source: {feature: ccc}}."""
    E = F["E"].loc[pats]
    res = {}
    for s in (sources or SOURCES[1:]):
        X = F[s].loc[pats]
        res[s] = {}
        for f in FEATS:
            ok = E[f].notna() & X[f].notna()
            res[s][f] = lins_ccc(E[f][ok], X[f][ok]) if ok.sum() > 2 and E[f][ok].std() > 0 else np.nan
    return res


def median_ccc(res):
    return {s: float(np.nanmedian(list(v.values()))) for s, v in res.items()}


def step_analyse(a, out):
    rng = np.random.default_rng(RULE["bootstrap"]["seed"])
    pp = pd.read_csv(out / "per_patient.csv")
    ft = pd.read_csv(out / "features.csv")
    F = {s: g.set_index("cache_id")[FEATS] for s, g in ft.groupby("source")}
    nonempty = pp[[f"{s}_ml" for s in SOURCES]].gt(0).all(axis=1)
    pats = pp.cache_id[nonempty].to_numpy()
    res = agreement(F, pats)
    med = median_ccc(res)
    pf = pd.DataFrame(res).rename_axis("feature")
    pf.to_csv(out / "agreement_per_feature.csv")
    pairs = [("S", "B"), ("S", "C"), ("H", "C"), ("C", "B")]
    boots = []
    for _ in range(RULE["bootstrap"]["n"]):
        b = rng.choice(pats, len(pats), replace=True)
        boots.append(median_ccc(agreement(F, b)))
    bt = pd.DataFrame(boots)
    summ = [dict(source=s, median_ccc=med[s], ci95_lo=float(bt[s].quantile(0.025)),
                 ci95_hi=float(bt[s].quantile(0.975)), n_patients=len(pats)) for s in SOURCES[1:]]
    paired = []
    for x, y in pairs:
        d = bt[x] - bt[y]
        paired.append(dict(comparison=f"{x} - {y}", diff=med[x] - med[y], ci95_lo=float(d.quantile(0.025)),
                           ci95_hi=float(d.quantile(0.975)), supported=bool(d.quantile(0.025) > 0)))
    fail = [dict(source=s, mean_dice=float(pp[f"{s}_dice"].mean()), median_dice=float(pp[f"{s}_dice"].median()),
                 gross_failures=int((pp[f"{s}_dice"] < 0.1).sum()), empty=int((pp[f"{s}_ml"] == 0).sum()))
            for s in SOURCES[1:]]
    pd.DataFrame(summ).to_csv(out / "agreement_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(fail).to_csv(out / "failures.csv", index=False)
    fb = {r["source"]: r["gross_failures"] for r in fail}
    pm = {r["comparison"]: r for r in paired}
    claims = dict(one_click_sam_removes_gross_failures=bool(fb["S"] < fb["B"]),
                  sam_better_biomarkers_than_automatic=pm["S - B"]["supported"],
                  sam_beats_non_sam_click_control=pm["S - C"]["supported"],
                  patients_where_H_used_sam=int(pp.h_used_sam.sum()))
    pcr = pcr_auc(F, pp, a.clinical) if a.clinical else None
    if pcr is not None:
        pcr.to_csv(out / "pcr_auc.csv", index=False)
    (out / "summary.json").write_text(json.dumps(dict(claims=claims, n=len(pp), n_all_nonempty=len(pats)),
                                                 indent=1))
    pd.set_option("display.width", 200)
    print("Dice vs expert / gross failures (all patients):")
    print(pd.DataFrame(fail).round(4).to_string(index=False))
    print(f"\nagreement with the expert (median CCC over {len(FEATS)} features, {len(pats)} patients):")
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print("\npaired differences:")
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    print("\nper-feature CCC:")
    print(pf.round(3).to_string())
    if pcr is not None:
        print("\npCR (EXPLORATORY):")
        print(pcr.round(3).to_string(index=False))
    print("\nCLAIMS:", json.dumps(claims, indent=1))


def nearest_blob(lab, p):
    """POST-HOC control: the candidate under the click, else the candidate nearest to it (Euclidean, voxels=mm).
    Returns (mask, distance_mm); empty mask and nan if there are no candidates."""
    k = int(lab[tuple(p)])
    if k > 0:
        return lab == k, 0.0
    idx = np.argwhere(lab > 0)
    if not len(idx):
        return np.zeros(lab.shape, bool), np.nan
    d = np.sqrt(((idx - np.asarray(p)) ** 2).sum(1))
    j = int(np.argmin(d))
    return lab == lab[tuple(idx[j])], float(d[j])


def step_posthoc(a, out):
    """Two checks added AFTER seeing the S3u results (labelled post-hoc, not part of the frozen rule):
    1. agreement on ALL patients for the always-non-empty sources (B, S, H) + C_near;
    2. a stronger non-SAM control C_near (nearest SegResNet candidate when the click misses every candidate),
       compared with H on the patients where H used SAM."""
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    pp = pd.read_csv(out / "per_patient.csv")
    ft = pd.read_csv(out / "features.csv")
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(pd.read_csv(a.splits), root, "val"))
    cdir = Path(a.val_cases).expanduser()
    cdir = cdir if any(cdir.glob("*.npz")) else next(cdir.rglob("*.npz")).parent
    rows, extra = [], []
    for _, r in pp.iterrows():
        cid = r.cache_id
        img, gt = load_case(paths[cid])
        gt = gt > 0
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        p = click_point(gt)
        saved = json.loads(r.click) if isinstance(r.click, str) else None
        if p is None or saved is None or list(p) != saved:
            raise SystemExit(f"{cid}: click differs from the masks step ({p} vs {saved})")
        m, dist = nearest_blob(lab, p)
        rows.append(dict(cache_id=cid, h_used_sam=bool(r.h_used_sam), Cn_dice=M.dice_count(m, gt), Cn_dist_mm=dist,
                         Cn_ml=float(m.sum()) / 1000, C_dice=r.C_dice, S_dice=r.S_dice, H_dice=r.H_dice,
                         B_dice=r.B_dice))
        extra.append(dict(cache_id=cid, source="Cn", **mask_features(img, m)))
    ph = pd.DataFrame(rows)
    ph.to_csv(out / "posthoc_per_patient.csv", index=False)
    ft = pd.concat([ft, pd.DataFrame(extra)], ignore_index=True)
    F = {s: g.set_index("cache_id")[FEATS] for s, g in ft.groupby("source")}
    srcs = ["B", "Cn", "S", "H"]
    ok = ph.set_index("cache_id").Cn_ml.gt(0)
    pats = pp.cache_id[pp[["B_ml", "S_ml", "H_ml"]].gt(0).all(axis=1).to_numpy() &
                       ok.reindex(pp.cache_id).to_numpy()].to_numpy()
    med = median_ccc(agreement(F, pats, srcs))
    rng = np.random.default_rng(RULE["bootstrap"]["seed"])
    bt = pd.DataFrame([median_ccc(agreement(F, rng.choice(pats, len(pats), replace=True), srcs))
                       for _ in range(RULE["bootstrap"]["n"])])
    summ = [dict(source=s, median_ccc=med[s], ci95_lo=float(bt[s].quantile(0.025)),
                 ci95_hi=float(bt[s].quantile(0.975)), n_patients=len(pats)) for s in srcs]
    paired = []
    for x, y in [("H", "B"), ("H", "Cn"), ("Cn", "B")]:
        d = bt[x] - bt[y]
        paired.append(dict(comparison=f"{x} - {y}", diff=med[x] - med[y], ci95_lo=float(d.quantile(0.025)),
                           ci95_hi=float(d.quantile(0.975)), supported=bool(d.quantile(0.025) > 0)))
    sam = ph[ph.h_used_sam]
    fail = [dict(source=s, mean_dice=float(ph[f"{s}_dice"].mean()), gross_failures=int((ph[f"{s}_dice"] < 0.1).sum()))
            for s in ("B", "Cn", "S", "H")]
    res = dict(label="POST-HOC (added after seeing the S3u results)",
               rescue_patients=len(sam), H_dice_on_rescue=float(sam.H_dice.mean()) if len(sam) else None,
               Cn_dice_on_rescue=float(sam.Cn_dice.mean()) if len(sam) else None,
               H_better_on_rescue=int((sam.H_dice > sam.Cn_dice).sum()),
               Cn_better_on_rescue=int((sam.Cn_dice > sam.H_dice).sum()))
    pd.DataFrame(summ).to_csv(out / "posthoc_agreement_all.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "posthoc_paired.csv", index=False)
    (out / "posthoc_summary.json").write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 200)
    print("POST-HOC checks (added after seeing the S3u results)\n")
    print("Dice / gross failures, all patients (Cn = nearest SegResNet candidate when the click misses):")
    print(pd.DataFrame(fail).round(4).to_string(index=False))
    print(f"\npatients where the click missed every candidate (H used SAM): {len(sam)}")
    print(sam[["cache_id", "Cn_dist_mm", "Cn_dice", "S_dice", "B_dice"]].round(3).to_string(index=False))
    print(f"\nagreement with the expert, ALL {len(pats)} patients (median CCC over {len(FEATS)} features):")
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print("\npaired differences:")
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    print("\nRESCUE:", json.dumps(res, indent=1))


def pcr_auc(F, pp, clinical, min_class=10):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = Path(clinical).expanduser()
    cl = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    y = cl.assign(cache_id=cl.patient_id.str.lower()).set_index("cache_id").pcr
    y = y.reindex(pp.cache_id).dropna().astype(int)
    if min(y.sum(), len(y) - y.sum()) < min_class:
        print(f"pCR skipped: {int(y.sum())} responders / {len(y)} patients (< {min_class} per class)")
        return None
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    for s in SOURCES:
        X = F[s].loc[y.index].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nan)
        aucs = []
        for tr, te in cv.split(X, y):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=3000))
            m.fit(X[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X[te])[:, 1]))
        rows.append(dict(source=s, auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), n=len(y),
                         responders=int(y.sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["masks", "analyse", "posthoc"], required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3k_out", required=True, help="S3k output (per_patient.csv with the val B0_rad selection)")
    ap.add_argument("--val_cases", default="", help="S3f val component maps (key lab)")
    ap.add_argument("--sam_ckpt", default="")
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--clinical", default="", help="clinical_and_imaging_info.xlsx (pCR), for --step analyse")
    ap.add_argument("--random_weights", action="store_true", help="tests only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "test" in Path(a.val_cases).name.lower():
        raise SystemExit("S3u is a development study on the validation patients: test files are refused")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "masks":
        if not a.val_cases:
            raise SystemExit("--val_cases is required")
        if not a.random_weights and not (a.sam_ckpt and a.turbo_ckpt):
            raise SystemExit("--sam_ckpt (S2b best.pt) and --turbo_ckpt are required")
        step_masks(a, out)
    else:
        if not (out / "per_patient.csv").exists():
            raise SystemExit("run --step masks first")
        if a.step == "posthoc":
            if not a.val_cases:
                raise SystemExit("--val_cases is required")
            step_posthoc(a, out)
        else:
            step_analyse(a, out)


if __name__ == "__main__":
    main()
