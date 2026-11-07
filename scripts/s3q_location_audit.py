"""S3q: location audit of B0_rad wrong picks (development data only) -- would whole-breast anatomy help the jury?

Question: how often did B0_rad keep tissue OUTSIDE the breast when a usable tumour candidate was available inside it?
Answers whether a whole-breast outline (released masks first, SAM-Med3D later) could give the radiomics jury useful
anatomical context, BEFORE any mask is downloaded or SAM is trained. Train out-of-fold patients only; test not read.

Population: the train-OOF patients where B0_rad Dice < 0.1 although at least one candidate touches the tumour and no
kept candidate does ("wrong pick"; 36 in S3k). Ceiling if every wrong pick were fixed: severe failures 60 -> 26.

Two-stage review
  --step images   Stage A (BLIND): per patient a whole-volume enhancement projection (post1 - pre, axial MIP) with every
                  candidate outlined and numbered, plus axial / coronal / sagittal slices through each candidate's centre
                  (full field of view). No expert mask, no jury decision.  -> stageA/<cid>.png
                  annotation_template.csv: one row per candidate; fill `location` with
                      inside_breast | outside_breast | boundary_uncertain   (posterior lesions, axillary tail -> uncertain)
                  Stage B (REVEAL, open only after Stage A is complete): the same views + expert tumour (yellow), the kept
                  candidates marked, single-candidate Dice.  -> stageB/<cid>.png ;  _reveal_key.csv (do not open early)
  --step tally    Reads the filled annotation + the key. Per kept blob: location. Per patient:
                  clear outside-breast error   = every kept blob outside_breast
                  uncertain                    = any kept blob boundary_uncertain (counted separately)
                  rescuable                    = clear outside-breast error AND an inside_breast candidate with single-
                                                 candidate Dice >= 0.1 (also reported: >= 0.5)
                  Practical threshold to proceed: >= 10 rescuable patients (uncertain counted separately). By cohort.

Usage (server)
  python scripts/s3q_location_audit.py --step images --cache_root ~/data --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --train_cases ~/CIDS/outputs/s3g_oof_masks/cases --out ~/CIDS/outputs/s3q_location_audit
  python scripts/s3q_location_audit.py --step tally --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --annotations ~/CIDS/outputs/s3q_location_audit/annotation_filled.csv --out ~/CIDS/outputs/s3q_location_audit
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCATIONS = ("inside_breast", "outside_breast", "boundary_uncertain")
COLOURS = ["#00e5ff", "#ff40ff", "#7cfc00", "#ff9f1a", "#9fa8ff"]


def kept_ids(s):
    if s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() in ("", "nan"):
        return set()
    return {int(float(x)) for x in str(s).split("+")}


def wrong_picks(s3k_out):
    """Patients with B0_rad Dice < 0.1, a tumour-touching candidate, and no kept candidate touching the tumour."""
    t = pd.read_csv(Path(s3k_out) / "train_oof_table.csv")
    pp = pd.read_csv(Path(s3k_out) / "per_patient.csv", dtype={"B0_rad_subset": str})
    pp = pp[pp.split == "train_oof"].set_index("cache_id")
    rows, key = [], []
    for cid, g in t.groupby("cache_id"):
        r = g[g.k >= 0]
        if not len(r):
            continue
        G = float(g.gt_vox.iloc[0])
        kept = kept_ids(pp.loc[cid, "B0_rad_subset"])
        d = float(pp.loc[cid, "B0_rad_dice"])
        touches = r.inter_vox > 0
        if d < 0.1 and touches.any() and not r[r.k.isin(kept)].inter_vox.gt(0).any():
            rows.append(dict(cache_id=cid, cohort=cid.split("_")[0].upper(), b0_rad_dice=d, n_cand=len(r),
                             n_kept=len(kept)))
            for _, c in r.sort_values("rank").iterrows():
                key.append(dict(cache_id=cid, k=int(c.k), kept=int(c.k) in kept, touches_tumour=bool(c.inter_vox > 0),
                                single_dice=float(2 * c.inter_vox / (c.size_vox + G)), size_ml=float(c.size_vox) / 1000))
    return pd.DataFrame(rows), pd.DataFrame(key)


def _norm(a):
    lo, hi = np.percentile(a, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _contour(ax, m2d, colour, lw=1.4, ls="-"):
    if m2d.any():
        ax.contour(m2d.T.astype(float), [0.5], colors=colour, linewidths=lw, linestyles=ls, origin="lower")


def patient_figure(png, cid, img, lab, cand_rows, gt=None, reveal=False):
    """Stage A (reveal=False): candidates only. Stage B: + expert mask and jury decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = img[1] - img[0]
    post = _norm(img[1])
    n = len(cand_rows)
    fig, axes = plt.subplots(n + 1, 3, figsize=(13, 4.2 * (n + 1)), squeeze=False)
    mip = _norm(sub.max(axis=2))
    ax = axes[0, 0]
    ax.imshow(mip.T, cmap="gray", origin="lower")
    for i, c in enumerate(cand_rows):
        m = lab == c["k"]
        _contour(ax, m.max(axis=2), COLOURS[i % len(COLOURS)], 1.6, "-" if not (reveal and c["kept"]) else "--")
        cx, cy = np.argwhere(m).mean(0)[:2]
        ax.text(cx + 4, cy + 4, f"#{i + 1}", color=COLOURS[i % len(COLOURS)], fontsize=11, weight="bold")
    if reveal and gt is not None:
        _contour(ax, gt.max(axis=2), "yellow", 1.8)
    ax.set_title(f"{cid}: enhancement MIP (axial; x = axis0, y = axis1)", fontsize=10)
    for j in (1, 2):
        axes[0, j].axis("off")
    if reveal:
        txt = "STAGE B (reveal)\nyellow = expert tumour\ndashed = KEPT by B0_rad\n\n" + "\n".join(
            f"#{i + 1} (k={c['k']}): {'KEPT' if c['kept'] else 'dropped'}, Dice {c['single_dice']:.2f}, "
            f"{c['size_ml']:.1f} ml" for i, c in enumerate(cand_rows))
    else:
        txt = ("STAGE A (blind)\nFor each numbered candidate record:\ninside_breast / outside_breast / boundary_uncertain\n"
               "(posterior lesions, axillary tail -> boundary_uncertain)")
    axes[0, 1].text(0, 1, txt, va="top", fontsize=10, family="monospace")
    for i, c in enumerate(cand_rows):
        m = lab == c["k"]
        x, y, z = np.round(np.argwhere(m).mean(0)).astype(int)
        col = COLOURS[i % len(COLOURS)]
        ls = "--" if (reveal and c["kept"]) else "-"
        for j, (name, sl2, msl, gsl) in enumerate((
                ("axial z", post[:, :, z], m[:, :, z], None if gt is None else gt[:, :, z]),
                ("coronal y", post[:, y, :], m[:, y, :], None if gt is None else gt[:, y, :]),
                ("sagittal x", post[x, :, :], m[x, :, :], None if gt is None else gt[x, :, :]))):
            ax = axes[i + 1, j]
            ax.imshow(sl2.T, cmap="gray", origin="lower")
            _contour(ax, msl, col, 1.6, ls)
            if reveal and gsl is not None:
                _contour(ax, gsl, "yellow", 1.4)
            ax.set_title(f"#{i + 1} {name}={[z, y, x][j]}", fontsize=9, color=col)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    plt.close(fig)


def step_images(a, out):
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    pats, key = wrong_picks(Path(a.s3k_out).expanduser())
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(pd.read_csv(a.splits), root, "train"))
    cdir = Path(a.train_cases).expanduser()
    (out / "stageA").mkdir(parents=True, exist_ok=True); (out / "stageB").mkdir(exist_ok=True)
    tmpl = []
    for cid in pats.cache_id:
        img, lbl = load_case(paths[cid])
        gt = lbl > 0
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        rows = key[key.cache_id == cid].to_dict("records")
        patient_figure(out / "stageA" / f"{cid}.png", cid, img, lab, rows, None, False)
        patient_figure(out / "stageB" / f"{cid}.png", cid, img, lab, rows, gt, True)
        for i, r in enumerate(rows):
            tmpl.append(dict(cache_id=cid, cohort=cid.split("_")[0].upper(), candidate=f"#{i + 1}", k=r["k"],
                             location="", notes=""))
    pd.DataFrame(tmpl).to_csv(out / "annotation_template.csv", index=False)
    key.to_csv(out / "_reveal_key.csv", index=False)
    pats.to_csv(out / "patients.csv", index=False)
    ceiling = dict(wrong_pick_patients=len(pats),
                   reach_01_with_perfect_pick=int(key[key.touches_tumour].groupby("cache_id").single_dice.max().ge(0.1).sum()),
                   reach_05_with_perfect_pick=int(key[key.touches_tumour].groupby("cache_id").single_dice.max().ge(0.5).sum()))
    (out / "ceiling.json").write_text(json.dumps(ceiling, indent=1))
    print(f"S3q images | {len(pats)} wrong-pick patients, {len(key)} candidates | by cohort "
          f"{pats.cohort.value_counts().to_dict()}")
    print("ceiling (single best candidate):", ceiling)
    print(f"Stage A (blind): {out / 'stageA'}  -> fill {out / 'annotation_template.csv'} (save as annotation_filled.csv)")
    print(f"Stage B (reveal, only afterwards): {out / 'stageB'} ; do not open _reveal_key.csv before Stage A is done")


def step_tally(a, out):
    ann = pd.read_csv(Path(a.annotations).expanduser())
    key = pd.read_csv(out / "_reveal_key.csv")
    bad = ann[~ann.location.isin(LOCATIONS)]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows without a valid location (use {LOCATIONS}), e.g. {bad.head(3).to_dict('records')}")
    d = key.merge(ann[["cache_id", "k", "location"]], on=["cache_id", "k"], how="left", validate="one_to_one")
    if d.location.isna().any():
        raise SystemExit("some candidates are not annotated")
    rows = []
    for cid, g in d.groupby("cache_id"):
        kept = g[g.kept]
        inside = g[(g.location == "inside_breast") & g.touches_tumour]
        best_in = float(inside.single_dice.max()) if len(inside) else 0.0
        kl = set(kept.location)
        cat = ("nothing_kept" if not kl else "uncertain" if "boundary_uncertain" in kl else "clear_outside" if kl == {"outside_breast"}
               else "all_inside" if kl == {"inside_breast"} else "mixed")
        rows.append(dict(cache_id=cid, cohort=cid.split("_")[0].upper(), n_kept=len(kept),
                         kept_locations="+".join(sorted(kept.location)), category=cat,
                         best_inside_alternative_dice=best_in,
                         rescuable_01=cat == "clear_outside" and best_in >= 0.1,
                         rescuable_05=cat == "clear_outside" and best_in >= 0.5,
                         uncertain_with_alternative_01=cat == "uncertain" and best_in >= 0.1))
    p = pd.DataFrame(rows)
    p.to_csv(out / "audit_patients.csv", index=False)
    blobs = d[d.kept].groupby("location").size().to_dict()
    by = p.groupby("cohort").agg(patients=("cache_id", "size"), clear_outside=("category", lambda s: int((s == "clear_outside").sum())),
                                 uncertain=("category", lambda s: int((s == "uncertain").sum())),
                                 rescuable_01=("rescuable_01", "sum"), rescuable_05=("rescuable_05", "sum"))
    tot = dict(patients=len(p), kept_blobs_by_location=blobs, categories=p.category.value_counts().to_dict(),
               rescuable_01=int(p.rescuable_01.sum()), rescuable_05=int(p.rescuable_05.sum()),
               uncertain_with_alternative_01=int(p.uncertain_with_alternative_01.sum()), threshold=10)
    tot["proceed_to_breast_mask_experiment"] = tot["rescuable_01"] >= 10
    by.to_csv(out / "audit_by_cohort.csv")
    (out / "audit_summary.json").write_text(json.dumps(tot, indent=1, default=int))
    pd.set_option("display.width", 200)
    print(by.to_string())
    print(json.dumps(tot, indent=1, default=int))
    print("\nDECISION:", "PROCEED (>= 10 clear, rescuable outside-breast errors)" if tot["proceed_to_breast_mask_experiment"]
          else "STOP (fewer than 10 clear, rescuable outside-breast errors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["images", "tally"], required=True)
    ap.add_argument("--s3k_out", required=True)
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--train_cases", default="")
    ap.add_argument("--annotations", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    if a.step == "images":
        if not a.train_cases:
            raise SystemExit("--train_cases required")
        step_images(a, out)
    else:
        if not a.annotations:
            raise SystemExit("--annotations required")
        step_tally(a, out)


if __name__ == "__main__":
    main()
