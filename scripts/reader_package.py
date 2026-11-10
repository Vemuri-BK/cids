"""Reader pilot package: 20 validation cases for one radiologist (accept/reject the automatic mask, one click).

Case selection (fixed before building; validation patients only, test refused)
  10 FLAGGED   = the 10 highest QC-risk validation patients (S3n val_review.csv risk_baseline; empty masks first)
  10 UNFLAGGED = 10 drawn at random (seed 0) from the remaining validation patients
  shuffled (seed 0) and renamed case01..case20; the key (case -> cache_id, flagged, B0_rad Dice) is written
  OUTSIDE the package and must not be sent.

Per case folder (NIfTI, 1 mm, the CIDS cache grid)
  caseNN_pre.nii.gz, caseNN_post.nii.gz, caseNN_sub.nii.gz   pre-contrast, post-contrast, subtraction
  caseNN_auto_mask.seg.nii.gz                                our automatic B0_rad mask (label 1)
No expert mask, no QC score, no hint of failures. Plus README.txt (instructions), responses.csv (sheet),
load_case_in_slicer.py (optional helper for 3D Slicer).

  python scripts/reader_package.py --cache_root ~/data --s3k_out ~/CIDS/outputs/s3k_radiomics \\
      --val_cases ~/CIDS/outputs/s3f_jury_masks/s3f_jury/cases --s3n_out ~/CIDS/outputs/s3n_qc \\
      --out ~/CIDS/outputs/reader_pilot
  -> <out>/reader_pilot_package/ (+ .zip), <out>/reader_key_DO_NOT_SEND.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3m_contour_study as M  # noqa: E402

N_FLAGGED, N_UNFLAGGED, SEED = 10, 10, 0
AFFINE = np.eye(4)

README = """CIDS reader pilot - instructions (about 1 hour, 20 cases)

What this is
  Each folder is one breast DCE-MRI scan with an AUTOMATIC tumour outline made by our software.
  We want to know whether the outline is correct and, if not, where the real tumour is.

Files in each case folder
  caseNN_pre.nii.gz        pre-contrast image
  caseNN_post.nii.gz       post-contrast image (first phase after contrast)
  caseNN_sub.nii.gz        subtraction image (post - pre)
  caseNN_auto_mask.seg.nii.gz   the automatic outline (load it as a segmentation / label map)

Viewer: 3D Slicer (free, https://www.slicer.org)
  Drag the four files of one case into Slicer. For the mask file choose "Segmentation" (or "LabelMap").
  Optional: in Slicer's Python console run  exec(open(r"<path>/load_case_in_slicer.py").read())
  after setting CASE_DIR in that file.

For EACH case
  1. Scroll through the scan with the outline shown (switch between post, subtraction and pre as you like).
  2. Decide: is the outline on the primary (main) tumour and acceptable?  -> ACCEPT or REJECT
  3. If REJECT (or unsure): place ONE point on the primary tumour
       Markups module -> create a Point List -> place one point in the centre of the tumour
       -> save it as  caseNN_click.mrk.json  in that case folder (File > Save Data).
  4. Write in responses.csv: decision (ACCEPT/REJECT), confidence 1-5, time in minutes, comments.

Please do not edit the outline itself. There is no expert outline in the package; please do not try to
find the patients in any database. Return the whole folder (with responses.csv and the click files).
"""

SLICER_HELPER = '''# 3D Slicer helper: set CASE_DIR, then in Slicer's Python console: exec(open(r"<this file>").read())
import glob, os, slicer
CASE_DIR = r"PUT/THE/CASE/FOLDER/HERE"
for f in sorted(glob.glob(os.path.join(CASE_DIR, "*.nii.gz"))):
    if f.endswith(".seg.nii.gz"):
        seg = slicer.util.loadSegmentation(f)
    else:
        slicer.util.loadVolume(f)
sub = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if n.GetName().endswith("_sub")]
if sub:
    slicer.util.setSliceViewerLayers(background=sub[0])
'''


def save_nii(arr, path, dtype):
    import nibabel as nib
    img = nib.Nifti1Image(np.ascontiguousarray(arr).astype(dtype), AFFINE)
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(path))


def select_cases(review: pd.DataFrame, val_ids):
    r = review[review.cache_id.isin(val_ids)].copy()
    if len(r) != len(val_ids):
        raise SystemExit(f"S3n val_review.csv covers {len(r)} of {len(val_ids)} validation patients")
    r["qc"] = np.where(r.is_empty == 1, np.inf, r.risk_baseline)
    r = r.sort_values(["qc", "cache_id"], ascending=[False, True])
    flagged = r.cache_id.iloc[:N_FLAGGED].tolist()
    rest = sorted(r.cache_id.iloc[N_FLAGGED:])
    rng = np.random.default_rng(SEED)
    unflagged = sorted(rng.choice(rest, size=min(N_UNFLAGGED, len(rest)), replace=False).tolist())
    cases = [(c, True) for c in flagged] + [(c, False) for c in unflagged]
    order = rng.permutation(len(cases))
    return [cases[i] for i in order]


def main():
    from cids.data.cache import find_cache_root, load_case
    from cids.provenance import require_cases
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--s3k_out", required=True, help="S3k per_patient.csv (val B0_rad selection)")
    ap.add_argument("--val_cases", required=True, help="S3f val component maps (key lab)")
    ap.add_argument("--s3n_out", required=True, help="S3n val_review.csv (QC risk)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in (a.val_cases, a.s3n_out):
        if "test" in Path(f).name.lower():
            raise SystemExit("the reader pilot uses validation patients only: test files are refused")
    out = Path(a.out).expanduser()
    pkg = out / "reader_pilot_package"
    if pkg.exists():
        raise SystemExit(f"{pkg} already exists: move it away first (the package is built once)")
    sp = pd.read_csv(a.splits)
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    paths = dict(require_cases(sp, root, "val"))
    pp = M.load_selection(Path(a.s3k_out).expanduser())
    pp = pp[pp.split == "val"].set_index("cache_id")
    review = pd.read_csv(Path(a.s3n_out).expanduser() / "val_review.csv")
    cdir = Path(a.val_cases).expanduser()
    cdir = cdir if any(cdir.glob("*.npz")) else next(cdir.rglob("*.npz")).parent
    cases = select_cases(review, sorted(paths))
    pkg.mkdir(parents=True)
    key, sheet = [], []
    for i, (cid, flagged) in enumerate(cases, 1):
        name = f"case{i:02d}"
        d = pkg / name
        d.mkdir()
        img, gt = load_case(paths[cid])
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        kept = M.kept_ids(pp.loc[cid, "B0_rad_subset"])
        auto = np.isin(lab, kept) if kept else np.zeros(gt.shape, bool)
        dice = M.dice_count(auto, gt > 0)
        if abs(dice - float(pp.loc[cid, "B0_rad_dice"])) > 1e-6:
            raise SystemExit(f"PARITY FAILED {cid}: rebuilt B0_rad Dice {dice} vs S3k {pp.loc[cid, 'B0_rad_dice']}")
        pre, post = img[0].astype(np.float32), img[1].astype(np.float32)
        for tag, arr in (("pre", pre), ("post", post), ("sub", post - pre)):
            save_nii(arr, d / f"{name}_{tag}.nii.gz", np.float32)
        save_nii(auto, d / f"{name}_auto_mask.seg.nii.gz", np.uint8)
        key.append(dict(case=name, cache_id=cid, flagged_by_qc=flagged, B0_rad_dice=dice,
                        auto_ml=float(auto.sum()) / 1000, auto_empty=bool(not auto.any())))
        sheet.append(dict(case=name, decision_ACCEPT_or_REJECT="", confidence_1to5="", time_minutes="",
                          click_saved_yes_no="", comments=""))
        print(f"  {name} written", flush=True)
    (pkg / "README.txt").write_text(README)
    (pkg / "load_case_in_slicer.py").write_text(SLICER_HELPER)
    pd.DataFrame(sheet).to_csv(pkg / "responses.csv", index=False)
    k = pd.DataFrame(key)
    k.to_csv(out / "reader_key_DO_NOT_SEND.csv", index=False)
    (out / "selection.json").write_text(json.dumps(dict(n_flagged=N_FLAGGED, n_unflagged=N_UNFLAGGED, seed=SEED,
                                                        rule="top QC risk (val) + random rest"), indent=1))
    z = shutil.make_archive(str(out / "reader_pilot_package"), "zip", root_dir=out, base_dir="reader_pilot_package")
    print(k.groupby("flagged_by_qc").agg(n=("case", "size"), mean_dice=("B0_rad_dice", "mean"),
                                         dice_lt_0_1=("B0_rad_dice", lambda x: int((x < 0.1).sum()))).to_string())
    print(f"PACKAGE DONE: {pkg}\nzip: {z} ({Path(z).stat().st_size / 1e6:.0f} MB)\n"
          f"key (keep, do NOT send): {out / 'reader_key_DO_NOT_SEND.csv'}")


if __name__ == "__main__":
    main()
