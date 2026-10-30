"""S3g: rebuild the TRAINING jury masks from out-of-fold SegResNet maps (refiner v2 input).

Why: notebook 12 (S3f) built the training masks M0 from SegResNet-100 candidates, but SegResNet-100 was trained
on those patients, so the masks are cleaner than on new patients (train table Dice 0.805 vs val 0.774). A
refiner trained on them practises on the wrong kind of mistakes. Here every training patient's candidates
come from the fold model that never saw it (notebook 20 / s1_train_segresnet --split_col fold3_k).

Pipeline per training patient (same settings as S3f):
  OOF probability map > 0.5 -> components (>= 0.1 ml, top 5) -> SAM-S2b crop + box -> 13 clues
  -> the S3f memory jury, cross-fitted by patient (CrossFitMemoryJury on the S3f training table: the jury of
     patient c never saw c, neither its trees nor its memory) -> keep / drop at the S3f val cut-off
Val / test are NOT touched: their masks, metrics and summary are copied from S3f, so the final result (test
Dice 0.7495) is unchanged and s7_refine's per-case parity check still holds.

Inputs
  --s3f_dir   notebook 12 output: train_features.csv, train_desc.npy, summary.json, masks/ (or masks.zip),
              val_metrics.csv
  --oof_dirs  comma-separated fold outputs (seg_fold0,seg_fold1,seg_fold2), each with heldout_pred/ or
              heldout_pred.zip of <cid>.npz (prob = probability x 255)
Outputs (<out>, same layout as S3f so s7_refine --s3f_dir can point here)
  masks/<cid>.npz       train: rebuilt from OOF candidates | val / test: copied from S3f
  cases/<cid>.npz       train: seg (OOF prob x255), lab (component labels)
  train_features.csv, train_desc.npy      OOF candidate table + SAM descriptors
  parts/<cid>.npz       per-case progress (a restarted run skips finished patients)
  summary.json          threshold (copied), train Dice old (S3f) vs new (OOF), AUC, provenance
  val_metrics.csv       copied from S3f
Example
  python scripts/s3g_oof_train_masks.py --cache_root ~/data --s3f_dir ~/CIDS/outputs/s3f_jury_masks/s3f_jury \\
      --oof_dirs ~/CIDS/outputs/seg_oof/fold0/seg_fold0,~/CIDS/outputs/seg_oof/fold1/seg_fold1,... \\
      --sam_ckpt ~/CIDS/checkpoints/sam_s2b_best.pt --turbo_ckpt ~/CIDS/checkpoints/sam_med3d_turbo.pth \\
      --out ~/CIDS/outputs/s3g_oof_masks
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.jury import CrossFitMemoryJury, keep_candidates  # noqa: E402
from cids.provenance import file_digest, require_cases, save_npz_atomic  # noqa: E402
from cids.refine.source import find_s3f, index_maps  # noqa: E402


def load_prob(path) -> np.ndarray:
    d = np.load(path)
    if "prob" not in d:
        raise SystemExit(f"{path} has no 'prob' (run s1_train_segresnet with --save_prob)")
    return d["prob"].astype(np.float32) / 255.0


def case_rows(cid, prob, img, gt, sam, box, amp, s3, s3d):
    """Candidate rows + SAM descriptors of one case, exactly as S3f's run_split (from a given probability)."""
    lab, cands = s3.components(prob > 0.5, top_k=5)
    if not cands:
        rows = s3d.case_columns([dict(cache_id=cid, k=-1, size_vox=0, inter_vox=0, gt_vox=int(gt.sum()),
                                      on_tumour=0)], lab, gt)
        return rows, np.zeros((1, 768), np.float32), lab
    refined = s3.sam_refine(sam, box, img, lab, cands, amp, descriptors=True)
    by_k = {d["k"]: d["desc"] for d in refined}
    rows = s3d.case_columns(s3d.candidate_rows(cid, prob, lab, cands, refined, gt), lab, gt)
    return rows, np.stack([by_k[r["k"]] for r in rows]).astype(np.float32), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--s3f_dir", required=True)
    ap.add_argument("--oof_dirs", required=True, help="comma-separated fold outputs with heldout_pred/")
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--turbo_ckpt", default="")
    ap.add_argument("--random_weights", action="store_true", help="CPU tests only")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10, help="nearest memories (as S3f)")
    ap.add_argument("--limit", type=int, default=0, help="training cases (tests)")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    s3f = find_s3f(str(Path(a.s3f_dir).expanduser()), out / "s3f_unpacked")
    s3f_summary = json.loads((s3f / "summary.json").read_text())
    t = float(s3f_summary["threshold"])
    root = find_cache_root(a.cache_root)
    sp = pd.read_csv(a.splits)
    train = require_cases(sp, root, "train")[:a.limit or None]
    maps = index_maps(a.oof_dirs.split(","), ("heldout_pred",), out / "oof_unpacked")
    miss = [c for c, _ in train if c not in maps]
    if miss:
        raise SystemExit(f"{len(miss)} training cases have no out-of-fold map (e.g. {miss[:5]})")
    extra = sorted(set(maps) - {c for c, _ in require_cases(sp, root, "train")})
    if extra:
        raise SystemExit(f"out-of-fold maps for non-training cases (e.g. {extra[:5]}): wrong folders?")
    print(f"cache {root} | S3f {s3f} (cut-off t = {t}) | {len(train)} training cases, OOF maps from "
          f"{len(a.oof_dirs.split(','))} folds")

    # 1. the S3f jury, cross-fitted by patient (identical to notebook 12's training masks)
    old_tr = pd.read_csv(s3f / "train_features.csv")
    old_D = np.load(s3f / "train_desc.npy").astype(np.float32)
    if len(old_tr) != len(old_D):
        raise SystemExit("S3f train_features.csv and train_desc.npy differ in length")
    cross = CrossFitMemoryJury(old_tr, old_D, k=a.k, allow_tiny=bool(a.limit))
    import s3d_learned_jury as s3d
    old_tab = s3d.rule_tables(old_tr, cross.proba(old_tr, old_D))[f"learned_t{t:.2f}"]
    ref = s3f_summary.get("train_oof_table_dice")
    print(f"S3f training masks (in-sample finder): table Dice {old_tab.mean():.4f} (notebook 12: {ref})")
    if ref is not None and not a.limit and abs(old_tab.mean() - ref) > 1e-6:
        raise SystemExit("cross-fitted jury does not reproduce notebook 12's training masks: different S3f files?")

    # 2. OOF candidates -> SAM -> rows (resumable per case)
    import s3a_seg2sam as s3
    import torch
    amp = s3.DEV == "cuda"
    from cids.models.sam_med3d import download_turbo
    turbo = None if a.random_weights else (a.turbo_ckpt or download_turbo("/tmp/sam_ckpt"))
    sam, box = _load_sam(s3, a.sam_ckpt, turbo)
    pdir, cdir, mdir = out / "parts", out / "cases", out / "masks"
    for d in (pdir, cdir, mdir):
        d.mkdir(exist_ok=True)
    for cid, path in tqdm(train, desc="train (OOF)", unit="case", file=sys.stdout, mininterval=30):
        if (pdir / f"{cid}.npz").exists():
            continue
        img, lbl = load_case(path); gt = lbl > 0
        prob = load_prob(maps[cid])
        if prob.shape != gt.shape:
            raise SystemExit(f"{cid}: map {prob.shape} vs image {gt.shape}")
        with torch.no_grad():
            rows, desc, lab = case_rows(cid, prob, img, gt, sam, box, amp, s3, s3d)
        save_npz_atomic(cdir / f"{cid}.npz", seg=np.round(prob * 255).astype(np.uint8), lab=lab.astype(np.int32))
        save_npz_atomic(pdir / f"{cid}.npz", rows=np.array(json.dumps(rows)), desc=desc.astype(np.float16))
    tr_rows, tr_D = [], []
    for cid, _ in train:
        d = np.load(pdir / f"{cid}.npz")
        tr_rows += json.loads(str(d["rows"])); tr_D.append(d["desc"])
    tr, D = pd.DataFrame(tr_rows), np.concatenate(tr_D).astype(np.float32)
    tr.to_csv(out / "train_features.csv", index=False); np.save(out / "train_desc.npy", D.astype(np.float16))

    # 3. jury (never the patient's own fold) -> masks
    p = cross.proba(tr, D)
    keep = keep_candidates(tr, p, t)
    for cid, _ in train:
        lab = np.load(cdir / f"{cid}.npz")["lab"]
        ks = keep.get(cid, [])
        m = np.isin(lab, ks) if ks else np.zeros(lab.shape, bool)
        save_npz_atomic(mdir / f"{cid}.npz", mask=m.astype(np.uint8), kept=np.array(ks, np.int32))
    new_tab = s3d.rule_tables(tr, p)
    new_dice = new_tab[f"learned_t{t:.2f}"]

    # 4. val / test unchanged: copy S3f masks + metrics
    n_copied = 0
    for split in ("val", "test"):
        for cid, _ in require_cases(sp, root, split):
            src = s3f / "masks" / f"{cid}.npz"
            if not src.exists():
                raise SystemExit(f"S3f has no mask for {split} case {cid}")
            shutil.copy(src, mdir / f"{cid}.npz"); n_copied += 1
    for f in ("val_metrics.csv", "test_metrics.csv"):
        if (s3f / f).exists():
            shutil.copy(s3f / f, out / f)

    from sklearn.metrics import roc_auc_score
    r = (tr.k >= 0).to_numpy(); y = tr.on_tumour.to_numpy()[r]
    old_c = old_tr.groupby("cache_id").size()
    new_c = tr[tr.k >= 0].groupby("cache_id").size()
    summary = dict(s3f_summary)                                     # threshold, val/test results: unchanged
    summary.update(
        s3g=dict(train_cases=len(train), train_table_dice_s3f=float(old_tab.reindex(new_dice.index).mean()),
                 train_table_dice_oof=float(new_dice.mean()),
                 train_seg_rel20_dice_oof=float(new_tab["seg_rel20"].mean()),
                 train_oracle_dice_oof=float(new_tab["oracle"].mean()),
                 auc_train_oof_candidates=float(roc_auc_score(y, p[r])) if 0 < y.sum() < len(y) else None,
                 candidates_per_case_s3f=float(old_c.mean()), candidates_per_case_oof=float(new_c.mean()),
                 empty_cases_oof=int((tr.groupby("cache_id").k.max() < 0).sum()),
                 val_test_masks_copied=n_copied, threshold=t,
                 provenance=dict(sam=file_digest(a.sam_ckpt), turbo=file_digest(turbo) if turbo else None,
                                 oof_dirs=a.oof_dirs.split(","), s3f=str(s3f)),
                 minutes=(time.time() - t0) / 60))
    summary["train_oof_table_dice"] = float(new_dice.mean())         # what the refiner now trains on
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    g = summary["s3g"]
    print(f"\nTRAINING MASKS ({len(train)} patients)")
    print(f"  table Dice, S3f (in-sample finder):   {g['train_table_dice_s3f']:.4f}")
    print(f"  table Dice, OOF finder (new):         {g['train_table_dice_oof']:.4f}   (val, for reference: "
          f"{s3f_summary.get('val_dice', float('nan')):.4f})")
    print(f"  SegResNet rel20 Dice, OOF finder:     {g['train_seg_rel20_dice_oof']:.4f}")
    print(f"  best possible subset (oracle), OOF:   {g['train_oracle_dice_oof']:.4f}")
    print(f"  candidates per case: S3f {g['candidates_per_case_s3f']:.2f} -> OOF {g['candidates_per_case_oof']:.2f}"
          f" | empty: {g['empty_cases_oof']} | jury AUC on OOF candidates: {g['auc_train_oof_candidates']}")
    print(f"val / test masks copied unchanged: {n_copied}")
    print(f"total {g['minutes']:.1f} min -> {out}")


def _load_sam(s3, sam_ckpt, turbo):
    """SAM-Med3D-turbo + S2b LoRA decoder + box corners (no SegResNet needed here)."""
    import torch
    sam = s3.load_sam_med3d(turbo, s3.DEV)
    s3.add_lora(sam.mask_decoder, ("q_proj", "v_proj"), 8, 16.0)
    box = s3.BoxCorners(sam.prompt_encoder)
    st = torch.load(sam_ckpt, map_location="cpu", weights_only=False)
    sam.mask_decoder.load_state_dict(st["mask_decoder"]); box.load_state_dict(st["box"])
    sam.to(s3.DEV).eval(); box.to(s3.DEV).eval()
    return sam, box


if __name__ == "__main__":
    main()
