"""Figures of the final pipeline: MRI | expert mask | proposed mask | error map (CPU only).

For each chosen patient one PNG with 4 panels on the axial slice where the tumour is largest:
  1. MRI (post-contrast 1), whole slice, with a box around the zoomed region
  2. expert (ground-truth) mask, green, zoomed
  3. proposed pipeline mask, red, zoomed
  4. error map, zoomed: yellow = correct (true positive), blue = missed tumour (false negative),
     red = extra (false positive)
plus one overview PNG with all chosen patients (one row each).

Patients: --cases id1,id2,... or automatically the best / median / worst test patient of every hospital
(by the proposed jury's Dice in metrics.csv).

Example (server)
  python scripts/s3h_figures.py --final_dir ~/CIDS/outputs/s3h_radiomics/final/jury13_rad_rel \\
      --cache_root ~/data --out ~/CIDS/outputs/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cids.data.cache import find_cache_root, load_case  # noqa: E402
from cids.provenance import require_cases  # noqa: E402

ZOOM_MM = 40          # margin around the tumour / prediction in the zoomed panels


def pick_cases(met: pd.DataFrame, split: str) -> list:
    """Best, median and worst patient of every hospital (by Dice)."""
    m = met[met.split == split].sort_values("dice")
    out = []
    for _, g in m.groupby("collection"):
        g = g.reset_index(drop=True)
        for i in (len(g) - 1, len(g) // 2, 0):
            cid = g.cache_id[i]
            if cid not in out:
                out.append(cid)
    return out


def best_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    """Axial slice (last axis) with the largest tumour area; no tumour -> largest prediction; else middle."""
    for m in (gt, pred):
        area = m.sum(axis=(0, 1))
        if area.max() > 0:
            return int(area.argmax())
    return gt.shape[2] // 2


def zoom_box(gt2: np.ndarray, pr2: np.ndarray, margin: int = ZOOM_MM):
    m = gt2 | pr2
    if not m.any():
        return 0, gt2.shape[0], 0, gt2.shape[1]
    xs, ys = np.where(m)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, gt2.shape[0])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, gt2.shape[1])
    return x0, x1, y0, y1


def error_rgb(gt2: np.ndarray, pr2: np.ndarray) -> np.ndarray:
    rgb = np.zeros(gt2.shape + (4,), np.float32)
    rgb[gt2 & pr2] = (1.0, 0.85, 0.0, 0.85)       # correct: yellow
    rgb[gt2 & ~pr2] = (0.1, 0.4, 1.0, 0.85)       # missed tumour: blue
    rgb[~gt2 & pr2] = (1.0, 0.1, 0.1, 0.85)       # extra: red
    return rgb


def _show(ax, img):
    """Cache arrays are (X, Y) in RAS; show with Y up the page."""
    ax.imshow(np.rot90(img), cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def draw_row(axes, cid, post, gt, pred, row: dict | None):
    import matplotlib.patches as mpatches
    z = best_slice(gt, pred)
    p2, g2, r2 = post[:, :, z], gt[:, :, z], pred[:, :, z]
    x0, x1, y0, y1 = zoom_box(g2, r2)
    lo, hi = np.percentile(p2, [1, 99.5])
    p2n = np.clip((p2 - lo) / (hi - lo + 1e-6), 0, 1)
    pz, gz, rz = p2n[x0:x1, y0:y1], g2[x0:x1, y0:y1], r2[x0:x1, y0:y1]

    _show(axes[0], p2n)
    H = p2n.shape[1]                                   # rot90: (x, y) -> (col = x, row = H - 1 - y)
    axes[0].add_patch(mpatches.Rectangle((x0 - 0.5, H - y1 - 0.5), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1))
    title = f"{cid}  (slice {z})"
    if row is not None:
        title += f"\nDice {row['dice']:.3f} | HD95 {row['hd95']:.1f} mm | NSD {row['nsd']:.3f}"
    axes[0].set_title(title, fontsize=9)
    for ax, mask, color, name in ((axes[1], gz, (0.1, 0.9, 0.2), "expert mask"),
                                  (axes[2], rz, (1.0, 0.15, 0.15), "proposed mask")):
        _show(ax, pz)
        ov = np.zeros(mask.shape + (4,), np.float32)
        ov[mask] = (*color, 0.5)
        ax.imshow(np.rot90(ov), interpolation="nearest")
        ax.set_title(name, fontsize=9)
    _show(axes[3], pz)
    axes[3].imshow(np.rot90(error_rgb(gz, rz)), interpolation="nearest")
    axes[3].set_title("error: yellow correct, blue missed, red extra", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", required=True, help="s3h final/<jury> folder (masks/, metrics.csv)")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--splits", default=str(ROOT / "configs" / "splits.csv"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--cases", default="", help="comma-separated cache ids (default: best/median/worst per hospital)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is missing: pip install matplotlib")

    fin, out = Path(a.final_dir).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(fin / "metrics.csv")
    root = find_cache_root(str(Path(a.cache_root).expanduser()) if a.cache_root else "")
    sp = pd.read_csv(a.splits)
    paths = dict(require_cases(sp, root, a.split))
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] or pick_cases(met, a.split)
    missing = [c for c in cases if c not in paths or not (fin / "masks" / f"{c}.npz").exists()]
    if missing:
        raise SystemExit(f"not in the {a.split} split or no saved mask: {missing}")
    rows = met[met.split == a.split].set_index("cache_id")

    big, axes_all = plt.subplots(len(cases), 4, figsize=(13, 3.4 * len(cases)), squeeze=False)
    for i, cid in enumerate(cases):
        img, lbl = load_case(paths[cid])
        pred = np.load(fin / "masks" / f"{cid}.npz")["mask"] > 0
        gt = lbl > 0
        if pred.shape != gt.shape:
            raise SystemExit(f"{cid}: mask {pred.shape} != cache {gt.shape}")
        row = rows.loc[cid].to_dict() if cid in rows.index else None
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        draw_row(axes, cid, img[1], gt, pred, row)
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=a.dpi)
        plt.close(fig)
        draw_row(axes_all[i], cid, img[1], gt, pred, row)
        print(f"{cid}: " + (f"Dice {row['dice']:.3f}" if row else "no metrics row") + f" -> {out / (cid + '.png')}")
    big.tight_layout()
    big.savefig(out / f"overview_{a.split}.png", dpi=a.dpi)
    print(f"overview -> {out / f'overview_{a.split}.png'} ({len(cases)} patients)")


if __name__ == "__main__":
    main()
