import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _ball(shape, c, r):
    g = np.indices(shape)
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r ** 2


def test_figures_end_to_end(tmp_path):
    cache, fin = tmp_path / "cache", tmp_path / "final" / "jury13_rad_rel"
    for d in (cache / "train", cache / "val", fin / "masks"):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    shape, rows, spl = (80, 70, 40), [], []
    for i, coll in enumerate(["DUKE", "DUKE", "ISPY2"]):
        cid = f"{coll.lower()}_{i:03d}"
        gt = _ball(shape, (30, 40, 20), 8)
        pred = _ball(shape, (33, 40, 20), 8) | _ball(shape, (65, 15, 20), 4)    # shifted + one false blob
        img = np.stack([0.2 + 0.02 * rng.standard_normal(shape), 0.2 + 0.02 * rng.standard_normal(shape)])
        img[1][gt] = 0.9
        np.savez_compressed(cache / "val" / f"{cid}.npz", image=img.astype(np.float16), label=gt[None].astype(np.uint8))
        np.savez_compressed(fin / "masks" / f"{cid}.npz", mask=pred.astype(np.uint8))
        rows.append(dict(cache_id=cid, split="test", collection=coll, dice=0.7 + 0.1 * i, hd95=5.0, nsd=0.8))
        spl.append(dict(cache_id=cid, split="test", official="test", collection=coll))
    pd.DataFrame(rows).to_csv(fin / "metrics.csv", index=False)
    pd.DataFrame(spl).to_csv(tmp_path / "splits.csv", index=False)
    out = tmp_path / "fig"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "s3h_figures.py"), "--final_dir", str(fin),
                        "--cache_root", str(cache), "--splits", str(tmp_path / "splits.csv"), "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["duke_000.png", "duke_001.png", "ispy2_002.png", "overview_test.png"]
    assert all((out / p).stat().st_size > 10_000 for p in pngs)
