import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import (crop_centers, crop_pad, list_cases, load_case,  # noqa: E402
                             make_sam_crops, resample, tumor_center, znorm)


def fake_case(shape=(150, 140, 90), c=(70, 60, 40), r=8):
    img = np.zeros((2,) + shape, np.float32)
    img[:, 10:-10, 10:-10, 5:-5] = np.random.rand(2, shape[0] - 20, shape[1] - 20, shape[2] - 10) * 0.5
    g = np.indices(shape)
    ball = ((g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 + (g[2] - c[2]) ** 2) <= r ** 2
    img[1][ball] = 0.95
    return img, ball.astype(np.uint8)


def test_crop_pad_inside_and_outside():
    v = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    c, o = crop_pad(v, (1, 1, 1), size=5)
    assert c.shape == (5, 5, 5) and (o == -1).all()
    assert c[1:4, 1:4, 1:4].sum() == v.sum() and c[0].sum() == 0


def test_resample_and_crops(tmp_path):
    img, lbl = fake_case()
    (tmp_path / "train").mkdir(); (tmp_path / "val").mkdir()
    np.savez_compressed(tmp_path / "train" / "duke_001.npz", image=img.astype(np.float16), label=lbl[None])
    np.savez_compressed(tmp_path / "val" / "duke_019.npz", image=img.astype(np.float16), label=lbl[None])
    cases = list_cases(tmp_path)
    assert [(c[0], c[1]) for c in cases] == [("duke_001", "train"), ("duke_019", "test")]
    im, lb = load_case(cases[0][2])
    im15, lb15 = resample(im, lb)
    assert lb15.shape == tuple(int(round(s / 1.5)) for s in lb.shape)
    assert abs(lb15.sum() * 1.5 ** 3 / lb.sum() - 1) < 0.15      # volume preserved
    cs = crop_centers(lb15, 3, 16, seed=0)
    assert cs.shape == (4, 3) and (cs[0] == tumor_center(lb15)).all()
    out = make_sam_crops(im15, lb15, cs)
    assert out["subtraction"].shape == (4, 1, 128, 128, 128)
    assert out["label"].shape == (4, 128, 128, 128)
    assert out["label"][0].sum() == lb15.sum()                     # centred crop holds whole tumour
    fg = out["post1"][0, 0] != out["post1"][0, 0].min()
    assert abs(out["post1"][0, 0][fg].mean()) < 0.5


def test_znorm_stats():
    x = np.random.rand(20, 20, 20).astype(np.float32) + 3
    z = znorm(x, np.ones_like(x))
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3
