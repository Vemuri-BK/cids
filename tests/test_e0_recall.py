import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e0_recall_ceiling as e0  # noqa: E402


def _ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((g[i] - c[i]) ** 2 for i in range(3)) <= r ** 2


def test_oracle_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 6)
        size = rng.integers(1, 100, n)
        inter = np.minimum(size, rng.integers(0, 60, n))
        g = int(inter.sum() + rng.integers(1, 80))
        best = max([0.0] + [2 * inter[list(s)].sum() / (size[list(s)].sum() + g)
                            for r in range(1, n + 1) for s in itertools.combinations(range(n), r)])
        assert abs(e0.oracle_dice(inter, size, g) - best) < 1e-12


def test_components_same_as_pipeline():
    sys.path.insert(0, str(ROOT))
    import s3a_seg2sam as s3
    rng = np.random.default_rng(1)
    m = rng.random((30, 30, 30)) > 0.7
    l1, c1 = e0.components(m, 20, 3)
    l2, c2 = s3.components(m, 20, 3)
    assert c1 == c2 and np.array_equal(l1, l2)


def test_case_rows_recall_first(tmp_path):
    shape = (60, 60, 60)
    gt = _ball(shape, (15, 15, 30), 5)
    false = _ball(shape, (45, 45, 30), 6)
    prob = np.zeros(shape, np.float32)
    prob[false] = 0.9                                   # confident false blob
    prob[gt] = 0.35                                     # tumour seen only weakly -> missed at 0.5
    np.savez(tmp_path / "m.npz", prob=np.round(prob * 255).astype(np.uint8))
    np.savez(tmp_path / "c.npz", label=gt[None].astype(np.uint8), image=np.zeros((2,) + shape, np.float16))
    settings = e0.settings_for([0.5, 0.3], [5])
    rows = e0.case_rows(("c", "val", tmp_path / "m.npz", tmp_path / "c.npz", settings, 100))
    r = {(x["mode"], x["t"], x["k"]): x for x in rows}
    assert not r[("plain", 0.5, 5)]["found"] and r[("plain", 0.5, 5)]["oracle"] == 0.0
    assert r[("plain", 0.3, 5)]["found"] and r[("plain", 0.3, 5)]["oracle"] == 1.0
    assert r[("addon", 0.3, 5)]["found"] and r[("addon", 0.3, 5)]["n_cand"] == 2
    s = e0.summarise(pd.DataFrame(rows))
    row = s[(s["mode"] == "addon") & (s.t == 0.3)].iloc[0]
    assert row.missed_at_ref == 1 and row.recovered == 1 and row.delta_vs_ref == 1.0
