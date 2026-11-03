import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s9_attack_pilot as s9  # noqa: E402


def _scene(shape=(48, 48, 48)):
    rng = np.random.default_rng(0)
    img = (0.3 + 0.05 * rng.standard_normal((2,) + shape)).astype(np.float32)
    img[:, :, :, :6] = 0.0                                   # padding slab (exact zeros)
    gt = s9.ball(shape, (12, 12, 24), 4)
    return img, gt


def test_ball_and_patch():
    b = s9.ball((20, 20, 20), (10, 10, 10), 3)
    assert b[10, 10, 10] and b.sum() == sum(1 for x in np.argwhere(b))
    assert not b[10, 10, 14]
    sl = s9.patch_slices((50, 70, 30), (2, 69, 15), 32)
    assert [s.stop - s.start for s in sl] == [32, 32, 24]     # clipped to the volume, multiple of 8
    assert sl[0].start == 0 and sl[1].stop == 70


def test_choose_decoy_constraints():
    img, gt = _scene()
    sprob = np.zeros(gt.shape, np.float32)
    sprob[30:, 30:, :] = 0.9                                 # finder already fires here -> excluded
    c = s9.choose_decoy(img, sprob, gt, np.random.default_rng(1), min_dist=15, r=3)
    assert c is not None
    assert sprob[c] < 0.1 and (img[0][c] != 0 or img[1][c] != 0)
    d = min(np.linalg.norm(np.array(c) - p) for p in np.argwhere(gt))
    assert d >= 15
    assert all(3 <= v < s - 3 for v, s in zip(c, gt.shape))
    assert s9.choose_decoy(img, np.ones(gt.shape), gt, np.random.default_rng(1)) is None


def test_pgd_is_bounded_keeps_padding_and_works():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv3d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv3d(4, 1, 3, padding=1))
    img, _ = _scene()
    target = s9.ball(img.shape[1:], (24, 24, 30), 4)
    sl = s9.patch_slices(img.shape[1:], (24, 24, 30), 32)
    eps = 8 / 255
    with torch.no_grad():
        x = torch.from_numpy(img); z0 = net(torch.stack([x[0], x[1], x[1] - x[0]])[None])[0, 0][torch.from_numpy(target)]
    adv, info = s9.pgd(net, img, target, "decoy", eps, 20, eps / 10, sl, amp=False)
    d = adv - img
    assert np.abs(d).max() <= eps + 1e-6 and info["linf"] <= eps + 1e-6
    assert np.all(adv[img == 0] == 0) or np.all(d[:, :, :, :6] == 0)   # padding untouched
    assert np.all(d[:, :sl[0].start] == 0) and np.all(d[:, sl[0].stop:] == 0)   # only inside the patch
    with torch.no_grad():
        xa = torch.from_numpy(adv); z1 = net(torch.stack([xa[0], xa[1], xa[1] - xa[0]])[None])[0, 0][torch.from_numpy(target)]
    assert z1.mean() > z0.mean() and info["loss"] < info["loss0"]
    adv_h, info_h = s9.pgd(net, img, target, "hide", eps, 20, eps / 10, sl, amp=False)
    with torch.no_grad():
        xh = torch.from_numpy(adv_h); z2 = net(torch.stack([xh[0], xh[1], xh[1] - xh[0]])[None])[0, 0][torch.from_numpy(target)]
    assert z2.mean() < z0.mean()


class _FixedJury:
    def __init__(self, p):
        self.p = np.asarray(p, float)

    def proba(self, df, D):
        return self.p


def test_judge_flags():
    shape = (30, 30, 30)
    gt = s9.ball(shape, (8, 8, 8), 3)
    decoy = s9.ball(shape, (22, 22, 22), 3)
    lab = np.zeros(shape, np.int32); lab[gt] = 1; lab[decoy] = 2
    df = pd.DataFrame(dict(cache_id="c", k=[1, 2], size_vox=[int(gt.sum()), int(decoy.sum())]))
    bundles = {"jury13": dict(jury=_FixedJury([0.9, 0.8]), threshold=0.7),      # keeps the decoy
               "jury13_rad_rel": dict(jury=_FixedJury([0.9, 0.1]), threshold=0.4)}  # rejects it
    out, hits, sc = s9.judge(bundles, df, None, lab, gt, decoy, "c")
    assert hits == [2]
    assert out["jury13"]["target_kept"] and not out["jury13_rad_rel"]["target_kept"]
    assert out["rel20"]["target_kept"]                                         # same size -> rel20 keeps both
    assert out["jury13_rad_rel"]["dice"] == 1.0 and out["jury13"]["dice"] < 1.0
    assert abs(out["jury13_rad_rel"]["p_target"] - 0.1) < 1e-9
    assert list(sc.columns[-2:]) == ["p_jury13", "p_jury13_rad_rel"]


def test_summarise_columns():
    rows = []
    for e in (0, 4):
        rows.append(dict(cache_id="a", mode="decoy", eps255=e, finder_hit=e > 0, rel20_target_kept=e > 0,
                         jury13_target_kept=e > 0, jury13_rad_rel_target_kept=False, rel20_dice=0.7,
                         jury13_dice=0.8, jury13_rad_rel_dice=0.8))
    agg = s9.summarise(pd.DataFrame(rows))
    assert agg.loc[("decoy", 4), "jury13_kept"] == 1.0 and agg.loc[("decoy", 4), "proposed_kept"] == 0.0
