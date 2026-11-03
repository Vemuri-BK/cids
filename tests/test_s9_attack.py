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


def test_ball():
    b = s9.ball((20, 20, 20), (10, 10, 10), 3)
    assert b[10, 10, 10] and not b[10, 10, 14]


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


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv3d(3, 2, 3, padding=1), torch.nn.Tanh(),
                               torch.nn.Conv3d(2, 1, 3, padding=1)).eval()


def _sw_prob(net, img):
    """The pipeline's finder output (s3a.seg_prob without GPU): MONAI sliding window on 3 channels."""
    from monai.inferers import sliding_window_inference
    from cids.data.volumes import pad_to, three_channel, unpad
    x, pads = pad_to(three_channel(img), (96, 96, 96))
    with torch.no_grad():
        z = sliding_window_inference(torch.from_numpy(x)[None], (96,) * 3, sw_batch_size=4, predictor=net,
                                     overlap=0.5, mode="gaussian")
    return unpad(torch.sigmoid(z.float())[0, 0].numpy(), pads)


def test_attack_model_equals_pipeline_and_attack_works():
    img = (0.3 + 0.05 * np.random.default_rng(0).standard_normal((2, 100, 90, 104))).astype(np.float32)
    img[:, :, :, :4] = 0.0                                               # padding slab
    net = _net()
    target = s9.ball(img.shape[1:], (50, 45, 60), 4)
    eps = 8 / 255
    p0 = _sw_prob(net, img)
    adv, info = s9.pgd_sw(net, img, target, "decoy", eps, 3, eps / 3, amp=False)
    zc = info["z_clean"]
    assert np.nanmax(np.abs(1 / (1 + np.exp(-zc[target])) - p0[target])) < 1e-5     # exact same function
    assert info["n_windows"] >= 1 and adv.shape == img.shape
    d = adv - img
    assert np.abs(d).max() <= eps + 1e-6 and np.all(d[:, :, :, :4] == 0)           # bounded, padding untouched
    p1 = _sw_prob(net, adv)
    assert p1[target].mean() > p0[target].mean() and info["loss"] < info["loss0"]  # transfers to the pipeline
    adv_h, info_h = s9.pgd_sw(net, img, target, "hide", eps, 3, eps / 3, amp=False)
    assert _sw_prob(net, adv_h)[target].mean() < p0[target].mean()


def test_restrict_target_caps_windows():
    shape = (200, 200, 200)
    big = s9.ball(shape, (100, 100, 100), 60)
    t, frac = s9.restrict_target(big, (100, 100, 100), 8, shape)
    assert len([w for w in s9.sw_windows(shape) if t[w].any()]) <= 8 and 0 < frac < 1
    small = s9.ball(shape, (100, 100, 100), 3)
    assert s9.restrict_target(small, (100, 100, 100), 64, shape)[1] == 1.0


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
    assert out["target_cand"]["ml"] == decoy.sum() / 1000 and not out["target_cand"]["touches_gt"]
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
                         jury13_dice=0.8, jury13_rad_rel_dice=0.8, target_cand_ml=0.5 if e else 0.0,
                         target_cand_touches_gt=False))
    agg = s9.summarise(pd.DataFrame(rows))
    assert agg.loc[("decoy", 4), "jury13_kept"] == 1.0 and agg.loc[("decoy", 4), "proposed_kept"] == 0.0


def test_local_region_and_containment():
    img = (0.3 + 0.05 * np.random.default_rng(1).standard_normal((2, 100, 96, 100))).astype(np.float32)
    net = _net()
    c, r, margin = (50, 48, 50), 4, 8
    target = s9.ball(img.shape[1:], c, r)
    region = s9.ball(img.shape[1:], c, r + margin)
    shell = region & ~s9.ball(img.shape[1:], c, r + 2)
    eps = 8 / 255
    adv, _ = s9.pgd_sw(net, img, target, "decoy", eps, 3, eps / 3, False, region, shell, 1.0)
    d = np.abs(adv - img).max(0)
    assert d[~region].max() == 0 and d[region].max() > 0                 # perturbation stays local
    adv0, _ = s9.pgd_sw(net, img, target, "decoy", eps, 3, eps / 3, False, region, None, 0.0)
    p_c, p_0, p_clean = _sw_prob(net, adv), _sw_prob(net, adv0), _sw_prob(net, img)
    assert p_c[target].mean() > p_clean[target].mean()                     # still raises the decoy
    assert p_c[shell].mean() < p_0[shell].mean()                           # containment keeps the shell lower
