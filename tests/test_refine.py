import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cids.refine.data import RefinePatchDataset, channels, perturb_mask, prepare_case  # noqa: E402
from cids.refine.model import BoundaryRefiner  # noqa: E402


def _ball(shape, c, r):
    g = np.indices(shape)
    return ((g - np.array(c)[:, None, None, None]) ** 2).sum(0) <= r * r


@pytest.mark.parametrize("mode", ["freq", "spatial"])
def test_starts_as_identity_and_twins_match_in_size(mode):
    net = BoundaryRefiner(mode)
    m = _ball((32, 32, 32), (16, 16, 16), 7)
    x = torch.from_numpy(channels(np.random.rand(3, 32, 32, 32).astype(np.float32), m))[None]
    assert torch.equal((net(x) > 0)[0, 0], torch.from_numpy(m))
    n = {k: sum(p.numel() for p in BoundaryRefiner(k).parameters()) for k in ("freq", "spatial")}
    assert abs(n["freq"] - n["spatial"]) < 0.002 * n["freq"]          # same capacity (+- the split layer)


def test_corrections_only_near_boundary():
    net = BoundaryRefiner("freq")
    torch.nn.init.constant_(net.delta.bias, 50.0)                      # force "grow everywhere"
    m = _ball((32, 32, 32), (16, 16, 16), 5)
    x = torch.from_numpy(channels(np.zeros((3, 32, 32, 32), np.float32), m))[None]
    out = (net(x) > 0)[0, 0].numpy()
    far = ~_ball((32, 32, 32), (16, 16, 16), 5 + 8.5)
    assert out.sum() > m.sum() and not out[far].any()


def test_roi_counts_and_training_learns_to_grow(tmp_path):
    shape = (40, 40, 40)
    gt = _ball(shape, (20, 20, 20), 8)
    m0 = _ball(shape, (20, 20, 20), 6)                                 # under-segmented by 2 voxels
    img = np.zeros((2,) + shape, np.float16); img[1][gt] = 1.0
    lbl = gt[None].astype(np.uint8); lbl[0, 2, 2, 2] = 1               # one GT voxel outside the ROI
    np.savez(tmp_path / "duke_001.npz", image=img, label=lbl)
    np.savez(tmp_path / "m.npz", mask=m0.astype(np.uint8))
    work = tmp_path / "work"; work.mkdir()
    meta = prepare_case("duke_001", tmp_path / "duke_001.npz", tmp_path / "m.npz", work, margin=4)
    assert meta["gt_out"] == 1 and meta["gt_total"] == gt.sum() + 1
    rng = np.random.default_rng(0)
    assert perturb_mask(m0, rng).shape == m0.shape
    torch.manual_seed(0)
    net = BoundaryRefiner("freq", width=8)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    ds = RefinePatchDataset(["duke_001"], work, patch=24, n_patches=2, augment=False)
    d = np.load(work / "duke_001.npz")
    x_full = torch.from_numpy(channels(d["img"], d["m0"].astype(bool)))[None]
    pad = [(0, (-s) % 4) for s in x_full.shape[-3:]]
    x_full = torch.nn.functional.pad(x_full, [p for a in reversed(pad) for p in a])
    def dice():
        with torch.no_grad():
            p = (net(x_full) > 0)[0, 0][: d["gt"].shape[0], : d["gt"].shape[1], : d["gt"].shape[2]].numpy()
        g = d["gt"].astype(bool)
        return 2 * (p & g).sum() / (p.sum() + g.sum())
    d0 = dice()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from s7_refine import dice_bce
    for _ in range(60):
        b = ds[0]
        z = net(b["x"])
        loss = dice_bce(z, b["y"], b["x"][:, 5:6])                     # the loss used in training
        opt.zero_grad(); loss.backward(); opt.step()
    assert dice() > d0 + 0.05


def test_loss_is_zero_for_a_perfect_boundary_and_ignores_fixed_voxels():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from s7_refine import dice_bce
    gt, m0 = _ball((48, 48, 48), (24, 24, 24), 10), _ball((48, 48, 48), (24, 24, 24), 8)
    x = torch.from_numpy(channels(np.zeros((3, 48, 48, 48), np.float32), m0))[None]
    y = torch.from_numpy(gt[None, None].astype(np.float32))
    z0 = (2 * x[:, 3:4] - 1) * np.log(3)
    perfect = torch.where(x[:, 5:6] > 0, (2 * y - 1) * 20.0, z0)          # exact inside the band
    assert dice_bce(perfect, y, x[:, 5:6]).item() < 0.01                  # old loss gave ~0.97 here
    assert dice_bce(z0, y, x[:, 5:6]).item() > dice_bce(perfect, y, x[:, 5:6]).item() + 0.3


def test_band_oracle_fixes_band_errors_only():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import s7_oracle as orc
    from cids.eval.metrics import dice
    g = _ball((48, 48, 48), (24, 24, 24), 10)
    m0 = _ball((48, 48, 48), (24, 24, 24), 7)                         # 3 voxels too small
    far = np.zeros_like(g); far[2:6, 2:6, 2:6] = True                  # a tumour part far from M0
    d = orc.band_distance(m0, 12)
    for r, expect_perfect in ((4, True), (2, False)):
        a = d <= r
        o = (m0 & ~a) | (g & a)
        assert (dice(o, g) == 1.0) == expect_perfect
    a = d <= 8
    o = (m0 & ~a) | ((g | far) & a)
    assert not o[far].any()                                            # unreachable: stays missed


def test_pad_min_rejects_patch_not_divisible_by_4():
    from cids.refine.data import pad_min
    with pytest.raises(AssertionError):
        pad_min(np.zeros((1, 10, 10, 10)), 30)


def _toy_work(tmp_path):
    shape = (40, 40, 40)
    gt = _ball(shape, (20, 20, 20), 8)
    m0 = _ball(shape, (22, 20, 20), 7)                                 # shifted: both missed and extra voxels
    img = np.random.default_rng(0).random((2,) + shape).astype(np.float16)
    np.savez(tmp_path / "duke_001.npz", image=img, label=gt[None].astype(np.uint8))
    np.savez(tmp_path / "m.npz", mask=m0.astype(np.uint8))
    work = tmp_path / "work"; work.mkdir()
    prepare_case("duke_001", tmp_path / "duke_001.npz", tmp_path / "m.npz", work, margin=4)
    return work


def test_augmented_subtraction_channel_stays_post_minus_pre(tmp_path):
    work = _toy_work(tmp_path)
    ds = RefinePatchDataset(["duke_001"], work, patch=24, n_patches=4, augment=True, p_perturb=0.0)
    for e in range(3):
        ds.set_epoch(e)
        x = ds[0]["x"].numpy()
        assert np.allclose(x[:, 2], x[:, 1] - x[:, 0], atol=1e-5)


def test_error_sampling_centres_patches_on_correctable_errors(tmp_path):
    work = _toy_work(tmp_path)
    def err_frac(es):
        ds = RefinePatchDataset(["duke_001"], work, patch=8, n_patches=64, augment=False, error_sampling=es)
        b = ds[0]
        m, g, band = b["x"][:, 3] > 0.5, b["y"][:, 0] > 0.5, b["x"][:, 5] > 0.5
        return ((m ^ g) & band).float().mean().item()
    assert err_frac(True) > err_frac(False)                            # more error voxels per patch


@pytest.mark.parametrize("mode,gate", [("freq_lll", True), ("freq", False), ("spatial", False)])
def test_variants_start_as_identity_and_lll_matches_freq_size(mode, gate):
    net = BoundaryRefiner(mode, gate=gate)
    m = _ball((32, 32, 32), (16, 16, 16), 7)
    x = torch.from_numpy(channels(np.random.rand(3, 32, 32, 32).astype(np.float32), m))[None]
    assert torch.equal((net(x) > 0)[0, 0], torch.from_numpy(m))
    n = lambda k: sum(p.numel() for p in BoundaryRefiner(k).parameters())
    assert n("freq_lll") == n("freq")


def test_freq_lll_ignores_high_frequency_detail():
    net = BoundaryRefiner("freq_lll")
    ctx, det = net.bands(torch.rand(1, 3, 16, 16, 16))
    assert ctx.abs().sum() > 0 and det.abs().sum() == 0 and det.shape[1] == 21


def test_balanced_loss_counts_each_error_group_equally():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from s7_refine import balanced_loss
    shape = (1, 1, 8, 8, 8)
    m0 = torch.zeros(shape); m0[..., :4] = 1
    y = m0.clone(); y[..., 0, 0, 4] = 1                                # one missed voxel among many correct
    x = torch.cat([torch.zeros((1, 3, 8, 8, 8)), m0, torch.zeros(shape), torch.ones(shape)], 1)
    z = (2 * m0 - 1) * 3.0
    base = balanced_loss(z, y, x).item()
    z2 = z.clone(); z2[..., 0, 0, 4] = 3.0                              # fix the single missed voxel
    assert base - balanced_loss(z2, y, x).item() > 1.0                  # its group weighs as much as all correct ones


def test_fold_columns_hold_out_every_training_patient_once():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import pandas as pd
    from make_folds import fold_columns
    sp = fold_columns(pd.read_csv(Path(__file__).resolve().parents[1] / "configs" / "splits.csv"))
    tr = sp.split == "train"
    held = sum((sp[f"fold3_{k}"] == "heldout").astype(int) for k in range(3))
    assert (held[tr] == 1).all() and (held[~tr] == 0).all()
    for k in range(3):
        assert (sp.loc[~tr, f"fold3_{k}"] == sp.loc[~tr, "split"]).all()        # val / test untouched
        share = sp[sp[f"fold3_{k}"] == "heldout"].collection.value_counts() / sp[tr].collection.value_counts()
        assert share.between(0.30, 0.37).all()                                    # stratified by collection


def test_probability_channel_roundtrip_and_identity(tmp_path):
    from cids.refine.data import roi_channels
    shape = (40, 40, 40)
    gt = _ball(shape, (20, 20, 20), 8); m0 = _ball(shape, (21, 20, 20), 7)
    img = np.random.default_rng(0).random((2,) + shape).astype(np.float16)
    np.savez(tmp_path / "duke_001.npz", image=img, label=gt[None].astype(np.uint8))
    np.savez(tmp_path / "m.npz", mask=m0.astype(np.uint8))
    prob = np.round(np.clip(0.9 * gt + 0.05, 0, 1) * 255).astype(np.uint8)
    np.savez(tmp_path / "p.npz", prob=prob, mask=(prob > 127).astype(np.uint8))
    work = tmp_path / "work"; work.mkdir()
    meta = prepare_case("duke_001", tmp_path / "duke_001.npz", tmp_path / "m.npz", work, margin=4,
                        prob_path=tmp_path / "p.npz")
    d = np.load(work / "duke_001.npz")
    lo, hi = meta["bbox"]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    assert np.array_equal(d["prob"], prob[sl])                          # same crop as image and mask
    x = roi_channels(d)
    assert x.shape[0] == 7 and np.allclose(x[6], prob[sl] / 255.0)
    net = BoundaryRefiner("freq", width=8, extra_ch=1)
    pad = [(0, (-s) % 4) for s in x.shape[1:]]
    xt = torch.nn.functional.pad(torch.from_numpy(x)[None], [q for p_ in reversed(pad) for q in p_])
    out = (net(xt) > 0)[0, 0, : x.shape[1], : x.shape[2], : x.shape[3]].numpy()
    assert np.array_equal(out, d["m0"].astype(bool))                  # v2 still starts exactly at M0
    ds = RefinePatchDataset(["duke_001"], work, patch=24, n_patches=2, augment=True)
    assert ds[0]["x"].shape[1] == 7
    with pytest.raises(AssertionError):                                 # v2 model refuses v1 inputs
        net(xt[:, :6])


def test_index_maps_rejects_overlapping_folds(tmp_path):
    from cids.refine.source import index_maps
    for k in (0, 1):
        d = tmp_path / f"seg_fold{k}" / "heldout_pred"; d.mkdir(parents=True)
        np.savez(d / f"duke_00{k}.npz", prob=np.zeros(1, np.uint8))
    full = tmp_path / "seg_foldfull"
    for sub in ("val_pred", "test_pred"):
        (full / sub).mkdir(parents=True); np.savez(full / sub / f"x_{sub}.npz", prob=np.zeros(1, np.uint8))
    idx = index_maps([tmp_path / "seg_fold0", tmp_path / "seg_fold1", full])
    assert set(idx) == {"duke_000", "duke_001", "x_val_pred", "x_test_pred"}
    np.savez(tmp_path / "seg_fold1" / "heldout_pred" / "duke_000.npz", prob=np.zeros(1, np.uint8))
    with pytest.raises(SystemExit, match="two places"):
        index_maps([tmp_path / "seg_fold0", tmp_path / "seg_fold1"])
