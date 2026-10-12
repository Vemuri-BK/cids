import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.cache import make_sam_crops, resample  # noqa: E402
from cids.eval.postproc import rel20  # noqa: E402
from cids.fcouncil import BAND_GROUPS, band_energy, split_groups  # noqa: E402
from cids.fcouncil.adapter import SamFreqAdapter  # noqa: E402
from cids.fcouncil.data import CouncilPatchDataset, CouncilVolumeDataset  # noqa: E402
from cids.fcouncil.losses import council_loss  # noqa: E402
from cids.fcouncil.model import FreqCouncil  # noqa: E402
from cids.fcouncil.sam_crops import sam_crop  # noqa: E402


def test_band_groups_cover_all_8_bands_once():
    assert sorted(b for g in BAND_GROUPS for b in g) == list(range(8))
    bands = torch.randn(2, 3, 8, 4, 4, 4)
    gs = split_groups(bands)
    assert [g.shape[1] for g in gs] == [3, 9, 9, 3]
    assert torch.equal(gs[3], bands[:, :, 7]) and band_energy(bands).shape == (2, 4)


def test_adapter_starts_at_zero_and_has_unit_band_weights():
    a = SamFreqAdapter()
    r = a(torch.randn(2, 1, 128, 128, 128))
    assert r.shape == (2, 384, 8, 8, 8) and torch.count_nonzero(r) == 0      # SAM unchanged at step 0
    assert all(abs(v - 1.0) < 1e-6 for v in a.readout().values()) and len(a.readout()) == 8


def test_sam_crop_matches_s0_crops():
    rng = np.random.default_rng(0)
    img = rng.random((2, 90, 80, 70)).astype(np.float32)
    lbl = np.zeros((90, 80, 70), np.uint8); lbl[30:50, 30:45, 20:40] = 1
    img15, lbl15 = resample(img, lbl)
    ref = make_sam_crops(img15, lbl15, np.array([[30, 25, 20]]), modes=("post1",))
    mine = sam_crop(img15, ref["origin"][0])
    assert np.allclose(mine, ref["post1"][0, 0], atol=1e-5)


def test_council_independent_experts_equal_weights_at_init():
    torch.manual_seed(0)
    m = FreqCouncil(width=8)
    x = torch.rand(2, 5, 32, 32, 32)
    o = m(x)
    assert o["final"].shape == (2, 1, 32, 32, 32) and o["experts"].shape == (2, 4, 32, 32, 32)
    assert torch.allclose(o["weights"], torch.full_like(o["weights"], 0.25))       # gate starts uniform
    assert torch.allclose(o["final"], o["experts"].mean(1, keepdim=True), atol=1e-5)
    ids = [{id(p) for p in e.parameters()} for e in m.experts]
    assert all(not (ids[i] & ids[j]) for i in range(4) for j in range(i + 1, 4))  # no shared weights
    assert o["uncertainty"].shape == (2, 1, 32, 32, 32)


def test_dual_supervision_loss_and_ablation():
    torch.manual_seed(0)
    m = FreqCouncil(width=8)
    x = torch.rand(2, 5, 32, 32, 32)
    gt = torch.zeros(2, 1, 32, 32, 32); gt[:, :, 8:20, 8:20, 8:20] = 1
    o = m(x)
    L = council_loss(o, gt, x[:, 3:4], x[:, 4:5])
    assert torch.isfinite(L["total"]) and L["kd"] > 0
    L["total"].backward()
    assert all(e.out.weight.grad is not None for e in m.experts)
    assert council_loss(m(x), gt, x[:, 3:4], x[:, 4:5], use_teachers=False)["kd"] == 0
    far = council_loss(m(x), gt, torch.zeros_like(gt), torch.ones_like(gt))      # teachers disagree everywhere
    assert far["kd"] == 0


def test_rel20_keeps_large_components():
    m = np.zeros((40, 40, 40), bool)
    m[2:12, 2:12, 2:12] = True; m[20:24, 20:24, 20:24] = True; m[30:37, 30:37, 30:37] = True
    r = rel20(m)
    assert r[5, 5, 5] and not r[21, 21, 21] and r[33, 33, 33]


def test_council_data_patches_and_volumes(tmp_path):
    cache = tmp_path / "train"; cache.mkdir(); maps = tmp_path / "maps"; maps.mkdir()
    img = np.random.rand(2, 100, 100, 100).astype(np.float16)
    lbl = np.zeros((1, 100, 100, 100), np.uint8); lbl[0, 40:60, 40:60, 40:60] = 1
    np.savez(cache / "duke_001.npz", image=img, label=lbl)
    np.savez(maps / "duke_001.npz", seg=(lbl[0] * 255).astype(np.uint8), sam=(lbl[0] * 200).astype(np.uint8))
    cases = [("duke_001", cache / "duke_001.npz")]
    it = CouncilPatchDataset(cases, maps, patch=48, n_patches=5, map_aug=False)[0]
    assert it["image"].shape == (5, 5, 48, 48, 48) and it["label"].shape == (5, 1, 48, 48, 48)
    assert torch.equal(it["image"][:, 3:4] > 0.5, it["label"] > 0.5)               # finder map aligned (flips)
    v = CouncilVolumeDataset(cases, maps, 96)[0]
    assert v["image"].shape == (5, 100, 100, 100)
