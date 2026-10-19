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
