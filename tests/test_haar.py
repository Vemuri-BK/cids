import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.models.haar import HaarDWT3D, HaarIDWT3D, InstructedShrink  # noqa: E402


def test_perfect_reconstruction():
    x = torch.randn(2, 4, 16, 16, 8)
    y = HaarDWT3D()(x)
    assert y.shape == (2, 4, 8, 8, 8, 4)
    assert torch.allclose(HaarIDWT3D()(y), x, atol=1e-5)


def test_energy_preserved():
    x = torch.randn(1, 2, 8, 8, 8)
    y = HaarDWT3D()(x)
    assert torch.allclose(x.pow(2).sum(), y.pow(2).sum(), rtol=1e-4)


def test_constant_volume_only_in_LLL():
    y = HaarDWT3D()(torch.ones(1, 1, 4, 4, 4))
    assert y[:, :, 1:].abs().max() < 1e-6


def test_shrink_shape_and_sparsity():
    band, t = torch.randn(2, 4, 8, 8, 8), torch.randn(2, 256)
    out = InstructedShrink(4)(band, t)
    assert out.shape == band.shape
    assert (out == 0).float().mean() > 0.3
