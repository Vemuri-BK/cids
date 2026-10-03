import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.data.volumes import PatchDataset, pad_to, three_channel, unpad  # noqa: E402
from cids.eval.metrics import all_metrics, dice, hd95, nsd  # noqa: E402
from cids.models.segresnet_text import TextSegResNet  # noqa: E402
from cids.prompts.sampling import pick_prompt  # noqa: E402


def test_film_zero_init_equals_image_only():
    torch.manual_seed(0)
    a = TextSegResNet(text=False)
    b = TextSegResNet(text=True, text_dim=16)
    b.load_state_dict(a.state_dict(), strict=False)          # same conv weights
    x = torch.randn(2, 3, 32, 32, 32)
    t = torch.randn(2, 16)
    a.eval(); b.eval()
    with torch.no_grad():
        assert torch.allclose(a(x), b(x, t), atol=1e-6)          # FiLM starts as identity
    assert b(x, t).shape == (2, 1, 32, 32, 32)


def test_film_learns_something():
    m = TextSegResNet(text=True, text_dim=8)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    x, t = torch.randn(1, 3, 32, 32, 32), torch.randn(1, 8)
    for _ in range(3):
        loss = m(x, t).mean(); opt.zero_grad(); loss.backward(); opt.step()
    assert m.film_bott.net[-1].weight.abs().sum() > 0


def test_metrics_basic():
    g = np.zeros((40, 40, 40), bool); g[10:20, 10:20, 10:20] = True
    assert dice(g, g) == 1.0 and hd95(g, g) == 0.0 and nsd(g, g) == 1.0
    p = np.roll(g, 3, axis=0)
    assert 0.6 < dice(p, g) < 0.8 and 2.5 <= hd95(p, g) <= 3.5
    m = all_metrics(np.zeros_like(g), g)
    assert m["dice"] == 0 and np.isnan(m["hd95"]) and m["nsd"] == 0


def test_patches_and_padding(tmp_path):
    img = np.random.rand(2, 50, 60, 40).astype(np.float16)
    lbl = np.zeros((1, 50, 60, 40), np.uint8); lbl[0, 20:30, 25:35, 15:25] = 1
    np.savez_compressed(tmp_path / "x.npz", image=img, label=lbl)
    ds = PatchDataset([("duke_001", tmp_path / "x.npz")], patch=48, n_patches=4, pos_ratio=1.0)
    it = ds[0]
    assert it["image"].shape == (4, 3, 48, 48, 48) and it["label"].shape == (4, 1, 48, 48, 48)
    assert all(it["label"][k].sum() > 0 for k in range(4))       # positive patches contain tumour
    v, pads = pad_to(three_channel(img.astype(np.float32)), (48, 48, 48))
    assert v.shape == (3, 50, 60, 48) and unpad(v, pads).shape == (3, 50, 60, 40)


def test_prompt_sampling_dropout():
    row = dict(age_bin="35-45", menopause="pre-menopausal", ethnicity="Asian", subtype="triple-negative",
               field_strength="3T", manufacturer="GE", view="axial", laterality="bilateral",
               fat_suppressed="yes", implant="no", num_phases=4,
               narrative_1="n1", narrative_2="n2", narrative_3="n3")
    assert "35-45" in pick_prompt(row, "S", 0.0)
    assert pick_prompt(row, "N", 0.0, random.Random(0)) in ("n1", "n2", "n3")
    txt = pick_prompt(row, "S", 1.0, random.Random(0))
    assert "unknown" in txt and "GE" not in txt and "35-45" not in txt
