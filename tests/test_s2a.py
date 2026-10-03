import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("medim")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.models.sam_lora import BoxCorners, add_lora, decode, gt_boxes, sample_clicks  # noqa: E402
from cids.models.sam_med3d import load_sam_med3d  # noqa: E402


@pytest.fixture(scope="module")
def sam():
    torch.manual_seed(0)
    m = load_sam_med3d(None, "cpu")
    m.image_encoder = torch.nn.Identity()
    return m


def test_lora_zero_init_and_size(sam):
    import copy
    m = copy.deepcopy(sam)
    emb = torch.randn(2, 384, 8, 8, 8)
    c = torch.tensor([[[60., 64., 70.]], [[10., 20., 30.]]]); l = torch.ones(2, 1, dtype=torch.long)
    with torch.no_grad():
        ref = decode(m, None, emb, c, l, None, None)
        names = add_lora(m.mask_decoder, ("q_proj", "v_proj"), 8, 16)
        out = decode(m, None, emb, c, l, None, None)
    assert len(names) == 14
    n = sum(p.numel() for n_, p in m.named_parameters() if n_.endswith((".A", ".B")))
    assert 50_000 < n < 100_000
    assert torch.allclose(ref, out, atol=1e-5) and out.shape == (2, 1, 32, 32, 32)


def test_box_and_mixed_prompts(sam):
    box = BoxCorners(sam.prompt_encoder)
    gt = torch.zeros(2, 128, 128, 128, dtype=torch.bool)
    gt[0, 40:60, 50:70, 30:45] = True; gt[1, 5:9, 100:120, 60:61] = True
    b = gt_boxes(gt)
    assert b[0].tolist() == [[40, 50, 30], [59, 69, 44]]
    bj = gt_boxes(gt, jitter=5, gen=torch.Generator().manual_seed(0))
    assert (bj[:, 1] >= bj[:, 0]).all() and bj.min() >= 0 and bj.max() <= 127
    emb = torch.randn(2, 384, 8, 8, 8)
    c, l = sample_clicks(torch.zeros_like(gt), gt, torch.Generator().manual_seed(1))
    low = decode(sam, box, emb, None, None, b, None)                    # box only
    low2 = decode(sam, box, emb, c, l, b, low)                          # box + click + previous mask
    assert low.shape == low2.shape == (2, 1, 32, 32, 32)
    low2.mean().backward()
    assert box.corner.grad is not None


def test_click_rules():
    g = torch.Generator().manual_seed(0)
    gt = torch.zeros(1, 128, 128, 128, dtype=torch.bool); gt[0, 50:60, 50:60, 50:60] = True
    c, l = sample_clicks(torch.zeros_like(gt), gt, g)                   # first click: inside tumour, positive
    i = c[0, 0].long()
    assert gt[0, i[0], i[1], i[2]] and l.item() == 1
    prev = gt.clone(); prev[0, 50:60, 50:60, 50:55] = False             # missed half -> positive in the FN part
    c, l = sample_clicks(prev, gt, g); i = c[0, 0].long()
    assert l.item() == 1 and gt[0, i[0], i[1], i[2]] and not prev[0, i[0], i[1], i[2]]
    prev = gt.clone(); prev[0, 0:5, 0:5, 0:5] = True                    # false positive only -> negative there
    c, l = sample_clicks(prev, gt, g); i = c[0, 0].long()
    assert l.item() == 0 and (i < 5).all()
