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


def test_first_step_matches_official_zero_mask(sam):
    """decode(low_mask=None) must equal the official path prompt_encoder(points, masks=zeros) used in S0."""
    emb = torch.randn(1, 384, 8, 8, 8)
    c = torch.tensor([[[60., 64., 70.]]]); l = torch.ones(1, 1, dtype=torch.long)
    with torch.no_grad():
        ours = decode(sam, None, emb, c, l, None, None)
        sp, de = sam.prompt_encoder(points=[c, l], boxes=None, masks=torch.zeros(1, 1, 32, 32, 32))
        ref, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                  sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=False)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_realistic_boxes_messy_overlap_and_negatives_tumour_free():
    import random
    from cids.models.sam_lora import messy_box, negative_box, realistic_boxes
    rng = random.Random(0)
    gt = torch.zeros(4, 128, 128, 128, dtype=torch.bool)
    gt[:, 50:70, 40:60, 60:75] = True
    gt[3] = False                                               # empty crop -> must become a negative
    for _ in range(50):
        m = messy_box(gt[0], rng)
        a, b = m[0].long(), m[1].long()
        assert (m >= 0).all() and (m <= 127).all() and (b >= a).all()
        assert gt[0, a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].any()
        n = negative_box(gt[0], rng)
        a, b = n[0].long(), n[1].long()
        assert not gt[0, a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].any()
    boxes, neg = realistic_boxes(gt, rng, p_messy=0.4, p_neg=0.2)
    assert boxes.shape == (4, 2, 3) and bool(neg[3])
    _, neg_all = realistic_boxes(gt[:3], random.Random(1), p_messy=0.0, p_neg=1.0)
    assert neg_all.all()
    _, neg_none = realistic_boxes(gt[:3], random.Random(1), p_messy=0.5, p_neg=0.0)
    assert not neg_none.any()
