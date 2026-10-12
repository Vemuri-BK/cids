import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s2a_sam_finetune as s2  # noqa: E402
import s2c_mine_candidates as mine  # noqa: E402
import s3a_seg2sam as s3  # noqa: E402


class _FakeSAM:
    @staticmethod
    def image_encoder(x):
        return torch.zeros(len(x), 384, 8, 8, 8)


def _case(tmp_path):
    img = np.zeros((2, 100, 100, 90), np.float16)
    lbl = np.zeros((1, 100, 100, 90), np.uint8); lbl[0, 20:35, 20:35, 20:32] = 1
    img[1, 20:35, 20:35, 20:32] = 1.0          # tumour
    img[1, 70:80, 65:75, 55:63] = 1.0          # tumour-like blob elsewhere (a hard negative)
    img[:, 5:95, 5:95, 5:85] += np.float16(0.1)
    p = tmp_path / "duke_001.npz"; np.savez(p, image=img, label=lbl)
    return p, lbl[0]


def test_mining_labels_true_and_false_candidates(tmp_path, monkeypatch):
    path, lbl = _case(tmp_path)
    monkeypatch.setattr(s3, "seg_prob", lambda seg, image, amp, overlap=0.5: (image[1].numpy() > 0.9).astype(np.float32))
    from cids.data.volumes import VolumeDataset
    item = VolumeDataset([("duke_001", path)], 96)[0]
    d = mine.mine_case(None, _FakeSAM(), item, path, 0.5, 5, False, 0.5)
    assert d["emb"].shape == (2, 384, 8, 8, 8) and d["box"].shape == (2, 2, 3)
    assert sorted(d["is_true"].tolist()) == [False, True]
    t = int(np.flatnonzero(d["is_true"])[0])
    gt = np.unpackbits(d["gt_bits"][t], count=128 ** 3).reshape(128, 128, 128).astype(bool)
    lo, hi = d["box"][t].astype(int) + np.array([[-1], [2]])     # 1 voxel: nearest vs trilinear 1.5 mm grids
    inside = gt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum()
    assert inside > 0.95 * gt.sum() > 0                      # true candidate's box covers the tumour crop

    out = tmp_path / "cands" / "train"; out.mkdir(parents=True)
    np.savez(out / "duke_001.npz", **d)
    c = s2.load_cands(tmp_path / "cands", "train")
    assert len(c["false_idx"]) == 1 and len(c["true_idx"]) == 1
    emb, bx, tg = s2.cand_batch(c, np.array([c["false_idx"][0], c["true_idx"][0]]))
    assert tg[0].sum() == 0 and tg[1].sum() == gt.sum()          # false -> EMPTY target, true -> tumour


def test_candidate_crops_same_as_pipeline_without_label(tmp_path):
    path, lbl = _case(tmp_path)
    from cids.data.cache import load_case
    img, _ = load_case(path)
    lab, cands = s3.components(img[1] > 0.9)
    a = s3.candidate_crops(img, lab, cands)
    b = s3.candidate_crops(img, lab, cands, lbl)
    assert a["gt"] is None and np.array_equal(a["crops"], b["crops"]) and np.array_equal(a["boxes"], b["boxes"])
