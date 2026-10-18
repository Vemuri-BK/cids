import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    from cids.data.cache import load_case                          # origin rebuilds the exact SAM input crop
    from cids.fcouncil.sam_crops import sam_crop_from_case
    img, lb = load_case(path)
    lab, cands = s3.components(img[1] > 0.9)
    ref = s3.candidate_crops(img, lab, cands, lb)
    for j in range(2):
        assert np.allclose(sam_crop_from_case(img, lb, d["origin"][j]), ref["crops"][j, 0], atol=1e-5)
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


def test_learned_jury_table_dice_equals_mask_dice():
    import s3d_learned_jury as jury
    from cids.eval.metrics import dice
    rng = np.random.default_rng(0)
    gt = np.zeros((40, 40, 40), bool); gt[5:15, 5:15, 5:15] = True
    lab = np.zeros((40, 40, 40), np.int32)
    lab[8:18, 8:18, 8:18] = 1; lab[25:30, 25:30, 25:30] = 2; lab[2:5, 30:35, 30:35] = 3
    cands = [(k, float((lab == k).sum())) for k in (1, 2, 3)]
    refined = [dict(k=k, agree=a, sam_ml=s, sam_soft_ml=s) for k, a, s in ((1, 0.8, 1.0), (2, 0.2, 0.0), (3, 0.75, 0.2))]
    sprob = rng.random(gt.shape)
    df = pd.DataFrame(jury.candidate_rows("duke_001", sprob, lab, cands, refined, gt))
    assert df.on_tumour.tolist() == [1, 0, 0]
    tab = jury.rule_tables(df, proba=np.array([0.9, 0.1, 0.6]))
    assert np.isclose(tab["segjury_0.7"].iloc[0], dice(np.isin(lab, [1, 3]), gt))
    assert np.isclose(tab["oracle"].iloc[0], dice(lab == 1, gt))
    assert np.isclose(tab["seg_rel20"].iloc[0], dice(lab == 1, gt))     # 2 and 3 are < 20 % of the largest
    assert np.isclose(tab["learned_t0.50"].iloc[0], dice(np.isin(lab, [1, 3]), gt))
    assert np.isclose(tab["learned_t0.95"].iloc[0], dice(lab == 1, gt))             # none >= t -> most likely


def test_memory_knn_and_no_same_patient_leakage():
    import s3e_memory_jury as mem
    e = np.eye(4, dtype=np.float32)
    # query close to memory 0 (real) -> high score; close to memory 1 (false) -> low score
    assert mem.knn_score(e[[0]], e[[0, 1]], np.array([1., 0.]), k=2)[0] > 0.99
    assert mem.knn_score(e[[1]], e[[0, 1]], np.array([1., 0.]), k=2)[0] < 0.01
    # two patients with identical-looking candidates: A all real, B all false. Cross-fitted training scores
    # must come only from the OTHER patient -> A gets ~0, B gets ~1 (its own patient would give the opposite)
    D = np.stack([e[0]] * 4).astype(np.float16)
    tr = pd.DataFrame(dict(cache_id=["a", "a", "b", "b"], k=[1, 2, 1, 2], on_tumour=[1, 1, 0, 0]))
    tr2, _ = mem.memory_scores(tr, D, [], k=2, n_folds=2)
    assert np.allclose(tr2.mem_knn.to_numpy(), [0, 0, 1, 1])
