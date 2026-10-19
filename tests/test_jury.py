import numpy as np
import pandas as pd

from cids.jury import JURY13, CrossFitMemoryJury, MemoryJury, keep_candidates, knn_score


def _table(n_pat=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, descs = [], []
    for p in range(n_pat):
        for k, real in ((1, 1), (2, 0), (3, 0)):
            r = dict(cache_id=f"duke_{p:03d}", k=k, on_tumour=real)
            for f in JURY13[:-2]:
                r[f] = rng.normal(real, 1.0)
            rows.append(r)
            d = rng.normal(0, 1, 16) + (3 if real else 0) * np.eye(16)[0]
            descs.append(d / np.linalg.norm(d))
    rows.append(dict(cache_id="duke_999", k=-1, on_tumour=0))       # a case without candidates
    descs.append(np.zeros(16))
    return pd.DataFrame(rows), np.stack(descs).astype(np.float32)


def test_memory_jury_scores_and_keeps_at_least_one():
    df, D = _table()
    j = MemoryJury(k=5).fit(df, D)
    p = j.proba(df, D)
    assert p[df.k.to_numpy() < 0].max() == 0
    real = df.k >= 0
    assert p[real & (df.on_tumour == 1)].mean() > p[real & (df.on_tumour == 0)].mean()
    keep = keep_candidates(df, p, 0.99)
    assert all(len(v) == 1 for c, v in keep.items() if c != "duke_999") and keep["duke_999"] == []


def test_crossfit_never_uses_own_patient():
    df, D = _table()
    cf = CrossFitMemoryJury(df, D, n_folds=4, k=5)
    for c, f in cf.fold_of.items():
        assert c not in cf.models[f].train_ids               # the jury of case c never saw c
    assert len(cf.fold_of) == 40


def test_knn_weights_nearest():
    e = np.eye(3, dtype=np.float32)
    assert knn_score(e[[0]], e, np.array([1.0, 0, 0]), k=3)[0] > 0.99


def test_crossfit_fails_closed_when_a_fold_is_too_small():
    import pytest
    df, D = _table(n_pat=2)                                   # each fold keeps 1 patient
    with pytest.raises(Exception):
        CrossFitMemoryJury(df, D, n_folds=5, k=5)
    cf = CrossFitMemoryJury(df, D, n_folds=5, k=5, allow_tiny=True)
    assert cf.leaky_folds                                                # recorded, not silent
