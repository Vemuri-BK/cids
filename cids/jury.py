"""Memory-augmented learned jury (S3e): decides which of SegResNet's candidates to keep.

13 clues per candidate = the 11 S3d clues + 2 memory clues computed from SAM's frozen embedding descriptor
(768-d, s3a.embedding_descriptor):
  mem_knn    similarity-weighted share of real tumours among the k nearest TRAINING candidates (cosine)
  mem_probe  logistic regression on the descriptor
Decision: keep candidates with P(real tumour) >= threshold (chosen on val); none -> keep the most likely one.

  MemoryJury.fit(train_rows, train_desc)   the deployable jury (memory = all training candidates)
  CrossFitMemoryJury(...)                  out-of-fold jury for TRAINING patients (5 folds by patient): every
                                           training case is judged by a jury whose memory, probe and trees never
                                           saw that patient -> its masks are as noisy as on new cases
Both are plain sklearn/numpy objects (pickle). No torch here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

JURY11 = ["log_seg_ml", "rel_size", "is_largest", "rank", "n_cand", "seg_prob_mean", "seg_prob_max",
          "agree", "log_sam_ml", "log_sam_soft_ml", "log_sam_over_seg"]
JURY13 = JURY11 + ["mem_knn", "mem_probe"]


def knn_score(query, mem, mem_y, k=10, temp=0.05):
    """Similarity-weighted share of real tumours among the k nearest memories (descriptors are unit vectors)."""
    sims = query.astype(np.float32) @ mem.astype(np.float32).T
    k = min(k, mem.shape[0])
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / temp)
    return (w * mem_y[idx]).sum(1) / w.sum(1)


def _probe():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")


def _trees():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def _real(df):
    return (df.k >= 0).to_numpy()


class MemoryJury:
    def __init__(self, k: int = 10, temp: float = 0.05, n_folds: int = 5, features=JURY13, post=None):
        """post: optional function rows -> rows applied after the memory clues (S3h: within-patient relative
        clues, cids.radiomics.add_relative). None (default) = the S3e/S3f jury unchanged."""
        self.k, self.temp, self.n_folds, self.features = k, temp, n_folds, list(features)
        self.post = post
        self.threshold = None

    def _post(self, rows):
        p = getattr(self, "post", None)              # juries pickled before S3h have no `post`
        return p(rows) if p is not None else rows

    # ---------------------------------------------------------------- training
    def fit(self, rows: pd.DataFrame, desc: np.ndarray):
        """rows: candidate table (k >= 0 = real candidate), desc: descriptors aligned with rows."""
        from sklearn.model_selection import GroupKFold
        r = _real(rows)
        tr, D = rows[r].reset_index(drop=True), np.asarray(desc)[r].astype(np.float32)
        y = tr.on_tumour.to_numpy().astype(float)
        groups = tr.cache_id.to_numpy()
        if len(set(groups)) < 2 or set(y) != {0.0, 1.0}:
            raise ValueError("memory needs >= 2 patients and both real and false candidates")
        # memory clues of the training candidates themselves: cross-fitted (never their own patient)
        mk, mp = np.zeros(len(tr)), np.zeros(len(tr))
        for fit, held in GroupKFold(min(self.n_folds, len(set(groups)))).split(D, groups=groups):
            mk[held] = knn_score(D[held], D[fit], y[fit], self.k, self.temp)
            pr = _probe().fit(D[fit], y[fit]) if len(set(y[fit])) > 1 else None
            mp[held] = pr.predict_proba(D[held])[:, 1] if pr is not None else 0.5
        tr = self._post(tr.assign(mem_knn=mk, mem_probe=mp))
        self.mem, self.mem_y = D.astype(np.float16), y
        self.probe = _probe().fit(D, y)
        self.trees = _trees().fit(tr[self.features].to_numpy(float), y.astype(int))
        self.train_ids = sorted(set(groups))
        return self

    # ---------------------------------------------------------------- inference
    def memory_clues(self, rows: pd.DataFrame, desc: np.ndarray) -> pd.DataFrame:
        rows = rows.copy(); r = _real(rows)
        rows["mem_knn"] = 0.0; rows["mem_probe"] = 0.0
        if r.any():
            D = np.asarray(desc)[r].astype(np.float32)
            rows.loc[r, "mem_knn"] = knn_score(D, self.mem, self.mem_y, self.k, self.temp)
            rows.loc[r, "mem_probe"] = self.probe.predict_proba(D)[:, 1]
        return self._post(rows)

    def proba(self, rows: pd.DataFrame, desc: np.ndarray) -> np.ndarray:
        """P(real tumour) per row (0 for the k = -1 'no candidate' rows)."""
        rows = self.memory_clues(rows, desc)
        p = np.zeros(len(rows)); r = _real(rows)
        if r.any():
            p[r] = self.trees.predict_proba(rows.loc[r, self.features].to_numpy(float))[:, 1]
        return p


class CrossFitMemoryJury:
    """Out-of-fold juries for training patients (folds by patient); other cases use the full jury."""

    def __init__(self, rows: pd.DataFrame, desc: np.ndarray, n_folds: int = 5, allow_tiny: bool = False, **kw):
        from sklearn.model_selection import GroupKFold
        r = _real(rows)
        ids = rows.cache_id.to_numpy()
        pats = np.array(sorted(set(ids[r])))
        self.full = MemoryJury(**kw).fit(rows, desc)
        self.fold_of, self.models, self.leaky_folds = {}, [], []
        for f, (fit, held) in enumerate(GroupKFold(min(n_folds, len(pats))).split(pats, groups=pats)):
            keep = np.isin(ids, pats[fit])
            try:
                self.models.append(MemoryJury(**kw).fit(rows[keep], np.asarray(desc)[keep]))
            except ValueError:
                if not allow_tiny:                   # fail closed: never let a held-out patient's own jury leak in
                    raise
                print(f"[warn] cross-fit fold {f}: too few candidates -> full jury (allow_tiny, tests only)")
                self.models.append(self.full)
                self.leaky_folds.append(f)
            self.fold_of.update({c: f for c in pats[held]})

    def proba(self, rows: pd.DataFrame, desc: np.ndarray) -> np.ndarray:
        """Rows may mix cases; each case is scored by the model that never saw it."""
        p = np.zeros(len(rows))
        ids = rows.cache_id.to_numpy()
        for f, m in [(None, self.full)] + list(enumerate(self.models)):
            sel = np.array([self.fold_of.get(c) == f for c in ids])
            if sel.any():
                p[sel] = m.proba(rows[sel], np.asarray(desc)[sel])
        return p


def keep_candidates(rows: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    """{cache_id: [component labels to keep]}; none >= threshold -> the most likely one."""
    out = {}
    rows = rows.assign(_p=p)
    for cid, g in rows.groupby("cache_id", sort=False):
        g = g[g.k >= 0]
        if not len(g):
            out[cid] = []
            continue
        ks = g.loc[g._p >= threshold, "k"].astype(int).tolist()
        out[cid] = ks or [int(g.loc[g._p.idxmax(), "k"])]
    return out
