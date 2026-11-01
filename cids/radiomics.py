"""Radiomics clues for the candidate jury (S3h).

For every SegResNet candidate (connected component kept by the S3 pipeline) we measure, from the MRI cache only
(no ground truth), how its shape, contrast against the surrounding tissue and inner texture look. The jury
(cids.jury.MemoryJury) uses them as extra clues next to the finder / checker / memory clues.

Cache facts that shaped the feature set (checked on the server, 2026-11-01):
  * pre and post1 were scaled to [0, 1] SEPARATELY (1-99 percentiles, top clipped at 1.0) -> absolute
    enhancement values are not quantitative; every intensity feature is a CONTRAST against the candidate's own
    surroundings (ring / local box) or a within-patient comparison.
  * background is not reliably 0 (0.4-29 % exact zeros) -> no whole-body / whole-breast mask; the reference is
    local (4 mm ring, 15 mm box). Exact zeros (padding) and voxels of any SegResNet component (other
    candidates, satellites) are excluded from the reference regions (v2).
  * field of view differs strongly between and within hospitals -> no absolute position features (they would
    identify the hospital); only the distance to the patient's ANCHOR candidate (v3: the candidate SegResNet is
    most confident about -- highest mean probability, ties -> larger; v1/v2 used the largest candidate, which
    fails when a large background-enhancement blob is the biggest component).

Groups (1 mm isotropic voxels, so voxel units = mm):
  RAD_SHAPE     elongation, flatness (inertia eigenvalues), sphericity, solidity (convex hull), surface / volume
  RAD_CONTRAST  ring contrast-to-noise (subtraction), lesion / ring ratio (post1), rim - core (subtraction),
                z-score of the lesion against its 15 mm box (subtraction), enhancement relative to the local box
  RAD_TEXTURE   entropy, IQR, skewness of the subtraction inside the candidate; edge sharpness
  RAD_POS       distance to the patient's anchor (most confident) candidate; extraction stores the candidate
                centre (cx, cy, cz), the distance is computed from the candidate table (add_position)
  REL           within-patient relative clues (added after the memory clues: cids.radiomics.add_relative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

RAD_VERSION = 3          # bump whenever a feature definition changes -> cached radiomics tables are recomputed

RAD_SHAPE = ["rad_elongation", "rad_flatness", "rad_sphericity", "rad_solidity", "rad_surf_to_vol"]
RAD_CONTRAST = ["rad_ring_cnr", "rad_ring_ratio", "rad_rim_minus_core", "rad_local_z_sub", "rad_rel_enh"]
RAD_TEXTURE = ["rad_sub_entropy", "rad_sub_iqr", "rad_sub_skew", "rad_edge_sharpness"]
RAD_POS = ["rad_dist_to_dom_mm"]
RAD_ALL = RAD_SHAPE + RAD_CONTRAST + RAD_TEXTURE + RAD_POS
CENTRE = ["cx", "cy", "cz"]
RAD_EXTRACTED = RAD_SHAPE + RAD_CONTRAST + RAD_TEXTURE + CENTRE     # columns written by the extraction step
REL = ["rel_seg_prob", "rel_sam_soft", "rel_agree", "rel_mem_knn", "rel_ring_cnr"]
GROUPS = dict(shape=RAD_SHAPE, contrast=RAD_CONTRAST, texture=RAD_TEXTURE, position=RAD_POS, relative=REL)

RING_MM, RIM_MM, BOX_MM = 4.0, 2.0, 15.0
MARGIN = int(BOX_MM) + 2             # crop margin around the candidate's bounding box
EPS = 0.01


# ---------------------------------------------------------------------------- helpers
def _crop(shape, comp_idx, margin=MARGIN):
    lo = np.maximum(comp_idx.min(1) - margin, 0)
    hi = np.minimum(comp_idx.max(1) + margin + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _mean(x, m):
    return float(x[m].mean()) if m.any() else np.nan


def _std(x, m):
    return float(x[m].std()) if m.sum() > 1 else np.nan


# ---------------------------------------------------------------------------- groups
def shape_features(comp: np.ndarray) -> dict:
    """comp: boolean 3D mask (1 mm voxels)."""
    from skimage.measure import marching_cubes, mesh_surface_area
    idx = np.argwhere(comp).astype(np.float64)
    vol = float(len(idx))
    out = dict.fromkeys(RAD_SHAPE, np.nan)
    if vol < 2:
        return out
    ev = np.sort(np.linalg.eigvalsh(np.cov(idx.T)))[::-1]          # l1 >= l2 >= l3
    ev = np.clip(ev, 1e-9, None)
    out["rad_elongation"] = float(np.sqrt(ev[1] / ev[0]))
    out["rad_flatness"] = float(np.sqrt(ev[2] / ev[0]))
    padded = np.pad(comp, 1).astype(np.float32)
    try:
        verts, faces, _, _ = marching_cubes(padded, 0.5)
        area = float(mesh_surface_area(verts, faces))
    except (ValueError, RuntimeError):
        return out
    out["rad_sphericity"] = float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area) if area > 0 else np.nan
    out["rad_surf_to_vol"] = area / vol
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(verts).volume
        out["rad_solidity"] = float(min(vol / hull, 1.0)) if hull > 0 else np.nan
    except Exception:                                                # flat / degenerate component
        out["rad_solidity"] = np.nan
    return out


def contrast_features(pre: np.ndarray, post: np.ndarray, comp: np.ndarray, valid: np.ndarray) -> dict:
    """All arrays cropped around the candidate. valid = not padding (exact-zero voxels excluded)."""
    sub = post - pre
    d_out = ndimage.distance_transform_edt(~comp)
    d_in = ndimage.distance_transform_edt(comp)
    ring = (d_out > 0) & (d_out <= RING_MM) & valid
    box = (d_out > RING_MM) & (d_out <= BOX_MM) & valid
    rim, core = comp & (d_in <= RIM_MM), comp & (d_in > RIM_MM)
    if not core.any():                                               # thin candidate: no core
        core = comp
    m_in = _mean(sub, comp)
    return dict(
        rad_ring_cnr=(m_in - _mean(sub, ring)) / (_std(sub, ring) + EPS),
        rad_ring_ratio=(_mean(post, comp) + EPS) / (_mean(post, ring) + EPS),
        rad_rim_minus_core=_mean(sub, rim) - _mean(sub, core),
        rad_local_z_sub=(m_in - _mean(sub, box)) / (_std(sub, box) + EPS),
        rad_rel_enh=m_in - _mean(sub, box),
    ), (ring, d_in)


def texture_features(post: np.ndarray, sub: np.ndarray, comp: np.ndarray, ring: np.ndarray, d_in) -> dict:
    from scipy.stats import skew
    v = sub[comp]
    hist, _ = np.histogram(np.clip(v, -1, 1), bins=16, range=(-1, 1))
    p = hist[hist > 0] / hist.sum()
    g = np.sqrt(sum(gi ** 2 for gi in np.gradient(post)))
    edge = comp & (d_in <= 1.0)                                      # boundary shell of the candidate
    return dict(
        rad_sub_entropy=float(-(p * np.log2(p)).sum()),
        rad_sub_iqr=float(np.subtract(*np.percentile(v, [75, 25]))),
        rad_sub_skew=float(skew(v)) if len(v) > 2 and v.std() > 0 else 0.0,
        rad_edge_sharpness=(_mean(g, edge) + EPS) / (_mean(g, ring) + EPS),
    )


def candidate_features(img: np.ndarray, lab: np.ndarray, k: int) -> dict:
    """All RAD features (except position) of component k. img: (2, X, Y, Z) [pre, post1] from the cache."""
    full = lab == k
    idx = np.argwhere(full).T
    if idx.size == 0:
        return dict.fromkeys(RAD_SHAPE + RAD_CONTRAST + RAD_TEXTURE, np.nan)
    sl = _crop(lab.shape, idx)
    comp = full[sl]
    pre, post = img[0][sl], img[1][sl]
    non_pad = (pre != 0) | (post != 0)
    bg = non_pad & (lab[sl] == 0)                   # reference = tissue outside EVERY SegResNet component
    valid = bg if bg.any() else non_pad             # (a nearby satellite / other blob must not brighten the ring)
    out = shape_features(comp)
    c, (ring, d_in) = contrast_features(pre, post, comp, valid)
    out.update(c)
    out.update(texture_features(post, post - pre, comp, ring, d_in))
    return out


def case_features(cid: str, img: np.ndarray, lab: np.ndarray, ks) -> list:
    """Rows (cache_id, k, RAD_EXTRACTED) for the candidates ks of one patient: shape / contrast / texture features
    and the candidate centre (voxels = mm). The distance clue is computed later from the table (add_position)."""
    rows = []
    for k in ks:
        r = dict(cache_id=cid, k=int(k), **candidate_features(img, lab, int(k)))
        idx = np.argwhere(lab == k)
        c = idx.mean(0) if len(idx) else np.full(3, np.nan)
        r.update(cx=float(c[0]), cy=float(c[1]), cz=float(c[2]))
        rows.append(r)
    return rows


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """rad_dist_to_dom_mm = distance (mm) from each real candidate's centre to the patient's ANCHOR candidate:
    highest seg_prob_mean (ties -> larger size_vox). Needs columns seg_prob_mean, size_vox, cx, cy, cz."""
    df = df.copy()
    df["rad_dist_to_dom_mm"] = np.nan
    r = (df.k >= 0).to_numpy()
    if not r.any():
        return df
    real = df[r]
    order = real.sort_values(["cache_id", "seg_prob_mean", "size_vox"], ascending=[True, False, False])
    anchor = order.groupby("cache_id")[CENTRE].first()
    a = anchor.reindex(real.cache_id).to_numpy()
    df.loc[r, "rad_dist_to_dom_mm"] = np.linalg.norm(real[CENTRE].to_numpy() - a, axis=1)
    return df


def extract_case(args):
    """Worker for multiprocessing: (cid, cache_path, case_npz_path, ks) -> rows."""
    from cids.data.cache import load_case
    cid, cache_path, case_path, ks = args
    img, _ = load_case(cache_path)                 # the label is NOT used
    lab = np.load(case_path)["lab"]
    if lab.shape != img.shape[1:]:
        raise ValueError(f"{cid}: component labels {lab.shape} != cache {img.shape[1:]}")
    return case_features(cid, img, lab, list(ks))


# ---------------------------------------------------------------------------- relative clues
def versions() -> dict:
    """Library versions a saved jury depends on (pickled scikit-learn models do not load across versions)."""
    import platform
    import sklearn
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scikit_learn=sklearn.__version__, scipy=scipy.__version__)


def load_bundle(path) -> dict:
    """Load a saved jury bundle (s3h --final). Refuses with a clear message when the scikit-learn version
    differs from the one that saved it (S3f's Kaggle bundle failed on the server with a cryptic error);
    the bundle's score CSVs remain the version-independent record."""
    import pickle
    import sklearn
    path = str(path)
    meta_v = None
    try:
        with open(path, "rb") as fh:
            b = pickle.load(fh)
        meta_v = b.get("versions", {}).get("scikit_learn")
    except Exception as e:
        raise RuntimeError(f"cannot load {path} with scikit-learn {sklearn.__version__} ({type(e).__name__}: {e});"
                           f" install the version recorded at save time or use the saved score CSVs") from e
    if meta_v and meta_v != sklearn.__version__:
        raise RuntimeError(f"{path} was saved with scikit-learn {meta_v}, this environment has "
                           f"{sklearn.__version__}: results may differ -- install scikit-learn=={meta_v}")
    return b


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Within-patient relative clues over real candidates (k >= 0): the candidate compared with the strongest
    candidate of the SAME patient. Applied after the memory clues (cids.jury.MemoryJury post hook)."""
    df = df.copy()
    for c in REL:
        df[c] = 0.0
    r = (df.k >= 0).to_numpy()
    if not r.any():
        return df
    g = df[r].groupby("cache_id")
    mx = lambda col: g[col].transform("max").to_numpy()  # noqa: E731
    df.loc[r, "rel_seg_prob"] = df.loc[r, "seg_prob_mean"].to_numpy() / np.maximum(mx("seg_prob_mean"), 1e-6)
    df.loc[r, "rel_sam_soft"] = (df.loc[r, "sam_soft_ml"].to_numpy() + EPS) / (mx("sam_soft_ml") + EPS)
    df.loc[r, "rel_agree"] = df.loc[r, "agree"].to_numpy() - mx("agree")
    if "mem_knn" in df:
        df.loc[r, "rel_mem_knn"] = df.loc[r, "mem_knn"].to_numpy() - mx("mem_knn")
    if "rad_ring_cnr" in df:
        df.loc[r, "rel_ring_cnr"] = df.loc[r, "rad_ring_cnr"].to_numpy() - mx("rad_ring_cnr")
    return df


# ---------------------------------------------------------------------------- agreement (biomarker study)
def icc_31(a, b) -> float:
    """ICC(3,1): two-way mixed, consistency, single rater (Shrout & Fleiss) for paired measurements."""
    x = np.stack([np.asarray(a, float), np.asarray(b, float)], 1)
    n, k = x.shape
    gm = x.mean()
    ss_r = k * ((x.mean(1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(0) - gm) ** 2).sum()
    ss_e = ((x - gm) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


def lins_ccc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return float(2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))
