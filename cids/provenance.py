"""Provenance helpers: make sure cached / exported results belong to the experiment that reuses them.

  file_digest(path)        content hash of a checkpoint / table (sha256, computed once per process)
  cases_digest(cases)      identity of the input cases: ids + cache file sizes (cheap; catches a different or
                           re-preprocessed cache, or a partially mounted one)
  archive(path)            move a stale file aside under a UNIQUE name (never overwrites, never deletes)
  require_cases(...)       fail loudly when requested split cases are missing from the cache
  require_paths(...)       the same for any list of (cache_id, path, ...) tuples (e.g. SAM embeddings)
  save_npz_atomic(...)     write <file>.npz via a temp file + rename (no half-written files after a crash)
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

PREPROC_VERSION = "cache-2ch-1mm-v1"      # bump if the 2-channel cache preprocessing ever changes
_DIGESTS: dict = {}


def file_digest(path) -> dict:
    """{name, bytes, sha256} of a file (sha256 over the full content, first 16 hex digits)."""
    if not path:
        return dict(name="")
    p = Path(path)
    if not p.exists():
        return dict(name=str(path), missing=True)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _DIGESTS[key] = h.hexdigest()[:16]
    return dict(name=p.name, bytes=st.st_size, sha256=_DIGESTS[key])


def cases_digest(cases: Iterable[Tuple[str, Path]]) -> dict:
    cases = list(cases)
    h = hashlib.md5()
    for cid, path in cases:
        p = Path(path)
        h.update(f"{cid}:{p.stat().st_size if p.exists() else -1}\n".encode())
    return dict(n_cases=len(cases), cases_md5=h.hexdigest(), preproc=PREPROC_VERSION)


def archive(path) -> Path:
    """Rename `path` to `path.stale-<timestamp>[-n]` (unique; works on Windows and POSIX)."""
    p = Path(path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst, n = p.with_name(f"{p.name}.stale-{stamp}"), 1
    while dst.exists():
        dst, n = p.with_name(f"{p.name}.stale-{stamp}-{n}"), n + 1
    p.rename(dst)
    return dst


def require_cases(split_df, root: Path, split: str) -> List[Tuple[str, Path]]:
    """All cases of one split as (cache_id, path); folder follows the OFFICIAL split (train/ or val/).
    Missing files -> SystemExit with a report (a partially mounted cache must not shrink an evaluation)."""
    rows = split_df[split_df.split == split] if "split" in split_df else split_df
    lst = [(c, Path(root) / ("train" if o == "train" else "val") / f"{c}.npz")
           for c, o in zip(rows.cache_id, rows.official)]
    missing = [c for c, p in lst if not p.exists()]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} of {len(lst)} cases missing in {root} "
                         f"(e.g. {missing[:5]}); attach the complete cache")
    return lst


def require_paths(lst, split: str, what: str = "files"):
    """lst = [(cache_id, path, ...)]: every path must exist, else SystemExit with a report."""
    missing = [x[0] for x in lst if not all(Path(q).exists() for q in x[1:])]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} of {len(lst)} cases have missing {what} "
                         f"(e.g. {missing[:5]}); attach the complete inputs")
    return lst


def save_npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
