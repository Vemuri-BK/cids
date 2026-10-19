import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cids.provenance import archive, cases_digest, file_digest, require_cases, require_paths, save_npz_atomic


def test_digest_sees_same_size_content_change(tmp_path):
    f = tmp_path / "best.pt"
    f.write_bytes(b"abcd1234")
    a = file_digest(f)
    f.write_bytes(b"abcd9999")                 # same size
    b = file_digest(f)
    assert a["bytes"] == b["bytes"] and a["sha256"] != b["sha256"]


def test_archive_never_overwrites(tmp_path):
    for _ in range(3):
        f = tmp_path / "t.csv"; f.write_text("x")
        archive(f)
    assert len(list(tmp_path.glob("t.csv.stale-*"))) == 3 and not (tmp_path / "t.csv").exists()


def test_cases_digest_changes_with_cache_content(tmp_path):
    p = tmp_path / "duke_001.npz"; p.write_bytes(b"1" * 10)
    a = cases_digest([("duke_001", p)])
    p.write_bytes(b"1" * 11)
    assert cases_digest([("duke_001", p)]) != a


def test_missing_cases_fail_loudly(tmp_path):
    (tmp_path / "train").mkdir(); (tmp_path / "val").mkdir()
    (tmp_path / "train" / "duke_001.npz").write_bytes(b"x")
    sp = pd.DataFrame(dict(cache_id=["duke_001", "duke_002"], official=["train", "train"], split=["train", "train"]))
    with pytest.raises(SystemExit, match="1 of 2 cases missing"):
        require_cases(sp, tmp_path, "train")
    (tmp_path / "train" / "duke_002.npz").write_bytes(b"x")
    assert len(require_cases(sp, tmp_path, "train")) == 2
    with pytest.raises(SystemExit):
        require_paths([("duke_001", tmp_path / "nope.npz")], "train")


def test_atomic_npz(tmp_path):
    save_npz_atomic(tmp_path / "a.npz", x=np.arange(3))
    assert np.array_equal(np.load(tmp_path / "a.npz")["x"], np.arange(3))
    assert not list(tmp_path.glob("*.partial.npz"))
