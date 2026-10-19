"""Locate notebook 12 (S3f) output: summary.json + masks of every case, in any layout Kaggle produces."""
from __future__ import annotations

from pathlib import Path


def find_s3f(path: str, unpack_to: Path) -> Path:
    """Folder with summary.json + masks/<cid>.npz. Accepts every way Kaggle ships notebook 12's masks:
    masks/ with the files, masks.zip (unpacked here), or a zip that Kaggle already extracted into a folder
    (masks/masks/*.npz, masks.zip/masks/*.npz, ...). Nested layouts are linked to <unpack_to>/masks."""
    import shutil
    import zipfile
    p = Path(path)
    cands = [q.parent for q in p.rglob("summary.json")] if p.exists() else []
    if p.name == "summary.json":
        cands = [p.parent]
    for c in cands:
        if (c / "masks").is_dir() and any((c / "masks").glob("*.npz")):
            return c
    dst = Path(unpack_to)
    for c in cands:
        for name in ("masks", "masks.zip"):
            q = c / name
            if q.is_dir():                                   # extracted by Kaggle, possibly nested
                hit = next(q.rglob("*.npz"), None)
                if hit is not None:
                    dst.mkdir(parents=True, exist_ok=True)
                    if not (dst / "masks").exists():
                        (dst / "masks").symlink_to(hit.parent.resolve(), target_is_directory=True)
                    shutil.copy(c / "summary.json", dst / "summary.json")
                    print(f"masks found in {hit.parent} ({len(list(hit.parent.glob('*.npz')))} files)")
                    return dst
            elif q.is_file() and zipfile.is_zipfile(q):
                if not (dst / "masks").is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(q) as z:
                        z.extractall(dst)
                    print(f"unpacked {q} -> {dst / 'masks'} ({len(list((dst / 'masks').glob('*.npz')))} masks)")
                shutil.copy(c / "summary.json", dst / "summary.json")
                return dst
    listing = [str(x.relative_to(p)) for x in list(p.rglob("*"))[:30]] if p.exists() else []
    raise SystemExit(f"no notebook 12 output (summary.json + masks) under {path}; found: {listing}")
