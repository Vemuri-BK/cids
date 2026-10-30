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
                    if (c / "val_metrics.csv").exists():
                        shutil.copy(c / "val_metrics.csv", dst / "val_metrics.csv")
                    print(f"masks found in {hit.parent} ({len(list(hit.parent.glob('*.npz')))} files)")
                    return dst
            elif q.is_file() and zipfile.is_zipfile(q):
                if not (dst / "masks").is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(q) as z:
                        z.extractall(dst)
                    print(f"unpacked {q} -> {dst / 'masks'} ({len(list((dst / 'masks').glob('*.npz')))} masks)")
                shutil.copy(c / "summary.json", dst / "summary.json")
                if (c / "val_metrics.csv").exists():
                    shutil.copy(c / "val_metrics.csv", dst / "val_metrics.csv")
                return dst
    listing = [str(x.relative_to(p)) for x in list(p.rglob("*"))[:30]] if p.exists() else []
    raise SystemExit(f"no notebook 12 output (summary.json + masks) under {path}; found: {listing}")


def index_maps(dirs, subs=("heldout_pred", "val_pred", "test_pred"), unpack_to=None) -> dict:
    """{cid: npz path} of SegResNet probability maps (notebook 20 / s1_train_segresnet --save_prob) found as
    <dir>/<sub>/ folders or <dir>/<sub>.zip (unpacked once under unpack_to). A case in two folders -> error."""
    import zipfile
    idx = {}
    for d in [Path(x).expanduser() for x in dirs if str(x).strip()]:
        found = False
        for sub in subs:
            folder, zp = d / sub, d / f"{sub}.zip"
            if not folder.is_dir() and zp.exists():
                if unpack_to is None:
                    raise SystemExit(f"{zp} is zipped: pass unpack_to")
                folder = Path(unpack_to) / d.name / sub
                if not folder.is_dir():
                    with zipfile.ZipFile(zp) as z:
                        z.extractall(Path(unpack_to) / d.name)
            if not folder.is_dir():
                continue
            found = True
            for q in folder.glob("*.npz"):
                if q.stem in idx:
                    raise SystemExit(f"{q.stem} has maps in two places ({idx[q.stem]}, {q}): folds must not overlap")
                idx[q.stem] = q
        if not found:
            raise SystemExit(f"no map folder ({', '.join(subs)}) in {d}")
    return idx
