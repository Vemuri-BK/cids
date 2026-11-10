"""Tests for scripts/s3y_backup_sam.py and the s2a --select click1 option."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3y_backup_sam as Y  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "s3y_backup_sam.py")


def _data(n=40):
    tr = [f"duke_{i:03d}" for i in range(n)]
    sp = pd.DataFrame(dict(cache_id=tr + ["nact_v", "ispy2_t"], split=["train"] * n + ["val", "test"],
                           official=["train"] * (n + 1) + ["test"]))
    folds = pd.DataFrame(dict(cache_id=tr, outer_fold=[i % 5 for i in range(n)]))
    rows = []
    for i, c in enumerate(tr):
        for click in ("deep", "offset"):
            rows.append(dict(split="train_oof", cache_id=c, click=click, Cn_dice=0.3 if i % 4 == 0 else 0.8,
                             B_dice=0.2 if i % 10 == 1 else 0.9, S_dice=0.5, sam_decoder=f"fold{i % 5}"))
    for click in ("deep", "offset"):
        rows.append(dict(split="val", cache_id="nact_v", click=click, Cn_dice=0.7, B_dice=0.7, S_dice=0.6,
                         sam_decoder="main"))
    return sp, folds, pd.DataFrame(rows)


def test_make_splits():
    sp, folds, oc = _data()
    hard = Y.hard_flags(oc)
    assert hard.sum() == 14                                    # i%4==0 (10) + i%10==1 (4: 1,11,21,31)
    df = Y.make_splits(sp, folds, hard)
    for k in range(5):
        col = df[f"s3y_f{k}"]
        own = set(folds.cache_id[folds.outer_fold == k])
        tr = set(df.cache_id[col == "train"])
        assert not tr & own                                    # out-of-fold
        h = {c for c in tr if hard[c]}
        assert h == {c for c in hard.index[hard] if c not in own}
        assert len(tr) == 2 * len(h)                           # + the same number of non-hard
        assert col[df.cache_id == "nact_v"].item() == "val" and col[df.cache_id == "ispy2_t"].item() == "test"
    assert (df.s3y_full == "train").sum() == 28


def test_steps(tmp_path):
    sp, folds, oc = _data()
    sp.to_csv(tmp_path / "splits.csv", index=False); folds.to_csv(tmp_path / "folds.csv", index=False)
    cur = tmp_path / "cur"; cur.mkdir(); oc.to_csv(cur / "outcomes.csv", index=False)
    pd.DataFrame(dict(cache_id=["duke_000", "duke_004", "duke_008"], tool2=["S", "S", "Cn"])).to_csv(
        cur / "choices_train_oof_deep.csv", index=False)
    r = subprocess.run([sys.executable, SCRIPT, "--step", "splits", "--splits", str(tmp_path / "splits.csv"),
                        "--s3v_out", str(cur), "--folds", str(tmp_path / "folds.csv"), "--out", str(tmp_path / "y")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((tmp_path / "y" / "decision_rule.json").read_text())["recipe"]["select"] == "click1"
    new = tmp_path / "new"; new.mkdir()
    o2 = oc.copy()
    o2.loc[o2.cache_id.isin(["duke_000", "duke_004"]), "S_dice"] = [0.8, 0.8, 0.7, 0.7]
    o2.to_csv(new / "outcomes.csv", index=False)
    (new / "rules.json").write_text(json.dumps(dict(rules=dict(train_oof=dict(R4=dict(passes=True, gain_on_chosen=0.25))))))
    r = subprocess.run([sys.executable, SCRIPT, "--step", "analyse", "--s3v_out", str(cur), "--new_s3v_out", str(new),
                        "--out", str(tmp_path / "y")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "y" / "rules.json").read_text())
    assert res["R6a"]["n"] == 2 and abs(res["R6a"]["diff"] - 0.25) < 1e-9 and res["R6a"]["passes"]
    assert res["R6b"]["passes"]
    o2["sam_decoder"] = "main"
    o2.to_csv(new / "outcomes.csv", index=False)
    r = subprocess.run([sys.executable, SCRIPT, "--step", "analyse", "--s3v_out", str(cur), "--new_s3v_out", str(new),
                        "--out", str(tmp_path / "y")], capture_output=True, text=True)
    assert r.returncode != 0 and "not out-of-fold" in r.stdout + r.stderr


def test_s2a_select_option():
    src = (ROOT / "scripts" / "s2a_sam_finetune.py").read_text()
    assert '"--select", choices=["default", "click1"], default="default"' in src
    assert 'm["score"] = m["click1"]' in src
