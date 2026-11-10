"""S3z-b: confirm the S3z exploratory finding -- when the agent sends a click to SAM, use F_max = SAM u SegResNet.

S3z (train OOF, development) showed, AFTER the rules were run, that on the 30 patients the S3v-OOF agent sends to
SAM, F_max = max(P_sam, P_seg) > 0.5 (component at the click) scored 0.538 vs SAM alone 0.441. That observation
created this hypothesis, so train OOF cannot confirm it. S3z-b freezes ONE fixed change and tests it on data that
did not create it.

The change (nothing is re-trained or tuned)
  QC triage, the click and the S3v stage-2 decision (blob or SAM) are unchanged; only the mask delivered when the
  decision is SAM changes from S (SAM alone) to F_max.
Frozen rule (decision_rule.json, written by --step freeze BEFORE --step val reads any val outcome)
  R9 (primary)  population = val + test patients for which the S3v stage-2 agent (fitted on all train) picks SAM
                at the deep click (every patient clicked); mean Dice(F_max) - Dice(S), patient-bootstrap 95% CI
                lower bound > 0. Evaluated once, when the test part is approved by the guide (not run here).
  reported      val alone (direction check, few patients); the same at the 10% QC budget; harm counts
                (patients where F_max < S); train OOF numbers labelled 'development (hypothesis-generating)'.
Steps
  --step freeze  CPU: decision_rule.json (+ development numbers from the S3z train choices)
  --step val     CPU: val check from the S3z run (choices_val_deep.csv) -> val_result.json, val_patients.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import s3v_review_agent as V  # noqa: E402

BOOT = dict(n=5000, seed=0)
RULE = dict(
    change="when the S3v stage-2 agent picks SAM, deliver F_max = max(P_sam, P_seg) > 0.5 (component at the click) "
           "instead of SAM alone; triage, click and decision unchanged; nothing re-trained",
    decision="S3v stage-2 agent fitted on all 1,080 train patients (tool2 in S3v choices_<split>_deep.csv)",
    R9="PRIMARY: val + test patients with tool2 == S (deep click, every patient clicked): mean Dice(F_max) - "
       "Dice(S), patient-bootstrap 95% CI lower bound > 0; evaluated once, after guide approval of the test run",
    reported="val alone (direction); 10% QC budget subset; harm = #patients with F_max < S; train OOF = development",
    bootstrap=BOOT)


def boot(x, n=BOOT["n"], seed=BOOT["seed"]):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    v = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def compare(d: pd.DataFrame) -> dict:
    """F_max vs S on the patients the agent sends to SAM (all clicked, and the 10% QC-clicked subset)."""
    sel10 = V.select(d.qc.to_numpy(), d.cache_id.to_numpy(), 0.10)
    out = {}
    for name, m in (("all_clicked", d.tool2 == "S"), ("qc10_clicked", (d.tool2 == "S") & sel10)):
        g = d[m]
        diff = g.FM_dice - g.S_dice
        out[name] = dict(n=int(len(g)), dice_S=float(g.S_dice.mean()) if len(g) else None,
                         dice_F_max=float(g.FM_dice.mean()) if len(g) else None,
                         dice_blob=float(g.Cn_dice.mean()) if len(g) else None,
                         diff=float(diff.mean()) if len(g) else None, ci95=boot(diff),
                         better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum()))
    agent_s = np.where(d.tool2 == "S", d.S_dice, d.Cn_dice)
    agent_f = np.where(d.tool2 == "S", d.FM_dice, d.Cn_dice)
    out["mean_dice_everyone_clicked"] = dict(agent_with_S=float(agent_s.mean()), agent_with_F_max=float(agent_f.mean()))
    return out


def load(s3z: Path, split: str) -> pd.DataFrame:
    f = s3z / f"choices_{split}_deep.csv"
    if not f.exists():
        raise SystemExit(f"{f} missing: run s3z_fusion.py --step analyse first")
    d = pd.read_csv(f)
    need = {"cache_id", "tool2", "qc", "S_dice", "FM_dice", "Cn_dice"}
    if not need <= set(d.columns):
        raise SystemExit(f"{f}: missing columns {sorted(need - set(d.columns))}")
    return d


def step_freeze(a, out):
    rj = out / "decision_rule.json"
    if rj.exists():
        raise SystemExit(f"{rj} already exists: the rule is frozen (move it to outputs/obsolete/ to re-freeze)")
    dev = compare(load(Path(a.s3z_out).expanduser(), "train_oof"))
    rule = dict(RULE, frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                development_train_oof=dict(note="hypothesis-generating (S3z), not evidence", **dev))
    rj.write_text(json.dumps(rule, indent=1, default=float))
    print(json.dumps(rule, indent=1, default=float))
    print(f"\nFROZEN -> {rj}\nnext: --step val")


def step_val(a, out):
    rj = out / "decision_rule.json"
    if not rj.exists():
        raise SystemExit("run --step freeze first (the rule must be frozen before val is read)")
    d = load(Path(a.s3z_out).expanduser(), "val")
    res = dict(rule_frozen_utc=json.loads(rj.read_text())["frozen_utc"], val=compare(d),
               note="direction check only; R9 is decided on val + test pooled after the guide approves the test run")
    g = d[d.tool2 == "S"][["cache_id", "qc", "Cn_dice", "S_dice", "FM_dice"]].copy()
    g["diff"] = g.FM_dice - g.S_dice
    g.to_csv(out / "val_patients.csv", index=False)
    (out / "val_result.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 200)
    print("val patients the agent sends to SAM (every patient clicked):")
    print(g.round(4).to_string(index=False) if len(g) else "  none")
    print("\n" + json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["freeze", "val"], required=True)
    ap.add_argument("--s3z_out", required=True, help="the S3z run (choices_<split>_deep.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "test" in Path(a.s3z_out).name.lower():
        raise SystemExit("test files are refused here (the test part needs the guide's approval)")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    step_freeze(a, out) if a.step == "freeze" else step_val(a, out)


if __name__ == "__main__":
    main()
