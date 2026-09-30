"""Re-apply the CURRENT filters to already-generated narrative candidates (no GPU, no LLM).

Every candidate the LLM produced is stored in <target>_progress.jsonl with its text and
similarity score S. When the fact check changes, we re-check every candidate, recompute
diversity, and re-select the top-k -- instead of regenerating.

Usage:
    python scripts/refilter_narratives.py --prompts data/prompts/cids_prompts.csv \
        --progress data/narratives/train_style_progress.jsonl --target train_style \
        --out data/narratives/refiltered_train_style.csv
Chain the two targets by passing the first output as --prompts of the second run.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cids.prompts.narrative import (Candidate, FilterConfig, check_facts,  # noqa: E402
                                    jaccard_distance, select_top_k)

FIELD_COLS = ["age_bin", "menopause", "ethnicity", "subtype", "field_strength", "manufacturer",
              "view", "laterality", "fat_suppressed", "implant", "num_phases"]


def refilter(prompts: pd.DataFrame, recs: list, cfg: FilterConfig):
    by_id = prompts.set_index("patient_id")
    chosen, stats, rows = {}, collections.Counter(), []
    n_c = n_ok = 0
    for r in recs:
        pid = r["patient_id"]
        row = by_id.loc[pid]
        ref, f = row["prompt_S"], {c: str(row[c]) for c in FIELD_COLS}
        cands = []
        for c in r["candidates"]:
            d = jaccard_distance(ref, c["text"])
            cand = Candidate(text=c["text"], diversity=d, similarity=float(c["S"]),
                             fact_problems=check_facts(c["text"], f))
            common = (not cand.fact_problems) and cand.similarity >= cfg.tau and len(c["text"].split()) >= 8
            cand.valid_strict = common and cfg.d_min <= d <= cfg.d_max
            cand.valid_relaxed = common and cfg.d_min_relaxed <= d <= cfg.d_max_relaxed
            cand.score = cfg.lam * d + (1 - cfg.lam) * cand.similarity
            cands.append(cand)
            n_c += 1
            n_ok += not cand.fact_problems
        sel, status = select_top_k(cands, cfg)
        chosen[pid] = [c.text for c in sel]
        stats[status.split("(")[0]] += 1
        changed = [c.text for c in sel] != r["chosen"]
        rows.append(dict(patient_id=pid, status=status, n_candidates=len(cands), changed=changed,
                         model=r.get("model", "")))
    return chosen, stats, rows, n_c, n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--target", choices=["train_style", "heldout"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    cfg = FilterConfig(k=a.k)
    prompts = pd.read_csv(a.prompts, keep_default_na=False)
    recs = {}
    for line in open(a.progress, encoding="utf-8"):
        r = json.loads(line)
        recs[r["patient_id"]] = r          # last record wins if a case was redone
    chosen, stats, rows, n_c, n_ok = refilter(prompts, list(recs.values()), cfg)

    prefix = "narrative_" if a.target == "train_style" else "narrative_heldout_"
    for j in range(1, cfg.k + 1):
        prompts[f"{prefix}{j}"] = [
            (chosen[p][j - 1] if p in chosen and len(chosen[p]) >= j else "")
            for p in prompts["patient_id"]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(out, index=False, encoding="utf-8")
    log = pd.DataFrame(rows)
    log.to_csv(out.with_name(out.stem + "_log.csv"), index=False)

    expected = prompts if a.target == "train_style" else prompts[prompts.split == "test"]
    missing = sorted(set(expected.patient_id) - set(recs))
    short = log[~log.status.str.startswith(("strict", "relaxed"))].patient_id.tolist()
    print(f"{a.target}: {len(recs)} cases re-filtered, {len(missing)} not generated yet")
    print("status:", dict(stats))
    print(f"candidates {n_c} | fact-check pass {n_ok / max(n_c, 1):.1%} | "
          f"selection changed for {int(log.changed.sum())} cases")
    print(f"cases with < {cfg.k} valid narratives: {len(short)} {short[:20]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
