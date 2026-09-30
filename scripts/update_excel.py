"""Put narrative columns generated on Kaggle back into the Excel sheet.

Usage:
    python scripts/update_excel.py --csv path/to/cids_prompts_with_heldout.csv \
                                   --xlsx data/prompts/cids_prompts.xlsx
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_prompts import write_excel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--xlsx", default="data/prompts/cids_prompts.xlsx")
    a = ap.parse_args()
    new = pd.read_csv(a.csv, keep_default_na=False)
    fd = pd.read_excel(a.xlsx, sheet_name="field_dictionary")
    sm = pd.read_excel(a.xlsx, sheet_name="summary")
    filled = {c: int((new[c].astype(str).str.len() > 0).sum())
              for c in new.columns if c.startswith("narrative_")}
    sm = pd.concat([sm, pd.DataFrame([(f"filled {k}", v) for k, v in filled.items()],
                                     columns=["item", "value"])], ignore_index=True)
    write_excel(new, fd, sm, Path(a.xlsx))
    new.to_csv(Path(a.xlsx).with_suffix(".csv"), index=False, encoding="utf-8")
    print("filled:", filled)


if __name__ == "__main__":
    main()
