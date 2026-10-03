"""Choose the instruction text for one patient at training / test time.

mode 'S'  : structured prompt, rebuilt with field dropout (each field -> 'unknown' with prob p)
mode 'N'  : one of narrative_1..3 at random
mode 'SN' : S or N with probability 1/2
At evaluation use p=0 and mode 'S' (or a fixed column, e.g. 'narrative_heldout_1').
"""
from __future__ import annotations

import random
from typing import Dict, Optional

from .fields import UNKNOWN
from .structured import structured_prompt

FIELD_COLS = ["age_bin", "menopause", "ethnicity", "subtype", "field_strength", "manufacturer",
              "view", "laterality", "fat_suppressed", "implant", "num_phases"]


def fields_of(row: Dict) -> Dict[str, str]:
    f = {}
    for c in FIELD_COLS:
        v = str(row.get(c, UNKNOWN))
        f[c] = UNKNOWN if v in ("", "nan") else (v[:-2] if v.endswith(".0") else v)
    return f


def pick_prompt(row: Dict, mode: str = "S", p_drop: float = 0.0,
                rng: Optional[random.Random] = None, column: Optional[str] = None) -> str:
    rng = rng or random
    if column:
        return str(row[column])
    if mode == "SN":
        mode = "S" if rng.random() < 0.5 else "N"
    if mode == "N":
        opts = [str(row[c]) for c in ("narrative_1", "narrative_2", "narrative_3") if str(row.get(c, ""))]
        if opts:
            return rng.choice(opts)
        mode = "S"
    f = fields_of(row)
    if p_drop > 0:
        f = {k: (UNKNOWN if rng.random() < p_drop else v) for k, v in f.items()}
    return structured_prompt(f)
