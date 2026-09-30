"""Type-S (structured) clinical instructions, knowledge hints and counterfactuals.

One patient -> one deterministic instruction string. Field dropout (replacing
fields with "unknown") is applied at TRAINING time in the dataloader, not here.
"""
from __future__ import annotations

import random
from typing import Dict, List

from .fields import UNKNOWN

TASK = "Segment the primary enhancing breast tumor in the dynamic contrast-enhanced volume."

# Order matters: this is the order sentences appear in the prompt.
DEMOGRAPHIC_KEYS = ("age_bin", "menopause", "ethnicity")
CLINICAL_KEYS = ("subtype",)
ACQUISITION_KEYS = ("manufacturer", "field_strength", "view", "laterality",
                    "fat_suppressed", "implant", "num_phases")


def _patient_sentence(f: Dict[str, str], with_ethnicity: bool = True) -> str:
    age = f["age_bin"]
    age_txt = "Patient of unknown age" if age == UNKNOWN else f"Patient aged {age}"
    meno = f["menopause"]
    meno_txt = "unknown menopausal status" if meno == UNKNOWN else meno
    s = f"{age_txt}, {meno_txt}"
    if with_ethnicity:
        eth = f["ethnicity"]
        s += ", unknown ethnicity" if eth == UNKNOWN else f", {eth}"
    return s + "."


def _tumor_sentence(f: Dict[str, str]) -> str:
    return f"Tumor subtype: {f['subtype']}."


def _acq_sentence(f: Dict[str, str]) -> str:
    scanner = " ".join(v for v in (f["manufacturer"], f["field_strength"]) if v != UNKNOWN)
    scanner = scanner or "unknown scanner"
    parts = [f"{scanner} MRI"]
    if f["laterality"] != UNKNOWN and f["view"] != UNKNOWN:
        parts.append(f"{f['laterality']} {f['view']} acquisition")
    if f["num_phases"] != UNKNOWN:
        parts.append(f"{f['num_phases']} dynamic phases")
    if f["fat_suppressed"] == "yes":
        parts.append("fat-suppressed")
    elif f["fat_suppressed"] == "no":
        parts.append("without fat suppression")
    if f["implant"] == "yes":
        parts.append("breast implant present")
    return "Acquired on " + ", ".join(parts) + "."


def structured_prompt(f: Dict[str, str], with_subtype: bool = True,
                      with_ethnicity: bool = True, hints: bool = False) -> str:
    sents = [_patient_sentence(f, with_ethnicity)]
    if with_subtype:
        sents.append(_tumor_sentence(f))
    sents.append(_acq_sentence(f))
    if hints:
        sents.extend(knowledge_hints(f))
    sents.append(TASK)
    return " ".join(sents)


# ---------------------------------------------------------------------------
# Knowledge hints (rule-based). Wording is deliberately hedged ("may"); have a
# radiologist review this table before submission.
# ---------------------------------------------------------------------------
def knowledge_hints(f: Dict[str, str]) -> List[str]:
    h: List[str] = []
    young = f["age_bin"] in ("under 35", "35-45")
    if f["menopause"] in ("pre-menopausal", "peri-menopausal") or (f["menopause"] == UNKNOWN and young):
        h.append("Background parenchymal enhancement may be high and can mimic tumor.")
    elif f["menopause"] == "post-menopausal":
        h.append("Background parenchymal enhancement is usually low, so the tumor is expected to be well contrasted.")
    st = f["subtype"]
    if st == "triple-negative":
        h.append("The tumor may show rim enhancement with relatively circumscribed margins.")
    elif st.startswith("luminal"):
        h.append("The tumor may show irregular or spiculated margins.")
    elif "HER2" in st:
        h.append("The tumor may show irregular margins with heterogeneous enhancement.")
    if f["field_strength"] == "3T":
        h.append("Higher signal-to-noise ratio; fine boundary detail is reliable.")
    elif f["field_strength"] == "1.5T":
        h.append("Lower signal-to-noise ratio; fine high-frequency detail may be noisy.")
    if f["fat_suppressed"] == "no":
        h.append("Without fat suppression, bright fat may be confused with enhancement.")
    if f["implant"] == "yes":
        h.append("A breast implant may distort the surrounding tissue.")
    if f["laterality"] == "bilateral":
        h.append("Both breasts are imaged; the tumor may be in either breast.")
    return h


# ---------------------------------------------------------------------------
# Counterfactuals (test-time analysis only)
# ---------------------------------------------------------------------------
AGE_SWAP = {"under 35": "over 65", "35-45": "over 65", "45-55": "over 65",
            "55-65": "35-45", "over 65": "35-45"}
MENO_SWAP = {"pre-menopausal": "post-menopausal", "peri-menopausal": "post-menopausal",
             "post-menopausal": "pre-menopausal"}
ETH_POOL = ["Caucasian", "African American", "Asian", "Hispanic"]
FS_SWAP = {"1.5T": "3T", "3T": "1.5T"}


def counterfactual_fields(f: Dict[str, str], kind: str, rng: random.Random) -> Dict[str, str]:
    g = dict(f)
    if kind == "age":            # demographic: mask should NOT change
        if g["age_bin"] in AGE_SWAP:
            g["age_bin"] = AGE_SWAP[g["age_bin"]]
            g["menopause"] = MENO_SWAP.get(g["menopause"], g["menopause"])
    elif kind == "ethnicity":    # demographic: mask should NOT change
        choices = [e for e in ETH_POOL if e != g["ethnicity"]]
        g["ethnicity"] = rng.choice(choices)
    elif kind == "field_strength":  # clinical/acquisition: mask MAY change
        g["field_strength"] = FS_SWAP.get(g["field_strength"], g["field_strength"])
    else:
        raise ValueError(kind)
    return g


COUNTERFACTUAL_KINDS = {"age": "demographic (mask should not change)",
                        "ethnicity": "demographic (mask should not change)",
                        "field_strength": "acquisition (mask may change)"}
