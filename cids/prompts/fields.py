"""Normalize MAMA-MIA clinical_and_imaging_info fields into prompt-ready values.

Every function returns a short human-readable string, or "unknown" when the
value is missing. Fields that leak the outcome or the label are never used:
see EXCLUDED_FIELDS.
"""
from __future__ import annotations

import math
from typing import Any

UNKNOWN = "unknown"

# Fields that must NEVER enter a prompt (outcome / post-treatment / label leakage)
EXCLUDED_FIELDS = {
    "pcr": "outcome (pathological complete response) - label leakage",
    "nac_agent": "treatment given after the scan",
    "endocrine_therapy": "treatment given after the scan",
    "anti_her2_neu_therapy": "treatment given after the scan",
    "mastectomy_post_nac": "post-treatment surgery",
    "days_to_follow_up": "follow-up outcome",
    "days_to_recurrence": "follow-up outcome",
    "days_to_metastasis": "follow-up outcome",
    "days_to_death": "follow-up outcome",
    "multifocal_cancer": "partly derived from the segmentation itself - leakage risk",
    "breast_density": "96% missing - not usable as a prompt field",
    "oncotype_score": "99% missing",
    "nottingham_grade": "87% missing",
    "acquisition_date": "irrelevant / anonymized",
    "tcia_series_uid": "identifier",
}

AGE_BINS = [(0, 35, "under 35"), (35, 45, "35-45"), (45, 55, "45-55"),
            (55, 65, "55-65"), (65, 200, "over 65")]


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip() == "" or str(x).strip().lower() == "nan"


def age_bin(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    a = float(x)
    for lo, hi, label in AGE_BINS:
        if lo <= a < hi:
            return label
    return UNKNOWN


def menopause(x: Any) -> str:
    """Raw values include long ISPY2 definitions such as
    'pre (<6 months since LMP AND ...)'. Keep only pre / peri / post."""
    if is_missing(x):
        return UNKNOWN
    s = str(x).strip().lower()
    for key in ("peri", "post", "pre"):  # 'peri' before 'pre' on purpose
        if s.startswith(key):
            return {"pre": "pre-menopausal", "peri": "peri-menopausal",
                    "post": "post-menopausal"}[key]
    return UNKNOWN


_ETHNICITY = {
    "caucasian": "Caucasian",
    "african american": "African American",
    "asian": "Asian",
    "hispanic": "Hispanic",
    "multiple race": "multiple race",
    "hawaiian/pacific islander": "Native Hawaiian or Pacific Islander",
    "hawaian": "Native Hawaiian or Pacific Islander",
    "american indian/alaskan native": "American Indian or Alaska Native",
    "american indian": "American Indian or Alaska Native",
    "native american": "American Indian or Alaska Native",
}


def ethnicity(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return _ETHNICITY.get(str(x).strip().lower(), UNKNOWN)


def ethnicity_group(x: Any) -> str:
    """Coarse group used for fairness metrics (small groups merged)."""
    e = ethnicity(x)
    if e in ("Caucasian", "African American", "Asian"):
        return e
    return "other/unknown"


_SUBTYPE = {
    "triple_negative": "triple-negative",
    "luminal_a": "luminal A (HR-positive, HER2-negative)",
    "luminal_b": "luminal B (HR-positive)",
    "luminal": "luminal (HR-positive)",
    "her2_enriched": "HER2-enriched",
    "her2_pure": "HER2-positive, HR-negative",
}


def subtype(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return _SUBTYPE.get(str(x).strip().lower(), UNKNOWN)


def field_strength(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    v = float(x)
    return "3T" if v >= 2.5 else "1.5T"


def manufacturer(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    s = str(x).strip().lower()
    return {"ge": "GE", "siemens": "Siemens", "philips": "Philips"}.get(s, str(x).strip())


def view(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return str(x).strip().lower()  # axial / sagittal


def laterality(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return "bilateral" if int(float(x)) == 1 else "unilateral"


def yes_no(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return "yes" if int(float(x)) == 1 else "no"


def num_phases(x: Any) -> str:
    if is_missing(x):
        return UNKNOWN
    return str(int(float(x)))


def normalize_row(row: dict) -> dict:
    """Return the prompt-ready fields for one patient."""
    return {
        "age_bin": age_bin(row.get("age")),
        "menopause": menopause(row.get("menopause")),
        "ethnicity": ethnicity(row.get("ethnicity")),
        "ethnicity_group": ethnicity_group(row.get("ethnicity")),
        "subtype": subtype(row.get("tumor_subtype")),
        "field_strength": field_strength(row.get("field_strength")),
        "manufacturer": manufacturer(row.get("manufacturer")),
        "view": view(row.get("view")),
        "laterality": laterality(row.get("bilateral_mri")),
        "fat_suppressed": yes_no(row.get("fat_suppressed")),
        "implant": yes_no(row.get("has_implant")),
        "num_phases": num_phases(row.get("num_phases")),
    }
