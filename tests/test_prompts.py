import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cids.prompts import counterfactual_fields, normalize_row, structured_prompt  # noqa: E402
from cids.prompts.fields import menopause  # noqa: E402
from cids.prompts.narrative import (FilterConfig, check_facts, jaccard_distance,  # noqa: E402
                                    score_candidates, select_top_k)

ROW = {"age": 41, "menopause": "pre (<6 months since LMP AND no prior bilateral ovariectomy)",
       "ethnicity": "african american", "tumor_subtype": "triple_negative",
       "field_strength": 1.5, "manufacturer": "GE", "view": "axial", "bilateral_mri": 1,
       "fat_suppressed": 1, "has_implant": 0, "num_phases": 4}
F = normalize_row(ROW)


def test_normalize():
    assert F["age_bin"] == "35-45"
    assert F["menopause"] == "pre-menopausal"
    assert menopause("peri (6-12 months since LMP ...)") == "peri-menopausal"
    assert F["subtype"] == "triple-negative"
    assert F["field_strength"] == "1.5T"


def test_structured_never_leaks_outcome():
    s = structured_prompt(F, hints=True).lower()
    for w in ("pcr", "response", "chemotherapy", "mastectomy"):
        assert w not in s


def test_counterfactual_changes_only_one_field():
    g = counterfactual_fields(F, "ethnicity", random.Random(0))
    assert g["ethnicity"] != F["ethnicity"]
    assert {k for k in F if F[k] != g[k]} == {"ethnicity"}


GOOD = ("Pre-menopausal African American woman aged 35-45 with biopsy-proven triple-negative "
        "carcinoma. Bilateral axial fat-suppressed DCE-MRI on a GE 1.5T scanner with four "
        "dynamic phases. Please delineate the enhancing primary tumor.")


def test_fact_check_accepts_faithful_paraphrase():
    assert check_facts(GOOD, F) == []


def test_fact_check_rejects_changed_subtype():
    bad = GOOD.replace("triple-negative", "HER2-positive")
    probs = check_facts(bad, F)
    assert any("subtype" in p for p in probs)


def test_fact_check_rejects_wrong_field_strength_and_outcome():
    bad = GOOD.replace("1.5T", "3T") + " Patient later achieved pCR."
    probs = check_facts(bad, F)
    assert any("field_strength" in p for p in probs)
    assert "mentions treatment/outcome" in probs


def test_fact_check_rejects_invented_value_for_unknown_field():
    f = dict(F, menopause="unknown")
    probs = check_facts(GOOD, f)
    assert any(p.startswith("invented menopause") for p in probs)


def test_selection_top_k():
    ref = structured_prompt(F)
    texts = [GOOD, GOOD.replace("Please delineate", "Kindly segment"),
             GOOD.replace("biopsy-proven", "known"), GOOD.replace("GE", "Siemens")]
    cfg = FilterConfig(k=2, tau=0.5)
    cands = score_candidates(ref, texts, F, lambda r, t: [0.95] * len(t), cfg)
    chosen, status = select_top_k(cands, cfg)
    assert len(chosen) == 2 and status in ("strict", "relaxed")
    assert all("Siemens" not in c.text for c in chosen)
    assert 0 <= jaccard_distance(ref, GOOD) <= 1


def test_bilateral_scan_is_not_bilateral_tumor():
    ok = GOOD  # "Bilateral axial ... DCE-MRI" = both breasts imaged
    assert "claims tumor in both breasts" not in check_facts(ok, F)
    bad = GOOD.replace("Please delineate the enhancing primary tumor.",
                       "Segment the enhancing breast tumor in both breasts.")
    assert "claims tumor in both breasts" in check_facts(bad, F)
    bad2 = GOOD.replace("triple-negative carcinoma", "triple-negative bilateral breast cancer")
    assert "claims tumor in both breasts" in check_facts(bad2, F)


def test_phase_count_as_int_and_word():
    f = dict(F, num_phases=4)                       # int, as read from CSV
    assert check_facts(GOOD, f) == []               # GOOD says "four dynamic phases"
    assert check_facts(GOOD.replace("four", "4"), f) == []
    assert any("num_phases" in p for p in check_facts(GOOD.replace("four", "five"), f))


def test_age_phrasing_variants():
    young = dict(F, age_bin="under 35", menopause="pre-menopausal")
    txt = GOOD.replace("aged 35-45", "under the age of 35")
    assert not any("age_bin" in p for p in check_facts(txt, young))
    old = dict(F, age_bin="over 65")
    assert not any("age_bin" in p for p in check_facts(GOOD.replace("aged 35-45", "over the age of 65"), old))
    assert any("age_bin" in p for p in check_facts(GOOD.replace("aged 35-45", "34-year-old"), young))
