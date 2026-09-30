"""Type-N (narrative) prompt generation and validity filters.

Follows FairVLM's SRCP recipe (generate m candidates, keep top-k by
lambda*diversity + (1-lambda)*semantic similarity) and adds a FACT-PRESERVATION
check, because an LLM can silently change e.g. "HER2-enriched" to "HR-positive".

The LLM itself lives in scripts/generate_narratives.py so that everything in
this file can be unit-tested on a CPU without downloading a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .fields import UNKNOWN

# ---------------------------------------------------------------------------
# LLM instruction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a breast radiologist writing a short clinical note that asks for a tumor "
    "segmentation on a DCE-MRI scan. Rewrite the given instruction in your own words, "
    "like a real radiology request. Rules: keep EVERY fact exactly (age range numbers, "
    "menopausal status, ethnicity, tumor subtype, scanner vendor, field strength, "
    "number of phases, view, laterality, fat suppression, implant). Do not add any new "
    "facts, findings, measurements or treatment information. Do not drop facts. You may "
    "change word order and sentence structure. Write one paragraph of at most 60 words "
    "and end with a request to segment the tumor. Output only the paragraph."
)


def facts_block(f: Dict[str, str]) -> str:
    names = {"age_bin": "age range", "menopause": "menopausal status", "ethnicity": "ethnicity",
             "subtype": "tumor subtype", "manufacturer": "scanner vendor",
             "field_strength": "field strength", "num_phases": "number of dynamic phases",
             "view": "acquisition plane", "laterality": "laterality",
             "fat_suppressed": "fat suppression", "implant": "breast implant"}
    return "\n".join(f"- {names[k]}: {f[k]}" for k in names if f.get(k, UNKNOWN) != UNKNOWN)


def build_user_message(instruction: str, f: Dict[str, str]) -> str:
    return (f"Instruction to rewrite:\n{instruction}\n\nFacts that must all appear unchanged:\n"
            f"{facts_block(f)}\n\nRewritten instruction:")


# ---------------------------------------------------------------------------
# Fact-preservation check
# ---------------------------------------------------------------------------
def _age_patterns(v: str) -> List[str]:
    age_of = r"(the\s+age\s+of\s+)?"
    if v == "under 35":
        return [rf"(under|below|younger than|less than)\s+{age_of}35", r"<\s*35"]
    if v == "over 65":
        return [rf"(over|above|older than|more than)\s+{age_of}65", r">\s*65", r"65\s*(\+|or older|and older)"]
    lo, hi = v.split("-")
    return [rf"{lo}\s*(-|–|—|to|and)\s*{hi}"]


FACT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "menopause": {
        "pre-menopausal": [r"pre-?\s?menopaus"],
        "peri-menopausal": [r"peri-?\s?menopaus"],
        "post-menopausal": [r"post-?\s?menopaus"],
    },
    "ethnicity": {
        "Caucasian": [r"caucasian", r"\bwhite\b"],
        "African American": [r"african[- ]american", r"\bblack\b"],
        "Asian": [r"\basian\b"],
        "Hispanic": [r"hispanic", r"latina"],
        "multiple race": [r"multiple race", r"multiracial", r"mixed race"],
        "Native Hawaiian or Pacific Islander": [r"hawaiian", r"pacific islander"],
        "American Indian or Alaska Native": [r"american indian", r"alaska native", r"native american"],
    },
    "subtype": {
        "triple-negative": [r"triple[- ]?negative", r"\btnbc\b"],
        "luminal A (HR-positive, HER2-negative)": [r"luminal[- ]?a\b"],
        "luminal B (HR-positive)": [r"luminal[- ]?b\b"],
        "luminal (HR-positive)": [r"luminal", r"\bhr[- ]?positive", r"hormone[- ]receptor[- ]positive"],
        "HER2-enriched": [r"her-?2[- ]?enriched"],
        "HER2-positive, HR-negative": [r"her-?2[- ]?(positive|\+)"],
    },
    "field_strength": {
        "1.5T": [r"1\.5\s*-?\s*(t\b|tesla)"],
        "3T": [r"\b3(\.0)?\s*-?\s*(t\b|tesla)"],
    },
    "manufacturer": {
        "GE": [r"\bge\b", r"general electric"],
        "Siemens": [r"siemens"],
        "Philips": [r"philips"],
    },
    "view": {"axial": [r"axial"], "sagittal": [r"sagittal"]},
    "laterality": {
        "bilateral": [r"bilateral", r"both breasts"],
        "unilateral": [r"unilateral", r"single breast", r"one breast"],
    },
    "fat_suppressed": {
        "yes": [r"fat[- ]?(suppress|sat)"],
        "no": [r"(without|no|non)[- ]?fat[- ]?(suppress|sat)", r"fat suppression was not"],
    },
    "implant": {"yes": [r"implant"]},
}

NUM_WORDS = {"3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
             "8": "eight", "9": "nine", "10": "ten", "11": "eleven"}


def _any(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def check_facts(text: str, f: Dict[str, str]) -> List[str]:
    """Return a list of problems (empty list = all facts preserved, none contradicted)."""
    t = text.lower()
    f = {k: str(v) for k, v in f.items()}
    problems: List[str] = []

    # age
    if f["age_bin"] != UNKNOWN and not _any(_age_patterns(f["age_bin"]), t):
        problems.append(f"missing age_bin={f['age_bin']}")

    # phases
    n = str(f.get("num_phases", UNKNOWN))   # may arrive as int from a CSV
    if n.endswith(".0"):
        n = n[:-2]
    if n != UNKNOWN and not _any([rf"\b{n}\b", rf"\b{NUM_WORDS.get(n, n)}\b"], t):
        problems.append(f"missing num_phases={n}")

    for key, table in FACT_PATTERNS.items():
        v = f.get(key, UNKNOWN)
        if v == UNKNOWN or v not in table:
            # unknown -> the text must not invent any value for this field
            if v == UNKNOWN:
                for other, pats in table.items():
                    if key in ("fat_suppressed", "implant"):
                        continue
                    if _any(pats, t):
                        problems.append(f"invented {key}={other}")
            continue
        if not _any(table[v], t):
            problems.append(f"missing {key}={v}")
        # contradictions: another value of the same field is mentioned
        for other, pats in table.items():
            if other == v:
                continue
            if key == "subtype" and ("luminal" in v and "luminal" in other):
                continue  # 'luminal' is shared by luminal A/B/generic
            if key == "subtype" and v.startswith("HER2") and other.startswith("HER2"):
                continue
            if key == "fat_suppressed" and v == "yes" and other == "no":
                if _any(pats, t):
                    problems.append("contradiction fat_suppressed")
                continue
            if key == "fat_suppressed":
                continue
            if _any(pats, t):
                problems.append(f"contradiction {key}: mentions {other}")
    # leakage words
    if re.search(r"complete response|\bpcr\b|chemotherapy|neoadjuvant|mastectomy|recurrence", t):
        problems.append("mentions treatment/outcome")
    # bilateral ACQUISITION must not become bilateral TUMOR (not in the metadata)
    if re.search(r"(tumou?rs?|cancers?|lesions?|masse?s?|carcinomas?)\s+(located\s+)?(in|of|within|across)\s+both\s+breasts"
                 r"|bilateral\s+(breast\s+)?(tumou?rs?|cancers?|lesions?|masses|carcinomas?)"
                 r"|(tumou?rs?|cancers?|lesions?)\s+(are\s+)?bilateral", t):
        problems.append("claims tumor in both breasts")
    return problems


# ---------------------------------------------------------------------------
# Diversity & semantic similarity
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def words(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def jaccard_distance(a: str, b: str) -> float:
    A, B = words(a), words(b)
    if not A and not B:
        return 0.0
    return 1.0 - len(A & B) / len(A | B)


@dataclass
class FilterConfig:
    m: int = 5                     # candidates generated per case
    k: int = 3                     # candidates kept per case
    d_min: float = 0.30            # FairVLM diversity band (Jaccard distance)
    d_max: float = 0.50
    d_min_relaxed: float = 0.20    # used only if fewer than k pass
    d_max_relaxed: float = 0.70
    tau: float = 0.90              # min cosine similarity
    lam: float = 0.40              # score = lam*D + (1-lam)*S


@dataclass
class Candidate:
    text: str
    diversity: float
    similarity: float
    fact_problems: List[str] = field(default_factory=list)
    score: float = 0.0
    valid_strict: bool = False
    valid_relaxed: bool = False


def clean_generation(text: str) -> str:
    t = text.strip().strip('"').strip()
    t = re.sub(r"^(rewritten instruction|instruction)\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    return t


def score_candidates(reference: str, texts: Sequence[str], f: Dict[str, str],
                     sim_fn: Callable[[str, Sequence[str]], Sequence[float]],
                     cfg: FilterConfig) -> List[Candidate]:
    texts = [clean_generation(t) for t in texts]
    sims = list(sim_fn(reference, texts)) if texts else []
    out = []
    for t, s in zip(texts, sims):
        d = jaccard_distance(reference, t)
        c = Candidate(text=t, diversity=d, similarity=float(s), fact_problems=check_facts(t, f))
        ok_common = (not c.fact_problems) and c.similarity >= cfg.tau and len(t.split()) >= 8
        c.valid_strict = ok_common and cfg.d_min <= d <= cfg.d_max
        c.valid_relaxed = ok_common and cfg.d_min_relaxed <= d <= cfg.d_max_relaxed
        c.score = cfg.lam * d + (1 - cfg.lam) * c.similarity
        out.append(c)
    return out


def select_top_k(cands: Sequence[Candidate], cfg: FilterConfig) -> tuple[List[Candidate], str]:
    """Top-k strict-valid; top up with relaxed-valid if needed. Returns (chosen, status)."""
    uniq, seen = [], set()
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if c.text.lower() not in seen:
            seen.add(c.text.lower())
            uniq.append(c)
    strict = [c for c in uniq if c.valid_strict]
    if len(strict) >= cfg.k:
        return strict[: cfg.k], "strict"
    relaxed = strict + [c for c in uniq if c.valid_relaxed and not c.valid_strict]
    if len(relaxed) >= cfg.k:
        return relaxed[: cfg.k], "relaxed"
    return relaxed, f"insufficient({len(relaxed)})"
