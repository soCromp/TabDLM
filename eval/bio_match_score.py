import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Set

def _expand_range(lo: float, hi: float, tol: float) -> tuple[float, float]:
    tol = float(max(0.0, tol))
    if tol < 1.0:
        return lo * (1.0 - tol), hi * (1.0 + tol)
    else:
        return lo - tol, hi + tol


def acceptable_age_desc(age: int, tol: float = 0.0) -> set[str]:
    age = float(age)
    ok = set()

    bins = [
        (21, 30, (
            "in an early phase of career development",
            "in the early career stage",
        )),
        (31, 40, (
            "in a career-building stage",
            "in the career-building stage",
        )),
        (41, 50, (
            "at an established professional stage",
            "in the established career stage",
        )),
        (51, 60, (
            "in an advanced career stage",
            "in the advanced career stage",
        )),
    ]

    for lo, hi, descs in bins:
        lo2, hi2 = _expand_range(lo, hi, tol)
        if lo2 <= age <= hi2:
            for d in descs:
                ok.add(d.lower())

    if tol < 1.0:
        if age <= 20 * (1.0 + tol) or age >= 61 * (1.0 - tol):
            ok.add("at the late career stage".lower())
            ok.add("in the late career stage".lower())
    else:
        if age <= (20 + tol) or age >= (61 - tol):
            ok.add("at the late career stage".lower())
            ok.add("in the late career stage".lower())

    return ok


def acceptable_salary_desc(sal: int, tol: float = 0.0) -> set[str]:
    sal = float(sal)

    ok = set()

    _, hi1 = _expand_range(100, 100, tol)
    if sal <= hi1:
        ok.add("comfortable income")

    lo2, hi2 = _expand_range(101, 150, tol)
    if lo2 <= sal <= hi2:
        ok.add("professional income")

    lo3, _ = _expand_range(151, 151, tol)
    if sal >= lo3:
        ok.add("a high-level executive income")
        ok.add("a high-level income")

    return {s.lower() for s in ok}


def expected_pronoun(sex: str) -> str:
    return "He" if str(sex).strip().lower() == "male" else "She"


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def norm_text(s: str) -> str:
    return normalize_space(s).lower()


def word_view(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", norm_text(s))


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def contains_substring(haystack: str, needle: str) -> bool:
    h = norm_text(haystack)
    n = norm_text(needle)
    if not n:
        return False
    return n in h


def fuzzy_hit(haystack: str, needle: str, min_token_hits: int = 2, min_ratio: float = 0.6) -> bool:
    n_tokens = [t for t in word_view(needle) if t]
    if not n_tokens:
        return False

    h_tokens = set(word_view(haystack))
    common = sum(1 for t in n_tokens if t in h_tokens)

    ratio = common / max(len(n_tokens), 1)
    return (common >= min_token_hits) and (ratio >= min_ratio)


def hit_field(haystack: str, value: str, *, allow_fuzzy: bool = True) -> bool:
    if contains_substring(haystack, value):
        return True
    if allow_fuzzy:
        toks = word_view(value)
        if len(toks) <= 1:
            return False
        if len(toks) == 2:
            return fuzzy_hit(haystack, value, min_token_hits=2, min_ratio=1.0)
        return fuzzy_hit(haystack, value, min_token_hits=2, min_ratio=0.6)
    return False


def pronoun_consistent(bio: str, sex: str) -> bool:
    b = norm_text(bio)
    he = bool(re.search(r"\bhe\b|\bhim\b|\bhis\b", b))
    she = bool(re.search(r"\bshe\b|\bher\b|\bhers\b", b))

    if not he and not she:
        return True  # no signal -> don't punish

    exp = expected_pronoun(sex)
    if exp == "He":
        return he and not she
    else:
        return she and not he


def sex_hit(bio: str, sex: str) -> bool:
    b = norm_text(bio)
    sex = norm_text(sex)

    explicit = False
    if sex == "male":
        explicit = bool(re.search(r"\bmale\b|\bman\b|\bmale individual\b", b))
    elif sex == "female":
        explicit = bool(re.search(r"\bfemale\b|\bwoman\b|\bfemale individual\b", b))

    if explicit:
        return True

    return pronoun_consistent(bio, sex)


def desc_hit_any(bio: str, acceptable_descs: Set[str]) -> bool:
    b = norm_text(bio)
    for d in acceptable_descs:
        if norm_text(d) in b:
            return True
    return False


def check_row(row: pd.Series, age_tol: int = 0, salary_tol: int = 0) -> Tuple[bool, Dict[str, Any]]:
    detail: Dict[str, Any] = {"ok": True, "errors": []}

    bio = safe_str(row.get("biography", ""))
    bio = normalize_space(bio)
    if not bio:
        detail["ok"] = False
        detail["errors"].append("bio_empty")
        return False, detail

    # expected from columns
    sex = normalize_space(safe_str(row.get("sex", ""))).lower()
    state = normalize_space(safe_str(row.get("birth_state", "")))
    college = normalize_space(safe_str(row.get("college", "")))
    degree = normalize_space(safe_str(row.get("degree", ""))).lower()
    occupation = normalize_space(safe_str(row.get("occupation", "")))

    try:
        age_raw = float(row.get("age"))
        salary_raw = float(row.get("salary"))
    except (TypeError, ValueError):
        detail["ok"] = False
        detail["errors"].append("invalid_numeric_value")
        return False, detail

    if not (np.isfinite(age_raw) and np.isfinite(salary_raw)):
        detail["ok"] = False
        detail["errors"].append("numeric_not_finite")
        return False, detail

    age = int(age_raw)
    salary = int(salary_raw)

    exp_age_desc_set = acceptable_age_desc(age, tol=age_tol)
    exp_sal_desc_set = acceptable_salary_desc(salary, tol=salary_tol)
    detail["expected_age_descs"] = " | ".join(sorted(exp_age_desc_set))
    detail["expected_salary_descs"] = " | ".join(sorted(exp_sal_desc_set))
    detail["expected_sex"] = sex
    detail["expected_birth_state"] = state
    detail["expected_college"] = college
    detail["expected_degree"] = degree
    detail["expected_occupation"] = occupation

    checks = {
        "sex": sex_hit(bio, sex),

        "birth_state": hit_field(bio, state, allow_fuzzy=False),
        "college": hit_field(bio, college, allow_fuzzy=True),
        "degree": hit_field(bio, degree, allow_fuzzy=False),
        "occupation": hit_field(bio, occupation, allow_fuzzy=True),

        "age_desc": desc_hit_any(bio, exp_age_desc_set),
        "salary_desc": desc_hit_any(bio, exp_sal_desc_set),

        "pronoun_consistency": pronoun_consistent(bio, sex),
    }

    detail.update({f"ok_{k}": bool(v) for k, v in checks.items()})

    for k, v in checks.items():
        if not v:
            detail["ok"] = False
            detail["errors"].append(f"mismatch_{k}")

    detail["failed_fields"] = [e.replace("mismatch_", "") for e in detail["errors"]]
    detail["failed_fields_text"] = ", ".join(detail["failed_fields"])

    return detail["ok"], detail


_AGE_TOL = 0.06
_SALARY_TOL = 0.06


def eval_bio_match_score(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    details: List[Dict[str, Any]] = []
    match_count = 0
    op_consistency_count = 0

    for idx, row in df.iterrows():
        ok, d = check_row(row, age_tol=_AGE_TOL, salary_tol=_SALARY_TOL)
        d["row_index"] = idx
        d["match"] = ok
        op_ok = bool(d.get("ok_pronoun_consistency", True))
        d["op_consistency"] = op_ok
        details.append(d)
        match_count += int(ok)
        op_consistency_count += int(op_ok)

    det = pd.DataFrame(details)
    n = len(df)
    metrics: Dict[str, Any] = {
        "match_rate": float(match_count / max(n, 1)),
        "op_consistency_rate": float(op_consistency_count / max(n, 1)),
    }

    return metrics, det
