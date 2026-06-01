import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_MATCH_REL_TOL = 0.07

_float_pat = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")   # numbers
_pow_pat = re.compile(r"\^\s*[-+]?\d+(?:\.\d+)?")  # ^2, ^3, ^2.0 ...


def extract_two_numbers_from_latex(s: str) -> Optional[Tuple[float, float]]:
    if s is None:
        return None
    t = str(s)

    t = _pow_pat.sub("", t)

    nums = _float_pat.findall(t)
    if len(nums) < 2:
        return None

    try:
        return float(nums[0]), float(nums[1])
    except Exception:
        return None


def infer_ops_from_latex(s: str) -> Dict[str, Optional[str]]:
    if s is None:
        return {"op_between": None, "op_x1": None, "op_x2": None}
    t = str(s)

    if r"\frac" in t:
        op_between = "div"
    elif r"\times" in t:
        op_between = "mul"
    elif " + " in t:
        op_between = "add"
    elif " - " in t:
        op_between = "sub"
    else:
        op_between = None

    def detect_unary(expr: str) -> str:
        if r"\log" in expr:
            return "log"
        if r"\exp" in expr:
            return "exp"
        if r"\sqrt" in expr:
            return "sqrt"
        if r"\sin" in expr:
            return "sin"
        if r"\cos" in expr:
            return "cos"
        if r"\tan" in expr:
            return "tan"
        if "^2" in expr:
            return "square"
        if "^3" in expr:
            return "cube"
        return "none"

    if op_between == "div":
        m = re.search(r"\\frac\{(.+?)\}\{(.+?)\}", t)
        left, right = (m.group(1), m.group(2)) if m else (t, t)
    elif op_between == "mul":
        parts = t.split(r"\times", 1)
        left, right = parts[0], parts[1] if len(parts) > 1 else ("", "")
    elif op_between in ("add", "sub"):
        token = " + " if op_between == "add" else " - "
        parts = t.split(token, 1)
        left, right = parts[0], parts[1] if len(parts) > 1 else ("", "")
    else:
        left, right = t, t

    return {"op_between": op_between, "op_x1": detect_unary(left), "op_x2": detect_unary(right)}


def eval_match_score(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    required = ["x1", "x2", "operation_x1", "operation_x2", "operation_between", "latex_expression"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Need {required}")

    out = df.copy()
    out["latex_expression"] = out["latex_expression"].astype(str)

    nums = out["latex_expression"].apply(extract_two_numbers_from_latex)
    out["parseable"] = nums.notna()
    out["x1_from_latex"] = nums.apply(lambda t: t[0] if t is not None else np.nan)
    out["x2_from_latex"] = nums.apply(lambda t: t[1] if t is not None else np.nan)

    x1 = pd.to_numeric(out["x1"], errors="coerce").to_numpy()
    x2 = pd.to_numeric(out["x2"], errors="coerce").to_numpy()
    x1l = pd.to_numeric(out["x1_from_latex"], errors="coerce").to_numpy()
    x2l = pd.to_numeric(out["x2_from_latex"], errors="coerce").to_numpy()
    out["x1_err"] = np.abs(x1 - x1l)
    out["x2_err"] = np.abs(x2 - x2l)

    inferred = out["latex_expression"].apply(infer_ops_from_latex)
    out["op_between_infer"] = inferred.apply(lambda d: d["op_between"])
    out["op_x1_infer"] = inferred.apply(lambda d: d["op_x1"])
    out["op_x2_infer"] = inferred.apply(lambda d: d["op_x2"])

    out["op_match_between"] = out["op_between_infer"] == out["operation_between"].astype(str)
    out["op_match_x1"] = out["op_x1_infer"] == out["operation_x1"].astype(str)
    out["op_match_x2"] = out["op_x2_infer"] == out["operation_x2"].astype(str)
    out["op_match_all"] = out["op_match_between"] & out["op_match_x1"] & out["op_match_x2"]
    out["match"] = (
        out["parseable"]
        & (out["x1_err"] / out["x1"] <= _MATCH_REL_TOL)
        & (out["x2_err"] / out["x2"] <= _MATCH_REL_TOL)
        & out["op_match_all"]
    )

    metrics = {
        "match_rate": float(out["match"].mean()),
        "op_consistency_rate": float(out["op_match_all"].mean()),
    }

    return metrics, out
