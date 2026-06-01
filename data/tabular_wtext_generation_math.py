import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
import os


def fmt_num(x: float) -> str:
    return f"{x:.1f}"

def unary_apply(name: str, x: float) -> float:
    if name == "none":
        return x
    if name == "log":
        return np.log(x)
    if name == "exp":
        return np.exp(x)
    if name == "sqrt":
        return np.sqrt(x)
    if name == "sin":
        return np.sin(x)
    if name == "cos":
        return np.cos(x)
    if name == "tan":
        return np.tan(x)
    if name == "abs":
        return np.abs(x)
    if name == "square":
        return x ** 2
    if name == "cube":
        return x ** 3
    raise ValueError(f"Unknown unary op: {name}")

def unary_latex(name: str, x_token: str) -> str:
    if name == "none":
        return x_token
    if name == "log":
        return rf"\log({x_token})"
    if name == "exp":
        return rf"\exp({x_token})"
    if name == "sqrt":
        return rf"\sqrt{{{x_token}}}"
    if name == "sin":
        return rf"\sin({x_token})"
    if name == "cos":
        return rf"\cos({x_token})"
    if name == "tan":
        return rf"\tan({x_token})"
    if name == "square":
        return rf"({x_token})^2"
    if name == "cube":
        return rf"({x_token})^3"
    raise ValueError(f"Unknown unary op: {name}")

def binary_apply(name: str, a: float, b: float, eps: float = 1e-12) -> float:
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    if name == "mul":
        return a * b
    if name == "div":
        return a / (b + eps)
    raise ValueError(f"Unknown binary op: {name}")

def binary_latex(op3: str, a_ltx: str, b_ltx: str) -> str:
    if op3 == "add":
        return f"{a_ltx} + {b_ltx}"
    if op3 == "sub":
        return f"{a_ltx} - {b_ltx}"
    if op3 == "mul":
        return rf"{a_ltx} \times {b_ltx}"
    if op3 == "div":
        return rf"\frac{{{a_ltx}}}{{{b_ltx}}}"
    raise ValueError(f"Unknown binary op: {op3}")


# -----------------------------
# Discrete Gaussian sampler for x1/x2
# -----------------------------
def build_candidates(lo: float = 2.0, hi: float = 8.0, step: float = 0.1) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 1)

def discrete_gaussian_probs(cands: np.ndarray, mu: float, sigma: float, alpha_uniform: float) -> np.ndarray:
    w = np.exp(-0.5 * ((cands - mu) / sigma) ** 2)
    w = w / w.sum()
    u = np.ones_like(w) / len(w)
    p = (1.0 - alpha_uniform) * w + alpha_uniform * u
    p = p / p.sum()
    return p

def sample_discrete(cands: np.ndarray, p: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(cands), size=n, replace=True, p=p)
    return cands[idx]

def singleton_ratio(vals: np.ndarray) -> float:
    _, counts = np.unique(vals, return_counts=True)
    return float(np.mean(counts == 1))


# -----------------------------
# Main dataset generator
# -----------------------------
@dataclass
class SynthConfig:
    num_rows: int = 10000
    seed: int = 0

    # x distributions
    mu1: float = 3.0
    mu2: float = 6.5
    sigma_init: float = 1.5
    alpha_uniform: float = 0.5
    singleton_ratio_target: float = 0.25
    max_sigma_adjust_iters: int = 20
    sigma_shrink: float = 0.90

    # value range
    x1_lo: float = 0.1
    x1_hi: float = 6.0
    x1_step: float = 0.1

    x2_lo: float = 3.0
    x2_hi: float = 9.9
    x2_step: float = 0.1

    # ops
    unary_ops: Tuple[str, ...] = ("none", "log", "exp", "sqrt", "sin", "cos", "tan", "square", "cube")
    binary_ops: Tuple[str, ...] = ("add", "sub", "mul", "div")

    # NEW: op distributions (must sum to 1)
    unary_p_x1: Dict[str, float] = None
    unary_p_x2: Dict[str, float] = None
    binary_p: Dict[str, float] = None

    # split
    valid_ratio: float = 0.1

def _probs_from_dict(ops, p_dict):
    p = np.array([p_dict[o] for o in ops], dtype=float)
    p = p / p.sum()
    return p


def generate_synth_table(cfg: SynthConfig, save_csv_path: Optional[str] = None) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    cands1 = build_candidates(cfg.x1_lo, cfg.x1_hi, cfg.x1_step)
    cands2 = build_candidates(cfg.x2_lo, cfg.x2_hi, cfg.x2_step)

    # adapt sigma to control singleton ratio for x1/x2
    sigma = cfg.sigma_init
    for _ in range(cfg.max_sigma_adjust_iters):
        p1 = discrete_gaussian_probs(cands1, cfg.mu1, sigma, cfg.alpha_uniform)
        p2 = discrete_gaussian_probs(cands2, cfg.mu2, sigma, cfg.alpha_uniform)
        x1_try = sample_discrete(cands1, p1, cfg.num_rows, rng)
        x2_try = sample_discrete(cands2, p2, cfg.num_rows, rng)
        s1 = singleton_ratio(x1_try)
        s2 = singleton_ratio(x2_try)
        if max(s1, s2) <= cfg.singleton_ratio_target:
            x1, x2 = x1_try, x2_try
            break
        sigma *= cfg.sigma_shrink
    else:
        x1, x2 = x1_try, x2_try

    if cfg.unary_p_x1 is None:
        cfg.unary_p_x1 = {
            "none": 0.18, "log": 0.16, "sqrt": 0.13, "square": 0.12,
            "sin": 0.10, "cos": 0.10, "tan": 0.07, "exp": 0.07, "cube": 0.07
        }
    if cfg.unary_p_x2 is None:
        cfg.unary_p_x2 = {
            "none": 0.22, "sin": 0.14, "cos": 0.14, "sqrt": 0.12,
            "log": 0.10, "square": 0.09, "tan": 0.07, "exp": 0.06, "cube": 0.06
        }
    if cfg.binary_p is None:
        cfg.binary_p = {"add": 0.35, "mul": 0.30, "sub": 0.20, "div": 0.15}

    p_unary_x1 = _probs_from_dict(cfg.unary_ops, cfg.unary_p_x1)
    p_unary_x2 = _probs_from_dict(cfg.unary_ops, cfg.unary_p_x2)
    p_binary = _probs_from_dict(cfg.binary_ops, cfg.binary_p)

    op_x1 = rng.choice(cfg.unary_ops, size=cfg.num_rows, replace=True, p=p_unary_x1)
    op_x2 = rng.choice(cfg.unary_ops, size=cfg.num_rows, replace=True, p=p_unary_x2)
    op_between = rng.choice(cfg.binary_ops, size=cfg.num_rows, replace=True, p=p_binary)

    latex_list: List[str] = []
    for i in range(cfg.num_rows):
        x1_tok = fmt_num(float(x1[i]))
        x2_tok = fmt_num(float(x2[i]))
        a_ltx = unary_latex(str(op_x1[i]), x1_tok)
        b_ltx = unary_latex(str(op_x2[i]), x2_tok)
        expr = binary_latex(str(op_between[i]), a_ltx, b_ltx)
        latex_list.append(expr)

    df = pd.DataFrame({
        "x1": x1.astype(float),
        "x2": x2.astype(float),
        "operation_x1": op_x1,
        "operation_x2": op_x2,
        "operation_between": op_between,
        "latex_expression": latex_list,
    })

    return df

def split_train_valid(df: pd.DataFrame, valid_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_valid = max(1, int(round(len(df) * valid_ratio)))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)

if __name__ == "__main__":
    cfg = SynthConfig(num_rows=10000, seed=42)
    df = generate_synth_table(cfg)

    train_df, valid_df = split_train_valid(df, cfg.valid_ratio, seed=cfg.seed)

    out_base = "data/tabular/math_latex"
    os.makedirs(out_base, exist_ok=True)
    train_path = os.path.join(out_base, "train_ori.csv")
    valid_path = os.path.join(out_base, "valid_ori.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(train_df.head())
    print("train rows:", len(train_df), "valid rows:", len(valid_df))
    print("x1 singleton ratio (train):", singleton_ratio(train_df["x1"].to_numpy()))
    print("x2 singleton ratio (train):", singleton_ratio(train_df["x2"].to_numpy()))
