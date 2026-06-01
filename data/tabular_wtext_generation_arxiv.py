import os
import json
import re
import argparse
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import html
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer


# -----------------------------
# RelBench loader
# -----------------------------
def load_relbench_arxiv(download: bool = True, upto_test_timestamp: bool = True):
    from relbench.datasets import get_dataset

    dataset = get_dataset(name="rel-arxiv", download=download)

    db = dataset.get_db() if upto_test_timestamp else dataset.get_db(upto_test_timestamp=False)

    keys = list(db.table_dict.keys())
    lower = {k.lower(): k for k in keys}

    def pick_table(possible_names):
        for name in possible_names:
            if name in lower:
                return db.table_dict[lower[name]].df
        for k in keys:
            kl = k.lower()
            if any(name in kl for name in possible_names):
                return db.table_dict[k].df
        raise KeyError(f"Cannot find table among candidates={possible_names}. Available tables={keys}")

    out = {
        "papers":          pick_table(["papers", "paper"]),
        "categories":      pick_table(["categories", "category"]),
        "citations":       pick_table(["citations", "citation"]),
        "paperCategories": pick_table(["papercategories", "paper_category", "paper_categories"]),
        "authors":         pick_table(["authors", "author"]),
        "paperAuthors":    pick_table(["paperauthors", "paper_author", "paper_authors"]),
    }

    for name, df in out.items():
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"Loaded `{name}` is not a pandas DataFrame.")
    return out


# -----------------------------
# Column-name helpers
# -----------------------------
def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Find the first column in `df` matching one of `candidates` (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns={list(df.columns)}")


# -----------------------------
# Cleaning helpers
# -----------------------------
def safe_text(x, max_chars: int = 4000) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    if isinstance(x, (np.generic,)):
        try:
            x = x.item()
        except Exception:
            pass

    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple, set)):
        s = " ".join([str(t) for t in list(x)[:50]])
    elif isinstance(x, dict):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)

    s = html.unescape(s).strip()
    if s.lower() in ("nan", "none", "<null>"):
        return ""
    return s[:max_chars]


def _clean_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _word_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([w for w in s.split() if w])


def _year_from_newstyle_arxiv_id(code: str) -> Optional[int]:
    raw = _clean_str(code).lower().replace("arxiv:", "").strip()
    m = re.search(r"\b(\d{4})\.\d{4,5}(?:v\d+)?\b", raw)
    if not m:
        return None
    yymm = int(m.group(1))
    yy = yymm // 100
    if yy < 0 or yy > 99:
        return None
    # e.g. 1703 -> 2017; 9207 -> 1992
    return 2000 + yy if yy < 90 else 1900 + yy


# -----------------------------
# Build merged single-paper-per-row DataFrame
# -----------------------------
def build_merged_df(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute per-paper features and return a flat DataFrame whose columns are
    exactly: [days_since_first_submit, paper_year, title_word_count, abstract_word_count,
             Category, Title, arXiv_Code, Abstract].
    """
    papers = tables["papers"]
    categories = tables["categories"]

    paper_id_col      = _resolve_col(papers, ["Paper_ID", "paper_id"])
    title_col         = _resolve_col(papers, ["Title", "title"])
    abstract_col      = _resolve_col(papers, ["Abstract", "abstract"])
    arxiv_col         = _resolve_col(papers, ["arXiv_Code", "arxiv_code", "Arxiv_Code", "ArXivCode"])
    primary_cat_col   = _resolve_col(papers, ["Primary_Category_ID", "primary_category_id"])
    submit_col        = _resolve_col(
        papers,
        ["Submission_Date", "submission_date", "Submit_Date", "submit_date", "timestamp"],
    )

    cat_id_col        = _resolve_col(categories, ["Category_ID", "category_id"])
    cat_name_col      = _resolve_col(categories, ["Category", "category"])

    cat_lookup = categories.set_index(cat_id_col)[cat_name_col]

    df = papers[[paper_id_col, title_col, abstract_col, arxiv_col, primary_cat_col, submit_col]].copy()
    df = df.rename(columns={
        paper_id_col:    "Paper_ID",
        title_col:       "Title",
        abstract_col:    "Abstract",
        arxiv_col:       "arXiv_Code",
        primary_cat_col: "Primary_Category_ID",
        submit_col:      "_submit_raw",
    })

    df["Category"] = df["Primary_Category_ID"].map(cat_lookup)

    # text cleanup (before word counts)
    for c in ["Title", "Abstract", "arXiv_Code"]:
        df[c] = df[c].map(safe_text)
    df["Category"] = df["Category"].map(lambda x: _clean_str(x).strip())

    t = pd.to_datetime(df["_submit_raw"], errors="coerce")
    t0 = t.min()
    print("The earliest submission time is", t0)
    df["days_since_first_submit"] = (t - t0).dt.days.astype("float32")

    year_from_date = t.dt.year
    year_from_id = df["arXiv_Code"].map(lambda c: _year_from_newstyle_arxiv_id(c))
    df["paper_year"] = year_from_date.where(year_from_date.notna(), year_from_id).astype("float32")

    df["title_word_count"] = df["Title"].map(_word_count).astype("int32")
    df["abstract_word_count"] = df["Abstract"].map(_word_count).astype("int32")

    df = df.drop(columns=["_submit_raw", "Paper_ID", "Primary_Category_ID"], errors="ignore")

    mask = np.ones(len(df), dtype=bool)
    mask &= df["days_since_first_submit"].notna()
    mask &= df["paper_year"].notna()
    mask &= df["title_word_count"] > 0
    mask &= df["abstract_word_count"] > 0
    for c in ["Title", "Abstract", "arXiv_Code", "Category"]:
        mask &= df[c].astype(str).str.len() > 0
    df = df[mask].reset_index(drop=True)

    df = df[[
        "days_since_first_submit", "paper_year", "title_word_count", "abstract_word_count",
        "Category",
        "Title", "arXiv_Code", "Abstract",
    ]]
    return df


# -----------------------------
# Top-K category filtering + re-encoding
# -----------------------------
def keep_top_k_categories(df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Keep only papers whose ``Category`` is among the top-``k`` most frequent
    categories, then re-encode those categories as ``"0"``..``"k-1"`` strings,
    sorted by descending frequency (so "0" is the most common class).

    Returns (filtered_df, mapping) where ``mapping`` is ``{original_cat: new_cat_str}``.
    """
    counts = df["Category"].value_counts()
    if k <= 0 or k >= len(counts):
        return df.copy(), {c: c for c in counts.index}

    top_cats = counts.head(k).index.tolist()
    print(f"[top_k] keeping top {k} categories out of {len(counts)}:")
    for new_id, cat in enumerate(top_cats):
        print(f"  {new_id} <- {cat} (count={counts[cat]})")

    mapping = {cat: str(i) for i, cat in enumerate(top_cats)}
    df = df[df["Category"].isin(top_cats)].copy()
    df["Category"] = df["Category"].map(mapping)
    df = df.reset_index(drop=True)
    return df, mapping


# -----------------------------
# Serialization + token budgeting
# -----------------------------
NUM_ORDER  = [
    "days_since_first_submit",
    "paper_year",
    "title_word_count",
    "abstract_word_count",
]
CAT_ORDER  = ["Category"]
TEXT_ORDER = ["Title", "arXiv_Code", "Abstract"]
CSV_COLS   = NUM_ORDER + CAT_ORDER + TEXT_ORDER


def serialize_row(row: pd.Series) -> str:
    """
    numerical | categorical | text  (raw values, joined by '|', no key names).
    Order matches CSV_COLS so that the serialized text mirrors the saved CSV.
    """
    parts: List[str] = []
    for c in CSV_COLS:
        parts.append(_clean_str(row.get(c, "")))
    return "|".join(parts)


def token_len(tokenizer, s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))


def truncate_to_fit(
    tokenizer,
    row: pd.Series,
    max_tokens: int,
    min_keep_chars: int = 32,
    truncate_order: Tuple[str, ...] = ("Abstract", "Title", "arXiv_Code"),
) -> Optional[Dict[str, Any]]:
    """
    Try to truncate text fields (by chars) until serialized text fits max_tokens.
    Returns a dict of possibly-updated fields if successful, else None.
    """
    d = row.to_dict()

    s = serialize_row(pd.Series(d))
    L = token_len(tokenizer, s)
    if L < max_tokens:
        d["_serialized"] = s
        d["_tok_len"] = L
        return d

    caps = {
        "Abstract":   1500,
        "Title":      300,
        "arXiv_Code": 64,
    }

    for _ in range(12):
        for k in truncate_order:
            val = _clean_str(d.get(k, ""))
            if len(val) > caps.get(k, 300):
                d[k] = val[: caps[k]].rstrip()

        s = serialize_row(pd.Series(d))
        L = token_len(tokenizer, s)
        if L < max_tokens:
            d["_serialized"] = s
            d["_tok_len"] = L
            return d

        for k in truncate_order:
            caps[k] = max(min_keep_chars, int(caps[k] * 0.75))

    return None


# -----------------------------
# Sample / split pipeline
# -----------------------------
def sample_and_split(
    df: pd.DataFrame,
    tokenizer,
    max_tokens: int,
    num_samples: int,
    test_size: int,
    seed: int,
    truncate: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter by token budget, sample num_samples for (train+val) using the FIRST
    num_samples accepted rows, then sample test_size from the NEXT accepted
    rows (no overlap). Train/val split is 9:1 on the first num_samples.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pool_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    target_pool = num_samples
    target_test = test_size

    for _, row in df.iterrows():
        if truncate:
            d = truncate_to_fit(tokenizer, row, max_tokens=max_tokens)
            if d is None:
                continue
            ser = d["_serialized"]
            toklen = d["_tok_len"]
        else:
            ser = serialize_row(row)
            toklen = token_len(tokenizer, ser)
            if toklen >= max_tokens:
                continue
            d = row.to_dict()

        d["text"] = ser
        d["tok_len"] = int(toklen)

        if len(pool_rows) < target_pool:
            pool_rows.append(d)
        elif len(test_rows) < target_test:
            test_rows.append(d)
        else:
            break

    if len(pool_rows) < target_pool:
        raise RuntimeError(
            f"After filtering by max_tokens<{max_tokens}, only got {len(pool_rows)} rows "
            f"for train+valid, but you requested {target_pool}. "
            f"Try increasing max_tokens or enabling --truncate."
        )

    if len(test_rows) < target_test:
        raise RuntimeError(
            f"After filtering by max_tokens<{max_tokens}, only got {len(test_rows)} rows "
            f"for test, but you requested {target_test}. "
            f"Try increasing max_tokens or enabling --truncate."
        )

    out_df = pd.DataFrame(pool_rows).reset_index(drop=True)
    out_df["id"] = np.arange(len(out_df), dtype=int)

    train_df, val_df = train_test_split(out_df, test_size=1/9, random_state=seed)
    train_df = train_df.sort_values("id").reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df), dtype=int)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    val_df["id"] = np.arange(len(val_df), dtype=int)

    test_df = pd.DataFrame(test_rows).reset_index(drop=True)
    test_df["id"] = np.arange(len(test_df), dtype=int)

    return train_df, val_df, test_df


def save_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    train_df[CSV_COLS].to_csv(os.path.join(out_dir, "train_ori.csv"), index=False)
    val_df[CSV_COLS].to_csv(os.path.join(out_dir, "valid_ori.csv"), index=False)
    test_df[CSV_COLS].to_csv(os.path.join(out_dir, "test_ori.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./data/tabular/rel_arxiv", help="output directory")
    ap.add_argument("--max_tokens", type=int, default=160, help="X: each serialized row must have < X tokens")
    ap.add_argument("--num_samples", type=int, default=4500, help="y: number of sampled rows total (train+val)")
    ap.add_argument("--test_size", type=int, default=2000, help="number of sampled rows for test (disjoint from train/val)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truncate", action="store_true", help="truncate long text fields to fit max_tokens instead of dropping")
    ap.add_argument("--top_k_categories", type=int, default=15,
                    help="keep only the top-K most frequent Category values and re-encode them to '0'..'K-1' (sorted by descending frequency); set to 0 to keep all 53 categories")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
    )

    tables = load_relbench_arxiv(download=False, upto_test_timestamp=True)
    merged = build_merged_df(tables)
    print(f"[merged] {len(merged)} papers after cleaning")
    print(f"[merged] Category unique={merged['Category'].nunique()}; top 10:")
    print(merged["Category"].value_counts().head(10).to_string())

    if args.top_k_categories > 0:
        merged, _ = keep_top_k_categories(merged, args.top_k_categories)
        print(f"[merged] {len(merged)} papers after top-{args.top_k_categories} filtering")

    train_df, val_df, test_df = sample_and_split(
        merged,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        test_size=args.test_size,
        seed=args.seed,
        truncate=args.truncate,
    )

    save_outputs(train_df, val_df, test_df, args.out_dir)

    print(f"[OK] saved to {args.out_dir}")
    print(f"  train: {len(train_df)}  valid: {len(val_df)}  test: {len(test_df)}")
    print(f"  avg tok_len train: {train_df['tok_len'].mean():.1f}  valid: {val_df['tok_len'].mean():.1f}  test: {test_df['tok_len'].mean():.1f}")
    print(f"  max tok_len train: {train_df['tok_len'].max()}  valid: {val_df['tok_len'].max()}  test: {test_df['tok_len'].max()}")


if __name__ == "__main__":
    main()

