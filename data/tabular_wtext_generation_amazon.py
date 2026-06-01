import os
import json
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
def load_relbench_amazon(download: bool = True, upto_test_timestamp: bool = True):
    from relbench.datasets import get_dataset

    dataset = get_dataset(name="rel-amazon", download=download)

    db = dataset.get_db() if upto_test_timestamp else dataset.get_db(upto_test_timestamp=False)

    keys = list(db.table_dict.keys())
    lower = {k.lower(): k for k in keys}

    def pick_table(possible_names):
        for name in possible_names:
            if name in lower:
                return db.table_dict[lower[name]].df
        # fallback: substring search
        for k in keys:
            kl = k.lower()
            if any(name in kl for name in possible_names):
                return db.table_dict[k].df
        raise KeyError(f"Cannot find table among candidates={possible_names}. Available tables={keys}")

    review_df = pick_table(["review", "reviews"])
    product_df = pick_table(["product", "products"])

    if not isinstance(review_df, pd.DataFrame) or not isinstance(product_df, pd.DataFrame):
        raise RuntimeError("Loaded tables are not pandas DataFrames. Check RelBench version/table objects.")

    return review_df, product_df


NUM_COLS = ["price", "review_time", "rating"]
CAT_COLS = ["category", "verified"]
TEXT_COLS = ["brand", "title", "description", "review_text", "summary"]

KEEP_COLS = CAT_COLS + TEXT_COLS + NUM_COLS


def _clean_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def serialize_row(row: pd.Series) -> str:
    """
    numerical | categorical | text
    Each field is emitted as raw value (no key names) to minimize tokens.
    """
    # numerical first
    parts: List[str] = []
    # price, rating as strings (keep concise)
    price = row.get("price", "")
    review_time = row.get("review_time", "")
    rating = row.get("rating", "")
    parts.append(_clean_str(price))
    parts.append(_clean_str(review_time))
    parts.append(_clean_str(rating))

    # categorical
    parts.append(_clean_str(row.get("brand", "")))
    parts.append(_clean_str(row.get("category", "")))
    # verified can be bool/int/str
    v = row.get("verified", "")
    if isinstance(v, (bool, np.bool_)):
        v = "true" if bool(v) else "false"
    parts.append(_clean_str(v))

    # text
    parts.append(_clean_str(row.get("title", "")))
    parts.append(_clean_str(row.get("description", "")))
    parts.append(_clean_str(row.get("review_text", "")))
    parts.append(_clean_str(row.get("summary", "")))

    # final join
    return "|".join(parts)


def token_len(tokenizer, s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))


def truncate_to_fit(
    tokenizer,
    row: pd.Series,
    max_tokens: int,
    min_keep_chars: int = 32,
    truncate_order: Tuple[str, ...] = ("review_text", "description", "summary", "title"),
) -> Optional[Dict[str, Any]]:
    """
    Try to truncate text fields (by chars) until serialized text fits max_tokens.
    Returns a dict of possibly-updated fields if successful, else None.
    """
    # Work on a mutable dict copy
    d = row.to_dict()

    # Fast path
    s = serialize_row(pd.Series(d))
    if token_len(tokenizer, s) < max_tokens:
        d["_serialized"] = s
        d["_tok_len"] = token_len(tokenizer, s)
        return d

    # progressively truncate long text fields
    # Start with aggressive caps; shrink further if needed.
    caps = {
        "review_text": 800,
        "description": 600,
        "summary": 300,
        "title": 200,
    }

    for _ in range(12):  # bounded attempts
        # apply current caps
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

        # if still too long, shrink caps
        for k in truncate_order:
            caps[k] = max(min_keep_chars, int(caps[k] * 0.75))

    return None


def safe_text(x, max_chars: int = 4000) -> str:
    if x is None:
        return ""
    # pandas/numpy NaN
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    # numpy scalar -> python scalar
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


def verified_to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return "true" if int(v) != 0 else "false"
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return "true"
    if s in ("false", "f", "0", "no", "n"):
        return "false"
    return ""


def category_to_last_k(x, k: int = 2) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        items = []
        for t in x:
            st = safe_text(t, max_chars=200).strip()
            if st:
                items.append(st)
        if not items:
            return ""
        return " > ".join(items[-k:])
    return safe_text(x, max_chars=200)


def build_merged_df(review_df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    needed_review = ["product_id", "review_text", "summary", "rating", "verified", "review_time"]
    needed_product = ["product_id", "brand", "title", "description", "price", "category"]

    for c in needed_review:
        if c not in review_df.columns:
            raise ValueError(f"review table missing column: {c}")
    for c in needed_product:
        if c not in product_df.columns:
            raise ValueError(f"product table missing column: {c}")

    r = review_df[needed_review].copy()
    p = product_df[needed_product].copy()
    df = r.merge(p, on="product_id", how="inner")

    # normalize columns
    df["category"] = df["category"].map(lambda x: category_to_last_k(x, k=2))
    df["verified"] = df["verified"].map(verified_to_str)

    for c in ["brand", "title", "description", "review_text", "summary"]:
        df[c] = df[c].map(safe_text)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # review_time -> relative days (float)
    t = pd.to_datetime(df["review_time"], errors="coerce")
    t0 = t.min()
    print("The earliest review time is", t0)
    df["review_time"] = (t - t0).dt.days.astype("float32")

    # require non-empty every column
    mask = np.ones(len(df), dtype=bool)
    for c in ["brand", "title", "description", "category", "review_text", "summary", "verified"]:
        mask &= df[c].str.len() > 0
    mask &= df["price"].notna()
    mask &= df["rating"].notna()
    mask &= df["review_time"].notna()

    df = df[mask].reset_index(drop=True)

    df = df[["brand","title","description","price","category","review_text","summary","rating","verified","review_time"]]
    return df


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
    Filter by token budget, sample num_samples for (train+val) using the FIRST num_samples
    accepted rows, then sample test_size from the NEXT accepted rows (no overlap).
    Train/val split is 9:1 on the first num_samples.
    """
    rng = np.random.default_rng(seed)

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

    # Train/val split 9:1
    train_df, val_df = train_test_split(out_df, test_size=0.1, random_state=seed)
    train_df = train_df.sort_values("id").reset_index(drop=True)
    train_df["id"] = np.arange(len(train_df), dtype=int)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    val_df["id"] = np.arange(len(val_df), dtype=int)

    test_df = pd.DataFrame(test_rows).reset_index(drop=True)
    test_df["id"] = np.arange(len(test_df), dtype=int)

    return train_df, val_df, test_df


NUM_ORDER  = ["price", "review_time"]
CAT_ORDER  = ["rating", "category", "verified"]
TEXT_ORDER = ["brand", "title", "description", "review_text", "summary"]
CSV_COLS = NUM_ORDER + CAT_ORDER + TEXT_ORDER

def save_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    train_df[CSV_COLS].to_csv(os.path.join(out_dir, "train_ori.csv"), index=False)
    val_df[CSV_COLS].to_csv(os.path.join(out_dir, "valid_ori.csv"), index=False)
    test_df[CSV_COLS].to_csv(os.path.join(out_dir, "test_ori.csv"), index=False)

# The earliest review time is 2008-01-01 00:00:00
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./data/tabular/rel_amazon", help="output directory")
    ap.add_argument("--max_tokens", type=int, default=128, help="X: each serialized row must have < X tokens")
    ap.add_argument("--num_samples", type=int, default=5000, help="y: number of sampled rows total (train+val)")
    ap.add_argument("--test_size", type=int, default=2250, help="number of sampled rows for test (disjoint from train/val)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truncate", action="store_true", help="truncate long text fields to fit max_tokens instead of dropping")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
    )

    review_df, product_df = load_relbench_amazon(download=False, upto_test_timestamp=True)
    merged = build_merged_df(review_df, product_df)

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