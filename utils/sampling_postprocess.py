from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# Columns that are stored as integers in raw output but map to categorical candidates.
_NUM2CAT_COLUMNS: Dict[str, List[str]] = {
    "shoppers": ["OperatingSystems", "Browser", "Region", "TrafficType"],
    "shoppers_dcr": ["OperatingSystems", "Browser", "Region", "TrafficType"],
    "beijing": ["year", "month", "day", "hour"],
    "beijing_dcr": ["year", "month", "day", "hour"],
    "default": [
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "default payment next month",
    ],
    "default_dcr": [
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "default payment next month",
    ],
    "rel_amazon": ["rating"],
    "rel_amazon_dcr": ["rating"],
    "rel_arxiv": ["Category"],
}

_BOL2CAT_COLUMNS: Dict[str, List[str]] = {
    "shoppers": ["Weekend", "Revenue"],
    "shoppers_dcr": ["Weekend", "Revenue"],
    "rel_amazon": ["verified"],
    "rel_amazon_dcr": ["verified"],
}

_NUMERIC_COLUMN_TYPES = frozenset({"float", "int", "number", "numeric"})


def _candidate_filter_lists(dataset_name: str) -> Tuple[List[str], List[str]]:
    return (
        list(_NUM2CAT_COLUMNS.get(dataset_name, [])),
        list(_BOL2CAT_COLUMNS.get(dataset_name, [])),
    )


def filter_not_in_candidates(
    df: pd.DataFrame,
    dataset_name: str,
    info: dict,
    num_samples_init: int,
) -> pd.DataFrame:
    df_filtered = df.copy()
    num2cat_list, bol2cat_list = _candidate_filter_lists(dataset_name)

    for col in num2cat_list:
        info[col]["candidates"] = [int(i) for i in info[col]["candidates"]]
        df_filtered[col] = (
            pd.to_numeric(df_filtered[col], errors="coerce").round().astype("Int64")
        )
    for col in bol2cat_list:
        df_filtered[col] = df_filtered[col].astype(str).str.strip()
    df_filtered = df_filtered.dropna(subset=num2cat_list)

    for col, meta in info.items():
        if meta["type"] == "category":
            valid_vals = set(meta["candidates"])
            df_filtered = df_filtered[df_filtered[col].isin(valid_vals)]
    # print(
    #     f"After dropping row not in candidate, synthetic counts from "
    #     f"{num_samples_init} -> {len(df_filtered)}."
    # )

    for col in bol2cat_list:
        df_filtered[col] = (
            df_filtered[col].map({"True": True, "False": False}).astype("boolean")
        )

    for col, meta in info.items():
        if meta["type"] in _NUMERIC_COLUMN_TYPES:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    numeric_cols = [
        col for col, meta in info.items() if meta["type"] in _NUMERIC_COLUMN_TYPES
    ]
    df_filtered = df_filtered.dropna(subset=numeric_cols)
    df_filtered = df_filtered.reset_index(drop=True)
    # print(
    #     f"After dropping row is not numerical, synthetic counts from "
    #     f"{num_samples_init} -> {len(df_filtered)}."
    # )
    return df_filtered


def build_filter_column_info(train_ds, *, all_numerical: bool, dataset_name: str) -> dict:
    if all_numerical:
        info = copy.deepcopy(train_ds.stas_info["column_info"])
        for col in train_ds.nonNumerical_columns:
            info[col]["type"] = "category"
        if dataset_name in _RAW_CLASS_COLUMN_TRANSFORMS:
            info["class"]["type"] = "category"
        return info
    return train_ds.stas_info["column_info"]


def _normalize_binary_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map rounded 0/1 class indices to dataset label strings."""
    out = df.copy()
    out["class"] = out["class"].round().astype(int)
    out = out[out["class"].isin([0, 1])].copy()
    out["class"] = out["class"].map({1: "g", 0: "h"})
    return out


_RAW_CLASS_COLUMN_TRANSFORMS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "magic": _normalize_binary_class_column,
}


def _map_integer_indices_to_candidates(df: pd.DataFrame, train_ds) -> pd.DataFrame:
    out = df.copy()
    column_info = train_ds.stas_info["column_info"]
    for col in train_ds.nonNumerical_columns:
        candidates = column_info[col]["candidates"]
        out[col] = out[col].round().astype(int)
        index_to_label = {idx: label for idx, label in enumerate(candidates)}
        valid_indices = list(index_to_label.keys())
        out = out[out[col].isin(valid_indices)].copy()
        out[col] = out[col].map(index_to_label)
    return out


def _strip_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(lambda v: v.strip() if isinstance(v, str) else v))


def postprocess_loaded_samples(
    df: pd.DataFrame,
    train_ds,
    *,
    all_numerical: bool,
    dataset_name: str,
) -> pd.DataFrame:
    out = df.dropna()

    class_transform = _RAW_CLASS_COLUMN_TRANSFORMS.get(dataset_name)
    if class_transform is not None:
        out = class_transform(out)

    if all_numerical:
        out = _map_integer_indices_to_candidates(out, train_ds)

    out = out[train_ds.ori_columns]
    out = _strip_string_cells(out)
    return out.dropna()


def _finalize_math_latex_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[columns] = out[columns].round(1)
    return out


def _finalize_biography_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[columns] = out[columns].round()
    return out


def _finalize_rel_arxiv_numerical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        series = pd.to_numeric(out[col], errors="coerce")
        if col == "paper_year":
            out[col] = series.clip(lower=1991, upper=2035).round()
        else:
            out[col] = series.clip(lower=0).round()
    return out


_NUMERICAL_FINALIZE: Dict[str, Callable[[pd.DataFrame, List[str]], pd.DataFrame]] = {
    "math_latex": _finalize_math_latex_numerical,
    "biography": _finalize_biography_numerical,
    "rel_arxiv": _finalize_rel_arxiv_numerical,
}


def finalize_numerical_columns(
    df: pd.DataFrame,
    dataset_name: str,
    numerical_columns: List[str],
) -> pd.DataFrame:
    finalize = _NUMERICAL_FINALIZE.get(dataset_name)
    if finalize is None or not numerical_columns:
        return df
    return finalize(df, numerical_columns)


def clean_and_filter_samples(
    df: pd.DataFrame,
    train_ds,
    *,
    all_numerical: bool,
    dataset_name: str,
    num_samples_init: int,
) -> pd.DataFrame:
    df = postprocess_loaded_samples(
        df,
        train_ds,
        all_numerical=all_numerical,
        dataset_name=dataset_name,
    )
    # print(f"After doing strip, synthetic counts from {num_samples_init} -> {len(df)}.")
    info = build_filter_column_info(
        train_ds,
        all_numerical=all_numerical,
        dataset_name=dataset_name,
    )
    return filter_not_in_candidates(df, dataset_name, info, num_samples_init)


def target_sample_count(train_df_len: int, proportion: int) -> int:
    return int((train_df_len + proportion - 0.01) // proportion)
