from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import os
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


DATASET_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rel_amazon": {
        "num_feats":  ["price", "review_time"],
        "cat_feats":  ["verified"],
        "text_feats": ["title", "brand", "category", "description", "review_text", "summary"],
        "label_col":  "rating",
        "label_is_int": True,
    },
    "rel_arxiv": {
        "num_feats":  [
            "days_since_first_submit",
            "paper_year",
            "title_word_count",
            "abstract_word_count",
        ],
        "cat_feats":  [],
        "text_feats": ["Title", "arXiv_Code", "Abstract"],
        "label_col":  "Category",
        "label_is_int": True,
    },
}


def _resolve_schema(dataset_name: Optional[str]) -> Dict[str, Any]:
    if dataset_name is None:
        # fall back to rel_amazon (legacy behaviour)
        dataset_name = "rel_amazon"
    if dataset_name not in DATASET_SCHEMAS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}' for evaluate_mle_with_text. "
            f"Known datasets: {list(DATASET_SCHEMAS)}"
        )
    return DATASET_SCHEMAS[dataset_name]


# Legacy module-level constants kept for backwards compatibility.
NUM_FEATS = DATASET_SCHEMAS["rel_amazon"]["num_feats"]
CAT_FEATS = DATASET_SCHEMAS["rel_amazon"]["cat_feats"]
TEXT_FEATS = DATASET_SCHEMAS["rel_amazon"]["text_feats"]
LABEL_COL = DATASET_SCHEMAS["rel_amazon"]["label_col"]


# -----------------------------
# Embedding (HF mean pooling + optional caching)
# -----------------------------
class HFTextEmbedder:
    """
    HF embedding wrapper:
      - mean pooling over last_hidden_state
      - L2 normalize
      - optional per-text caching to disk (recommended)
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1",
        device: Optional[str] = None,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    def _hash(self, s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _cache_path(self, s: str) -> str:
        assert self.cache_dir is not None
        return os.path.join(self.cache_dir, self._hash(s) + ".npy")

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out_vecs: List[np.ndarray] = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]

            # cache path (per string)
            if self.cache_dir:
                cached: List[Optional[np.ndarray]] = []
                miss_text: List[str] = []
                miss_pos: List[int] = []

                for j, t in enumerate(batch):
                    fp = self._cache_path(t)
                    if os.path.exists(fp):
                        cached.append(np.load(fp))
                    else:
                        cached.append(None)
                        miss_text.append(t)
                        miss_pos.append(j)

                if miss_text:
                    enc = self.tok(
                        miss_text,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)

                    last = self.model(**enc).last_hidden_state  # (B, L, H)
                    mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    pooled = torch.nn.functional.normalize(pooled, dim=1)
                    pooled_np = pooled.detach().cpu().float().numpy()

                    for k, pos in enumerate(miss_pos):
                        fp = self._cache_path(batch[pos])
                        np.save(fp, pooled_np[k])
                        cached[pos] = pooled_np[k]

                out_vecs.append(np.stack(cached, axis=0))
                continue

            # no-cache
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            last = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            out_vecs.append(pooled.detach().cpu().float().numpy())

        return np.concatenate(out_vecs, axis=0)


def _safe_str(s: Any) -> str:
    if s is None:
        return ""
    # handle NaN
    try:
        if isinstance(s, float) and np.isnan(s):
            return ""
    except Exception:
        pass
    return str(s)


def concat_text(df: pd.DataFrame, text_feats: Optional[List[str]] = None) -> List[str]:
    """One-pass embedding for all text columns.

    If ``text_feats`` is None the legacy rel_amazon column ordering/labels are
    used (kept for backwards compat). Otherwise we generically build
    ``"colA: ... [SEP] colB: ..."``.
    """
    if text_feats is None:
        t = (
            "title: " + df["title"].astype(str)
            + " [SEP] brand: " + df["brand"].astype(str)
            + " [SEP] category: " + df["category"].astype(str)
            + " [SEP] description: " + df["description"].astype(str)
            + " [SEP] review: " + df["review_text"].astype(str)
            + " [SEP] summary: " + df["summary"].astype(str)
        )
        return t.tolist()

    parts = None
    for c in text_feats:
        chunk = f"{c}: " + df[c].astype(str)
        parts = chunk if parts is None else parts + " [SEP] " + chunk
    return parts.tolist()


def _normalize_verified(x: Any) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "true" if bool(x) else "false"
    if isinstance(x, (int, np.integer)):
        return "true" if int(x) != 0 else "false"
    s = _safe_str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return "true"
    if s in ("false", "f", "0", "no", "n"):
        return "false"
    return ""


def _ensure_schema(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Normalize the dataframe according to ``schema``.

    ``schema`` is a dict with keys ``num_feats``, ``cat_feats``, ``text_feats``,
    ``label_col`` and ``label_is_int``. Defaults to the rel_amazon schema if
    not provided (for backwards compatibility).
    """
    if schema is None:
        schema = DATASET_SCHEMAS["rel_amazon"]

    num_feats: List[str]  = schema["num_feats"]
    cat_feats: List[str]  = schema["cat_feats"]
    text_feats: List[str] = schema["text_feats"]
    label_col: str        = schema["label_col"]
    label_is_int: bool    = schema.get("label_is_int", True)

    need = set(num_feats + cat_feats + text_feats + [label_col])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    out = df.copy()

    # numeric features
    for c in num_feats:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # label - parsed as numeric if ``label_is_int``, otherwise left as a string.
    if label_is_int:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    else:
        out[label_col] = out[label_col].map(_safe_str).str.strip()

    # categorical features: special-case the rel_amazon "verified" column.
    for c in cat_feats:
        if c == "verified":
            out[c] = out[c].map(_normalize_verified)
        else:
            out[c] = out[c].map(_safe_str).str.strip()

    # text features
    for c in text_feats:
        out[c] = out[c].map(_safe_str).str.strip()

    # drop rows with any empty/NaN
    mask = np.ones(len(out), dtype=bool)
    for c in num_feats:
        mask &= out[c].notna()
    if label_is_int:
        mask &= out[label_col].notna()
    else:
        mask &= out[label_col].astype(str).str.len() > 0
    for c in cat_feats:
        mask &= out[c].astype(str).str.len() > 0
    for c in text_feats:
        mask &= out[c].astype(str).str.len() > 0

    out = out[mask].reset_index(drop=True)

    if label_is_int:
        out[label_col] = out[label_col].round().astype(int)

    return out


def make_X_y(
    df: pd.DataFrame,
    embedder: HFTextEmbedder,
    label_encoder: Optional[LabelEncoder] = None,
    ohe: Optional[OneHotEncoder] = None,
    fit: bool = False,
    use_num: bool = True,
    use_cat: bool = True,
    use_text: bool = True,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray, LabelEncoder, OneHotEncoder]:
    if schema is None:
        schema = DATASET_SCHEMAS["rel_amazon"]
    num_feats: List[str]  = schema["num_feats"]
    cat_feats: List[str]  = schema["cat_feats"]
    text_feats: List[str] = schema["text_feats"]
    label_col: str        = schema["label_col"]

    df = _ensure_schema(df, schema=schema)

    # y
    y_raw = df[label_col].to_numpy()
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_raw)
    y = label_encoder.transform(y_raw)

    parts = []

    # numeric
    if use_num and len(num_feats) > 0:
        X_num = df[num_feats].to_numpy(dtype=np.float32)
        parts.append(sparse.csr_matrix(X_num))

    # categorical
    if use_cat and len(cat_feats) > 0:
        if ohe is None:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            fit = True
        X_cat = ohe.fit_transform(df[cat_feats]) if fit else ohe.transform(df[cat_feats])
        parts.append(X_cat)

    # text
    if use_text and len(text_feats) > 0:
        texts = concat_text(df, text_feats=text_feats)
        X_txt = embedder(texts)
        parts.append(sparse.csr_matrix(X_txt.astype(np.float32)))

    if not parts:
        raise ValueError("At least one feature group (num/cat/text) must be non-empty")

    X = sparse.hstack(parts, format="csr")
    return X, y, label_encoder, ohe

DEFAULT_XGB_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [6, 10],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "gamma": [0.0, 1.0],
    "reg_lambda": [1.0],
    "tree_method": ["gpu_hist"],  # if no GPU, override to 'hist'
}


@dataclass
class UtilityEvalConfig:
    embed_model: str = "nomic-ai/nomic-embed-text-v1"
    embed_max_length: int = 256
    embed_cache_dir: str = ""
    seed: int = 42

    val_ratio: float = 0.1

    xgb_grid: Dict[str, List[Any]] = None

    use_gpu: bool = True


def _fit_best_xgb_multiclass(
    Xtr, ytr, Xva, yva, num_class: int, seed: int, grid: Dict[str, List[Any]]
) -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, float]]:
    best = None

    for param in ParameterGrid(grid):
        model = XGBClassifier(
            **param,
            objective="multi:softprob",
            num_class=num_class,
            random_state=seed,
            nthread=-1,
            eval_metric="mlogloss",
        )
        model.fit(Xtr, ytr)

        pred = model.predict(Xva)
        proba = model.predict_proba(Xva)

        macro_f1 = f1_score(yva, pred, average="macro")
        w_f1 = f1_score(yva, pred, average="weighted")
        acc = accuracy_score(yva, pred)

        auc = multiclass_auroc_ovr(yva, proba)

        score = (auc["auc_ovr_macro"], macro_f1, acc)

        if (best is None) or (score > best["score"]):
            best = {
                "score": score,
                "model": model,
                "param": param,
                "val": {
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(w_f1),
                    "acc": float(acc),
                    **auc,
                },
            }

    assert best is not None
    return best["model"], best["param"], best["val"]


def multiclass_auroc_ovr(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    y_true: shape (N,), integer class ids in [0, C-1]
    proba : shape (N, C), predicted probabilities aligned to those class ids

    Returns macro/weighted OvR AUROC.
    Handles missing classes in y_true by restricting to present classes.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    C = proba.shape[1]
    present = np.unique(y_true)
    if len(present) <= 1:
        # AUROC undefined if only one class present
        return {"auc_ovr_macro": float("nan"), "auc_ovr_weighted": float("nan")}

    # restrict to present classes to avoid sklearn ValueError
    proba_sub = proba[:, present]
    # re-map y_true into 0..len(present)-1
    mapper = {c: i for i, c in enumerate(present.tolist())}
    y_sub = np.vectorize(mapper.get)(y_true)

    # one-hot
    Y = np.eye(len(present), dtype=np.float32)[y_sub]

    auc_macro = roc_auc_score(Y, proba_sub, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(Y, proba_sub, multi_class="ovr", average="weighted")
    return {"auc_ovr_macro": float(auc_macro), "auc_ovr_weighted": float(auc_weighted)}


def _eval_on_test(model: XGBClassifier, Xte, yte) -> Dict[str, float]:
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    auc = multiclass_auroc_ovr(yte, proba)
    return {
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "test_weighted_f1": float(f1_score(yte, pred, average="weighted")),
        "test_acc": float(accuracy_score(yte, pred)),
        **{f"test_{k}": v for k, v in auc.items()},
    }


def remap_labels_to_zero_based(ytr: np.ndarray, yva: np.ndarray, yte: np.ndarray):
    classes = np.unique(ytr)
    mapper = {c: i for i, c in enumerate(classes.tolist())}

    def _map(y):
        y = np.asarray(y)
        mask = np.isin(y, classes)
        return np.vectorize(mapper.get)(y[mask]), mask

    ytr2, mtr = _map(ytr)
    yva2, mva = _map(yva)
    yte2, mte = _map(yte)
    return ytr2, yva2, yte2, mtr, mva, mte, len(classes)


def evaluate_mle_with_text(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[UtilityEvalConfig] = None,
    do_ttr=False,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns dict containing:
      - TTR (train real -> test real)
      - TSTR (train syn  -> test real)

    ``dataset_name`` selects the column schema (rel_amazon, rel_arxiv, ...).
    Defaults to ``rel_amazon`` to preserve backwards-compatible behaviour.
    """
    cfg = cfg or UtilityEvalConfig()
    if cfg.xgb_grid is None:
        cfg.xgb_grid = dict(DEFAULT_XGB_GRID)

    if not cfg.use_gpu:
        cfg.xgb_grid = dict(cfg.xgb_grid)
        cfg.xgb_grid["tree_method"] = ["hist"]

    if dataset_name is None:
        dataset_name = "rel_amazon"
    schema = _resolve_schema(dataset_name)

    if not cfg.embed_cache_dir:
        cfg.embed_cache_dir = os.path.join("result", dataset_name, "embedding_cache")
        dir_name = cfg.embed_cache_dir
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    embedder = HFTextEmbedder(
        model_name=cfg.embed_model,
        max_length=cfg.embed_max_length,
        cache_dir=cfg.embed_cache_dir,
        trust_remote_code=True,
    )

    has_cat = len(schema["cat_feats"]) > 0
    test_df_clean = _ensure_schema(test_df, schema=schema)

    feature_sets: Dict[str, Dict[str, bool]] = {
        "num+cat+text": dict(use_num=True, use_cat=True, use_text=True),
    }
    if has_cat:
        feature_sets["cat+text"] = dict(use_num=False, use_cat=True, use_text=True)
    else:
        feature_sets["text"] = dict(use_num=False, use_cat=False, use_text=True)

    def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = _ensure_schema(df, schema=schema)
        tr, va = train_test_split(df, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    out: dict[str, Any] = {}
    for fs_name, fs_kwargs in feature_sets.items():
        real_tr, real_va = _split(train_df)
        Xtr, ytr, le, ohe = make_X_y(real_tr, embedder, fit=True, schema=schema, **fs_kwargs)
        Xva, yva, _, _ = make_X_y(real_va, embedder, label_encoder=le, ohe=ohe, fit=False, schema=schema, **fs_kwargs)
        Xte, yte, _, _ = make_X_y(test_df_clean, embedder, label_encoder=le, ohe=ohe, fit=False, schema=schema, **fs_kwargs)

        # ---- TTR: Train real -> Test real ----
        if do_ttr:
            num_class = len(le.classes_)
            ttr_model, ttr_param, ttr_val = _fit_best_xgb_multiclass(
                Xtr, ytr, Xva, yva, num_class=num_class, seed=cfg.seed, grid=cfg.xgb_grid
            )
            ttr_test = _eval_on_test(ttr_model, Xte, yte)
            out[f"{fs_name}/Real->Test"] = {**ttr_test}

        # ---- TSTR: Train synthetic -> Test real ----
        syn_tr, syn_va = _split(synthetic_df)
        Xtr_s, ytr_s, _, _ = make_X_y(syn_tr, embedder, label_encoder=le, ohe=ohe, fit=False, schema=schema, **fs_kwargs)
        Xva_s, yva_s, _, _ = make_X_y(syn_va, embedder, label_encoder=le, ohe=ohe, fit=False, schema=schema, **fs_kwargs)

        ytr_s2, yva_s2, yte2, mtr, mva, mte, num_class_s = remap_labels_to_zero_based(ytr_s, yva_s, yte)

        Xtr_s2 = Xtr_s[mtr]
        Xva_s2 = Xva_s[mva]
        Xte2 = Xte[mte]

        tstr_model, tstr_param, tstr_val = _fit_best_xgb_multiclass(
            Xtr_s2, ytr_s2, Xva_s2, yva_s2,
            num_class=num_class_s, seed=cfg.seed, grid=cfg.xgb_grid
        )
        tstr_test = _eval_on_test(tstr_model, Xte2, yte2)

        out[f"{fs_name}/Synthetic->Test"] = {**tstr_test}

    return out
