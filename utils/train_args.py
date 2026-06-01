import json
import os
from typing import Any, Iterable, Optional, Sequence

MODEL_INIT_ARGS: Sequence[str] = (
    "model_name",
    "dataset_name",
    "max_len",
    "answer_len",
    "mask_eps",
    "ae_hidden_dim",
    "dlm_hidden_dim",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_parameters",
    "normalization",
    "loss_type",
    "bf16",
    "template_probs",
    "all_numerical",
)


_BOOL_STORE_TRUE_KEYS = frozenset({"bf16", "all_numerical"})


def ckpt_dir_for_dataset(dataset_name: str) -> str:
    return os.path.join("ckpt", dataset_name)


def ckpt_tag(description: str, *, use_best: bool) -> str:
    return f"{description}_best" if use_best else f"{description}_last"


def train_args_path(
    dataset_name: str,
    description: str,
    *,
    use_best: bool = True,
) -> str:
    """Path to train_args.json co-located with a checkpoint directory."""
    return os.path.join(
        ckpt_dir_for_dataset(dataset_name),
        ckpt_tag(description, use_best=use_best),
        "train_args.json",
    )


def collect_model_init_args(args: Any, keys: Iterable[str] = MODEL_INIT_ARGS) -> dict:
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def dump_train_args(args: Any, *, save_dir: Optional[str] = None) -> str:
    """Write model-init fields to ``<checkpoint_dir>/train_args.json``."""
    if save_dir is None:
        save_dir = getattr(args, "save_dir", None)
        description = getattr(args, "description", None)
        if save_dir is None or description is None:
            raise ValueError(
                "dump_train_args needs save_dir or args.save_dir with args.description"
            )
        save_dir = os.path.join(save_dir, f"{description}_last")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "train_args.json")
    payload = collect_model_init_args(args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[train_args] persisted model-init args to {path}")
    return path


def populate_args_from_train_args(
    args: Any,
    *,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    use_best: Optional[bool] = None,
    keys: Sequence[str] = MODEL_INIT_ARGS,
) -> list[str]:
    if dataset_name is None:
        dataset_name = getattr(args, "dataset_name", None)
    if description is None:
        description = getattr(args, "description", None)
    if use_best is None:
        use_best = getattr(args, "use_best_ckp", True)

    if dataset_name is None or description is None:
        raise ValueError(
            "populate_args_from_train_args needs dataset_name and description"
        )

    path = train_args_path(dataset_name, description, use_best=use_best)
    if not os.path.exists(path):
        legacy = os.path.join(ckpt_dir_for_dataset(dataset_name), "train_args.json")
        if os.path.exists(legacy):
            path = legacy
            print(
                f"[train_args] warning: using legacy {path}; "
                f"prefer {train_args_path(dataset_name, description, use_best=use_best)}"
            )
        else:
            print(f"[train_args] {path} not found; using CLI defaults only.")
            return []

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    overridden: list[str] = []
    for k in keys:
        if k not in saved:
            continue
        if k in _BOOL_STORE_TRUE_KEYS:
            if not getattr(args, k, False):
                setattr(args, k, bool(saved[k]))
                overridden.append(k)
            continue
        if getattr(args, k, None) is None:
            setattr(args, k, saved[k])
            overridden.append(k)

    if overridden:
        print(f"[train_args] loaded {len(overridden)} args from {path}: {overridden}")
    return overridden
