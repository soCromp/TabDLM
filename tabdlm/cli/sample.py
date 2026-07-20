import os
import random
import json
import argparse
import time
from numbers import Number
from pathlib import Path

import torch
import pandas as pd
import wandb
from transformers import AutoTokenizer

from utils.word_dict import SPECIALS
from utils.dataset import TabularDataset
from utils.util import load_config
from utils.train_args import populate_args_from_train_args
from utils.sampling_postprocess import (
    build_filter_column_info,
    finalize_numerical_columns,
    filter_not_in_candidates,
    postprocess_loaded_samples,
)
from eval.metrics import TabMetrics
from tabdlm.model import TabDLM
from tabdlm.hFloatEmb import getfloatenc, getfloatdec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="adult")
    parser.add_argument("--description", type=str, default="testing")

    # ----- model init (defaults from ckpt/<dataset>/<description>_{best,last}/train_args.json) -----
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--answer_len", type=int, default=None)
    parser.add_argument("--mask_eps", type=float, default=None)
    parser.add_argument("--ae_hidden_dim", type=int, default=None)
    parser.add_argument("--dlm_hidden_dim", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_parameters", nargs="+", type=str, default=None)
    parser.add_argument("--normalization", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["no_divide_pmask", "dream_loss"])
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--template_probs", type=str, default=None)
    parser.add_argument("--all_numerical", action="store_true", default=False)

    # ----- sampling-only args -----
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--do_sampling", action="store_true", default=False)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--keep_nan_samples", action="store_true", default=False)
    parser.add_argument("--use_best_ckp", action="store_false", default=True)
    parser.add_argument("--sample_step", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Replaced proportion with n
    parser.add_argument("--n", type=int, default=1000,
                        help="exact number of samples to generate")
                        
    parser.add_argument("--remasking", type=str, default="random",
                        choices=["low_confidence", "random"])
    parser.add_argument("--save_description", type=str, default="")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--sample_prompt_template", type=str, default="B")
    parser.add_argument("--eval_metrics", nargs="*", type=str,
                        default=["density", "c2st", "mle"])

    parser.add_argument("--decoupled_steps", action="store_true", default=False)
    parser.add_argument("--text_steps", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)

    args = parser.parse_args()
    populate_args_from_train_args(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_raw_samples(args, train_ds):
    if args.all_numerical:
        raw_path = os.path.join(
            "result", args.dataset_name, "raw_sampling_result",
            f"{args.description}{args.save_description}.csv",
        )
        return pd.read_csv(raw_path)

    raw_path = os.path.join(
        "result", args.dataset_name, "raw_sampling_result",
        f"{args.description}{args.save_description}.txt",
    )
    return pd.read_csv(
        raw_path,
        sep="|",
        header=None,
        engine="python",
        names=train_ds.col_order,
        on_bad_lines="skip",
        skipinitialspace=True,
    )


def _to_float_if_numeric(value):
    if isinstance(value, Number):
        return float(value)
    return None


def _flatten_numeric_metrics(obj, prefix="", out=None):
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten_numeric_metrics(v, key, out)
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}/{i}" if prefix else str(i)
            _flatten_numeric_metrics(v, key, out)
        return out

    numeric_value = _to_float_if_numeric(obj)
    if numeric_value is not None and prefix:
        out[prefix] = numeric_value

    return out


def main():
    args = get_args()
    set_seed(args.seed)
    args.save_dir = os.path.join("ckpt", args.dataset_name)
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(
        project="tabdlm_evaluation",
        group=args.dataset_name,
        name=f"{args.dataset_name}_{args.description}{args.save_description}",
        config=vars(args),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_config = load_config(PROJECT_ROOT / "utils" / "configs.toml")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tpl_probs = json.loads(args.template_probs)
    train_ds = TabularDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        template_probs=tpl_probs,
        max_len=args.max_len,
        answer_len=args.answer_len,
        normalization=args.normalization,
        split="train",
        all_numerical=args.all_numerical,
    )

    if args.do_sampling:
        mask_token_id, num_token_id = 126336, 126090
        assert mask_token_id == tokenizer.encode(SPECIALS["MASK"])[0]
        assert num_token_id == tokenizer.encode(SPECIALS["NUMBER"])[0]

        floatenc = getfloatenc(hiddim=args.ae_hidden_dim, train=False)
        floatdec = getfloatdec(hiddim=args.ae_hidden_dim, train=False)

        model = TabDLM(
            args=args,
            model_name=args.model_name,
            num_numerical_features=train_ds.num_numerical_columns,
            mask_token_id=mask_token_id,
            num_token_id=num_token_id,
            eps=args.mask_eps,
            use_bf16=args.bf16,
            lora_parameters=args.lora_parameters,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            floatenc=floatenc,
            floatdec=floatdec,
            ae_hidden_dim=args.ae_hidden_dim,
            dlm_hidden_dim=args.dlm_hidden_dim,
            loss_type=args.loss_type,
            device=device,
            all_numerical=args.all_numerical,
            decoupled_steps=args.decoupled_steps,
            **raw_config["diffusion_params"],
        )
        if args.use_best_ckp:
            model.load_model(args.save_dir, f"{args.description}_best")
        else:
            model.load_model(args.save_dir, f"{args.description}_last")

        model.to(device)
        model.eval()

        if args.decoupled_steps:
            if args.num_steps is None or args.text_steps is None:
                raise ValueError("--decoupled_steps requires --text_steps and --num_steps.")
            model.num_timesteps = args.num_steps
        elif model.num_timesteps != args.sample_step:
            model.num_timesteps = args.sample_step

    else:
        model = None

    # Replaced proportional logic with direct n assignment
    num_samples_init = num_samples_targ = args.n
    num_samples_left = num_samples_init
    
    prompt, num_value_idx = train_ds.build_sample_prompt(template=args.sample_prompt_template)
    prompt_ids = torch.tensor(tokenizer(prompt)["input_ids"]).to(device)
    print(f"Start sampling/loading, total samples to generate = {num_samples_left}")
    start_time = time.time()

    while num_samples_left > 0:
        if args.do_sampling:
            with torch.no_grad():
                model.sample_synthetic_all(
                    num_samples=num_samples_left,
                    batch_size=args.sample_batch_size,
                    prompt_ids=prompt_ids,
                    num_value_idx=num_value_idx,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    steps=args.sample_step,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    tokenizer=tokenizer,
                    train_ds=train_ds,
                    description=args.description,
                    save_description=args.save_description,
                    top_k=args.top_k,
                    keep_nan_samples=args.keep_nan_samples,
                    text_steps=args.text_steps,
                    num_steps=args.num_steps,
                )

        df = _load_raw_samples(args, train_ds)
        # print(f"After filtering, synthetic counts from {num_samples_init} -> {len(df)}.")
        df.dropna(inplace=True)
        # print(f"After dropping nan, synthetic counts from {num_samples_init} -> {len(df)}.")

        df = postprocess_loaded_samples(
            df,
            train_ds,
            all_numerical=args.all_numerical,
            dataset_name=args.dataset_name,
        )
        # print(f"After doing strip, synthetic counts from {num_samples_init} -> {len(df)}.")

        info = build_filter_column_info(
            train_ds,
            all_numerical=args.all_numerical,
            dataset_name=args.dataset_name,
        )
        
        # force categorical columns to be strings
        for col, col_info in info.items():
            if col_info.get("type") == "category":
                df[col] = df[col].astype(str)
        
        # # DEBUG
        # print("\n--- COLUMN ALIGNMENT CHECK ---")
        # print(df.head(2)) 
        # print("\n--- EXPECTED CATEGORIES (INFO) ---")
        # print(info)
        # print("------------------------------\n")
        
        df = filter_not_in_candidates(df, args.dataset_name, info, num_samples_init)
        # print(f"Samples remaining after checking for invalid categories: {len(df)}")

        num_samples_left = num_samples_targ - len(df)
        if num_samples_left > 0:
            print(f"still have {num_samples_left} samples to generate.")
        else:
            df = df[:num_samples_targ]
            print(f"Use the first {num_samples_targ} samples for evaluation.")

    df = finalize_numerical_columns(df, args.dataset_name, train_ds.numerical_columns)

    syn_csv_path = os.path.join(
        "result", args.dataset_name, "synthetic_result",
        f"{args.description}{args.save_description}.csv",
    )
    dir_name = os.path.dirname(syn_csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(syn_csv_path):
        df.to_csv(syn_csv_path, index=False)
    else:
        print(f"The synthetic result already exists: {syn_csv_path}")
    end_time = time.time()
    print(f"End sampling/loading, total sampling time = {end_time - start_time:.2f}s")

    wandb.finish()

if __name__ == "__main__":
    main()
    