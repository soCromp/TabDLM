import os
import math
import random
import json
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb

from utils.word_dict import SPECIALS
from utils.dataset import TabularDataset, PadCollator
from utils.util import load_config
from utils.train_args import dump_train_args
from tabdlm.model import TabDLM
from tabdlm.hFloatEmb import getfloatenc, getfloatdec

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="adult")
    parser.add_argument("--description", type=str, default="testing")

    # ----- model init (persisted under ckpt/<dataset>/<description>_{best,last}/) -----
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--answer_len", type=int, default=64)
    parser.add_argument("--mask_eps", type=float, default=1e-3)
    parser.add_argument("--ae_hidden_dim", type=int, default=512)
    parser.add_argument("--dlm_hidden_dim", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_parameters", nargs="+", type=str,
                        default=["ff_proj", "up_proj", "ff_out", "q_proj", "k_proj", "v_proj", "attn_out"])
    parser.add_argument("--normalization", type=str, default="quantile")
    parser.add_argument("--loss_type", type=str, default="no_divide_pmask",
                        choices=["no_divide_pmask", "dream_loss"])
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="use bfloat16 for LLM and disable GradScaler")
    parser.add_argument("--template_probs", type=str,
                        default=json.dumps({"A": 0, "B": 1, "C": 0}))
    parser.add_argument("--all_numerical", action="store_true", default=False)

    # ----- training loop -----
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batch_accum", type=int, default=16,
                        help="effective batch size in samples; grad_accum is derived from this")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="gradient accumulation steps (overridden by batch_accum // batch_size)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="cosine LR warmup as a fraction of total steps")
    parser.add_argument("--warm", type=int, default=2000,
                        help="linear warmup steps for num_loss weight lambda")
    parser.add_argument("--random_length_ratio", type=float, default=0.01)

    # ----- optimizer -----
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # ----- data loading / reproducibility -----
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num_workers", type=int, default=8)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    args.save_dir = os.path.join("ckpt", args.dataset_name)
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(
        project="tabdlm_training",
        group=args.dataset_name,
        name=f"{args.dataset_name}_{args.description}",
        config=vars(args),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_config = load_config(PROJECT_ROOT / "utils" / "configs.toml")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    tpl_probs = json.loads(args.template_probs)
    train_ds = TabularDataset(dataset_name=args.dataset_name,
                              tokenizer=tokenizer,
                              template_probs=tpl_probs,
                              max_len=args.max_len,
                              answer_len=args.answer_len,
                              normalization=args.normalization,
                              split="train",
                              all_numerical=args.all_numerical)

    valid_ds = TabularDataset(dataset_name=args.dataset_name,
                              tokenizer=tokenizer,
                              template_probs=tpl_probs,
                              max_len=args.max_len,
                              answer_len=args.answer_len,
                              normalization=args.normalization,
                              split="valid",
                              all_numerical=args.all_numerical)

    if not args.all_numerical:
        print(f"[stats] computing answer length statistics on train set (answer_len={args.answer_len}) ...")
        ans_len_stats = train_ds.compute_answer_len_stats()
        for tpl, s in ans_len_stats.items():
            coverage = (s["max"] <= args.answer_len)
            print(f"[stats] template={tpl} | n={s['count']} prompt_len={s['prompt_len']} | "
                  f"answer_len mean={s['mean']:.2f} min={s['min']} max={s['max']} median={s['median']:.2f} | "
                  f"answer_len={args.answer_len} -> {'OK (covers all)' if coverage else 'WARNING: answer_len < max'}")

    collate = PadCollator(tokenizer, pad_to_multiple_of=0 if args.all_numerical else 8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)

    if args.grad_accum != args.batch_accum // args.batch_size:
        args.grad_accum = args.batch_accum // args.batch_size
        print(f'rest grad_accum to {args.grad_accum}')
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    print(f"Total steps: {total_steps}")

    mask_token_id, num_token_id = 126336, 126090
    assert mask_token_id == tokenizer.encode(SPECIALS['MASK'])[0]
    assert num_token_id == tokenizer.encode(SPECIALS['NUMBER'])[0]

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
        **raw_config["diffusion_params"],
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optim_params = [
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=(args.adam_beta1,args.adam_beta2), eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*args.warmup_ratio), total_steps)

    model.to(device)
    if args.bf16:
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    step = 0
    best_val = 1e9
    log_every = 10
    lam_max = 1.0
    warm = args.warm
    for ep in range(args.epochs):
        model.train()
        run_txt_loss, run_num_loss = 0.0, 0.0
        run_loss, masked_tok, window_step, start_time  = 0.0, 0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            with autocast_ctx:
                txt_loss, num_loss, nmask = model(batch)
                lam = lam_max * min(1.0, step / warm)
                loss = txt_loss + lam * num_loss
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += (txt_loss.item() + num_loss.item()) / args.grad_accum
            masked_tok += nmask / args.grad_accum
            run_txt_loss += txt_loss.item() / args.grad_accum
            run_num_loss += num_loss.item() / args.grad_accum

            if (it + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm = 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                window_step +=1
                if step % log_every == 0:
                    avg = run_loss / window_step; mtok = masked_tok / window_step
                    avg_txt_loss = run_txt_loss / window_step; avg_num_loss = run_num_loss / window_step
                    print(f"[train] ep={ep} step={step}/{total_steps} num_loss={avg_num_loss:.4f} txt_loss={avg_txt_loss:.4f} tot_loss={avg:.4f} masked_tok/batch={mtok:.1f} run_time={time.time() - start_time:.2f}s")
                    wandb.log({
                        "train/tot_loss": avg,
                        "train/num_loss": avg_num_loss,
                        "train/txt_loss": avg_txt_loss,
                        "train/masked_tok_per_batch": mtok,
                        "lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item()}, step=step)
                    run_loss = 0.0; masked_tok = 0; start_time=time.time(); window_step=0; run_txt_loss = 0.0; run_num_loss = 0.0

        model.eval()
        val_loss = 0.0
        val_txt_loss = 0.0
        val_num_loss = 0.0
        n_batch = 0
        with torch.no_grad():
            for batch in valid_loader:
                txt_loss, num_loss, _ = model(batch)
                val_loss += (txt_loss.item() + num_loss.item())
                val_txt_loss += txt_loss.item()
                val_num_loss += num_loss.item()
                n_batch += 1
        val_loss /= max(1,n_batch)
        val_txt_loss /= max(1,n_batch)
        val_num_loss /= max(1,n_batch)
        wandb.log({"valid/tot_loss": val_loss, "valid/num_loss": val_num_loss, "valid/txt_loss": val_txt_loss,}, step=step)
        print(f"[valid] ep={ep} val_num_loss={val_num_loss:.4f} val_txt_loss={val_txt_loss:.4f} val_tot_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_tag = f"{args.description}_best"
            model.save_model(args.save_dir, best_tag)
            dump_train_args(args, save_dir=os.path.join(args.save_dir, best_tag))
            wandb.log({"best/val_tot_loss": best_val}, step=step)
            print(f"[save] best -> {args.save_dir}/{best_tag}")

    last_tag = f"{args.description}_last"
    model.save_model(args.save_dir, last_tag)
    dump_train_args(args, save_dir=os.path.join(args.save_dir, last_tag))
    print(f"[done] last saved to {args.save_dir}/{last_tag}")

    wandb.finish()