"""TinyStories / OpenWebText 训练脚本，支持架构消融（variant）与可选云盘同步。

每次启动都会新建独立目录 runs/<variant>_<时间戳>/，互不覆盖。
续训必须显式指定 --resume_from，不指定就一定是全新训练。

用法示例:
    # 新跑一个基线
    python -m cs336_basics.train --variant base --train_data train.npy --val_data val.npy

    # 从某次训练续训（显式指定目录或 checkpoint 文件）
    python -m cs336_basics.train --variant base --train_data train.npy --val_data val.npy \
        --resume_from runs/base_20261114_153012

    # 四个消融，只换 --variant
    python -m cs336_basics.train --variant postnorm --total_steps 10000 ...
"""

import argparse
import contextlib
import csv
import datetime
import glob
import json
import os
import time

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.cloud_sync import upload_async
from cs336_basics.data import get_batch
from cs336_basics.modules import TransformerLM, cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule

# variant -> 架构开关；d_ff 留空则按变体自动推导
VARIANTS = {
    "base":      {"norm_style": "pre",  "use_rope": True,  "ffn_type": "swiglu"},
    "postnorm":  {"norm_style": "post", "use_rope": True,  "ffn_type": "swiglu"},
    "nope":      {"norm_style": "pre",  "use_rope": False, "ffn_type": "swiglu"},
    "silu":      {"norm_style": "pre",  "use_rope": True,  "ffn_type": "silu"},
    "norm_less": {"norm_style": "none", "use_rope": True,  "ffn_type": "swiglu"},
}

# 续训时必须一致的架构字段（头数改变权重形状不变，不校验会被静默接受）
ARCH_KEYS = ("variant", "vocab_size", "context_length", "d_model",
             "num_layers", "num_heads", "d_ff")


def parse_args():
    p = argparse.ArgumentParser()

    # 数据与输出
    p.add_argument("--train_data", type=str, required=True, help="训练集 token 的 .npy 路径")
    p.add_argument("--val_data", type=str, required=True, help="验证集 token 的 .npy 路径")
    p.add_argument("--run_root", type=str, default="runs", help="所有 run 的输出根目录")
    p.add_argument("--run_name", type=str, default=None,
                   help="自定义 run 目录名，默认 <variant>_<时间戳>")
    p.add_argument("--variant", type=str, default="base", choices=sorted(VARIANTS),
                   help="架构变体；base 之外都对应 PDF 7.3 的消融实验")
    p.add_argument("--resume_from", type=str, default=None,
                   help="显式续训：run 目录或 .pt 文件。不填则是全新训练")

    # 模型结构（默认值 = 文档 7.2.1 的 base model）
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=None,
                   help="不指定时: swiglu 用 1344, silu 用 4*d_model")

    # 训练超参
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)

    # 运行环境
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="混合精度（仅 CUDA 生效）")

    # 云盘同步（可选，缺 token 时自动跳过）
    p.add_argument("--gdrive", action="store_true", help="开启 checkpoint 自动上传")
    p.add_argument("--gdrive_token", type=str, default="/kaggle/working/token.json")
    p.add_argument("--gdrive_folder", type=str, default=None)
    p.add_argument("--gdrive_mode", choices=["overwrite", "snapshot"], default="overwrite",
                   help="overwrite: 云端固定文件名覆盖，只占一份; "
                        "snapshot: 每次带步数存档，保留历史")
    p.add_argument("--gdrive_every", type=int, default=None,
                   help="每隔多少步上传一次，默认与 save_interval 相同")

    args = p.parse_args()

    if args.d_ff is None:
        args.d_ff = 4 * args.d_model if VARIANTS[args.variant]["ffn_type"] == "silu" else 1344

    # ---- 决定 run 目录：全新训练 vs 续训 ----
    if args.resume_from:
        src = os.path.abspath(args.resume_from)
        run_dir = src if os.path.isdir(src) else os.path.dirname(src)
        if not os.path.isdir(run_dir):
            p.error(f"--resume_from 指向的目录不存在: {run_dir}")
        args.run_dir = run_dir
        args.resume_ckpt = src if os.path.isfile(src) else _pick_latest_ckpt(run_dir)
        print(f"[run] 续训模式: {run_dir}")
    else:
        name = args.run_name or f"{args.variant}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        args.run_dir = os.path.join(args.run_root, name)
        os.makedirs(args.run_dir, exist_ok=False)
        args.resume_ckpt = None
        print(f"[run] 全新训练: {args.run_dir}")

    args.run_name = os.path.basename(args.run_dir)
    args.ckpt_latest = os.path.join(args.run_dir, "checkpoint_latest.pt")
    args.ckpt_best = os.path.join(args.run_dir, "checkpoint_best.pt")
    args.ckpt_final = os.path.join(args.run_dir, "checkpoint_final.pt")
    args.log_file = os.path.join(args.run_dir, "experiment_log.csv")
    args.config_file = os.path.join(args.run_dir, "config.json")

    return args


def _pick_latest_ckpt(run_dir: str) -> str:
    """在 run 目录里挑一个 checkpoint：优先 latest，其次按步数最大的快照。"""
    latest = os.path.join(run_dir, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest

    candidates = glob.glob(os.path.join(run_dir, "checkpoint_step*.pt"))
    if not candidates:
        raise FileNotFoundError(f"{run_dir} 里没有可用的 checkpoint")

    def step_of(path):
        digits = os.path.basename(path).replace("checkpoint_step", "").replace(".pt", "")
        return int(digits) if digits.isdigit() else -1

    return max(candidates, key=step_of)


def check_arch_compatible(run_dir: str, args) -> None:
    """续训前校验架构一致，防止把头数不同的权重静默加载进来。"""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[warn] {config_path} 缺失，跳过架构一致性校验")
        return

    with open(config_path, encoding="utf-8") as f:
        old = json.load(f)

    diffs = [f"{k}: 存档={old[k]} vs 当前={getattr(args, k)}"
             for k in ARCH_KEYS if k in old and old[k] != getattr(args, k)]
    if diffs:
        raise ValueError(
            "续训配置与存档不一致，拒绝加载（权重形状相同但语义不同，会静默训坏）:\n  "
            + "\n  ".join(diffs)
        )


def build_model(args) -> TransformerLM:
    flags = VARIANTS[args.variant]
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
        norm_style=flags["norm_style"],
        use_rope=flags["use_rope"],
        ffn_type=flags["ffn_type"],
    ).to(args.device)


def autocast_ctx(use_amp: bool):
    """每次返回全新的 autocast 上下文，避免复用同一实例。"""
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def maybe_upload(args, local_path: str, cloud_name: str) -> None:
    if args.gdrive:
        upload_async(local_path, f"{args.run_name}/{cloud_name}",
                     args.gdrive_token, args.gdrive_folder)


def load_state(run_dir: str) -> dict:
    """读取 run 目录的附加状态（最优指标），用于续训时接续判断。"""
    path = os.path.join(run_dir, "state.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # 兼容旧 run：早期版本没写 state.json，尝试从 config.json 找
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                old = json.load(f)
            if "best_val" in old:
                return {"best_val": old["best_val"], "best_step": old.get("best_step", 0)}
        except (json.JSONDecodeError, OSError):
            pass

    return {}


def save_state(run_dir: str, best_val: float, best_step: int) -> None:
    """原子写入状态文件，避免中断时留下半截 JSON。"""
    path = os.path.join(run_dir, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"best_val": best_val, "best_step": best_step}, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, val_data, args, use_amp) -> float:
    model.eval()
    total = 0.0
    for _ in range(args.eval_batches):
        vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
        with autocast_ctx(use_amp):
            logits = model(vx)
            loss = cross_entropy(logits.view(-1, args.vocab_size), vy.view(-1))
        total += loss.item()
    model.train()
    return total / args.eval_batches


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = build_model(args)
    optimizer = AdamW(model.parameters(), lr=args.max_lr)

    # 续训前先校验架构
    start_step = 0
    if args.resume_ckpt:
        check_arch_compatible(args.run_dir, args)
        start_step = load_checkpoint(args.resume_ckpt, model, optimizer,
                                     map_location=args.device)
        print(f"[{args.variant}] 已恢复 {args.resume_ckpt}, step={start_step}")

    # 新训练才写 config；续训保留原文件
    if not args.resume_ckpt:
        config = vars(args).copy()
        config["variant_flags"] = VARIANTS[args.variant]
        config["tokens_processed"] = args.batch_size * args.context_length * args.total_steps
        config["num_params"] = sum(p.numel() for p in model.parameters())
        with open(args.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[{args.variant}] 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | "
          f"tokens {args.batch_size * args.context_length * args.total_steps / 1e6:.2f}M | "
          f"d_ff {args.d_ff} | heads {args.num_heads} | device {args.device}")
    print(f"[{args.variant}] 架构开关 {VARIANTS[args.variant]}")

    use_amp = bool(args.amp) and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    gdrive_every = args.gdrive_every or args.save_interval
    if args.gdrive:
        print(f"[cloud] 上传模式={args.gdrive_mode}, 每 {gdrive_every} 步一次 -> {args.gdrive_folder}")

    log_exists = os.path.exists(args.log_file)
    log_f = open(args.log_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if not log_exists:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "time_ms"])

    state = load_state(args.run_dir)
    best_val = state.get("best_val", float("inf"))
    best_step = state.get("best_step", 0)
    if best_val != float("inf"):
        print(f"[{args.variant}] 接续历史最优: val={best_val:.4f} @ step {best_step}")

    model.train()

    for step in range(start_step, args.total_steps):
        t0 = time.time()

        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_steps, args.total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        with autocast_ctx(use_amp):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        # 数值爆炸保护：无归一化等消融可能直接 NaN，留证并中止
        if not torch.isfinite(loss):
            print(f"[{args.variant}] step {step} loss={loss.item()}，训练发散，中止。")
            writer.writerow([step, "nan", "", lr, ""])
            log_f.flush()
            break

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)
            optimizer.step()

        dt = time.time() - t0

        if step % args.log_every == 0:
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | {dt * 1000:.1f}ms")
            writer.writerow([step, f"{loss.item():.6f}", "", f"{lr:.3e}", f"{dt * 1000:.1f}"])
            log_f.flush()

        if step == 100 and str(args.device).startswith("cuda"):
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"[{args.variant}] 显存峰值 {peak:.2f} GB")

        if step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data, args, use_amp)
            print(f"--- Val Loss @ {step}: {val_loss:.4f} ---")
            writer.writerow([step, "", f"{val_loss:.6f}", f"{lr:.3e}", ""])
            log_f.flush()

            if val_loss < best_val:
                best_val, best_step = val_loss, step
                save_checkpoint(model, optimizer, step, args.ckpt_best)
                save_state(args.run_dir, best_val, best_step)
                print(f"[{args.variant}] 最优 checkpoint 更新: val={val_loss:.4f} @ step {step}")
                maybe_upload(args, args.ckpt_best, "checkpoint_best.pt")

        if step % args.save_interval == 0 and step > 0:
            # 本地只保留 latest（滚动覆盖）
            save_checkpoint(model, optimizer, step, args.ckpt_latest)
            print(f"[{args.variant}] 已保存 {args.ckpt_latest} (step {step})")

            if step % gdrive_every == 0:
                cloud_name = (
                    "checkpoint_latest.pt" if args.gdrive_mode == "overwrite"
                    else f"checkpoint_step{step}.pt"
                )
                maybe_upload(args, args.ckpt_latest, cloud_name)
                # 日志很小（几十 KB），顺手传一份，session 结束也不会丢
                log_f.flush()
                maybe_upload(args, args.log_file, "experiment_log.csv")

    # 训练结束：无条件补一次最终验证与保存，避免丢最后一段权重
    final_val = evaluate(model, val_data, args, use_amp)
    print(f"=== Final Val Loss @ {args.total_steps}: {final_val:.4f} ===")
    writer.writerow([args.total_steps, "", f"{final_val:.6f}", "", ""])
    log_f.flush()

    if final_val < best_val:
        best_val, best_step = final_val, args.total_steps
        save_checkpoint(model, optimizer, args.total_steps, args.ckpt_best)
        print(f"[{args.variant}] 最优 checkpoint 更新: val={final_val:.4f} @ step {args.total_steps}")
        maybe_upload(args, args.ckpt_best, "checkpoint_best.pt")

    save_state(args.run_dir, best_val, best_step)

    save_checkpoint(model, optimizer, args.total_steps, args.ckpt_final)
    print(f"[final] Checkpoint saved: {args.ckpt_final}")
    log_f.close()

    maybe_upload(args, args.ckpt_final, "checkpoint_final.pt")
    maybe_upload(args, args.log_file, "experiment_log.csv")


if __name__ == "__main__":
    main()
