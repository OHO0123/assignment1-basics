import os
import time
import argparse
import numpy as np
import torch
import csv  # <--- 新增：导入 csv 库
from torch.nn import functional as F

# 导入你写好的模块
from cs336_basics.modules import TransformerLM, cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from tests.adapters import run_save_checkpoint as save_checkpoint
from tests.adapters import run_load_checkpoint as load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--train_data", type=str, required=True, help="Path to train data (e.g., .npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val data")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    
    # <--- 新增：实验日志的保存路径 --->
    parser.add_argument("--log_file", type=str, default="experiment_log.csv", help="Path to save the CSV log")
    
    # 模型超参
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 内存映射加载数据
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r')
    
    # 2. 初始化模型和优化器
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0
    ).to(args.device)
    
    optimizer = AdamW(model.parameters(), lr=args.max_lr)
    
    # 3. 尝试恢复 Checkpoint
    start_step = 0
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from step {start_step}")

    # =========================================================
    # <--- 新增：初始化 CSV 日志文件 --->
    # a 模式表示追加。这样即使断点续训，也不会覆盖之前的数据
    log_exists = os.path.exists(args.log_file)
    log_f = open(args.log_file, "a", newline="")
    writer = csv.writer(log_f)
    if not log_exists:
        writer.writerow(["Step", "Train_Loss", "Val_Loss", "LR", "Time_ms"])
    # =========================================================

    # 4. 训练循环
    model.train()
    for step in range(start_step, args.total_steps):
        t0 = time.time()
        
        # --- (1) 动态调整学习率 ---
        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_steps, args.total_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # --- (2) 取数据 ---
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        # --- (3) 前向传播 + 算 Loss ---
        logits = model(x)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        # --- (4) 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        
        # --- (5) 梯度裁剪 ---
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        # --- (6) 更新参数 ---
        optimizer.step()
        
        dt = time.time() - t0
        
        # --- (7) 日志打印与保存 ---
        if step % 10 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.1f}ms")
            # <--- 新增：每 10 步写入一行 CSV --->
            writer.writerow([step, loss.item(), "", lr, dt * 1000])
            log_f.flush() # 保证立刻写入硬盘
            
        # --- (8) 验证集评估 ---
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(10):
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    vlogits = model(vx)
                    val_loss += cross_entropy(vlogits.view(-1, args.vocab_size), vy.view(-1)).item()
            val_loss /= 10
            print(f"--- Validation Loss at Step {step}: {val_loss:.4f} ---")
            # <--- 新增：写入验证集 Loss --->
            writer.writerow([step, "", val_loss, lr, ""])
            log_f.flush()
            model.train()
            
        # --- (9) 保存 Checkpoint ---
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Checkpoint saved at Step {step}.")

    # <--- 新增：训练结束后关闭文件 --->
    log_f.close()

if __name__ == "__main__":
    main()