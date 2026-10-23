import os
import time
import argparse
import numpy as np
import torch
from torch.nn import functional as F

# 导入你写好的模块
from cs336_basics.modules import TransformerLM, cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--train_data", type=str, required=True, help="Path to train data (e.g., .npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val data")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
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
    
    # 1. 内存映射加载数据 (Memory-efficient)
    # mmap_mode='r' 意味着只读，用到哪加载哪，不占满内存
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
    
    # PyTorch 的 AdamW 只是做接口，底层调用你写的
    # 注意：如果你自己实现了完整的 AdamW 类，直接用你的
    optimizer = AdamW(model.parameters(), lr=args.max_lr)
    
    # 3. 尝试恢复 Checkpoint
    start_step = 0
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from step {start_step}")

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
        # 把 (B, T, V) 展平成 (B*T, V)，标签展平为 (B*T,)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        # --- (4) 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        
        # --- (5) 梯度裁剪（防爆炸）---
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        # --- (6) 更新参数 ---
        optimizer.step()
        
        dt = time.time() - t0
        
        # --- (7) 日志打印 ---
        if step % 10 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.1f}ms")
            
        # --- (8) 验证集评估 ---
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(10): # 跑 10 个 batch 取平均
                    vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    vlogits = model(vx)
                    val_loss += cross_entropy(vlogits.view(-1, args.vocab_size), vy.view(-1)).item()
            val_loss /= 10
            print(f"--- Validation Loss at Step {step}: {val_loss:.4f} ---")
            model.train()
            
        # --- (9) 保存 Checkpoint ---
        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Checkpoint saved at Step {step}.")

if __name__ == "__main__":
    main()