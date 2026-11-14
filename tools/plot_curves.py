#!/usr/bin/env python3
"""从各 run 的 experiment_log.csv 生成对比曲线。

用法:
    # 自动扫描 runs/ 下所有含 experiment_log.csv 的子目录
    python tools/plot_curves.py --runs-root runs --out curves

    # 只比较指定的几个
    python tools/plot_curves.py runs/base_full runs/nope_20261115_0915 --out curves
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    """解析 experiment_log.csv，返回 {step/train/val/lr/time: [...]}。"""
    series = {"step": [], "train": [], "val": [], "lr": [], "time": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
            except (KeyError, TypeError, ValueError):
                continue
            series["step"].append(step)

            for key, col in (("train", "train_loss"), ("val", "val_loss"),
                             ("lr", "lr"), ("time", "time_ms")):
                raw = row.get(col, "")
                try:
                    series[key].append(float(raw))
                except (TypeError, ValueError):
                    series[key].append(float("nan"))

    # 去掉整列都是 nan 的尾部空行
    return series


def tokens_per_step(run_dir):
    """从 config.json 读 batch_size × context_length，用于 token 数横轴。"""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["batch_size"]) * int(cfg["context_length"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def smooth(values, window):
    """简单移动平均，忽略 nan。"""
    out = []
    buf = []
    for v in values:
        if v == v:  # 非 nan
            buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf) if buf else float("nan"))
    return out


def discover(runs_root):
    found = []
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name, "experiment_log.csv")
        if os.path.exists(path):
            found.append(os.path.join(runs_root, name))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run 目录列表；留空则扫描 --runs-root")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="curves")
    ap.add_argument("--xaxis", choices=["tokens", "steps"], default="tokens")
    ap.add_argument("--smooth", type=int, default=15, help="train loss 平滑窗口（点数）")
    args = ap.parse_args()

    run_dirs = args.runs or discover(args.runs_root)
    if not run_dirs:
        raise SystemExit(f"{args.runs_root} 下没有找到 experiment_log.csv")

    os.makedirs(args.out, exist_ok=True)

    fig_val, ax_val = plt.subplots(figsize=(9, 5.5))
    fig_train, ax_train = plt.subplots(figsize=(9, 5.5))
    fig_lr, ax_lr = plt.subplots(figsize=(9, 5))

    summary = []
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        series = read_log(os.path.join(run_dir, "experiment_log.csv"))
        if not series["step"]:
            print(f"[skip] {name}: 日志为空")
            continue

        tps = tokens_per_step(run_dir)
        if args.xaxis == "tokens" and tps:
            x = [s * tps / 1e6 for s in series["step"]]
            xlabel = "Tokens processed (M)"
        else:
            x = series["step"]
            xlabel = "Training step"

        # 验证曲线
        vx = [xi for xi, v in zip(x, series["val"]) if v == v]
        vy = [v for v in series["val"] if v == v]
        (line,) = ax_val.plot(vx, vy, marker="o", markersize=3, label=name)
        color = line.get_color()

        # 训练曲线（同色虚线 + 平滑）
        tx = [xi for xi, v in zip(x, series["train"]) if v == v]
        ty = smooth([v for v in series["train"] if v == v], args.smooth)
        ax_train.plot(tx, ty, linestyle="--", alpha=0.75, color=color, label=name)

        # 学习率曲线
        lx = [xi for xi, v in zip(x, series["lr"]) if v == v]
        ly = [v for v in series["lr"] if v == v]
        ax_lr.plot(lx, ly, color=color, label=name)

        if len(vy) >= 2:
            best = min(vy)
            summary.append((name, series["step"][-1], vy[-1], best, tps))

    ax_val.set(xlabel=xlabel, ylabel="validation loss", title="Validation loss")
    ax_train.set(xlabel=xlabel, ylabel="training loss (smoothed)",
                 title="Training loss")
    ax_lr.set(xlabel=xlabel, ylabel="learning rate", title="Learning rate schedule")
    for ax in (ax_val, ax_train, ax_lr):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for fig, fname in ((fig_val, "val_loss.png"), (fig_train, "train_loss.png"),
                       (fig_lr, "lr_schedule.png")):
        path = os.path.join(args.out, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"[saved] {path}")

    print("\nrun                                  最后step   最终val   最优val   最终tokens")
    for name, last_step, final_val, best_val, tps in summary:
        tok = f"{last_step * tps / 1e6:.1f}M" if tps else "-"
        print(f"{name:<35} {last_step:>8} {final_val:>9.4f} {best_val:>9.4f} {tok:>11}")


if __name__ == "__main__":
    main()
