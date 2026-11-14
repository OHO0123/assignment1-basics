"""模型 checkpoint 的保存与恢复。

原先这两个函数写在 tests/adapters.py 里，导致 train.py 反向依赖测试脚手架。
移到包内部后，提交时只打包 cs336_basics/ 也能正常训练。
"""

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """把模型、优化器状态和迭代步数序列化到 out（路径或文件对象）。"""
    if isinstance(out, (str, os.PathLike)):
        parent = os.path.dirname(os.path.abspath(out))
        os.makedirs(parent, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
) -> int:
    """从 src 恢复模型与优化器，返回 checkpoint 中记录的迭代步数。"""
    checkpoint = torch.load(src, map_location=map_location)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # 跨设备恢复（如 CPU 存档 -> GPU 续训）时，把优化器状态搬到参数所在设备
    try:
        device = next(model.parameters()).device
    except StopIteration:  # 无参数模型，直接跳过
        return checkpoint["iteration"]

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.device != device:
                state[key] = value.to(device)

    return checkpoint["iteration"]
