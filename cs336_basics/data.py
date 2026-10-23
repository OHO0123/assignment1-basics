import numpy as np
import torch

def get_batch(
    dataset: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. 随机生成 batch_size 个起始索引
    # 为什么是 len(dataset) - context_length？
    # 因为我们要保证从起点往后切 context_length 个词时，不会越界。
    # 并且因为 Target 要往后错一位，所以更不会越界。
    max_start_idx = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=(batch_size,))
    
    # 2. 初始化输入和目标张量
    # 创建形状为 (batch_size, context_length) 的 numpy 数组
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)
    
    # 3. 按照起始索引，切片抽取数据
    for i, start_idx in enumerate(start_indices):
        # 输入: 从 start_idx 开始，取 context_length 个词
        x[i] = dataset[start_idx : start_idx + context_length]
        # 目标: 从 start_idx + 1 开始，取 context_length 个词
        y[i] = dataset[start_idx + 1 : start_idx + 1 + context_length]
        
    # 4. 转换为 PyTorch Tensor 并放到指定的设备上 (如 'cuda' 或 'cpu')
    return (
        torch.tensor(x, device=device),
        torch.tensor(y, device=device)
    )