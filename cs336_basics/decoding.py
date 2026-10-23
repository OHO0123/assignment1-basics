import torch

def top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 核心修复：找到第一个超过阈值的位置
    first_exceed_idx = (cumulative_probs > top_p).int().argmax()
    
    # 构建掩码：该位置及之前的保留（False），之后的删除（True）
    sorted_indices_to_remove = torch.zeros_like(sorted_probs, dtype=torch.bool)
    sorted_indices_to_remove[first_exceed_idx + 1:] = True
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits = logits.clone()
    logits[indices_to_remove] = float('-inf')
    return logits


def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    endoftext_id: int = -1,
    device: str = "cpu"
) -> list[int]:
    """
    让模型生成文本的核心函数。
    """
    # 1. 把 prompt 转成 tensor 并放到设备上
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)
    
    # 记录生成的完整序列
    generated = prompt_tokens.copy()
    
    model.eval() # 切换到推理模式
    
    with torch.no_grad(): # 推理不需要计算梯度
        for _ in range(max_new_tokens):
            # 2. 前向传播
            # 注意：如果输入长度超过了模型的最大长度，我们需要截断，这里假设模型支持动态输入
            logits = model(input_ids) # (1, seq_len, vocab_size)
            
            # 3. 只取最后一个位置的预测结果，并移除 batch 维度
            next_token_logits = logits[0, -1, :] # (vocab_size,)
            
            # 4. 温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # 5. Top-p 筛选
            if top_p is not None:
                next_token_logits = top_p_sampling(next_token_logits, top_p)
                
            # 6. 转成概率并采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # 7. 拼接到结果中
            generated.append(next_token)
            
            # 把新词拼到输入序列里，准备预测下一个词
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 8. 如果遇到结束符，提前停止
            if endoftext_id != -1 and next_token == endoftext_id:
                break
                
    return generated