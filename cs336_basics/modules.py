import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 根据 Xavier 初始化公式计算标准差
        std = math.sqrt(2.0 / (in_features + out_features))
        
        # 2. 创建权重矩阵，形状为 (out_features, in_features)
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        
        # 3. 使用截断正态分布初始化，限制在 [-3*std, 3*std]
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
        # 4. 包装成可训练参数
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播：x 的形状是 (..., in_features)
        # 我们需要计算 x 乘以 W 的转置
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 创建嵌入矩阵，形状为 (vocab_size, d_model)
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        
        # 2. 初始化：官方要求 N(0,1) 截断在 [-3, 3]
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
        # 3. 包装成可训练参数
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 前向传播：直接根据 token_ids 去矩阵里查表
        # PyTorch 的高级索引会自动处理形状，返回 (..., embedding_dim) 的张量
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # TODO 1: 定义 self.weight (即公式里的 gamma)，用 nn.Parameter 包装，并且用 torch.ones 初始化为全 1。
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        
        # 1. 算均方值 (x**2).mean()
        mean_square = (x_float ** 2).mean(dim=-1, keepdim=True)
        
        # 2. 算分母 (sqrt(均方值 + eps))
        denominator = torch.sqrt(mean_square + self.eps)
        
        # 3. x / 分母，再乘以权重 gamma
        result = (x_float / denominator) * self.weight.to(torch.float32)
        
        # 4. 最后统一转回原来的 dtype (比如 fp16)
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # 定义三个线性层 (注意：不包含 bias)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 算 W1 * x，然后过 SiLU 激活函数
        # SiLU(x) = x * sigmoid(x)，官方提示直接用 torch.sigmoid 稳定
        w1_out = self.w1(x)
        silu_out = w1_out * torch.sigmoid(w1_out)
        
        # 2. 算 W3 * x (门控通道)
        w3_out = self.w3(x)
        
        # 3. 两个通道逐元素相乘 (Hadamard 积)
        gated = silu_out * w3_out
        
        # 4. 最后过 W2 降维回 d_model
        return self.w2(gated)


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # 1. 预计算频率 (频度公式: 1 / (theta ^ (2i / d_k)))
        # 注意：i 的范围是 0 到 d_k//2 - 1
        i = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2 * i / d_k))  # 形状: (d_k/2,)
        
        # 2. 预计算位置 (0 到 max_seq_len - 1)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # 形状: (max_seq_len,)
        
        # 3. 外积得到每个位置、每个频率的旋转角
        # positions[:, None] 变成 (max_seq_len, 1), inv_freq[None, :] 变成 (1, d_k/2)
        # 相乘后 angles 形状是 (max_seq_len, d_k/2)
        angles = positions[:, None] * inv_freq[None, :]
        
        # 4. 算出 cos 和 sin，并存入 buffer (不参与训练)
        # persistent=False 表示不需要存进 state_dict 里
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x 的形状: (..., seq_len, d_k)
        # token_positions 的形状: (..., seq_len)
        
        # 1. 根据 token 的位置，从缓存里把对应的 cos 和 sin 切片取出来
        # cos_cached 形状是 (max_seq_len, d_k/2)，切出来变成 (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 2. 将 x 按最后一维两两分组 (x1, x2, x3, x4... 变成 (x1, x2), (x3, x4)...)
        # 使用 view 或者 reshape：把最后一维拆成 (seq_len, d_k/2, 2)
        # 这样 x_pairs[..., 0] 就是偶数位(x1, x3)，x_pairs[..., 1] 就是奇数位(x2, x4)
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x1 = x_pairs[..., 0]  # 偶数位
        x2 = x_pairs[..., 1]  # 奇数位
        
        # 3. 应用旋转公式 (利用复数乘法的二维旋转特性)
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        
        # 4. 重新拼接并恢复维度
        # torch.stack 拼接成一个新的最后一维 2
        x_rotated = torch.stack([x1_new, x2_new], dim=-1)
        # 展平回原来的形状 (..., seq_len, d_k)
        return x_rotated.view(*x.shape)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. 找到这一维上的最大值 (keepdim=True 是为了方便后续广播)
    max_val = x.max(dim=dim, keepdim=True).values
    
    # 2. 减去最大值，防止 exp 溢出
    x_shifted = x - max_val
    
    # 3. 算指数
    exp_x = torch.exp(x_shifted)
    
    # 4. 求和（分母）
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    # 5. 归一化
    return exp_x / sum_exp


def scaled_dot_product_attention(Q, K, V, mask=None):
    # 1. 获取 d_k (Q 的最后一维)
    d_k = Q.shape[-1]
    
    # 2. 计算 Q 和 K 的转置乘积，并除以根号 d_k
    # K 的转置：K.transpose(-2, -1) 会把最后两维对调
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 3. 处理 Mask
    if mask is not None:
        # 官方要求：把 mask 为 False 的地方填成负无穷 (-inf)
        # ~mask 是取反，意思是把 False 变成 True 去执行填充
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # 4. 使用你刚才写好的 Softmax (注意在最后一维做归一化)
    # 切记：这里必须用你手写的稳定版 softmax
    attn_weights = softmax(scores, dim=-1)
    
    # 5. 乘以 V，得到最终输出
    out = attn_weights @ V
    
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义四个线性层（不带 bias）
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (batch, seq_len, d_model)
        B, T, C = x.shape
        
        # 1. 线性投影，生成 Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 拆头，把 d_model 拆成 (num_heads, d_k)
        # 先 view 拆分，再 transpose 把 head 维度提前
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, T, d_k)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 生成因果掩码 (Causal Mask)
        # 使用 torch.tril 生成下三角矩阵，形状为 (T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        
        # 4. 调用你的 SDPA（注意：mask 的形状是 (T, T)，会自动广播到所有 batch 和 head 上）
        out = scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, T, d_k)
        
        # 5. 拼头（Concat）
        # 先转置回来 (B, T, num_heads, d_k)，再 reshape 变成 (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. 最后的输出线性层
        return self.output_proj(out)



class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # 实例化你刚才写好的 RoPE 模块
        # 注意：传入的 d_k 是每个头的维度，因为两两分组是在每个头内部进行的
        self.rope = RoPE(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # 1. 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 拆头 (B, T, d_model) -> (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 应用 RoPE (只对 q 和 k！)
        # 因为你的 RoPE forward 接收的是 (..., seq_len, d_k)，现在形状是 (B, heads, T, d_k)，完美匹配
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # 4. 因果掩码
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        
        # 5. 计算 Attention
        out = scaled_dot_product_attention(q, k, v, mask)
        
        # 6. 拼头
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. 输出线性层
        return self.output_proj(out)



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        
        # 1. 两个前置 LayerNorm
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
        # 2. 多头注意力 (带 RoPE)
        # 注意：你刚才写好的 MultiHeadAttentionWithRoPE 需要传入这 4 个参数
        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        
        # 3. FFN (SwiGLU)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入形状：批次大小 B，序列长度 T，模型维度 C
        B, T, C = x.shape
        
        # 动态生成 token 位置编码 (0 到 T-1)
        # 无论序列多长，都能自动适配
        token_positions = torch.arange(T, device=x.device).unsqueeze(0) # 形状: (1, T)
        
        # ==========================================
        # 第一个子层：Pre-Norm Attention + 残差连接
        # ==========================================
        # 先 Norm 再 Attention，最后加上原始 x
        attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out
        
        # ==========================================
        # 第二个子层：Pre-Norm FFN + 残差连接
        # ==========================================
        # 先 Norm 再 FFN，最后加上更新后的 x
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x



class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        
        # 1. 词嵌入层 (你之前写好的 Embedding)
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # 2. 堆叠 N 层 Transformer Block (使用 ModuleList 完美兼容官方权重键名)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        
        # 3. 最终归一化
        self.ln_final = RMSNorm(d_model)
        
        # 4. 输出层 (LM Head)，把维度映射回词表大小
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices 形状: (batch_size, sequence_length)
        
        # 1. 查表得到词向量
        x = self.token_embeddings(in_indices)
        
        # 2. 逐层前向传播
        for layer in self.layers:
            x = layer(x)
            
        # 3. 最终归一化
        x = self.ln_final(x)
        
        # 4. 映射到词表维度，得到 logits
        logits = self.lm_head(x)
        
        return logits


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    inputs: (batch_size, vocab_size) 预测的 logits
    targets: (batch_size,) 真实标签的索引
    """
    # 1. 减去最大值 (防溢出)
    # 找到每一行的最大值，保持维度以便广播
    max_val = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_val
    
    # 2. 计算 log_sum_exp 
    # 这一步对应公式里的 log( sum(exp(x)) )
    # 为什么用 logsumexp？因为 PyTorch 内部专门优化了这个操作，极度稳定
    # 等价于 torch.log(torch.sum(torch.exp(shifted), dim=-1))
    log_sum_exp = torch.logsumexp(shifted, dim=-1) 
    
    # 3. 拿出真实标签对应的 logit (分子)
    # 使用 gather 函数，根据 target 索引去每一行拿对应的数值
    target_logits = shifted.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # 4. 计算最终 Loss = log_sum_exp - target_logit
    loss_per_example = log_sum_exp - target_logits
    
    # 5. 返回 batch 的平均 loss
    return loss_per_example.mean()


def gradient_clipping(parameters, max_l2_norm: float):
    # 1. 计算所有参数梯度的总平方和
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            # 把当前参数的梯度平方求和，加到总和中
            total_norm_sq += torch.sum(p.grad ** 2)
            
    # 2. 开根号得到 L2 范数
    total_norm = torch.sqrt(total_norm_sq)
    
    # 3. 如果总范数超过了设定的最大值 max_l2_norm
    if total_norm > max_l2_norm:
        # 计算缩放系数，加上 1e-6 防止除以 0
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        
        # 4. 对所有参数的梯度进行原地缩放
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)