import torch
import math
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 把默认超参数打包成字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 2. 交给父类去建档
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data # 这就是伪代码里的 g

                # 获取或者初始化状态柜子
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data) # 一阶动量
                    state['v'] = torch.zeros_like(p.data) # 二阶动量

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # 伪代码第 4 行：更新 m = β1*m + (1-β1)*g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 伪代码第 5 行：更新 v = β2*v + (1-β2)*g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 伪代码第 6 行：计算偏差修正后的学习率
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 伪代码第 7 行：更新参数
                p.data -= alpha_t * m / (v.sqrt() + eps)
                
                # 伪代码第 8 行：AdamW 的核心，权重衰减（Decoupled Weight Decay）
                p.data -= lr * weight_decay * p.data

        return loss


# ==========================================
# 顺手把学习率调度器也写了（对应那个 1 分的测试）
# ==========================================
def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        # 预热阶段：线性增加
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # 余弦退火阶段
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        # 衰减结束，保持最低学习率
        return min_learning_rate