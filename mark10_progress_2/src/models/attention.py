# src/models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    自注意力模块。
    对输入的特征图应用自注意力机制，以捕捉全局依赖关系。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力权重
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k) * (c ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        # 应用注意力权重
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b, c, hw
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        return x + self.proj_out(h_)