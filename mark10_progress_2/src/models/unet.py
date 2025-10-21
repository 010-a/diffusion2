# src/models/unet.py

import torch
import torch.nn as nn
import math
from .attention import SelfAttention


class TimeEmbedding(nn.Module):
    """
    将时间步（t）编码为时间嵌入向量。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -self.emb)

    def forward(self, t):
        self.emb = self.emb.to(t.device)
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """
    带有时间嵌入的残差块。
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h += time_emb

        h = self.act2(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_connection(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    带有注意力的条件U-Net模型。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 参数
        img_size = config['model']['image_size']
        in_channels = config['model']['input_channels'] + config['model']['gt_channels']  # 条件输入
        base_channels = config['model']['base_channels']
        channel_mults = config['model']['channel_multipliers']
        time_emb_dim = config['model']['time_embedding_dim']
        attn_resolutions = config['model']['attention_resolutions']
        num_res_blocks = config['model']['num_res_blocks']
        dropout = config['model']['dropout']

        # 时间编码
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # 初始卷积层
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # U-Net的下采样部分
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(now_channels, out_channels, time_emb_dim, dropout))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(now_channels))
                channels.append(now_channels)

        # U-Net的中间部分
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)

        # U-Net的上采样部分
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(channels.pop() + now_channels, out_channels, time_emb_dim, dropout))
                now_channels = out_channels

            if i != 0:
                self.up_blocks.append(Upsample(now_channels))

        # 输出层
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, config['model']['gt_channels'], kernel_size=3, padding=1)

    def forward(self, noisy_gt, t, condition):
        # 将含噪的gt和作为条件的input图像在通道维度上拼接
        x = torch.cat((noisy_gt, condition), dim=1)

        # 时间编码
        time_emb = self.time_embedding(t)

        # 初始卷积
        x = self.init_conv(x)

        # 下采样
        skips = [x]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, time_emb)
            else:
                x = block(x)
            skips.append(x)

        # 中间部分
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # 上采样
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                x = block(x)
            else:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, time_emb)

        # 输出
        x = self.out_act(self.out_norm(x))
        return self.out_conv(x)