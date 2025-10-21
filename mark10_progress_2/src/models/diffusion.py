# src/models/diffusion.py

import torch
import torch.nn.functional as F


def get_beta_schedule(schedule_type, beta_start, beta_end, timesteps):
    """
    根据指定的类型生成beta调度表。
    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif schedule_type == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"不支持的调度类型: {schedule_type}")


def extract(a, t, x_shape):
    """
    从一个张量a中根据索引t提取值，并重塑以匹配x_shape。
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionModel:
    """
    封装了扩散模型的正向和反向过程。
    """

    def __init__(self, config):
        self.timesteps = config['diffusion']['timesteps']
        self.betas = get_beta_schedule(
            config['diffusion']['schedule_type'],
            config['diffusion']['beta_start'],
            config['diffusion']['beta_end'],
            self.timesteps
        )

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        正向过程：从x_0（干净图像）计算任意时间步t的x_t（加噪图像）。
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition):
        """
        反向采样一步：从x_t预测x_{t-1}。
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x_t.shape)

        # 使用模型预测噪声
        predicted_noise = model(x_t, t, condition)

        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels, condition):
        """
        完整的反向采样过程，从纯噪声生成去噪图像。
        """
        device = next(model.parameters()).device
        shape = (batch_size, channels, image_size, image_size)

        # 从纯高斯噪声开始
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, condition)
            imgs.append(img.cpu())

        # 将像素值从[-1, 1]转换回[0, 1]
        imgs[-1] = (imgs[-1] + 1) * 0.5
        return imgs[-1]