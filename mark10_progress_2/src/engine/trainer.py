# src/engine/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch, config):
    """
    训练一个epoch的逻辑。
    """
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['train']['epochs']}")
    total_loss = 0.0

    for i, batch in enumerate(pbar):
        # 准备数据
        gt_images = batch['gt'].to(device)
        input_images = batch['input'].to(device)  # 条件
        batch_size = gt_images.shape[0]

        # 1. 随机采样时间步 t
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

        # 2. 生成高斯噪声
        noise = torch.randn_like(gt_images)

        # 3. 使用正向过程加噪
        noisy_gt = diffusion.q_sample(x_start=gt_images, t=t, noise=noise)

        # 4. 模型预测噪声
        predicted_noise = model(noisy_gt, t, input_images)

        # 5. 计算损失 (通常使用L1或L2损失)
        loss = F.mse_loss(noise, predicted_noise)

        # 6. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % config['log']['log_freq'] == 0:
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    return avg_loss