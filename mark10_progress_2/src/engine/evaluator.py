# src/engine/evaluator.py

import torch
from src.utils.utils import save_plot_images
import os


def evaluate(model, diffusion, dataloader, device, output_dir, epoch, config):
    """
    在验证集上进行评估，并保存去噪结果图。
    """
    model.eval()

    # 只取一个batch进行可视化
    batch = next(iter(dataloader))
    input_images = batch['input'].to(device)
    gt_images = batch['gt'].to(device)

    print("开始生成验证图像...")
    with torch.no_grad():
        # 使用扩散模型进行采样去噪
        denoised_images = diffusion.sample(
            model=model,
            image_size=config['model']['image_size'],
            batch_size=input_images.shape[0],
            channels=config['model']['gt_channels'],
            condition=input_images
        )

    # 保存对比图像
    save_plot_images(
        input_tensor=input_images.cpu(),
        gt_tensor=gt_images.cpu(),
        output_tensor=denoised_images,  # 已经是cpu tensor
        path=os.path.join(output_dir, 'plots'),
        epoch=epoch
    )
    print(f"Epoch {epoch} 的验证图像已保存在 {os.path.join(output_dir, 'plots')}")