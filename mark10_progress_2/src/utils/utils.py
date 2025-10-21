# src/utils/utils.py

import os
import yaml
import shutil
from datetime import datetime
import torch
from torchvision.utils import save_image


def setup_output_dir(config):
    """
    根据当前时间创建输出目录，并备份配置文件。
    """
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(config['data']['output_dir'], time_str)

    # 创建子目录
    os.makedirs(os.path.join(output_dir, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # 备份配置文件
    shutil.copy('config.yaml', os.path.join(output_dir, 'config.yaml'))

    print(f"输出将保存在: {output_dir}")
    return output_dir


def save_plot_images(input_tensor, gt_tensor, output_tensor, path, epoch):
    """
    保存输入、GT和模型输出的对比图。
    """
    # 将Tensor从[-1, 1]转换回[0, 1]
    input_tensor = (input_tensor + 1) * 0.5
    gt_tensor = (gt_tensor + 1) * 0.5
    # output_tensor 已经在 sample 时转换

    # 将三者拼接在一起进行对比
    comparison = torch.cat([input_tensor, gt_tensor, output_tensor], dim=3)

    save_path = os.path.join(path, f"epoch_{epoch:04d}_comparison.png")
    save_image(comparison, save_path, nrow=1)