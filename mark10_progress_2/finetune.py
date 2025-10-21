# finetune.py

import yaml
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data_pre.dataset import MicroscopeDataset
from src.models.unet import UNet
from src.models.diffusion import DiffusionModel
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate
from src.utils.utils import setup_output_dir


def main():
    # 1. 加载微调的配置文件
    config_path = 'config_finetune.yaml'
    print(f"加载微调配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 创建新的输出目录用于存放微调结果
    output_dir = setup_output_dir(config)

    # 3. 设置设备
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 准备新的 B->C 数据集
    print("加载新的微调数据集 (Input: B, GT: C)")
    train_dataset = MicroscopeDataset(
        input_dir=config['data']['train_input_dir'],
        gt_dir=config['data']['train_gt_dir'],
        image_size=config['model']['image_size']
    )
    val_dataset = MicroscopeDataset(
        input_dir=config['data']['val_input_dir'],
        gt_dir=config['data']['val_gt_dir'],
        image_size=config['model']['image_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    print("数据集加载完成。")

    # 5. 初始化模型和优化器
    model = UNet(config).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['lr'])
    diffusion = DiffusionModel(config)

    # 6. [核心步骤] 加载预训练的 A->B 模型权重
    finetune_checkpoint_path = config['finetune']['checkpoint_path']
    if not os.path.exists(finetune_checkpoint_path):
        raise FileNotFoundError(f"找不到指定的预训练权重: {finetune_checkpoint_path}")

    print(f"从预训练权重加载模型: {finetune_checkpoint_path}")
    checkpoint = torch.load(finetune_checkpoint_path, map_location=device)

    # 只需要加载模型的权重，不需要加载优化器状态和epoch
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型权重加载成功！")

    # 7. 开始微调训练
    print("开始微调训练...")
    # 微调从 epoch 1 开始计数
    for epoch in range(1, config['train']['epochs'] + 1):
        # 训练逻辑复用原来的 trainer
        train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, config)

        # 评估和可视化
        if epoch % config['log']['plot_freq'] == 0:
            evaluate(model, diffusion, val_loader, device, output_dir, epoch, config)

        # 保存新的微调模型
        if epoch % config['log']['checkpoint_freq'] == 0 or epoch == config['train']['epochs']:
            checkpoint_path = os.path.join(output_dir, 'checkpoint', f'finetuned_model_epoch_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"微调模型已保存至: {checkpoint_path}")

    print("微调完成！")


if __name__ == '__main__':
    main()