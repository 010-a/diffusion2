# train.py
import csv
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
    # 1. 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 创建输出目录
    output_dir = setup_output_dir(config)
    # --- 创建并初始化日志文件 ---
    log_file_path = os.path.join(output_dir, 'logs', 'training_log.csv')
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'avg_loss'])  # 写入表头
    print(f"训练日志将保存到: {log_file_path}")

    # 3. 设置设备
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 准备数据集
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
        batch_size=config['train']['batch_size'],  # 验证时可以适当增大
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    print("数据加载完成。")

    # 5. 初始化模型和扩散过程
    model = UNet(config).to(device)
    diffusion = DiffusionModel(config)
    optimizer = Adam(model.parameters(), lr=config['train']['lr'])
    # [新增] 创建一个列表来存储当前epoch的所有损失值
    epoch_losses = []
    start_epoch = 1
    # 如果设置了，则加载预训练模型
    if config['train']['resume_checkpoint']:
        print(f"从 checkpoint 加载模型: {config['train']['resume_checkpoint']}")
        checkpoint = torch.load(config['train']['resume_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"模型加载完毕，将从 Epoch {start_epoch} 开始训练。")

    # 6. 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, config['train']['epochs'] + 1):
        # 训练
        #train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, config)
        # [修改] 接收 train_one_epoch 返回的平均损失
        avg_loss = train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, config)

        # [新增] 将损失写入文件
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])
        # 评估和可视化
        if epoch % config['log']['plot_freq'] == 0:
            evaluate(model, diffusion, val_loader, device, output_dir, epoch, config)

        # 保存模型
        if epoch % config['log']['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoint', f'model_epoch_{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"模型已保存至: {checkpoint_path}")

    print("训练完成！")


if __name__ == '__main__':
    main()