# train.py (最终完整版 - 集成了日志记录与安全中断功能)

import csv
import yaml
import os
import sys  # <-- [新增] 导入sys库以便干净地退出
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# 原始导入保持不变
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

    # 2. 创建输出目录并初始化日志文件
    output_dir = setup_output_dir(config)
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
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    print("数据加载完成。")

    # 5. 初始化模型和扩散过程
    model = UNet(config).to(device)
    diffusion = DiffusionModel(config)
    optimizer = Adam(model.parameters(), lr=config['train']['lr'])

    start_epoch = 1
    # 如果设置了，则加载预训练模型
    if config['train']['resume_checkpoint']:
        print(f"从 checkpoint 加载模型: {config['train']['resume_checkpoint']}")
        checkpoint = torch.load(config['train']['resume_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"模型加载完毕，将从 Epoch {start_epoch} 开始训练。")

    # --- [核心修改] 使用 try...except 结构包装整个训练循环 ---
    try:
        # 6. 训练循环
        print("\n" + "=" * 50)
        print("开始训练... (在训练过程中随时按 Ctrl+C 可以安全中断并保存模型)")
        print("=" * 50)

        for epoch in range(start_epoch, config['train']['epochs'] + 1):
            # 训练并接收返回的平均损失
            avg_loss = train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, config)

            # 将损失写入日志文件
            with open(log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss])

            # 评估和可视化
            if epoch % config['log']['plot_freq'] == 0:
                evaluate(model, diffusion, val_loader, device, output_dir, epoch, config)

            # 保存模型 (增加了最后一轮的保存判断)
            if epoch % config['log']['checkpoint_freq'] == 0 or epoch == config['train']['epochs']:
                checkpoint_path = os.path.join(output_dir, 'checkpoint', f'model_epoch_{epoch:04d}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"模型已保存至: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 50)
        print("检测到手动中断 (Ctrl+C)...")

        if 'epoch' in locals() and epoch > start_epoch:
            # 保存的是上一个完整训练完的epoch的状态
            save_epoch = epoch - 1
            print(f"正在保存第 {save_epoch} 轮训练完成后的模型状态...")

            checkpoint_path = os.path.join(output_dir, 'checkpoint', f'interrupted_model_epoch_{save_epoch:04d}.pth')

            torch.save({
                'epoch': save_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            print(f"✔ 模型已成功保存至: {checkpoint_path}")
            print("您可以使用此文件恢复训练。")
        else:
            print("训练尚未完成任何一个完整的epoch，没有可保存的模型状态。")

        print("程序安全退出。")
        print("=" * 50)

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    # --- 修改结束 ---

    print("训练完成！")


if __name__ == '__main__':
    main()