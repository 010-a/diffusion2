# test.py

import yaml
import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel


def denoise_image(model, diffusion, input_image_tensor, device, config):
    """
    对单张输入图像进行去噪。
    """
    model.eval()

    # 图像tensor需要是 [batch_size, channels, height, width]
    # 这里 batch_size = 1
    condition = input_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        denoised_image = diffusion.sample(
            model=model,
            image_size=config['model']['image_size'],
            batch_size=1,
            channels=config['model']['gt_channels'],
            condition=condition
        ).squeeze(0)  # 移除batch维度

    return denoised_image


def main():
    # --- 需要手动指定的参数 ---
    # 指定要测试的模型checkpoint路径
    checkpoint_path = "output/2025-10-22_17-25-28/checkpoint/model_epoch_0200.pth"
    # 指定包含测试图像的文件夹
    test_input_dir = "C:/Users/Guo_lab/Desktop/lxy/data-64-128/predict/lr"  # 或者其他测试文件夹
    # -------------------------

    # 1. 从训练时的文件夹加载配置文件
    # checkpoint_path 示例: 'output/2025-09-21_15-00-00/checkpoint/model_epoch_0200.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到指定的模型文件: {checkpoint_path}")

    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"在模型目录下找不到配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 创建本次测试的输出目录
    test_output_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"测试结果将保存在: {test_output_dir}")

    # 3. 设置设备
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 初始化模型并加载权重
    model = UNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {checkpoint_path}")

    # 5. 初始化扩散过程
    diffusion = DiffusionModel(config)

    # 6. 准备图像变换
    transform = transforms.Compose([
        transforms.Resize((config['model']['image_size'], config['model']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]
    ])

    to_pil = transforms.ToPILImage()

    # 7. 遍历测试文件夹中的所有tif图像并进行去噪
    test_files = glob.glob(os.path.join(test_input_dir, '*.tif'))

    for image_path in tqdm(test_files, desc="正在处理测试图像"):
        # 加载和预处理图像
        input_image = Image.open(image_path).convert("L")
        input_tensor = transform(input_image)

        # 执行去噪
        denoised_tensor = denoise_image(model, diffusion, input_tensor, device, config)

        # 后处理并保存
        denoised_image = to_pil(denoised_tensor.cpu())

        base_name = os.path.basename(image_path)
        save_path = os.path.join(test_output_dir, f"denoised_{base_name}")
        denoised_image.save(save_path)

    print("所有测试图像处理完毕！")


if __name__ == '__main__':
    main()
