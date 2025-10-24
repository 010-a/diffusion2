# test.py (已修正数据预处理和16-bit保存逻辑)

import yaml
import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
import tifffile  # <-- [新增] 导入tifffile库，用于保存16-bit图像

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel


# (这个函数是正确的，保持不变)
def normalize_image_to_01(img_np):
    img_np = img_np.astype(np.float32)
    if np.issubdtype(img_np.dtype, np.integer):
        try:
            info = np.iinfo(img_np.dtype)
            max_val = info.max
        except ValueError:  # 如果是float转来的int，可能没有iinfo
            max_val = 65535.0  # 默认按16-bit处理
        if max_val > 0:
            img_np = img_np / max_val
    else:
        min_val = img_np.min()
        max_val = img_np.max()
        if max_val - min_val > 1e-6:
            img_np = (img_np - min_val) / (max_val - min_val)
        else:
            img_np.fill(0.5)
    return img_np


# (这个函数是正确的，保持不变)
def denoise_image(model, diffusion, input_image_tensor, device, config):
    model.eval()
    condition = input_image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_image = diffusion.sample(
            model=model,
            image_size=config['model']['image_size'],
            batch_size=1,
            channels=config['model']['gt_channels'],
            condition=condition
        ).squeeze(0)
    return denoised_image


def main():
    # --- 您需要手动指定的参数 ---
    # [重要!] 请确保这个路径指向一个使用正确配置 (image_size: 128) 训练出的模型！
    checkpoint_path = "output/2025-10-22_17-25-28/checkpoint/model_epoch_0200.pth"
    test_input_dir = "C:/Users/Guo_lab/Desktop/lxy/data-64-128/predict-10/lr"
    # -------------------------

    # 1. 加载配置文件 (逻辑不变)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到指定的模型文件: {checkpoint_path}")
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"在模型目录下找不到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # [检查] 确认加载的配置是正确的
    if config['model']['image_size'] != 128:
        print("=" * 60)
        print(f"警告: 模型训练时的 image_size 为 {config['model']['image_size']}, 而不是期望的 128。")
        print(
            "预测结果的尺寸将是 " + str(config['model']['image_size']) + "x" + str(config['model']['image_size']) + "。")
        print("请确保您加载了使用正确配置训练出的模型！")
        print("=" * 60)

    # 2. 创建输出目录 (逻辑不变)
    test_output_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)),
                                   f"test_results_epoch_{os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]}")
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"测试结果将保存在: {test_output_dir}")

    # (3, 4, 5. 初始化部分完全不变)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = UNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {checkpoint_path}")
    diffusion = DiffusionModel(config)

    # 6. 图像变换 (逻辑不变，是正确的)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['model']['image_size'], config['model']['image_size']), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 7. 遍历测试文件夹中的所有tif图像并进行去噪
    test_files = glob.glob(os.path.join(test_input_dir, '*.tif'))
    for image_path in tqdm(test_files, desc="正在处理测试图像"):
        # (加载和预处理部分是正确的，保持不变)
        input_img_np = np.array(Image.open(image_path))
        if input_img_np.ndim == 3:
            input_img_np = np.squeeze(input_img_np, axis=2)
        input_img_norm = normalize_image_to_01(input_img_np)
        input_tensor = transform(input_img_norm)

        # 执行去噪 (逻辑不变)
        denoised_tensor = denoise_image(model, diffusion, input_tensor, device, config)

        # --- [核心修正] 使用 tifffile 保存为 16-bit TIF ---
        # 1. 移除旧的、错误的PIL转换
        #    denoised_image = to_pil(denoised_tensor.cpu()) <-- 删除

        # 2. 将范围[0,1]的浮点数Tensor，转换回范围[0, 65535]的16-bit无符号整数 (uint16)
        output_image_np = (denoised_tensor.clamp(0, 1).cpu().numpy().squeeze() * 65535).astype(np.uint16)

        # 3. 使用 tifffile.imwrite 进行专业保存
        base_name = os.path.basename(image_path)
        save_path = os.path.join(test_output_dir, f"denoised_{base_name}")
        tifffile.imwrite(save_path, output_image_np)
        # --- 修正结束 ---

    print("所有测试图像处理完毕！")


if __name__ == '__main__':
    main()