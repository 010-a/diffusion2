# test.py (已修正数据预处理逻辑)

import yaml
import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np  # <-- [新增] 导入numpy库

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel


# --- [新增] 从dataset.py复制过来的、完全相同的归一化函数 ---
def normalize_image_to_01(img_np):
    """
    将Numpy图像数组归一化到[0, 1]范围。
    能处理 uint8, uint16, float32 等多种类型。
    """
    img_np = img_np.astype(np.float32)
    if np.issubdtype(img_np.dtype, np.integer):
        # 根据数据类型的理论最大值进行缩放 (例如 uint16 -> 65535)
        info = np.iinfo(img_np.dtype)
        max_val = info.max
        if max_val > 0:
            img_np = img_np / max_val
    else:  # 如果是浮点数类型
        min_val = img_np.min()
        max_val = img_np.max()
        if max_val - min_val > 1e-6:
            img_np = (img_np - min_val) / (max_val - min_val)
        else:
            img_np.fill(0.5)
    return img_np


# (denoise_image 函数保持不变)
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
    checkpoint_path = "output/2025-10-21_21-07-08/checkpoint/model_epoch_1600.pth"
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

    # 2. 创建输出目录 (逻辑不变)
    test_output_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)),
                                   f"test_results_epoch_{os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]}")
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"测试结果将保存在: {test_output_dir}")

    # 3. 设置设备 (逻辑不变)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 4. 初始化模型并加载权重 (逻辑不变)
    model = UNet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {checkpoint_path}")

    # 5. 初始化扩散过程 (逻辑不变)
    diffusion = DiffusionModel(config)

    # 6. [核心修正] 定义与训练时完全一致的图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将[0,1]的numpy数组转为Tensor
        transforms.Resize((config['model']['image_size'], config['model']['image_size']), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]范围
    ])

    to_pil = transforms.ToPILImage()

    # 7. 遍历测试文件夹中的所有tif图像并进行去噪
    test_files = glob.glob(os.path.join(test_input_dir, '*.tif'))

    for image_path in tqdm(test_files, desc="正在处理测试图像"):
        # --- [核心修正] 替换掉旧的加载和预处理逻辑 ---
        # 1. 使用 PIL.Image 打开图像，然后立即转换为 Numpy 数组
        #    这样可以保留原始的位深 (例如 16-bit)
        input_img_np = np.array(Image.open(image_path))

        # 2. 确保是单通道 (H, W)
        if input_img_np.ndim == 3:
            input_img_np = np.squeeze(input_img_np, axis=2)

        # 3. 使用与训练时完全相同的函数，将图像归一化到 [0.0, 1.0]
        input_img_norm = normalize_image_to_01(input_img_np)

        # 4. 应用torchvision的变换
        input_tensor = transform(input_img_norm)
        # --- 修正结束 ---

        # 执行去噪 (逻辑不变)
        denoised_tensor = denoise_image(model, diffusion, input_tensor, device, config)

        # 后处理并保存
        # diffusion.sample 的输出已经是 [0, 1]
        denoised_image = to_pil(denoised_tensor.cpu())

        base_name = os.path.basename(image_path)
        save_path = os.path.join(test_output_dir, f"denoised_{base_name}")
        denoised_image.save(save_path)

    print("所有测试图像处理完毕！")


if __name__ == '__main__':
    main()