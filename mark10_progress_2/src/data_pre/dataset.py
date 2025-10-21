# src/data_pre/dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob


def normalize_image_to_01(img_np):
    """
    将Numpy图像数组归一化到[0, 1]范围。
    能处理 uint8, uint16, float32 等多种类型。
    """
    # 转换为浮点数进行计算
    img_np = img_np.astype(np.float32)

    if np.issubdtype(img_np.dtype, np.integer):
        # 如果是整数类型，根据理论最大值进行缩放
        max_val = np.iinfo(img_np.dtype).max
        if max_val > 0:
            img_np = img_np / max_val
    else:  # 如果是浮点数类型
        # 使用Min-Max进行缩放
        min_val = img_np.min()
        max_val = img_np.max()
        if max_val - min_val > 1e-6:  # 避免除以零
            img_np = (img_np - min_val) / (max_val - min_val)
        else:  # 如果图像是平的（所有像素值都一样）
            img_np.fill(0.5)  # 可以设为中间值

    return img_np


class MicroscopeDataset(Dataset):
    """
    显微图像去噪数据集的自定义Dataset类。
    (已更新，可处理32-bit float及多种数据类型)
    """

    def __init__(self, input_dir, gt_dir, image_size=128):
        """
        初始化数据集。
        Args:
            input_dir (str): 含噪输入图像的文件夹路径。
            gt_dir (str): 真实干净图像（Ground Truth）的文件夹路径。
            image_size (int): 图像的目标尺寸。
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_size = image_size

        self.input_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.tif')))

        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将[0,1]的numpy数组转为Tensor
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]范围
        ])

        if len(self.input_files) != len(self.gt_files):
            raise ValueError("输入图像和GT图像的数量不匹配！")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        gt_path = self.gt_files[idx]

        # 1. 打开图像并转为Numpy数组
        # Pillow可以正确处理32位浮点TIF
        input_img_np = np.array(Image.open(input_path))
        gt_img_np = np.array(Image.open(gt_path))

        # 2. 检查并处理单通道情况
        # 有些TIF可能是 (H, W, 1)，需要转为 (H, W)
        if input_img_np.ndim == 3:
            input_img_np = np.squeeze(input_img_np, axis=2)
        if gt_img_np.ndim == 3:
            gt_img_np = np.squeeze(gt_img_np, axis=2)

        # 3. 使用辅助函数将图像的数值范围动态地归一化到 [0.0, 1.0]
        # 注意：input和gt需要独立归一化，因为它们的动态范围可能不同
        input_img_norm = normalize_image_to_01(input_img_np)
        gt_img_norm = normalize_image_to_01(gt_img_np)

        # 4. 应用定义好的transform
        input_tensor = self.transform(input_img_norm)
        gt_tensor = self.transform(gt_img_norm)

        return {"input": input_tensor, "gt": gt_tensor}