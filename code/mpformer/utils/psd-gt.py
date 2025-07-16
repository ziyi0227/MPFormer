import os
import torch
from PIL import Image
import numpy as np

# PSD 计算函数
def calculate_psd(image_tensor):
    """
    计算单张图片的功率谱密度 (PSD)。
    :param image_tensor: (C, H, W) 或 (H, W) 的 PyTorch 张量
    :return: 单张图片的 PSD 张量
    """
    # 如果有通道维度，取第一个通道
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor[0, :, :]  # 选择单通道
    elif len(image_tensor.shape) != 2:
        raise ValueError("输入图像的形状不正确，应为 (C, H, W) 或 (H, W)")
    
    # 转换到频域
    freq_domain = torch.fft.fft2(image_tensor)
    power_spectrum = torch.abs(freq_domain) ** 2

    # 对数功率谱密度
    psd = torch.log(power_spectrum + 1e-5)
    return psd

# 数据集父文件夹路径
root_dir = "/public/home/wangdongjing/zhengjingyuan/extreme_weather/NowcastViT/data/dataset/mrms/figure"

# 获取所有文件夹名称
folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

# 图像加载辅助工具
def load_image_as_tensor(image_path):
    """
    加载图片并转换为 PyTorch 张量 (1, H, W)。
    :param image_path: 图片路径
    :return: PyTorch 张量
    """
    image = Image.open(image_path).convert("L")  # 转为灰度
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32)  # 转为张量
    return image_tensor.unsqueeze(0)  # 增加通道维度

psd_sum = 0.0
num_samples = 0

for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    # 获取前 9 张和后 20 张图片路径
    first_9_images = images[:9]
    last_20_images = images[9:]

    # 加载并计算后 20 张图片的 PSD
    for image_name in last_20_images:
        image_path = os.path.join(folder_path, image_name)
        image_tensor = load_image_as_tensor(image_path)

        # 计算单张图片的 PSD
        psd = calculate_psd(image_tensor)
        psd_sum += torch.mean(psd)  # 累积平均 PSD
        num_samples += 1

# 数据集级别的 PSD 平均值
psd_ground_truth = psd_sum / num_samples
print("Dataset PSD Ground Truth:", psd_ground_truth.item())
