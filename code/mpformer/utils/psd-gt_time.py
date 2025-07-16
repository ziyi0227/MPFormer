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
root_dir = "/public/home/wangdongjing/zjy/NowcastViT/NowcastViT/data/mrms_test_demo"

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

# 初始化：保存每个时间步的 PSD 累积值和计数
num_timesteps = 20
psd_sums = torch.zeros(num_timesteps)
counts = torch.zeros(num_timesteps, dtype=torch.int)

# 遍历文件夹，计算每个时间步的 PSD
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    # 我们只对后 20 张图片计算 PSD
    last_20_images = images[9:]

    for idx, image_name in enumerate(last_20_images):
        image_path = os.path.join(folder_path, image_name)
        image_tensor = load_image_as_tensor(image_path)

        # 计算单张图片的 PSD 并取平均
        psd_mean = torch.mean(calculate_psd(image_tensor))
        psd_sums[idx] += psd_mean
        counts[idx] += 1

# 计算每个时间步的平均 PSD
psd_per_timestep = psd_sums / counts.float()

# 计算整体平均 PSD
psd_overall = psd_per_timestep.mean()

# 输出结果
for t in range(num_timesteps):
    print(f"Time step {t+1:2d}: average PSD = {psd_per_timestep[t].item():.6f}")
print(f"\nOverall average PSD across all timesteps: {psd_overall.item():.6f}")
