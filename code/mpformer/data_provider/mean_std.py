import torch
import numpy as np
import loader

def compute_mean_std(dataloader):
    total_sum = 0.0
    total_squared_sum = 0.0
    total_pixels = 0

    all_data = []

    for batch in dataloader:
        radar_frames = batch['radar_frames']
        data = radar_frames[..., 0].float()

        total_sum += torch.sum(data)
        total_squared_sum += torch.sum(data ** 2)
        total_pixels += data.numel()
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=None)

    mean = total_sum / total_pixels
    std = torch.sqrt(total_squared_sum / total_pixels - mean ** 2)
    qu_50 = np.percentile(all_data, 50)

    # qu_50 = (qu_50 - mean) / std

    return mean.item(), std.item(), qu_50.item()

if __name__ == '__main__':
    input_param = {
        'input_data_type': 'float32',
        'output_data_type': 'float32',
        'image_width': 512,
        'image_height': 512,
        'total_length': 10,
        'data_path': 'data/dataset/mrms/figure',
        'type': 'train'
    }
    dataset = loader.InputHandle(input_param)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    mean, std, qu_50 = compute_mean_std(dataloader)
    print(mean)
    print(std)
    print(qu_50)