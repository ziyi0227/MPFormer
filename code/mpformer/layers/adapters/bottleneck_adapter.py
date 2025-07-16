import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckAdapter(nn.Module):
    def __init__(self, in_channels, bottleneck_dim):
        super(BottleneckAdapter, self).__init__()
        self.down_proj = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, stride=1, bias=False)
        self.act = nn.ReLU()
        self.up_proj = nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x + residual
