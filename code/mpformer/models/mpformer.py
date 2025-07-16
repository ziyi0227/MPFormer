import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mpformer.layers.utils import warp, make_grid
from mpformer.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from mpformer.layers.evolution.evolution_network import Evolution_Network
from mpformer.layers.generation.noise_projector import Noise_Projector

class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, self.configs.img_height, self.configs.img_width)
        self.grid = make_grid(sample_tensor)
        
        # 是否冻结主干网络
        if self.configs.adapter:
            self._freeze_backbone()

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result/128
        
        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32).cuda()
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1)
    
    def _freeze_backbone(self):
        # 冻结 Evolution_Network 和 Generative_Encoder 的主干层
        for name, param in self.evo_net.named_parameters():
            if 'adapter' not in name:  # 保证 Bottleneck Adapter 未被冻结
                param.requires_grad = False

        for name, param in self.gen_enc.named_parameters():
            if 'adapter' not in name:  # 同样保留 Bottleneck Adapter 的训练
                param.requires_grad = False
                
        for name, param in self.gen_dec.named_parameters():
            if 'adapter' not in name:  # 同样保留 Bottleneck Adapter 的训练
                param.requires_grad = False
                
        for name, param in self.proj.named_parameters():
            if 'adapter' not in name:  # 同样保留 Bottleneck Adapter 的训练
                param.requires_grad = False