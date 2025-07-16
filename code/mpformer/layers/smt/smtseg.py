import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from .model import SMT, UPerHead
from mmseg.models.decode_heads import FCNHead
from torchinfo import summary

class SMTWrapper(nn.Module):
    def __init__(self, num_classes=32, pretrained=None):
        super(SMTWrapper, self).__init__()

        # 构建主干网络 (SMT)
        self.backbone = SMT(
            embed_dims=[64, 128, 256, 512],
            ca_num_heads=[4, 4, 4, -1],
            sa_num_heads=[-1, -1, 8, 16],
            mlp_ratios=[4, 4, 4, 2],
            qkv_bias=True,
            drop_path_rate=0.2,
            depths=[3, 4, 18, 2],
            ca_attentions=[1, 1, 1, 0],
            num_stages=4,
            head_conv=3,
            expand_ratio=2,
            use_convnext_path=False,
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        )

        # 构建解码头 (UPerHead)
        self.decode_head = UPerHead(
            in_channels=[64, 128, 256, 512],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        )

        # 构建辅助头 (FCNHead)
        self.auxiliary_head = FCNHead(
            in_channels=256,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 通过主干网络提取特征
        features = self.backbone(x)

        # 解码头生成分割结果
        decode_out = self.decode_head(features)

        # 辅助头生成中间监督
        auxiliary_out = self.auxiliary_head(features)

        return decode_out, auxiliary_out
