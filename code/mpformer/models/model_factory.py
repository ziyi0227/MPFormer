import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from mpformer.models import mpformer
from mpformer.utils import metrics
from mpformer.utils import loss_assemble

class CustomLoss(nn.Module):
    """
        Loss in NocastNet
        TotalLoss = MSE + Proto + Contrastive
    """
    def __init__(self, configs):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, prototypes, positive_pairs, negative_pairs):
        # MSE Loss
        mse_loss = self.mse_loss(outputs, targets)

        # Prototype Loss
        proto_loss = self.mse_loss(prototypes, targets.mean(dim=1, keepdim=True))

        # Contrastive Loss (Hinge Loss)
        pos_dist = F.pairwise_distance(positive_pairs[:, 0, :], positive_pairs[:, 1, :])
        neg_dist = F.pairwise_distance(negative_pairs[:, 0, :], negative_pairs[:, 1, :])
        contrastive_loss = torch.mean(torch.relu(pos_dist - neg_dist + 1.0)) # Margin of 1.0

        # Total Loss
        total_loss = mse_loss + proto_loss + contrastive_loss

        return total_loss

class SASTLoss(nn.Module):
    """
        Loss by design
        losstype = single:
        losstype = multi:
    """
    def __init__(self, losstype='single'):
        super(SASTLoss, self).__init__()
        self.losstype = losstype
        self.weights = [1, 2, 5, 10, 30]
        self.thresholds = [0.1, 0.5, 1, 5, 20]
        self.use_wavelet = True
        
        # 初始化损失函数
        self.Loss_func1 = loss_assemble.BMSELoss(weights=self.weights, thresholds=self.thresholds)
        self.Loss_func2 = loss_assemble.BMAELoss(weights=self.weights, thresholds=self.thresholds)
        
        if self.losstype == 'multi':
            # iterate越大，关注的越细，算的越慢
            self.Loss_func3 = loss_assemble.WSloss().cuda()
            self.Loss_func4 = loss_assemble.MTloss(scaler=1.0).cuda()
        else:
            self.Loss_func3 = loss_assemble.WSloss_linear_add_adhoc().cuda() # [0.25, 1., 4.]
            if self.use_wavelet:
                self.Loss_func4 = loss_assemble.WTloss().cuda()
            else:
                self.Loss_func4 = loss_assemble.MTloss_add_linear(scaler=0.1, iterate=self.configs['t_iter']).cuda()
        
    def forward(self, preds, targets):
        if self.losstype == 'multi':
            loss1 = self.Loss_func1(preds, targets)
            loss2 = self.Loss_func2(preds, targets)
            loss3 = self.Loss_func3(preds, targets)
            loss4 = self.Loss_func4(preds, targets)
            return loss1, loss2, *loss3, *loss4
        elif self.losstype == 'single':
            loss1 = self.Loss_func1(preds, targets)
            loss2 = self.Loss_func2(preds, targets)
            loss3 = self.Loss_func3(preds, targets)
            loss4 = self.Loss_func4(preds, targets)
            return loss1 + loss2 + loss3 + loss4


class Model(pl.LightningModule):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.network = mpformer.Net(configs)
        self.criterion = SASTLoss(losstype='single')

        self.hss = metrics.HSS()
        self.neigh_csi = metrics.NeighbourhoodCSI(kernel_size=3)
        self.neigh_csi2 = metrics.NeighbourhoodCSI2(kernel_size=3)
        self.psd = metrics.PSD()
        self.rmse = metrics.RMSE()
        self.crps = metrics.CRPS()
        self.fss = metrics.FSS()
        self.mae = metrics.MAE()
    
    def forward(self, x):
        x = x.float()
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        radar_frames = batch['radar_frames'].float()
        inputs = radar_frames[:, :self.configs.input_length]
        targets = radar_frames[:, self.configs.input_length:]
        outputs = self(inputs)

        prototypes = self.get_prototypes(outputs, targets)
        positive_pairs, negative_pairs = self.get_pairs(outputs, targets)

        loss = self.criterion(outputs, targets) # SASTLoss
        self.log('train_loss', loss)

        self.hss.update(outputs, targets)
        self.neigh_csi.update(outputs, targets)
        self.neigh_csi2.update(outputs, targets)
        self.psd.update(outputs, targets)
        self.rmse.update(outputs, targets)
        self.crps.update(outputs, targets)
        self.fss.update(outputs, targets)
        self.mae.update(outputs, targets)

        return loss
    
    def validation_step(self, batch, batch_idx):
        radar_frames = batch['radar_frames'].float()
        inputs = radar_frames[:, :self.configs.input_length]
        targets = radar_frames[:, self.configs.input_length:]
        outputs = self(inputs)

        prototypes = self.get_prototypes(outputs, targets)
        positive_pairs, negative_pairs = self.get_pairs(outputs, targets)

        loss = self.criterion(outputs, targets) # SASTLoss
        self.log('val_loss', loss)
        
        self.hss.update(outputs, targets)
        self.neigh_csi.update(outputs, targets)
        self.neigh_csi2.update(outputs, targets)
        self.psd.update(outputs, targets)
        self.rmse.update(outputs, targets)
        self.crps.update(outputs, targets)
        self.fss.update(outputs, targets)
        self.mae.update(outputs, targets)

        return loss
    
    def on_training_epoch_end(self, outputs):
        self.log('val_hss', self.hss.compute())
        self.log('val_neigh_csi2', self.neigh_csi2.compute())
        self.log('val_rmse', self.rmse.compute())
        self.log('val_crps', self.crps.compute())
        self.log('val_fss', self.fss.compute())
        self.log('val_mae', self.mae.compute())

        self.hss.reset()
        self.neigh_csi.reset()
        self.neigh_csi2.reset()
        self.psd.reset()
        self.rmse.reset()
        self.crps.reset()
        self.fss.reset()
        self.mae.reset()

    def on_validation_epoch_end(self):
        # 计算并记录每个时间步的CSI
        overall_csi, time_step_csi = self.neigh_csi.compute()
        self.log('val_neigh_csi', overall_csi)

        # 将每个时间步的CSI值记录到日志中，记录每个时间步的均值
        # Initialize a list to store the last 20 means
        last_20_means_csin = []

        for idx, time_step in enumerate(time_step_csi):
            # Convert time_step to a tensor
            time_step_tensor = torch.tensor(time_step)
            
            # Calculate the mean
            mean_value = time_step_tensor.mean()
            
            # Append the mean to the list
            last_20_means_csin.append(mean_value)
            
            # Keep only the last 20 means
            if len(last_20_means_csin) > 20:
                last_20_means_csin = last_20_means_csin[-20:]

        # After the loop, log the last 20 means
        for i, mean_val in enumerate(last_20_means_csin):
            self.log(f'val_neigh_csi_time_step_{i}', mean_val)
            
        # 计算并记录每个时间步的 PSD
        overall_psd, time_step_psd = self.psd.compute()
        self.log('val_neigh_psd', overall_psd)

        last_20_means_psd = []

        for idx, time_step in enumerate(time_step_psd):            
            stacked_tensor = torch.stack(time_step, dim=0)
            mean_value = torch.mean(stacked_tensor).item()
            last_20_means_psd.append(mean_value)
            
            # 仅保留最后20个均值
            if len(last_20_means_psd) > 20:
                last_20_means_psd = last_20_means_psd[-20:]

        # 在循环结束后，记录最后20个均值
        for i, mean_val in enumerate(last_20_means_psd):
            self.log(f'val_psd_time_step_{i}', mean_val)

        self.log('val_hss', self.hss.compute())
        self.log('val_neigh_csi2', self.neigh_csi2.compute())
        self.log('val_rmse', self.rmse.compute())
        self.log('val_crps', self.crps.compute())
        self.log('val_fss', self.fss.compute())
        self.log('val_mae', self.mae.compute())

        self.hss.reset()
        self.neigh_csi.reset()
        self.neigh_csi2.reset()
        self.psd.reset()
        self.rmse.reset()
        self.crps.reset()
        self.fss.reset()
        self.mae.reset()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=self.configs.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.configs.step_size, gamma=0.5)
        return [optimizer], [scheduler]
    
    def get_prototypes(self, outputs, targets):
        """
        获取原型: 所有输出中被targets选中的区域的均值
        outputs: 形状为 [batch, time, height, width, channel]
        targets: 形状为 [batch, time, height, width, channel]，二值化掩码
        """
        # 确保掩码与输出相同形状
        mask = (targets > 0).float() # 将掩码二值化并转换为 float

        # 计算被掩码选中的区域的加权和
        weighted_outputs = outputs * mask

        # 计算被掩码选中的区域的总和和掩码的总和
        sum_weighted_outputs = weighted_outputs.sum(dim=(2, 3, 4), keepdim=True) # 在 batch、height、width 和 channel 维度上求和
        sum_mask = mask.sum(dim=(2, 3, 4), keepdim=True) # 掩码求和

        # 防止除以0的情况
        prototypes = sum_weighted_outputs / (sum_mask + 1e-8)

        return prototypes
    
    def get_pairs(self, outputs, targets):
        """
        获取正样本对和负样本对。
        outputs: 模型的输出 [batch_size, seq_length, height, width, channels]
        targets: 目标 [batch_size, seq_length, height, width, channels]
        """
        # 确保 targets 和 outputs 的形状一致
        # batch_size, seq_length, height, width, channels = outputs.size()

        # 将 targets 转换为 [batch_size, seq_length, 1, height, width, channels]
        targets = targets.unsqueeze(2)

        # 广播生成 mask，判断哪些像素属于正样本
        positive_mask = (targets > 0).float() # [batch_size, seq_length, 1, height, width, channels]

        # 生成负样本 mask，通过取反操作
        negative_mask = 1.0 - positive_mask # [batch_size, seq_length, 1, height, width, channels]

        # 计算正样本对
        positive_pairs = outputs.unsqueeze(2) * positive_mask # [batch_size, seq_length, 1, height, width, channels]

        # 计算负样本对
        negative_pairs = outputs.unsqueeze(2) * negative_mask # [batch_size, seq_length, 1, height, width, channels]

        return positive_pairs, negative_pairs