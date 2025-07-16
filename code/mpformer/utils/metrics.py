import torch
import torchmetrics
from torchmetrics import Metric
from scipy.ndimage import label
import torch.nn.functional as F

class CSI(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        preds = (outputs > 0.5).int()
        targets = (target[..., 0:1] > 0.5).int()
        # print(preds.shape, targets.shape)

        self.tp += torch.sum(preds * targets)
        self.fp += torch.sum(preds * (1 - targets))
        self.fn += torch.sum((1 - preds) * targets)

    def compute(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-8)
    

class POD(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        preds = (outputs > 0.5).int()
        targets = (target[..., 0:1] > 0.5).int()

        self.tp += torch.sum(preds * targets)
        self.fn += torch.sum((1 - preds) * targets)

    def compute(self):
        return self.tp / (self.tp + self.fn + 1e-8)


class FAR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        preds = (outputs > 0.5).int()
        targets = (target[..., 0:1] > 0.5).int()

        self.fp += torch.sum(preds * (1 - targets))
        self.tp += torch.sum(preds * targets)

    def compute(self):
        return self.fp / (self.fp + self.tp + 1e-8)
    

# class HSS(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, outputs, target):
#         preds = (outputs > 0.5).int()
#         targets = (target[..., 0:1] > 0.5).int()

#         self.tp += torch.sum(preds * targets)
#         self.fp += torch.sum(preds * (1 - targets))
#         self.fn += torch.sum((1 - preds) * targets)
#         self.tn += torch.sum((1 - preds) * (1 - targets))

#     def compute(self):
#         num = 2 * (self.tp * self.tn - self.fp * self.fn)
#         den = (self.tp + self.fn) * (self.fn + self.tn) + (self.tp + self.fp) * (self.fp + self.tn)
#         return num / (den + 1e-8)

# HSS（Heidke Skill Score）三个阈值
class HSS(Metric):
    def __init__(self, kernel_size=3, thresholds=[0.5, 7, 20]):
        super().__init__()
        self.kernel_size = kernel_size
        self.thresholds = thresholds

        # 初始化TP, TN, FP, FN
        for thresh in self.thresholds:
            self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"tn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        """
        更新每个阈值的 TP, TN, FP, FN。
        """
        for thresh in self.thresholds:
            preds = (outputs >= thresh).float()
            targets = (target[..., 0:1] >= thresh).float()

            # 邻域扩展
            preds = torch.nn.functional.max_pool2d(
                preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            targets = torch.nn.functional.max_pool2d(
                targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            # 计算TP, TN, FP, FN
            tp = torch.sum(preds * targets)
            tn = torch.sum((1 - preds) * (1 - targets))
            fp = torch.sum(preds * (1 - targets))
            fn = torch.sum((1 - preds) * targets)

            # 累积TP, TN, FP, FN
            self.__dict__[f"tp_{thresh}"] += tp
            self.__dict__[f"tn_{thresh}"] += tn
            self.__dict__[f"fp_{thresh}"] += fp
            self.__dict__[f"fn_{thresh}"] += fn

    def compute(self):
        """
        计算每个阈值的 HSS 并返回均值。
        """
        hss_values = []
        for thresh in self.thresholds:
            tp = self.__dict__[f"tp_{thresh}"]
            tn = self.__dict__[f"tn_{thresh}"]
            fp = self.__dict__[f"fp_{thresh}"]
            fn = self.__dict__[f"fn_{thresh}"]

            # 计算 HSS
            numerator = 2 * (tp * tn - fp * fn)
            denominator = (tp + fp) * (fp + tn) + (fn + tn) * (tp + tn) + 1e-8  # 加上小常数避免除零错误
            hss = numerator / denominator

            hss_values.append(hss)

        # 返回所有时间步、所有阈值的 HSS 均值
        return torch.mean(torch.tensor(hss_values))


class AUC(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, outputs, target):
        self.scores.append(outputs.flatten())
        self.labels.append(target.flatten())

    def compute(self):
        scores = torch.cat(self.scores)
        labels = torch.cat(self.labels)
        return torchmetrics.functional.auroc(scores, labels.int(), task="binary")


# class NeighbourhoodCSI(Metric):
#     def __init__(self, kernel_size=3):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def update(self, outputs, target):
#         preds = (outputs > 0.5).float()
#         # print(preds.shape)
#         targets = (target[..., 0:1] > 0.5).float()
#         # print(targets.shape)

#         preds = torch.nn.functional.max_pool2d(
#             preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
#             kernel_size=self.kernel_size,
#             stride=1,
#             padding=self.kernel_size // 2
#         )

#         targets = torch.nn.functional.max_pool2d(
#             targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
#             kernel_size=self.kernel_size,
#             stride=1,
#             padding=self.kernel_size // 2
#         )

#         # 如果preds矩阵中有大于1的元素就输出，一个if语句判断print的输出
#         # if torch.max(preds) > 1:
#         #     print(torch.max(preds))

#         self.tp += torch.sum(preds * targets)
#         # print(self.tp)
#         self.fp += torch.sum(preds * (1 - targets))
#         # print(self.fp)
#         self.fn += torch.sum((1 - preds) * targets)
#         # print(self.fn)

#     def compute(self):
#         return self.tp / (self.tp + self.fp + self.fn + 1e-8)

# 方案3：使用多级阈值 CSI 计算
class NeighbourhoodCSI(Metric): 
    def __init__(self, kernel_size=3, thresholds=[0.5, 7, 20]):
        super().__init__()
        self.kernel_size = kernel_size
        self.thresholds = thresholds

        # 初始化阈值对应的TP, FP, FN
        for thresh in self.thresholds:
            self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        # 用于保存每个时间步的 CSI 值
        self.time_step_csi = []

    def update(self, outputs, target):
        """
        计算每个时间步和每个阈值的CSI。
        """
        # 确保输出和目标是 [B, T, C, H, W] 形状，其中 T 是时间步维度
        csi_values_time_step = []

        # 逐时间步计算CSI
        for t in range(outputs.shape[1]):  # [B, T, C, H, W] -> T
            csi_values_threshold = []
            for thresh in self.thresholds:
                preds = (outputs[:, t, :, :, :] >= thresh).float()
                targets = (target[:, t, :, :, 0:1] >= thresh).float()

                # 邻域扩展
                preds = torch.nn.functional.max_pool2d(
                    preds.permute(0, 3, 1, 2).reshape(-1, 1, preds.shape[2], preds.shape[3]),
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                )

                targets = torch.nn.functional.max_pool2d(
                    targets.permute(0, 3, 1, 2).reshape(-1, 1, targets.shape[2], targets.shape[3]),
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                )

                # 计算每个阈值的 TP, FP, FN
                self.__dict__[f"tp_{thresh}"] = torch.sum(preds * targets)
                self.__dict__[f"fp_{thresh}"] = torch.sum(preds * (1 - targets))
                self.__dict__[f"fn_{thresh}"] = torch.sum((1 - preds) * targets)
                
                # 计算每个阈值的 CSI
                tp = self.__dict__[f"tp_{thresh}"]
                fp = self.__dict__[f"fp_{thresh}"]
                fn = self.__dict__[f"fn_{thresh}"]
                csi = tp / (tp + fp + fn + 1e-8)
                csi_values_threshold.append(csi.item())
            
            # 保存每个时间步的阈值 CSI
            csi_values_time_step.append(csi_values_threshold)

        # 每个时间步的 CSI 计算结果，只保存一次（不是重复追加）
        self.time_step_csi.append(csi_values_time_step)

    def compute(self):
        """
        计算并返回每个时间步的 CSI 均值和所有时间步的总体均值。
        """
        # 计算每个时间步的 CSI 均值
        time_step_csi_mean = []
        for time_step_csi in self.time_step_csi:
            time_step_csi_mean.append(torch.mean(torch.tensor(time_step_csi)))

        # 计算所有时间步的 CSI 均值
        overall_csi = torch.mean(torch.tensor(time_step_csi_mean))
        
        return overall_csi, self.time_step_csi

class NeighbourhoodCSI2(Metric):
    def __init__(self, kernel_size=3, thresholds=[0.5, 7, 20]):
        super().__init__()
        self.kernel_size = kernel_size
        self.thresholds = thresholds

        # 保存不同阈值下的 TP, FP, FN
        for thresh in self.thresholds:
            self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        """
        对每个阈值进行处理，分别计算 TP, FP, FN。
        """
        for thresh in self.thresholds:
            preds = (outputs >= thresh).float()
            targets = (target[..., 0:1] >= thresh).float()

            # 邻域扩展
            preds = torch.nn.functional.max_pool2d(
                preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            targets = torch.nn.functional.max_pool2d(
                targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            # 累积 TP, FP, FN
            self.__dict__[f"tp_{thresh}"] += torch.sum(preds * targets)
            self.__dict__[f"fp_{thresh}"] += torch.sum(preds * (1 - targets))
            self.__dict__[f"fn_{thresh}"] += torch.sum((1 - preds) * targets)

    def compute(self):
        """
        计算每个阈值的 CSI 并返回均值。
        """
        csi_values = []
        for thresh in self.thresholds:
            tp = self.__dict__[f"tp_{thresh}"]
            fp = self.__dict__[f"fp_{thresh}"]
            fn = self.__dict__[f"fn_{thresh}"]
            csi = tp / (tp + fp + fn + 1e-8)
            csi_values.append(csi)

        # print(csi_values)
        # 返回所有时间步、所有阈值的 CSI 均值
        return torch.mean(torch.tensor(csi_values))
        # csi_values = torch.stack(csi_values, dim=0)  # [num_thresholds]
        # return csi_values.mean()

# 方案3.1：使用多级阈值 CSI 计算+对数归一化
# class NeighbourhoodCSI(Metric):
#     def __init__(self, kernel_size=3, thresholds=[1, 2, 4]):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.thresholds = thresholds

#         # 保存不同阈值下的 TP, FP, FN
#         for thresh in self.thresholds:
#             self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def log_normalize(self, tensor, eps=1e-6):
#         """
#         对输入进行 log1p 归一化，避免零值问题。
#         """
#         return torch.log1p(tensor + eps)

#     def update(self, outputs, target):
#         """
#         对每个阈值进行处理，分别计算 TP, FP, FN。
#         """
#         # 对 outputs 和 target 进行对数归一化
#         outputs = self.log_normalize(outputs)
#         target = self.log_normalize(target)

#         for thresh in self.thresholds:
#             preds = (outputs >= thresh).float()
#             targets = (target[..., 0:1] >= thresh).float()

#             # 邻域扩展
#             preds = torch.nn.functional.max_pool2d(
#                 preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )

#             targets = torch.nn.functional.max_pool2d(
#                 targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )

#             # 累积 TP, FP, FN
#             self.__dict__[f"tp_{thresh}"] += torch.sum(preds * targets)
#             self.__dict__[f"fp_{thresh}"] += torch.sum(preds * (1 - targets))
#             self.__dict__[f"fn_{thresh}"] += torch.sum((1 - preds) * targets)

#     def compute(self):
#         """
#         计算每个阈值的 CSI 并返回均值。
#         """
#         csi_values = []
#         for thresh in self.thresholds:
#             tp = self.__dict__[f"tp_{thresh}"]
#             fp = self.__dict__[f"fp_{thresh}"]
#             fn = self.__dict__[f"fn_{thresh}"]
#             csi = tp / (tp + fp + fn + 1e-8)
#             csi_values.append(csi)

#         # 返回所有时间步、所有阈值的 CSI 均值
#         return torch.mean(torch.tensor(csi_values))


# # 方案2：2. 引入概率加权的 CSI 计算
# class NeighbourhoodCSI(Metric):
#     def __init__(self, kernel_size=3, thresholds=[16, 32, 64]):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.thresholds = thresholds

#         # 保存不同阈值下的加权 TP, FP, FN
#         for thresh in self.thresholds:
#             self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def update(self, outputs, target):
#         """
#         更新加权 TP, FP, FN 的逻辑。
#         """
#         for thresh in self.thresholds:
#             # 将输出概率与阈值进行比较，得到概率权重
#             preds = torch.sigmoid(outputs)  # 确保 outputs 是 [0, 1] 概率
#             targets = (target[..., 0:1] >= thresh).float()

#             # 邻域扩展以增强容忍度
#             preds = torch.nn.functional.max_pool2d(
#                 preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )

#             targets = torch.nn.functional.max_pool2d(
#                 targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )

#             # 使用连续值计算加权 TP, FP, FN
#             self.__dict__[f"tp_{thresh}"] += torch.sum(preds * targets)
#             self.__dict__[f"fp_{thresh}"] += torch.sum(preds * (1 - targets))
#             self.__dict__[f"fn_{thresh}"] += torch.sum((1 - preds) * targets)

#     def compute(self):
#         """
#         计算每个阈值的概率加权 CSI 并返回均值。
#         """
#         csi_values = []
#         for thresh in self.thresholds:
#             tp = self.__dict__[f"tp_{thresh}"]
#             fp = self.__dict__[f"fp_{thresh}"]
#             fn = self.__dict__[f"fn_{thresh}"]
#             csi = tp / (tp + fp + fn + 1e-8)
#             csi_values.append(csi)

#         # 返回所有阈值的加权 CSI 均值
#         return torch.mean(torch.tensor(csi_values))

# 方案4：4. 采用区域级邻域 CSI
# class NeighbourhoodCSI(Metric):
#     def __init__(self, kernel_size=3, thresholds=[16, 32, 64]):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.thresholds = thresholds

#         # 保存不同阈值下的 TP, FP, FN
#         for thresh in self.thresholds:
#             self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
#             self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

#     def _extract_regions(self, binary_map):
#         """
#         提取连通区域并返回区域掩码和区域数量。
#         """
#         # 转为 numpy 格式以便使用 scipy 的连通组件分析
#         binary_map_np = binary_map.cpu().numpy()
#         labeled_map, num_features = label(binary_map_np)  # 连通组件分析
#         return labeled_map, num_features

#     def _calculate_region_metrics(self, preds, targets):
#         """
#         基于区域级别计算 TP, FP 和 FN。
#         """
#         tp, fp, fn = 0.0, 0.0, 0.0

#         # 提取预测和目标的连通区域
#         pred_labels, pred_count = self._extract_regions(preds)
#         target_labels, target_count = self._extract_regions(targets)

#         # 转为 Tensor 格式
#         pred_labels = torch.tensor(pred_labels, device=preds.device)
#         target_labels = torch.tensor(target_labels, device=targets.device)

#         # 遍历预测区域，检查与目标区域的重叠
#         for pred_region_id in range(1, pred_count + 1):  # 注意区域 ID 从 1 开始
#             pred_region = (pred_labels == pred_region_id).float()

#             # 计算与每个目标区域的重叠
#             overlap = pred_region.unsqueeze(0) * target_labels
#             matched_targets = torch.unique(overlap)

#             if len(matched_targets) > 1:  # 如果存在交集
#                 tp += 1  # 匹配区域计为 TP
#             else:
#                 fp += 1  # 无匹配区域计为 FP

#         # 遍历目标区域，检查是否未被匹配
#         for target_region_id in range(1, target_count + 1):
#             target_region = (target_labels == target_region_id).float()
#             overlap = target_region.unsqueeze(0) * pred_labels
#             if torch.sum(overlap) == 0:
#                 fn += 1  # 未匹配区域计为 FN

#         return tp, fp, fn

#     def update(self, outputs, target):
#         """
#         更新 TP, FP, FN 的逻辑。
#         """
#         for thresh in self.thresholds:
#             preds = (outputs >= thresh).float()
#             targets = (target[..., 0:1] >= thresh).float()

#             # 邻域扩展
#             preds = F.max_pool2d(
#                 preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )
#             targets = F.max_pool2d(
#                 targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
#                 kernel_size=self.kernel_size,
#                 stride=1,
#                 padding=self.kernel_size // 2,
#             )

#             # 基于区域计算 TP, FP, FN
#             tp, fp, fn = self._calculate_region_metrics(preds.squeeze(), targets.squeeze())
#             self.__dict__[f"tp_{thresh}"] += tp
#             self.__dict__[f"fp_{thresh}"] += fp
#             self.__dict__[f"fn_{thresh}"] += fn

#     def compute(self):
#         """
#         计算每个阈值的区域级 CSI 并返回均值。
#         """
#         csi_values = []
#         for thresh in self.thresholds:
#             tp = self.__dict__[f"tp_{thresh}"]
#             fp = self.__dict__[f"fp_{thresh}"]
#             fn = self.__dict__[f"fn_{thresh}"]
#             csi = tp / (tp + fp + fn + 1e-8)
#             csi_values.append(csi)

#         # 返回所有阈值的 CSI 均值
#         return torch.mean(torch.tensor(csi_values))


class PSD(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # 用于保存每个时间步的 PSD 值
        self.time_step_psd = []

    def update(self, outputs, target):
        """
        计算每个时间步的 PSD。
        """
        psd_values_time_step = []

        # 逐时间步计算PSD
        for t in range(outputs.shape[1]):  # [B, T, C, H, W] -> T
            outputs_t = outputs[:, t, ..., 0]  # 获取当前时间步的输出
            target_t = target[:, t, ..., 0]  # 获取当前时间步的目标

            # 转换到频域
            outputs_freq = torch.fft.fft2(outputs_t)
            target_freq = torch.fft.fft2(target_t)

            # 计算功率谱密度
            P_output = torch.abs(outputs_freq) ** 2
            P_target = torch.abs(target_freq) ** 2 + 1e-5

            # 计算PSD
            psd = torch.mean((torch.log(P_output + 1e-5) - torch.log(P_target + 1e-5)) ** 2, dim=(-2, -1))

            psd_values_time_step.append(psd)

        # 保存每个时间步的 PSD
        self.time_step_psd.append(psd_values_time_step)

        # 累加 PSD 总和
        self.psd_sum += torch.mean(torch.stack(psd_values_time_step))  # 计算当前批次的平均值
        self.num_samples += outputs.shape[0]

    def compute(self):
        """
        计算并返回每个时间步的 PSD 均值和所有时间步的总体均值。
        """
        # 计算每个时间步的 PSD 均值
        time_step_psd_mean = []
        for time_step_psd in self.time_step_psd:
            time_step_psd_mean.append(torch.mean(torch.stack(time_step_psd)).item())

        # 计算所有时间步的 PSD 均值
        overall_psd = torch.mean(torch.tensor(time_step_psd_mean))

        return overall_psd, self.time_step_psd

    
# RMSE（均方根误差）
class RMSE(Metric):
    def __init__(self):
        super().__init__()
        # 保存累计的平方误差和样本数
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        """
        更新每个batch的平方误差
        """
        # 假设outputs和target都是float类型
        squared_error = torch.square(outputs - target)
        
        # 累加平方误差
        self.sum_squared_error += torch.sum(squared_error)
        
        # 累加样本数
        self.num_samples += squared_error.numel()

    def compute(self):
        """
        计算并返回RMSE
        """
        # 计算RMSE
        rmse = torch.sqrt(self.sum_squared_error / self.num_samples)
        return rmse

# CRPS（Continuous Ranked Probability Score）
class CRPS(Metric):
    def __init__(self, kernel_size=3, thresholds=[0.5, 7, 20]):
        super().__init__()
        self.kernel_size = kernel_size
        self.thresholds = thresholds

    def update(self, outputs, target):
        """
        在这个方法中计算输出的CDF和目标的CDF。
        """
        # 假设outputs是模型预测的概率分布，target是真实的值
        # 如果是概率分布，我们计算离散化的CDF差异
        
        # outputs 和 target 假设都是形状为 [batch_size, time_steps, height, width, 1]
        self.outputs = outputs
        self.target = target

    def compute(self):
        """
        计算 CRPS，基于预测的累积分布函数 (CDF) 和真实的 CDF。
        """
        crps_values = []

        # 遍历每个 batch 对象，计算 CRPS
        for i in range(self.outputs.shape[0]):
            # 获取当前 batch 中第i个预测值和真实目标
            pred = self.outputs[i]
            true = self.target[i]
            
            # 离散化处理：将预测和真实值分成多个点，然后计算累积分布函数
            # 计算预测分布的CDF
            pred_cdf = torch.cumsum(pred, dim=-1)  # 累加分布
            true_cdf = torch.cumsum(true, dim=-1)  # 对目标也做同样的操作

            # 计算CRPS: 这里通过欧氏距离度量预测CDF与真实CDF的差异
            crps = torch.sum((pred_cdf - true_cdf)**2, dim=-1)

            # 将计算出的CRPS值加入结果列表
            crps_values.append(crps)

        # 返回所有 batch 的平均CRPS
        return torch.mean(torch.stack(crps_values))

# FSS（Fréchet Skill Score）
class FSS(Metric):
    def __init__(self, kernel_size=3, thresholds=[0.5, 7, 20]):
        super().__init__()
        self.kernel_size = kernel_size
        self.thresholds = thresholds

        # 保存不同阈值下的 TP, FP, FN
        for thresh in self.thresholds:
            self.add_state(f"tp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fp_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"fn_{thresh}", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        """
        对每个阈值进行处理，分别计算 TP, FP, FN。
        """
        for thresh in self.thresholds:
            preds = (outputs >= thresh).float()
            targets = (target[..., 0:1] >= thresh).float()

            # 邻域扩展
            preds = torch.nn.functional.max_pool2d(
                preds.permute(0, 4, 1, 2, 3).reshape(-1, 1, preds.shape[2], preds.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            targets = torch.nn.functional.max_pool2d(
                targets.permute(0, 4, 1, 2, 3).reshape(-1, 1, targets.shape[2], targets.shape[3]),
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )

            # 累积 TP, FP, FN
            self.__dict__[f"tp_{thresh}"] += torch.sum(preds * targets)
            self.__dict__[f"fp_{thresh}"] += torch.sum(preds * (1 - targets))
            self.__dict__[f"fn_{thresh}"] += torch.sum((1 - preds) * targets)

    def compute(self):
        """
        计算每个阈值的 FSS 并返回均值。
        """
        fss_values = []
        for thresh in self.thresholds:
            tp = self.__dict__[f"tp_{thresh}"]
            fp = self.__dict__[f"fp_{thresh}"]
            fn = self.__dict__[f"fn_{thresh}"]

            # 计算CSI (Critical Success Index)，作为FSS的一部分
            fss = tp / (tp + fp + fn + 1e-8)
            fss_values.append(fss)

        # 返回所有时间步、所有阈值的 FSS 均值
        return torch.mean(torch.tensor(fss_values))

# MAE（Mean Absolute Error，平均绝对误差）
class MAE(Metric):
    def __init__(self):
        super().__init__()
        # 保存累积的绝对误差和样本数量
        self.add_state("absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, target):
        """
        计算每个预测值与目标值之间的绝对误差，并累积
        """
        # 计算绝对误差
        absolute_error = torch.abs(outputs - target)
        
        # 累积误差和样本数量
        self.absolute_error += torch.sum(absolute_error)
        self.num_samples += absolute_error.numel()

    def compute(self):
        """
        计算MAE
        """
        return self.absolute_error / self.num_samples

class ReliabilityDiagram(Metric):
    """
    可靠性图（Calibration Curve）指标。
    将预测概率离散到若干 bin 中，统计每个 bin 的平均预测概率 vs. 实测频率。
    """
    def __init__(self, num_bins: int = 10, ignore_index: int = None):
        super().__init__(dist_sync_on_step=False)
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        # 为每个 bin 维持两个累计量：sum_pred_probs, sum_true_counts，以及样本数
        self.add_state("sum_prob", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("sum_true", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("count",    default=torch.zeros(num_bins), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds: 概率张量，shape [B, T, H, W, 1]，取值 [0,1]
        target: 二值张量 {0,1}，同 shape
        """
        # flatten
        p = preds.flatten()
        t = target.flatten()
        # 组 bin
        bins = torch.clamp((p * self.num_bins).long(), 0, self.num_bins - 1)
        for b in range(self.num_bins):
            mask = (bins == b)
            if self.ignore_index is not None:
                mask = mask & (t != self.ignore_index)
            n = mask.sum()
            if n > 0:
                self.sum_prob[b] += p[mask].sum()
                self.sum_true[b] += t[mask].sum()
                self.count[b]    += n

    def compute(self):
        """
        返回两个长度为 num_bins 的向量：
        - avg_pred: 每个 bin 的平均预测概率
        - freq_true: 每个 bin 的实际观测频率
        """
        avg_pred  = self.sum_prob / (self.count + 1e-8)
        freq_true = self.sum_true / (self.count + 1e-8)
        return avg_pred, freq_true