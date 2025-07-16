import os.path
import datetime
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入你在 metrics.py 中定义的各个指标类
from mpformer.utils.metrics import NeighbourhoodCSI, HSS, PSD, RMSE, CRPS, FSS, MAE


def test_pytorch_loader(model, test_input_handle, configs, itr):
    """
    批量预测并计算每个预测时刻的各项指标，避免一次性保存所有结果造成内存占用过高。
    同时针对 AUC 等需要单通道输入的指标，显式提取第一通道进行计算。
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

    # 输出目录
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    # 预测步数
    T_out = configs.total_length - configs.input_length

    # 1) 按时间步初始化指标列表
    csi_ts  = [NeighbourhoodCSI()   for _ in range(T_out)]
    # pod_ts  = [POD()   for _ in range(T_out)]
    # far_ts  = [FAR()   for _ in range(T_out)]
    hss_ts  = [HSS()   for _ in range(T_out)]
    # auc_ts  = [AUC()   for _ in range(T_out)]
    psd_ts  = [PSD()   for _ in range(T_out)]
    rmse_ts = [RMSE()  for _ in range(T_out)]
    crps_ts = [CRPS()  for _ in range(T_out)]
    fss_ts  = [FSS()   for _ in range(T_out)]
    mae_ts  = [MAE()   for _ in range(T_out)]

    # 2) 遍历所有 batch，逐时间步 update
    for batch_id, batch in enumerate(test_input_handle):
        # 获取 numpy 数据和模型预测
        test_ims = batch['radar_frames'].numpy()                   # [B, T_in+T_out, H, W, C]
        preds    = model.test(test_ims)                            # [B, T_out,       H, W, C]
        gt       = test_ims[:, configs.input_length:, ...]         # [B, T_out,       H, W, C]

        # 转 torch
        pr_t = torch.from_numpy(preds)
        gt_t = torch.from_numpy(gt)

        # 对每个时间步分别更新指标
        for t in range(T_out):
            # 先取出时间切片
            out_full = pr_t[:, t:t+1, ...]   # shape [B,1,H,W,C]
            tgt_full = gt_t[:, t:t+1, ...]
            # 只保留第1通道，符合大部分指标的设计
            out_t = out_full[..., 0:1]
            tgt_t = tgt_full[..., 0:1]

            # 更新指标
            csi_ts[t].update(out_t, tgt_t)
            # pod_ts[t].update(out_t, tgt_t)
            # far_ts[t].update(out_t, tgt_t)
            hss_ts[t].update(out_t, tgt_t)
            # auc_ts[t].update(out_t, tgt_t)
            psd_ts[t].update(out_t, tgt_t)
            rmse_ts[t].update(out_t, tgt_t)
            crps_ts[t].update(out_t, tgt_t)
            fss_ts[t].update(out_t, tgt_t)
            mae_ts[t].update(out_t, tgt_t)

    # 3) compute 并保存每个时间步的指标结果
    metrics_per_t = {
        'CSIN':  [m.compute()[0].item() for m in csi_ts],
        # 'POD':  [m.compute().item() for m in pod_ts],
        # 'FAR':  [m.compute().item() for m in far_ts],
        'HSS':  [m.compute().item() for m in hss_ts],
        # 'AUC':  [m.compute().item() for m in auc_ts],
        'PSD':  [m.compute()[0].item() for m in psd_ts],
        'RMSE': [m.compute().item() for m in rmse_ts],
        'CRPS': [m.compute().item() for m in crps_ts],
        'FSS':  [m.compute().item() for m in fss_ts],
        'MAE':  [m.compute().item() for m in mae_ts]
    }

    # 整体均值
    summary = {k: float(np.mean(v)) for k, v in metrics_per_t.items()}

    # 4) 打印与保存
    print("===== Per-Timestep Validation Metrics =====")
    for metric, vals in metrics_per_t.items():
        print(f"{metric}: ", ", ".join([f"{x:.3f}" for x in vals]))
        print(f"  mean {metric}: {summary[metric]:.3f}")

    # 保存为 pickle
    with open(os.path.join(res_path, 'metrics_per_t.pkl'), 'wb') as f:
        pickle.dump({'per_t': metrics_per_t, 'mean': summary}, f)

    print('finished!')
