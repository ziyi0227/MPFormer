import os
import datetime
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from cmweather import cm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.colors as mcolors

from mpformer.utils.metrics import CRPS, ReliabilityDiagram

def test_pytorch_loader_calib(model, test_input_handle, configs, itr):
    """
    仅计算 CRPS 与可靠性图，并将可靠性图保存为 PNG。
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    # 预测步数
    T_out = configs.total_length - configs.input_length

    # 只保留 CRPS 以及 可靠性图 指标
    crps_ts = [CRPS() for _ in range(T_out)]
    calib = [ReliabilityDiagram(num_bins=10) for _ in range(T_out)]

    for batch in test_input_handle:
        test_ims = batch['radar_frames'].numpy()
        preds    = model.test(test_ims)  # shape [B, T_out, H, W, C]
        gt       = test_ims[:, configs.input_length:, ...]
        
        pr_t = torch.from_numpy(preds)[..., 0:1]   # 只用第1通道概率
        gt_t = torch.from_numpy(gt)[..., 0:1]
        # 针对 CRPS 需要连续值，ReliabilityDiagram 需要二值化概率
        bin_gt = (gt_t > 0.5).int()

        for t in range(T_out):
            pr_step = pr_t[:, t:t+1, ...]
            gt_step = gt_t[:, t:t+1, ...]
            crps_ts[t].update(pr_step, gt_step)
            calib[t].update(pr_step, bin_gt[:, t:t+1, ...])

    # compute 并保存
    crps_per_t = [m.compute().item() for m in crps_ts]
    # 可靠性图数据
    cal_data = [m.compute() for m in calib]  # 每个元素 (avg_pred, freq_true)

    # 保存 CRPS
    with open(os.path.join(res_path, 'crps_per_t.pkl'), 'wb') as f:
        pickle.dump(crps_per_t, f)

    # 绘制并保存 可靠性图
    for t, (avg_p, freq_t) in enumerate(cal_data, start=1):
        plt.figure(figsize=(4,4))
        plt.plot([0,1], [0,1], 'k--', label='Perfect')
        plt.plot(avg_p.numpy(), freq_t.numpy(), 'o-', label=f'Lead {t}')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Observed frequency')
        plt.title(f'Reliability Diagram (t+{t})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(res_path, f'reliability_t{t}.png'))
        plt.close()

    # 打印 CRPS
    print("CRPS per lead time:", ", ".join([f"{x:.3f}" for x in crps_per_t]))
    print('finished!')

def test_pytorch_loader(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)

    for batch_id, test_ims in enumerate(test_input_handle):

        test_ims = test_ims['radar_frames'].numpy()
        img_gen = model.test(test_ims)
        output_length = configs.total_length - configs.input_length

        def save_plots(field, labels, res_path, figsize=None,
                       vmin=0, vmax=10, cmap="viridis", npy=False, **imshow_args):

            for i, data in enumerate(field):
                fig = plt.figure(figsize=figsize)
                ax = plt.axes()
                ax.set_axis_off()
                alpha = data[..., 0] / 1
                alpha[alpha < 1] = 0
                alpha[alpha > 1] = 1

                img = ax.imshow(data[..., 0], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                plt.savefig('{}/{}.png'.format(res_path, labels[i]))
                plt.close()  
                if npy:
                    with open( '{}/{}.npy'.format(res_path, labels[i]), 'wb') as f:
                        np.save(f, data[..., 0])


        data_vis_dict = {
            'radar': {'vmin': 1, 'vmax': 40},
        }
        vis_info = data_vis_dict[configs.dataset_name]

        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            if configs.case_type == 'normal':
                test_ims_plot = test_ims[0][:-2, 256-192:256+192, 256-192:256+192]
                img_gen_plot = img_gen[0][:-2, 256-192:256+192, 256-192:256+192]
            else:
                test_ims_plot = test_ims[0][:-2]
                img_gen_plot = img_gen[0][:-2]
            save_plots(test_ims_plot,
                       labels=['gt{}'.format(i + 1) for i in range(configs.total_length)],
                       res_path=path, vmin=vis_info['vmin'], vmax=vis_info['vmax'])
            save_plots(img_gen_plot,
                       labels=['pd{}'.format(i + 1) for i in range(9, configs.total_length)],
                       res_path=path, vmin=vis_info['vmin'], vmax=vis_info['vmax'])

    print('finished!')

def test_pytorch_loader2(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    def save_plots(field, labels, res_path, figsize=None,
                   vmin=0, vmax=10, cmap="viridis", npy=False, **imshow_args):

        for i, data in enumerate(field):
            arr = data[..., 0]
            # 对雨量数据做高斯模糊以美化边缘
            blurred = gaussian_filter(arr, sigma=1.0)

            # 根据原始方法计算 alpha
            alpha = arr.copy()
            alpha[alpha < 1] = 0
            alpha[alpha >= 1] = 1

            fig, ax = plt.subplots(figsize=figsize)
            ax.set_axis_off()
            ax.imshow(
                blurred,
                alpha=alpha,
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                interpolation='bicubic',  # 双三次插值，让图像更平滑
                **imshow_args
            )

            # 保存图像
            filepath = os.path.join(res_path, f'{labels[i]}.png')
            fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # 可选保存 numpy 数据
            if npy:
                np.save(os.path.join(res_path, f'{labels[i]}.npy'), arr)

    data_vis_dict = {
        'radar': {'vmin': 1, 'vmax': 40},
    }
    vis_info = data_vis_dict.get(configs.dataset_name, {'vmin': 0, 'vmax': 10})

    for batch_id, test_ims in enumerate(test_input_handle):

        test_ims = test_ims['radar_frames'].numpy()
        img_gen = model.test(test_ims)

        if batch_id <= configs.num_save_samples:
            batch_dir = os.path.join(res_path, str(batch_id))
            os.makedirs(batch_dir, exist_ok=True)

            if configs.case_type == 'normal':
                gt_seq = test_ims[0][:-2, 256-192:256+192, 256-192:256+192]
                pd_seq = img_gen[0][:-2, 256-192:256+192, 256-192:256+192]
            else:
                gt_seq = test_ims[0][:-2]
                pd_seq = img_gen[0][:-2]

            # 保存 Ground Truth 序列
            save_plots(
                gt_seq,
                labels=[f'gt{i+1}' for i in range(configs.total_length)],
                res_path=batch_dir,
                vmin=vis_info['vmin'], vmax=vis_info['vmax'],
            )
            # 保存预测序列
            save_plots(
                pd_seq,
                labels=[f'pd{i+1}' for i in range(configs.input_length+1, configs.total_length+1)],
                res_path=batch_dir,
                vmin=vis_info['vmin'], vmax=vis_info['vmax'],
            )

    print('finished!')