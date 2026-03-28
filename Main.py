import re
import time
import imageio
import torch.cuda
import torch.optim as optim
import os
import random
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from Dataset import *
from Net import *
from utils.Sundries import *
from utils.Mainloss_manage import *


def train_model(model, train_loader, device, num_epochs=50, save_dir='data/results', num_frames=5,
                val_loader=None):
    optimizer = optim.Adam([
        {'params': model.optical_flow_model.parameters(), 'lr': 1e-4},
        {'params': model.object_reconstructor.parameters(), 'lr': 1e-4},
    ])
    criterion = CombinedLoss(mode='train')

    train_losses = defaultdict(list)
    best_loss = float('inf')
    val_flow_mag_history = {'fw': [], 'bw': []}
    val_epe_history = {'fw': [], 'bw': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            object_losses = []

            speckle_seq = batch['speckle_seq'].to(device)  # [T, C, H, W]
            object_seq = batch['object_seq'].to(device)  # [T, C, H, W]
            flow_seq = batch['flow_seq'].to(device)
            t_random = random.randint(0, num_frames - 1)

            optimizer.zero_grad()

            output = model(speckle_seq)
            target = {
                'flow': flow_seq.to(device),
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }

            loss, loss_dict = criterion(output, target, epoch, t_random)
            object_losses.append(loss_dict)

            visualize_debug_images(output, target, epoch, batch_idx, save_dir, t_random, step=15, save_every=250)

            loss.backward()
            optimizer.step()

            # 处理批次损失
            batch_losses = process_batch_losses(object_losses, epoch_losses)
            # 生成并打印批次损失摘要
            batch_losses_summary = batch_loss_summary(epoch_losses, batch_losses, batch_idx)
            print(format_batch_loss_table(epoch, num_epochs, batch_idx, len(train_loader), batch_losses_summary))

        # 更新周期损失并获取归一化值
        normalized_epoch_losses = update_epoch_losses(epoch_losses, len(train_loader))
        # 更新训练损失用于绘图
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])
        # 保存最佳模型
        best_loss = save_best_model(model, normalized_epoch_losses, best_loss, save_dir, epoch)
        # 记录周期结果
        log_epoch_results(epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir)
        # 绘制损失曲线
        plot_losses(train_losses, save_dir)

        # ── 验证集光流均值 ──────────────────────────────
        if val_loader is not None:
            fw_mag, bw_mag, fw_epe, bw_epe = validate_metrics(model, val_loader, device)
            val_flow_mag_history['fw'].append(fw_mag)
            val_flow_mag_history['bw'].append(bw_mag)
            val_epe_history['fw'].append(fw_epe)
            val_epe_history['bw'].append(bw_epe)
            print(f"[Epoch {epoch + 1}/{num_epochs}] "
                  f"Magnitude FW: {fw_mag:.4f}  BW: {bw_mag:.4f} | "
                  f"EPE FW: {fw_epe:.4f}  BW: {bw_epe:.4f}")
            plot_flow_magnitude(val_flow_mag_history, save_dir)
            plot_val_epe(val_epe_history, save_dir)




def test_model(model, test_loader, device, save_dir='data/results'):
    # 创建保存目录
    directories = {
        'flow_arrow_fw': os.path.join(save_dir, 'flowdata/flow_arrow_fw'),
        'flow_colorimage_fw': os.path.join(save_dir, 'flowdata/flow_colorimage_fw'),
        'flow_arrow_bw': os.path.join(save_dir, 'flowdata/flow_arrow_bw'),
        'flow_colorimage_bw': os.path.join(save_dir, 'flowdata/flow_colorimage_bw'),
        'gt_flow_arrow_fw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_fw'),
        'gt_flow_colorimage': os.path.join(save_dir, 'flowdata/gt_flow_colorimage'),
        'gt_flow_arrow_bw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_bw'),
        'gt_flow_colorimage_bw': os.path.join(save_dir, 'flowdata/gt_flow_colorimage_bw'),
        'object_flow_arrow_fw': os.path.join(save_dir, 'flowdata/object_flow_arrow_fw'),
        'object_flow_colorimage_fw': os.path.join(save_dir, 'flowdata/object_flow_colorimage_fw'),
        'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
        'groundtruth_fw': os.path.join(save_dir, 'flowdata/groundtruth_fw'),
        'origin_object': os.path.join(save_dir, 'origin_object'),
        'recon_object': os.path.join(save_dir, 'recon_object'),
        'diff_recon_vs_gt':os.path.join(save_dir, 'diff_recon_vs_gt'),
        'overlay_results_origin_object': os.path.join(save_dir, 'overlay_results_origin_object'),
        'overlay_results_nl_origin_object': os.path.join(save_dir, 'overlay_results_nl_origin_object'),
        'overlay_results_recon_object': os.path.join(save_dir, 'overlay_results_recon_object'),
        'overlay_results_nl_recon_object': os.path.join(save_dir, 'overlay_results_nl_recon_object'),
        'diff_results_recon_obj_origin_obj': os.path.join(save_dir, 'diff_results_recon_obj_origin_obj'),
        'warp_from_each_t_nl_overlay': os.path.join('data/results/newcode', 'warp_from_each_t_nl_overlay'),
        'warp_from_each_t_overlay': os.path.join('data/results/newcode', 'warp_from_each_t_overlay'),
        'overlay_gt_each_t': os.path.join('data/results/newcode', 'overlay_gt_each_t'),
        'diff_overlay_each_t': os.path.join('data/results/newcode', 'diff_overlay_each_t'),
        'single_overlay_gt': os.path.join('data/results/newcode', 'single_overlay_gt'),
        'single_overlay_unet': os.path.join('data/results/newcode', 'single_overlay_unet'),
        'single_overlay_warp': os.path.join('data/results/newcode', 'single_overlay_warp'),
        'single_overlay_nl_warp': os.path.join('data/results/newcode', 'single_overlay_nl_warp'),
        'single_overlay_nl_nc_warp': os.path.join('data/results/newcode', 'single_overlay_nl_nc_warp'),
        'single_overlay_nl_gt': os.path.join('data/results/newcode', 'single_overlay_nl_gt'),
        'single_overlay_nl_unet': os.path.join('data/results/newcode', 'single_overlay_nl_unet'),
        'single_overlay_nl_warp_all_t': os.path.join('data/results/newcode', 'single_overlay_nl_warp_all_t'),
        'diff_single_each_t': os.path.join('data/results/newcode', 'diff_single_each_t'),
        'flow_fw_diff_each': os.path.join('data/results/newcode', 'flow_fw_diff_each'),
        'flow_bw_diff_each': os.path.join('data/results/newcode', 'flow_bw_diff_each')
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    # 定义需要清空的CSV文件路径
    csv_files = [
        os.path.join(save_dir, "test_batch_losses.csv"),
        os.path.join(save_dir, "test_batch_losses_warp.csv"),
        os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
        os.path.join(save_dir, "test_batch_losses_warp_psnr.csv")
    ]
    # 清空文件
    clear_csv_files(csv_files)

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses = defaultdict(float)
    remove_existing_test_logs(save_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)  # [T, C, H, W]
            object_seq = batch['object_seq'].to(device)  # [T, C, H, W]
            flow_seq = batch['flow_seq'].to(device)  # [T-1, 2, H, W]

            target = {
                'flow': flow_seq,
                'object': object_seq,
                'speckle': speckle_seq,
            }

            output = model(speckle_seq, speckle_for_unet=None, test=True)

            # 保存可视化结果
            save_all_results(batch_idx, output, target, directories)

            # 直接一次性计算 loss（不分帧）
            _, loss_dict = criterion(output, target)
            test_losses = process_test_frame_losses(test_losses, loss_dict)

            # 打印当前batch损失（不分帧）
            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

            # --- 写到 CSV ---
            append_test_csv(
                os.path.join(save_dir, "test_batch_losses.csv"),
                batch_idx,
                loss_dict,
                loss_dict.get("speckle_object_items", []),
                loss_dict.get("object_warp_items", [])
            )
            append_warp_csv(
                os.path.join(save_dir, "test_batch_losses_warp.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )
            append_warp_csv1(
                os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )
            append_warp_csv2(
                os.path.join(save_dir, "test_batch_losses_warp_psnr.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )

    # 处理测试结果并获取归一化值
    normalized_test_losses = process_test_results(test_losses, len(test_loader))
    # 记录测试结果
    log_test_results(normalized_test_losses, save_dir)

    save_path = os.path.join(save_dir, "test_metrics_summary.txt")
    with open(save_path, "w") as f:
        # ===== 所有 batch 的 per-object 平均 MSE =====
        if criterion.object_mse_sum is not None:
            mean_object_mse = (criterion.object_mse_sum / criterion.object_mse_count).tolist()
            print("=== Mean per-object MSE over ALL test batches ===")
            for i, v in enumerate(mean_object_mse):
                print(f"Object {i}: {v:.6f}")
            # 可选：保存为 txt / csv，方便画论文图
            write_metric_block(f, title="Mean per-object MSE over ALL test batches",
                values=mean_object_mse, prefix="Object",
            )
        # ===== 所有 batch 的 per-flow mean EPE =====
        if criterion.flow_fw_epe_sum is not None:
            mean_flow_fw_epe = (
                criterion.flow_fw_epe_sum / torch.clamp(criterion.flow_fw_epe_count, min=1)
            ).tolist()
            print("=== Mean per-flow FORWARD EPE over ALL test batches ===")
            for i, v in enumerate(mean_flow_fw_epe):
                print(f"Flow FW {i}: {v:.6f}")
            write_metric_block(f, title="Mean per-flow FORWARD EPE over ALL test batches",
                values=mean_flow_fw_epe, prefix="Flow FW",)
        if criterion.flow_bw_epe_sum is not None:
            mean_flow_bw_epe = (
                criterion.flow_bw_epe_sum / torch.clamp(criterion.flow_bw_epe_count, min=1)
            ).tolist()
            print("=== Mean per-flow BACKWARD EPE over ALL test batches ===")
            for i, v in enumerate(mean_flow_bw_epe):
                print(f"Flow BW {i}: {v:.6f}")
            write_metric_block(f, title="Mean per-flow BACKWARD EPE over ALL test batches",
                values=mean_flow_bw_epe, prefix="Flow BW",)


def train_simple(model, train_loader, device, num_epochs=50, save_dir='data/results'):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SimpleLoss()

    train_losses = defaultdict(list)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        for batch_idx, batch in enumerate(train_loader):
            object_losses = []
            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)

            optimizer.zero_grad()
            outputs = model(speckle_seq)
            target = {
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }
            loss, loss_dict = criterion(outputs, target)
            object_losses.append(loss_dict)

            # visualize_debug_images(outputs, target, epoch, batch_idx, save_dir, t_random, step=15, save_every=500)
            loss.backward()
            optimizer.step()

            # 处理批次损失
            batch_losses = process_batch_losses(object_losses, epoch_losses)
            # 生成并打印批次损失摘要
            batch_losses_summary = batch_loss_summary(epoch_losses, batch_losses, batch_idx)
            print(format_batch_loss_table(epoch, num_epochs, batch_idx, len(train_loader), batch_losses_summary))

        # 更新周期损失并获取归一化值
        normalized_epoch_losses = update_epoch_losses(epoch_losses, len(train_loader))
        # 更新训练损失用于绘图
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])
        # 保存最佳模型
        best_loss = save_best_model(model, normalized_epoch_losses, best_loss, save_dir, epoch)
        # 记录周期结果
        log_epoch_results(epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir)
        # 绘制损失曲线
        plot_losses(train_losses, save_dir)


def test_simple(model, test_loader, device, save_dir='data/results'):
    # 创建保存目录
    directories = {
        'origin_object1': os.path.join(save_dir, 'origin_object1'),
        'origin_object2': os.path.join(save_dir, 'origin_object2'),
        'recon_object2': os.path.join(save_dir, 'recon_object2'),
        'recon_object1': os.path.join(save_dir, 'recon_object1'),
        'gifs_object1': os.path.join(save_dir, 'gifs_object1'),
        'gifs_object2': os.path.join(save_dir, 'gifs_object2'),
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses = defaultdict(float)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)  # [T, C, H, W]
            object_seq = batch['object_seq'].to(device)  # [T, C, H, W]

            output = model(speckle_seq)
            target = {
                'object': object_seq,
                'speckle': speckle_seq,
            }

            # 保存可视化结果
            save_all_results(batch_idx, output, target, directories)

            # 直接一次性计算 loss（不分帧）
            _, loss_dict = criterion(output, target)
            test_losses = process_test_frame_losses(test_losses, loss_dict)

            # 打印当前batch损失（不分帧）
            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

    # 处理测试结果并获取归一化值
    normalized_test_losses = process_test_results(test_losses, len(test_loader))
    # 记录测试结果
    log_test_results(normalized_test_losses, save_dir)


def finetune_model(model, finetune_loader, device, num_epochs=10, save_dir='data/finetune_results', num_frames=5):
    optimizer = optim.Adam([
        {'params': model.optical_flow_model.parameters(), 'lr': 1e-6},  # 非常小
        {'params': model.object_reconstructor.parameters(), 'lr': 1e-5},
    ])

    criterion = CombinedLoss(mode='train')

    train_losses = defaultdict(list)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        for batch_idx, batch in enumerate(finetune_loader):
            object_losses = []

            speckle_seq = batch['speckle_seq'].to(device)  # [T, C, H, W]
            object_seq = batch['object_seq'].to(device)  # [T, C, H, W]
            t_random = random.randint(0, num_frames - 1)

            optimizer.zero_grad()

            output = model(speckle_seq)
            target = {
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }

            loss, loss_dict = criterion(output, target, epoch, t_random)
            object_losses.append(loss_dict)

            visualize_debug_images(output, target, epoch, batch_idx, save_dir, t_random, step=15,
                                   save_every=500)

            loss.backward()
            optimizer.step()

            # 处理批次损失
            batch_losses = process_batch_losses(object_losses, epoch_losses)
            # 生成并打印批次损失摘要
            batch_losses_summary = batch_loss_summary(epoch_losses, batch_losses, batch_idx)
            print(format_batch_loss_table(epoch, num_epochs, batch_idx, len(finetune_loader), batch_losses_summary))

        # 更新周期损失并获取归一化值
        normalized_epoch_losses = update_epoch_losses(epoch_losses, len(finetune_loader))
        # 更新训练损失用于绘图
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])
        # 保存最佳模型
        best_loss = save_best_model(model, normalized_epoch_losses, best_loss, save_dir, epoch)
        # 记录周期结果
        log_epoch_results(epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir)
        # 绘制损失曲线
        plot_losses(train_losses, save_dir)


# def test_experimental_data(model, data_loader, device, save_dir='data/experimental_results'):
#     directories = {
#         'speckle1': os.path.join(save_dir, 'speckle1'),
#         'speckle2': os.path.join(save_dir, 'speckle2'),
#         'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
#         'flow_colorimage': os.path.join(save_dir, 'flowdata/flow_colorimage'),
#         'flow_arrow': os.path.join(save_dir, 'flowdata/flow_arrow'),
#         'reconstructed_object1': os.path.join(save_dir, 'reconstructed_object1'),
#         'reconstructed_object2': os.path.join(save_dir, 'reconstructed_object2'),
#     }
#
#     for dir_path in directories.values():
#         os.makedirs(dir_path, exist_ok=True)
#
#     model.eval()
#
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(data_loader):
#             speckle_seq = batch['speckle_seq'].to(device)  # [T, 1, H, W]
#             speckle_raw_seq = batch['speckle_raw_seq'].to(device)
#
#             output = model(speckle_raw_seq, speckle_seq)  # 返回的 output 包含 reconstructed_object1 / 2
#
#             # Save results
#             save_experimental_results(batch_idx, output, speckle_seq, directories)
#
#             print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")


def test_experimental_data(model, data_loader, device, save_dir='data/experimental_results'):
    directories = {
        'speckle1': os.path.join(save_dir, 'speckle1'),
        'speckle2': os.path.join(save_dir, 'speckle2'),
        'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
        'flow_colorimage': os.path.join(save_dir, 'flowdata/flow_colorimage'),
        'flow_arrow': os.path.join(save_dir, 'flowdata/flow_arrow'),
        'reconstructed_object1': os.path.join(save_dir, 'reconstructed_object1'),
        'reconstructed_object2': os.path.join(save_dir, 'reconstructed_object2'),
    }

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            speckle_seq = batch['speckle_seq'].to(device)
            speckle_raw_seq = batch['speckle_raw_seq'].to(device)

            output = model(speckle_raw_seq, speckle_seq)  # 返回的 output 包含 reconstructed_object1 / 2

            # Save results
            save_experimental_results(batch_idx, output, speckle_seq, directories)

            print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")


            # crop_x = batch.get('crop_x', 0)
            # crop_y = batch.get('crop_y', 0)
            #
            # # 对每个序列分别处理
            # num_sequences = speckle_seq.shape[0]
            # for seq_idx in range(num_sequences):
            #     seq_speckle = speckle_seq[seq_idx]  # [frames_per_sequence, 1, H, W]
            #     seq_speckle_raw = speckle_raw_seq[seq_idx]  # [frames_per_sequence, 1, H, W]
            #     output = model(seq_speckle_raw, seq_speckle)
            #
            #     # 保存结果，添加序列索引
            #     save_experimental_results(
            #         batch_idx, output, seq_speckle, directories,
            #         # crop_x, crop_y, seq_idx
            #     )
            #
            # print(
            #     f"Processed position {batch_idx + 1}/{len(data_loader)} (crop_x={crop_x}, crop_y={crop_y}) - {num_sequences} sequences")


def test_experimental_data_withobj(model, test_loader, device, save_dir='data/experiment_results_withobjandflow', num_frames=5):
    # 创建保存目录
    directories = {
        'speckle1': os.path.join(save_dir, 'speckle1'),
        'speckle2': os.path.join(save_dir, 'speckle2'),
        'flow_arrow_fw': os.path.join(save_dir, 'flowdata/flow_arrow_fw'),
        'flow_colorimage_fw': os.path.join(save_dir, 'flowdata/flow_colorimage_fw'),
        'flow_arrow_bw': os.path.join(save_dir, 'flowdata/flow_arrow_bw'),
        'flow_colorimage_bw': os.path.join(save_dir, 'flowdata/flow_colorimage_bw'),
        'gt_flow_arrow_fw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_fw'),
        'gt_flow_colorimage': os.path.join(save_dir, 'flowdata/gt_flow_colorimage'),
        'gt_flow_arrow_bw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_bw'),
        'gt_flow_colorimage_bw': os.path.join(save_dir, 'flowdata/gt_flow_colorimage_bw'),
        'object_flow_arrow_fw': os.path.join(save_dir, 'flowdata/object_flow_arrow_fw'),
        'object_flow_colorimage_fw': os.path.join(save_dir, 'flowdata/object_flow_colorimage_fw'),
        'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
        'groundtruth_fw': os.path.join(save_dir, 'flowdata/groundtruth_fw'),
        'origin_object': os.path.join(save_dir, 'origin_object'),
        'recon_object': os.path.join(save_dir, 'recon_object'),
        'diff_recon_vs_gt':os.path.join(save_dir, 'diff_recon_vs_gt'),
        'overlay_results_origin_object': os.path.join(save_dir, 'overlay_results_origin_object'),
        'overlay_results_nl_origin_object': os.path.join(save_dir, 'overlay_results_nl_origin_object'),
        'overlay_results_recon_object': os.path.join(save_dir, 'overlay_results_recon_object'),
        'overlay_results_nl_recon_object': os.path.join(save_dir, 'overlay_results_nl_recon_object'),
        'diff_results_recon_obj_origin_obj': os.path.join(save_dir, 'diff_results_recon_obj_origin_obj'),
        'diff_results_warp_origin_obj': os.path.join(save_dir, 'diff_results_warp_origin_obj'),
        'warp_from_each_t_nl_overlay': os.path.join('data/experiment_results_withobjandflow/newcode', 'warp_from_each_t_nl_overlay'),
        'warp_from_each_t_overlay': os.path.join('data/experiment_results_withobjandflow/newcode', 'warp_from_each_t_overlay'),
        'overlay_gt_each_t': os.path.join('data/experiment_results_withobjandflow/newcode', 'overlay_gt_each_t'),
        'diff_overlay_each_t': os.path.join('data/experiment_results_withobjandflow/newcode', 'diff_overlay_each_t'),
        'single_overlay_gt': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_gt'),
        'single_overlay_unet': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_unet'),
        'single_overlay_warp': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_warp'),
        'single_overlay_nl_warp': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_nl_warp'),
        'single_overlay_nl_nc_warp': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_nl_nc_warp'),
        'single_overlay_nl_gt': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_nl_gt'),
        'single_overlay_nl_unet': os.path.join('data/experiment_results_withobjandflow/newcode', 'single_overlay_nl_unet'),
        'diff_single_each_t': os.path.join('data/experiment_results_withobjandflow/newcode', 'diff_single_each_t'),
        'flow_fw_diff_each': os.path.join('data/experiment_results_withobjandflow/newcode', 'flow_fw_diff_each'),
        'flow_bw_diff_each': os.path.join('data/experiment_results_withobjandflow/newcode', 'flow_bw_diff_each')
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses = defaultdict(float)
    remove_existing_test_logs(save_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)  # [T, C, H, W]
            speckle_raw_seq = batch['speckle_raw_seq'].to(device)
            object_seq = batch['object_seq'].to(device)  # [T, C, H, W]
            flow_seq = batch['flow_seq'].to(device)  # [T-1, 2, H, W]

            output = model(speckle_raw_seq, speckle_seq)
            output = rotate_output(output)

            target = {
                'flow': flow_seq,
                'object': object_seq,
                'speckle': speckle_seq,
            }

            # 保存可视化结果
            save_all_results(batch_idx, output, target, directories, True, True)

            # 直接一次性计算 loss（不分帧）
            _, loss_dict = criterion(output, target)
            test_losses = process_test_frame_losses(test_losses, loss_dict)

            # 打印当前batch损失（不分帧）
            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

            # --- 写到 CSV ---
            append_test_csv(
                os.path.join(save_dir, "test_batch_losses.csv"),
                batch_idx,
                loss_dict,
                loss_dict.get("speckle_object_items", []),
                loss_dict.get("object_warp_items", [])
            )
            append_warp_csv(
                os.path.join(save_dir, "test_batch_losses_warp.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )
            append_warp_csv1(
                os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )
            append_warp_csv2(
                os.path.join(save_dir, "test_batch_losses_warp_psnr.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t']
            )

        # 处理测试结果并获取归一化值
        normalized_test_losses = process_test_results(test_losses, len(test_loader))
        # 记录测试结果
        log_test_results(normalized_test_losses, save_dir)

        save_path = os.path.join(save_dir, "test_metrics_summary.txt")
        with open(save_path, "w") as f:
            # ===== 所有 batch 的 per-object 平均 MSE =====
            if criterion.object_mse_sum is not None:
                mean_object_mse = (criterion.object_mse_sum / criterion.object_mse_count).tolist()
                print("=== Mean per-object MSE over ALL test batches ===")
                for i, v in enumerate(mean_object_mse):
                    print(f"Object {i}: {v:.6f}")
                # 可选：保存为 txt / csv，方便画论文图
                write_metric_block(f, title="Mean per-object MSE over ALL test batches",
                                   values=mean_object_mse, prefix="Object",
                                   )
            # ===== 所有 batch 的 per-flow mean EPE =====
            if criterion.flow_fw_epe_sum is not None:
                mean_flow_fw_epe = (
                        criterion.flow_fw_epe_sum / torch.clamp(criterion.flow_fw_epe_count, min=1)
                ).tolist()
                print("=== Mean per-flow FORWARD EPE over ALL test batches ===")
                for i, v in enumerate(mean_flow_fw_epe):
                    print(f"Flow FW {i}: {v:.6f}")
                write_metric_block(f, title="Mean per-flow FORWARD EPE over ALL test batches",
                                   values=mean_flow_fw_epe, prefix="Flow FW", )
            if criterion.flow_bw_epe_sum is not None:
                mean_flow_bw_epe = (
                        criterion.flow_bw_epe_sum / torch.clamp(criterion.flow_bw_epe_count, min=1)
                ).tolist()
                print("=== Mean per-flow BACKWARD EPE over ALL test batches ===")
                for i, v in enumerate(mean_flow_bw_epe):
                    print(f"Flow BW {i}: {v:.6f}")
                write_metric_block(f, title="Mean per-flow BACKWARD EPE over ALL test batches",
                                   values=mean_flow_bw_epe, prefix="Flow BW", )


def main():
    start_time = time.time()

    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = 'data/datasets'
    pos = 'obj_128_bg_256_move_16'

    # 创建数据集和数据加载器
    train_dataset = SpeckleDataset_New(base_path, mode='train', pos=pos)
    test_dataset = SpeckleDataset_New(base_path, mode='test', pos=pos)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=12)
    val_dataset = SpeckleDataset_New(base_path, mode='val', pos=pos)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=12)

    # 创建模型
    model = CompleteModel().to(device)

    # ============ 新增：统计参数量 ============
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # ============ 训练模型 ============
    torch.cuda.reset_peak_memory_stats(device)  # 重置显存统计
    # train_model(model, train_loader, device, num_frames=5)
    train_model(model, train_loader, device, num_frames=5, val_loader=val_loader)

    # ============ 显存统计 ============
    if device.type == "cuda":
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"训练期间最大显存占用: {max_memory_allocated:.2f} MB")
    else:
        max_memory_allocated = 0

    # ============ 测试模型 ============
    # model.load_state_dict(torch.load('data/lunwen_sim_checkpoints/checkpoint_epoch_50.pth'))
    model.load_state_dict(torch.load('data/results/checkpoints/checkpoint_epoch_20.pth'))
    test_model(model, test_loader, device)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"代码运行总时间: {total_time / 60:.2f}分钟, {total_time / 3600:.2f}小时")

    # ============ 写入日志 ============
    log_file = os.path.join('data/results/result_log.csv')
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + f"运行配置: {pos}")
        log.write(f"\n模型总参数量: {total_params:,}")
        log.write(f"\n可训练参数量: {trainable_params:,}")
        log.write(f"\n最大显存占用: {max_memory_allocated:.2f} MB")
        log.write(f"\n运行总时间: {total_time / 60:.2f}分钟 ({total_time / 3600:.2f}小时)\n")

    # # ============ 备份代码 ============
    # current_path = os.path.dirname(os.path.abspath(__file__))
    # specific_files = ['Dataset.py', 'Main.py', 'Net_Unet.py', 'Net.py']
    # create_backup_zip(current_path, specific_files, pos)


def main_onlyunet():
    start_time = time.time()  # 记录开始时间

    # 创建设备, 数据集路径, 数据加载器和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = 'data/datasets'
    pos = 'obj_128_bg_256_move_16'
    train_dataset = SpeckleDataset_New(base_path, mode='train', pos=pos)
    test_dataset = SpeckleDataset_New(base_path, mode='test', pos=pos)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=12)
    model = SimpleReconstructionModel().to(device)

    # ============ 新增：统计参数量 ============
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    torch.cuda.reset_peak_memory_stats(device)  # 重置显存统计

    # 训练模型
    train_simple(model, train_loader, device, save_dir='data/results_onlyunet')

    # ============ 显存统计 ============
    if device.type == "cuda":
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"训练期间最大显存占用: {max_memory_allocated:.2f} MB")
    else:
        max_memory_allocated = 0

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('data/results_onlyunet/checkpoints/checkpoint_epoch_50.pth'))
    test_simple(model, test_loader, device, save_dir='data/results_onlyunet')

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"代码运行总时间: {total_time / 60:.2f}分钟, {total_time / 3600:.2f}小时")

    # 定义日志文件路径
    log_file = os.path.join('data/results_onlyunet/result_log.csv')
    # 打印最终结果
    with open(log_file, 'a', encoding = 'utf-8') as log:
        log.write("\n" + f"代码运行总时间: {total_time / 60:.2f}分钟, {total_time / 3600:.2f}小时")

    # current_path = os.path.dirname(os.path.abspath(__file__))
    # # 指定要备份的特定代码文件
    # specific_files = ['Dataset.py', 'Main.py', 'Net_Unet.py', 'Net.py']
    # create_backup_zip(current_path, specific_files, pos)


def finetune_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_speckle_data_dir = 'data/datasets/LCDMnist'
    experiment_object_data_dir = 'data/datasets/obj_128obj_movetimes4'
    dataset = SpeckleOnlySequenceDatasetWithObjectAndFlow(experiment_speckle_data_dir, experiment_object_data_dir)
    finetune_loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=4)
    model = CompleteModel().to(device)
    model.load_state_dict(torch.load('data/results/checkpoints/checkpoint_epoch_50.pth'))
    finetune_model(model, finetune_loader, device, num_epochs=150, save_dir='data/finetune_results', num_frames=5)


def main_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_speckle_data_dir = 'data/datasets/LCDSpeckle_1202'
    dataset = SpeckleOnlySequenceDataset(experiment_speckle_data_dir)
    test_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=12)
    model = CompleteModel().to(device)
    model.load_state_dict(torch.load('data/results/checkpoints/checkpoint_epoch_15.pth'))
    test_experimental_data(model, test_loader, device, save_dir='data/experiment_results')


# def main_experiment():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     experiment_speckle_data_dir = 'data/datasets/LCDMnist'
#
#     # 使用滑动窗口数据集
#     dataset = SpeckleOnlySequenceDataset(
#         experiment_speckle_data_dir,
#         # crop_size=896,
#         # target_size=256,
#         # step_size=32  # 可以调整步长，比如32会更密集
#     )
#
#     test_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=12)
#     model = CompleteModel().to(device)
#     model.load_state_dict(torch.load('data/results/best_model.pth'))
#     test_experimental_data(model, test_loader, device, save_dir='data/experiment_results')


def main_experiment_withobjandflow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_speckle_data_dir = 'data/datasets/LCDSpeckle_0630'
    experiment_object_data_dir = 'data/datasets/LCDobj_128_movetimes4_LeftUp10'
    experiment_flow_data_dir = 'data/datasets/flow_times4_0630'
    dataset = SpeckleOnlySequenceDatasetWithObjectAndFlow(experiment_speckle_data_dir, experiment_object_data_dir,
                                                          experiment_flow_data_dir)
    test_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=12)
    model = CompleteModel().to(device)
    model.load_state_dict(torch.load('data/lunwen_exp_checkpoints/checkpoint_epoch_15.pth'))
    test_experimental_data_withobj(model, test_loader, device, save_dir='data/experiment_results_withobjandflow', num_frames=5)


def run_all_checkpoints(
        checkpoint_dir='data/lunwen_exp_checkpoints/checkpoints',
        save_root='data/results_all_models',
        start_epoch=20,
        end_epoch=50,
):
    os.makedirs(save_root, exist_ok=True)
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    all_files = os.listdir(checkpoint_dir)
    epoch_files = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            ep = int(m.group(1))
            if start_epoch <= ep <= end_epoch:
                epoch_files.append((ep, f))
    epoch_files.sort()  # sort by epoch number
    print("找到模型文件：", epoch_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载测试集
    base_path = 'data/datasets'
    pos = 'obj_128_bg_256_move_16_psf_1024'
    test_dataset = SpeckleDataset_New(base_path, mode='test', pos=pos)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=12)
    for ep, file in epoch_files:
        print(f"\n== == == == == 开始测试epoch{ep} == == == == == ")
        # ⭐ 每个模型独立一个文件夹 ⭐
        save_dir = os.path.join(save_root, f"results_epoch_{ep}")
        os.makedirs(save_dir, exist_ok=True)
        # 加载模型
        model_path = os.path.join(checkpoint_dir, file)
        print(f"加载模型：{model_path}")
        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(model_path))
        # 测试并自动保存 csv
        test_model(model, test_loader, device, save_dir=save_dir)
        print(f"epoch {ep} 测试完毕，结果已保存到：{save_dir}")


def run_all_experiment_checkpoints(
        checkpoint_dir='data/lunwen_exp_checkpoints/checkpoints',
        save_root='data/experiment_results_withobjandflow_all',
        start_epoch=15,
        end_epoch=25
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ================================
    #  1. 加载实验数据（一次即可）
    # ================================
    experiment_speckle_data_dir = 'data/datasets/LCDSpeckle_0630'
    experiment_object_data_dir = 'data/datasets/LCDobj_128_movetimes4_LeftUp10'
    experiment_flow_data_dir = 'data/datasets/flow_times4_0630'
    dataset = SpeckleOnlySequenceDatasetWithObjectAndFlow(
        experiment_speckle_data_dir,
        experiment_object_data_dir,
        experiment_flow_data_dir
    )
    test_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=12)
    # ================================
    #  2. 搜索 checkpoint 列表
    # ================================
    import re
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    all_files = os.listdir(checkpoint_dir)
    epoch_files = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            ep = int(m.group(1))
            if start_epoch <= ep <= end_epoch:
                epoch_files.append((ep, f))
    epoch_files.sort()
    print("找到实验测试模型文件：", epoch_files)
    # ================================
    #  3. 依次加载 checkpoint 进行实验测试
    # ================================
    for ep, file in epoch_files:
        print(f"\n== == == == == 开始测试实验数据epoch{ep} == == == == == ")
        # 每个 epoch 一个独立保存目录
        save_dir = os.path.join(save_root, f"experiment_results_epoch_{ep}")
        os.makedirs(save_dir, exist_ok=True)
        # 加载模型
        model_path = os.path.join(checkpoint_dir, file)
        print(f"加载模型：{model_path}")
        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(model_path))
        # 调用你现有的测试（不需修改 test_experimental_data_withobj）
        test_experimental_data_withobj(
            model,
            test_loader,
            device,
            save_dir=save_dir,
            num_frames=5
        )
        print(f"epoch {ep} 实验测试完成！结果保存于：{save_dir}")


if __name__ == "__main__":
    main()
    # main_experiment_withobjandflow()
    # main_onlyunet()
    # finetune_experiment()
    # main_experiment()
    # run_all_checkpoints(
    #     checkpoint_dir='data/lunwen_exp_checkpoints/checkpoints',
    #     save_root='data/results_multi_models',
    #     start_epoch=15,
    #     end_epoch=20
    # )
    # run_all_experiment_checkpoints(
    #     checkpoint_dir='data/lunwen_exp_checkpoints/checkpoints',
    #     save_root='data/experiment_results_withobjandflow_all',
    #     start_epoch=15,
    #     end_epoch=25
    # )

