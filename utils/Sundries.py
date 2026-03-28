import csv
import os
from collections import defaultdict
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import torch.nn.functional as F
from torchvision.utils import save_image, flow_to_image
from torch import nn
from utils.Visual_utils import *
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# 总loss
class CombinedLoss(nn.Module):
    def __init__(self, mode='train'):
        super().__init__()
        self.mode = mode
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        if self.mode == 'test':
            self.seq_warp_sum = defaultdict(lambda: defaultdict(float))
            self.seq_warp_count = defaultdict(lambda: defaultdict(int))
            # ===== 新增：跨 batch 的 per-object 累计 =====
            self.object_mse_sum = None    # Tensor, shape [T]
            self.object_mse_count = 0     # batch 数
            # ===== 新增：flow fw / bw 的 per-flow 累计 =====
            self.flow_fw_epe_sum = None   # Tensor [T-1]
            self.flow_fw_epe_count = None
            self.flow_bw_epe_sum = None   # Tensor [T-1]
            self.flow_bw_epe_count = None

    def forward(self, outputs, groundtruth, epoch=None, t_speckle=None):
        losses = {}
        if self.mode == 'test':
            extra_logs = {}  # 新增：存放 list 类的数据
            if 'flow_forward' in outputs and 'flow' in groundtruth:
                flow_pred, flow_gt = outputs['flow_forward'], groundtruth['flow']
                eps = 1e-6
                losses['flow_fw_loss_epe'], extra_logs['each_flow_fw_epe'] = (
                    compute_flow_epe(flow_pred, flow_gt))

                # ===== 跨 batch 的 per-flow FW EPE 累计 =====
                per_flow_fw_epe = torch.tensor(
                    extra_logs['each_flow_fw_epe'],
                    device=flow_pred.device
                ).detach()
                valid_mask = torch.isfinite(per_flow_fw_epe)  # True where not inf / nan
                if self.flow_fw_epe_sum is None:
                    self.flow_fw_epe_sum = torch.zeros_like(per_flow_fw_epe)
                    self.flow_fw_epe_count = torch.zeros_like(per_flow_fw_epe)
                self.flow_fw_epe_sum[valid_mask] += per_flow_fw_epe[valid_mask]
                self.flow_fw_epe_count[valid_mask] += 1

                pred_norm = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + eps)  # [B,H,W]
                gt_norm = torch.sqrt(torch.sum(flow_gt ** 2, dim=1) + eps)
                dot = torch.sum(flow_pred * flow_gt, dim=1)  # [B,H,W]
                cos_theta = dot / (pred_norm * gt_norm + eps)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值不稳定
                angle_rad = torch.acos(cos_theta)  # [B,H,W] 单位：弧度
                angle_deg = angle_rad * 180.0 / 3.1415926
                angle_loss = angle_deg.mean() / 180.0  # 归一化到 0~1
                losses['flow_fw_loss_angle'] = angle_loss

                losses['flow_fw_fl_all'] = compute_flow_fl_all(flow_pred, flow_gt)
                losses['flow_fw_1px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 1.0)
                losses['flow_fw_3px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 3.0)
                losses['flow_fw_5px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 5.0)

            if 'flow_backward' in outputs and 'flow' in groundtruth:
                flow_pred, flow_gt = outputs['flow_backward'], -groundtruth['flow']
                eps = 1e-6
                losses['flow_bw_loss_epe'], extra_logs['each_flow_bw_epe'] = (
                    compute_flow_epe(flow_pred, flow_gt))

                # ===== 跨 batch 的 per-flow BW EPE 累计 =====
                per_flow_bw_epe = torch.tensor(
                    extra_logs['each_flow_bw_epe'],
                    device=flow_pred.device
                ).detach()
                valid_mask = torch.isfinite(per_flow_bw_epe)
                if self.flow_bw_epe_sum is None:
                    self.flow_bw_epe_sum = torch.zeros_like(per_flow_bw_epe)
                    self.flow_bw_epe_count = torch.zeros_like(per_flow_bw_epe)
                self.flow_bw_epe_sum[valid_mask] += per_flow_bw_epe[valid_mask]
                self.flow_bw_epe_count[valid_mask] += 1

                pred_norm = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + eps)  # [B,H,W]
                gt_norm = torch.sqrt(torch.sum(flow_gt ** 2, dim=1) + eps)
                dot = torch.sum(flow_pred * flow_gt, dim=1)  # [B,H,W]
                cos_theta = dot / (pred_norm * gt_norm + eps)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值不稳定
                angle_rad = torch.acos(cos_theta)  # [B,H,W] 单位：弧度
                angle_deg = angle_rad * 180.0 / 3.1415926
                angle_loss = angle_deg.mean() / 180.0  # 归一化到 0~1
                losses['flow_bw_loss_angle'] = angle_loss

                losses['flow_bw_fl_all'] = compute_flow_fl_all(flow_pred, flow_gt)
                losses['flow_bw_1px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 1.0)
                losses['flow_bw_3px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 3.0)
                # losses['flow_bw_5px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 5.0)

            if 'reconstructed_object' in outputs and 'object' in groundtruth:
                losses['speckle_object_loss'] = self.mse_loss(outputs['reconstructed_object'], groundtruth['object'])
                # losses['speckle_object_ssim_loss'] = (1 - ms_ssim(outputs['reconstructed_object'], groundtruth['object'], data_range=1.0))
                pred = outputs['reconstructed_object']
                gt = groundtruth['object']
                mask = (gt > -1).float()
                ssim_val = masked_ssim(pred, gt, mask, data_range=1.0)
                psnr_val = masked_psnr(pred, gt, mask, data_range=1.0)
                losses['speckle_object_ssim'] = ssim_val
                losses['speckle_object_psnr'] = psnr_val

            if 'reconstructed_object' in outputs and 'object' in groundtruth:
                # [T,1,H,W] -> [T]
                diff = outputs['reconstructed_object'] - groundtruth['object']
                per_obj_mse = diff.pow(2).mean(dim=(1, 2, 3))  # Tensor, shape [T]
                # ---------- 1️ 当前 batch 的 per-object（你原来就有的） ----------
                extra_logs['speckle_object_items'] = per_obj_mse.tolist()
                # ---------- 2️ 跨所有 batch 的累计（新增的） ----------
                per_obj_mse_detached = per_obj_mse.detach()
                if self.object_mse_sum is None:
                    self.object_mse_sum = per_obj_mse_detached.clone()
                else:
                    self.object_mse_sum += per_obj_mse_detached
                self.object_mse_count += 1

            if 'flow_forward' in outputs:
                warp_all_refs = {}
                T = outputs['reconstructed_object'].shape[0]
                for ref_t in range(T):
                    # tgt, mse = compute_warp_losses_from_ref(
                    #     outputs, groundtruth, ref_t
                    # )
                    # warp_all_refs[ref_t] = {
                    #     'target_t': tgt,
                    #     'mse': mse,
                    # }
                    warp_metrics = compute_warp_losses_from_ref(outputs, groundtruth, ref_t)
                    warp_all_refs[ref_t] = warp_metrics
                extra_logs['object_warp_from_each_t'] = warp_all_refs

        elif self.mode == 'train':
            if 'warped_speckle1' in outputs and 'speckle' in groundtruth:
                # losses['speckle1_warp_loss'] = self.mse_loss(
                #     outputs['warped_speckle1'], groundtruth['speckle'][:-1])
                losses['speckle1_warp_loss'] = Charbonnier_loss(
                    outputs['warped_speckle1'], groundtruth['speckle'][:-1])
                if epoch >= 2:
                    # 取非遮挡区域
                    valid_mask = 1.0 - outputs['fwd_occ']  # [B, 1, H, W]
                    diff = (outputs['warped_speckle1'] - groundtruth['speckle'][:-1]) ** 2
                    loss_warp1 = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                    losses['speckle1_valid_region_loss'] = loss_warp1

            if 'warped_speckle2' in outputs and 'speckle' in groundtruth:
                # losses['speckle2_warp_loss'] = self.mse_loss(
                #     outputs['warped_speckle2'], groundtruth['speckle'][1:])
                losses['speckle2_warp_loss'] = Charbonnier_loss(
                    outputs['warped_speckle2'], groundtruth['speckle'][1:])
                if epoch >= 2:
                    # 取非遮挡区域
                    valid_mask = 1.0 - outputs['bwd_occ']  # [B, 1, H, W]
                    diff = (outputs['warped_speckle2'] - groundtruth['speckle'][1:]) ** 2
                    loss_warp2 = (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                    losses['speckle2_valid_region_loss'] = loss_warp2

            if 'reconstructed_object' in outputs and 'object' in groundtruth:
                recon_object_t = outputs['reconstructed_object'][t_speckle].unsqueeze(0)
                gt_object_t = groundtruth['object'][t_speckle].unsqueeze(0)
                # 基础的重建误差
                losses['speckle_object_mse_loss'] = self.mse_loss(recon_object_t, gt_object_t)

                num_frames = len(outputs['reconstructed_object'])
                object_warp_loss = 0.0
                object_cycle_loss = 0.0
                count = 0

                if epoch >= 3:
                    # ---------- 前向方向（warp到过去） ----------
                    if t_speckle > 0:
                        warped_recon_fw = recon_object_t.clone()
                        warped_gt_fw = gt_object_t.clone()
                        for step in range(1, t_speckle + 1):  # e.g., t=2 时 -> step=1,2
                            flow_fw = outputs['flow_forward'][t_speckle - step].unsqueeze(0)
                            warped_recon_fw = warp(warped_recon_fw, flow_fw)
                            warped_gt_fw = warp(warped_gt_fw, flow_fw)
                            # 对应过去的真实帧索引
                            target_idx = t_speckle - step
                            object_warp_loss += self.mse_loss(warped_recon_fw, warped_gt_fw)
                            # object_cycle_loss += self.mse_loss(
                            #     warped_recon_fw, outputs['reconstructed_object'][target_idx].unsqueeze(0)
                            # )
                            count += 1

                    # ---------- 反向方向（warp到未来） ----------
                    if t_speckle < num_frames - 1:
                        warped_recon_bw = recon_object_t.clone()
                        warped_gt_bw = gt_object_t.clone()
                        for step in range(1, num_frames - t_speckle):  # e.g., t=2 时 -> step=1,2
                            flow_bw = outputs['flow_backward'][t_speckle + step - 1].unsqueeze(0)
                            warped_recon_bw = warp(warped_recon_bw, flow_bw)
                            warped_gt_bw = warp(warped_gt_bw, flow_bw)
                            target_idx = t_speckle + step
                            object_warp_loss += self.mse_loss(warped_recon_bw, warped_gt_bw)
                            # object_cycle_loss += self.mse_loss(
                            #     warped_recon_bw, outputs['reconstructed_object'][target_idx].unsqueeze(0)
                            # )
                            count += 1
                    # 平均化loss，避免帧数不平衡
                    if count > 0:
                        losses['object_warp_loss'] = object_warp_loss / count
                        # losses['object_cycle_loss'] = object_cycle_loss / count

        total = sum(losses.values())
        losses['total_loss'] = total

        if self.mode == 'test':
            # 将 list 项加入结果（但不参与 sum）
            for k, v in extra_logs.items():
                losses[k] = v
            # 返回：Tensor 用 item()，list 原样返回
            return total, {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()}
        else:
            return total, {k: v.item() for k, v in losses.items()}


def save_simple_results(batch_idx, output, target, directories):
    recon_objects = output['reconstructed_object']  # [T, C, H, W]
    object_seq = target['object']                   # [T, C, H, W]
    T = recon_objects.shape[0]
    for t in range(T):
        save_image(recon_objects[t:t+1],
                   os.path.join(directories['recon_object1'], f'batch{batch_idx:04d}_t{t}.png'))
        save_image(object_seq[t:t+1],
                   os.path.join(directories['origin_object1'], f'batch{batch_idx:04d}_t{t}.png'))

# def masked_ssim(pred, gt, mask, data_range=1.0):
#     """
#     pred, gt, mask: [T,1,H,W]
#     """
#     eps = 1e-6
#     pred_m = pred * mask
#     gt_m = gt * mask
#     # 防止全零 mask
#     valid = mask.sum(dim=(1,2,3)) > 0
#     ssim_vals = []
#     for t in range(pred.shape[0]):
#         if valid[t]:
#             ssim_t = ssim(
#                 pred_m[t:t+1],
#                 gt_m[t:t+1],
#                 data_range=data_range,
#                 size_average=True
#             )
#             ssim_vals.append(ssim_t)
#     if len(ssim_vals) == 0:
#         return torch.tensor(0.0, device=pred.device)
#     return torch.stack(ssim_vals).mean()


# def masked_psnr(pred, gt, mask, data_range=1.0):
#     mse = ((pred - gt) ** 2) * mask
#     mse = mse.sum() / (mask.sum() + 1e-6)
#     return 10 * torch.log10(data_range**2 / (mse + 1e-8))


def masked_ssim(pred, gt, mask, data_range=1.0):
    """
    pred, gt, mask: [T, 1, H, W]  (torch.Tensor)
    行为：等价于 skimage 的逐帧 SSIM，再对有效帧取平均
    """
    T = pred.shape[0]
    ssim_vals = []
    for t in range(T):
        # 如果这一帧没有目标，直接跳过
        if mask[t].sum() < 1:
            continue
        p = pred[t, 0].detach().cpu().numpy()
        g = gt[t, 0].detach().cpu().numpy()
        # 和你第二段代码完全一致
        p = np.clip(p, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)
        ssim_t = sk_ssim(
            g,
            p,
            data_range=data_range
        )
        ssim_vals.append(ssim_t)
    if len(ssim_vals) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.tensor(
        float(np.mean(ssim_vals)),
        device=pred.device
    )


def masked_psnr(pred, gt, mask, data_range=1.0):
    """
    pred, gt, mask: [T, 1, H, W]  (torch.Tensor)
    行为：等价于 skimage 的逐帧 PSNR，再对有效帧取平均
    """
    T = pred.shape[0]
    psnr_vals = []
    for t in range(T):
        if mask[t].sum() < 1:
            continue
        p = pred[t, 0].detach().cpu().numpy()
        g = gt[t, 0].detach().cpu().numpy()
        p = np.clip(p, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)
        psnr_t = sk_psnr(
            g,
            p,
            data_range=data_range
        )
        psnr_vals.append(psnr_t)
    if len(psnr_vals) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.tensor(
        float(np.mean(psnr_vals)),
        device=pred.device
    )


def compute_warp_losses_from_ref(outputs, groundtruth, ref_t):
    recon_seq = outputs['reconstructed_object']
    flows_fw = outputs['flow_forward']
    flows_bw = outputs['flow_backward']
    T = recon_seq.shape[0]
    ref = recon_seq[ref_t].unsqueeze(0)
    mse_list, ssim_list, psnr_list, tgt_list = [], [], [], []
    for t in reversed(range(ref_t)):
        ref = warp(ref, flows_fw[t].unsqueeze(0))
        gt = groundtruth['object'][t].unsqueeze(0)
        mask = (gt > -1).float()
        mse_list.append(F.mse_loss(ref, gt).item())
        ssim_list.append(masked_ssim(ref, gt, mask).item())
        psnr_list.append(masked_psnr(ref, gt, mask).item())
        tgt_list.append(t)
    ref = recon_seq[ref_t].unsqueeze(0)
    for t in range(ref_t + 1, T):
        ref = warp(ref, flows_bw[t - 1].unsqueeze(0))
        gt = groundtruth['object'][t].unsqueeze(0)
        mask = (gt > -1).float()
        mse_list.append(F.mse_loss(ref, gt).item())
        ssim_list.append(masked_ssim(ref, gt, mask).item())
        psnr_list.append(masked_psnr(ref, gt, mask).item())
        tgt_list.append(t)
    return {
        'target_t': tgt_list,
        'mse': mse_list,
        'ssim': ssim_list,
        'psnr': psnr_list
    }


# def compute_warp_losses_from_ref(outputs, groundtruth, ref_t):
#     """
#     从 reference frame ref_t warp 到其它帧，返回逐帧 loss
#     """
#     recon_seq = outputs['reconstructed_object']
#     flows_fw = outputs['flow_forward']
#     flows_bw = outputs['flow_backward']
#     T = recon_seq.shape[0]
#     ref = recon_seq[ref_t].unsqueeze(0)
#     mse_list, tgt_list = [], []
#     # ---- warp 到过去 ----
#     cur = ref.clone()
#     for t in reversed(range(ref_t)):
#         cur = warp(cur, flows_fw[t].unsqueeze(0))
#         mse_list.append(
#             F.mse_loss(cur, groundtruth['object'][t].unsqueeze(0)).item()
#         )
#         tgt_list.append(t)
#     # ---- warp 到未来 ----
#     cur = ref.clone()
#     for t in range(ref_t + 1, T):
#         cur = warp(cur, flows_bw[t - 1].unsqueeze(0))
#         mse_list.append(
#             F.mse_loss(cur, groundtruth['object'][t].unsqueeze(0)).item()
#         )
#         tgt_list.append(t)
#     return tgt_list, mse_list


# 先定义清空CSV的辅助函数（放在test_model函数外，或函数内开头）
def clear_csv_files(file_paths):
    """
    清空指定路径的CSV文件（创建空文件）
    :param file_paths: CSV文件路径列表
    """
    for file_path in file_paths:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 清空文件（以写入模式打开后立即关闭，会清空内容）
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            # 如果需要保留表头，可以在这里写入，否则直接创建空文件
            pass


def append_warp_csv(csv_path, batch_idx, warp_dict):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, mse in zip(
                item['target_t'], item['mse']
            ):
                writer.writerow([
                    batch_idx, ref_t, tgt_t, mse
                ])


def append_warp_csv1(csv_path, batch_idx, warp_dict):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, ssim in zip(
                item['target_t'], item['ssim']
            ):
                writer.writerow([
                    batch_idx, ref_t, tgt_t, ssim
                ])


def append_warp_csv2(csv_path, batch_idx, warp_dict):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, ssim in zip(
                item['target_t'], item['psnr']
            ):
                writer.writerow([
                    batch_idx, ref_t, tgt_t, ssim
                ])


def append_test_csv(csv_path, batch_idx, loss_dict, speckle_item_losses, warp_item_losses, digits=6):
    import os, csv

    def _round_value(v):
        """单个值的 rounding 规则：float → round，list → 每个元素 round"""
        if isinstance(v, float):
            return round(v, digits)
        elif isinstance(v, list):
            return [round(float(x), digits) for x in v]
        else:
            try:
                return round(float(v), digits)
            except:
                return v
    # 对 loss_dict 做 round
    rounded_loss_dict = {k: _round_value(v) for k, v in loss_dict.items()}
    # 对 list 类型的 item losses 做 round
    speckle_item_losses = [_round_value(v) for v in speckle_item_losses]
    warp_item_losses = [_round_value(v) for v in warp_item_losses]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # --- 首次写文件，写 header ---
        if not file_exists:
            header = ["batch"] + list(rounded_loss_dict.keys())
            header += [f"speckle_object_item_{i}" for i in range(len(speckle_item_losses))]
            header += [f"warp_item_{i}" for i in range(len(warp_item_losses))]
            writer.writerow(header)
        # --- 写入实际数据（全部已经 round 完成） ---
        row = [batch_idx] + list(rounded_loss_dict.values())
        row += speckle_item_losses
        row += warp_item_losses
        writer.writerow(row)


def Charbonnier_loss(x, y, eps=1e-3, alpha=0.5):
    z = x - y
    loss = (z * z + eps * eps).pow(alpha).mean()
    return loss


# --- 1. EPE ---
def compute_flow_epe(pred_flow, gt_flow):
    """ 标准 EPE，光流论文通用 """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    epe_mean = epe_map.mean()
    epe_per_sample_tensor = epe_map.mean(dim=(1, 2))
    epe_per_sample = epe_per_sample_tensor.detach().cpu().tolist()
    return epe_mean, epe_per_sample


# --- 2. Fl-all (KITTI) ---
def compute_flow_fl_all(pred_flow, gt_flow, abs_thresh=3.0, rel_thresh=0.05):
    """
    KITTI 标准异常点比例:
    EPE > abs_thresh AND 相对误差 > rel_thresh
    """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    mag = torch.sqrt((gt_flow ** 2).sum(dim=1))

    relative_err = epe_map / (mag + 1e-6)
    outlier = (epe_map > abs_thresh) & (relative_err > rel_thresh)
    return outlier.float().mean()


# --- 3. N-px Accuracy ---
def compute_flow_px_accuracy(pred_flow, gt_flow, threshold=1.0):
    """ epe < threshold 像素比例 """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    return (epe_map < threshold).float().mean()


class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, groundtruth):
        losses = {}
        losses['reconstruction_loss'] = self.mse(outputs['reconstructed_object'], groundtruth['object'])
        losses['total_loss'] = losses['reconstruction_loss']
        return losses['total_loss'], {k: v.item() for k, v in losses.items()}


# 扭曲函数
def warp(image, flow):
    batch_size, _, h, w = image.size()
    # 生成采样网格
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float().unsqueeze(0).repeat(batch_size, 1, 1, 1).to(image.device)

    # 根据光流进行偏移
    flow = flow.permute(0, 2, 3, 1)  # 将 flow 转换为 [batch_size, height, width, 2]
    new_coords = grid + flow  # 添加光流偏移

    # 归一化到[-1, 1]范围，供F.grid_sample使用
    new_coords[..., 0] = (new_coords[..., 0] / (w - 1)) * 2 - 1  # x 方向从[0,w-1]归一化到[-1,1]
    new_coords[..., 1] = (new_coords[..., 1] / (h - 1)) * 2 - 1  # y 方向从[0,h-1]归一化到[-1,1]

    # 利用F.grid_sample进行插值warp
    warped_image = F.grid_sample(image, new_coords, mode='bilinear', padding_mode='zeros', align_corners=True)

    mask = torch.autograd.Variable(torch.ones(image.size())).cuda()
    mask = F.grid_sample(mask, new_coords, align_corners=True)
    mask[mask < 0.99] = 0
    mask[mask > 0] = 1
    warped_image = warped_image * mask

    return warped_image


def length_sq(x):
    return torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, H, W]


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.1, beta=5.0):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2

    warped_bwd_flow = warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = warp(fwd_flow, bwd_flow)  # [B, 2, H, W]
    flow_mag_fw = length_sq(fwd_flow) + length_sq(warped_bwd_flow)   # [B, 1, H, W]
    flow_mag_bw = length_sq(bwd_flow) + length_sq(warped_fwd_flow)   # [B, 1, H, W]
    diff_fwd = length_sq(fwd_flow + warped_bwd_flow)  # [B, 1, H, W]
    diff_bwd = length_sq(bwd_flow + warped_fwd_flow)
    threshold_fw = alpha * flow_mag_fw + beta
    threshold_bw = alpha * flow_mag_bw + beta
    fwd_occ = (diff_fwd > threshold_fw).float()   # [B, 1, H, W]
    bwd_occ = (diff_bwd > threshold_bw).float()

    return fwd_occ, bwd_occ


def inpaint_flow(flow, occ_mask):
    """
    修正无效光流区域：
    - 计算有效光流的统计分布
    - 选择主流光流值
    - 在无效区域填充主流光流 ± 0.5 的随机扰动
    """
    B, C, H, W = flow.shape  # B=batch, C=2 (光流通道), H=height, W=width
    valid_mask = 1 - occ_mask  # 1 表示有效光流，0 表示无效光流
    inpainted_flow = flow.clone()
    for b in range(B):
        for c in range(C):  # x 方向和 y 方向分别计算
            valid_flow = flow[b, c][valid_mask[b, 0] > 0]  # 提取有效光流
            if valid_flow.numel() > 0:  # 确保有有效值
                median_flow = torch.median(valid_flow)  # 计算中位数
                noise = (torch.rand_like(flow[b, c]) - 0.5) * 1.0  # 生成 [-0.5, 0.5] 之间的随机扰动
                inpainted_flow[b, c][valid_mask[b, 0] == 0] = median_flow + noise[valid_mask[b, 0] == 0]  # 填充无效区域
    return inpainted_flow


def remove_existing_test_logs(save_dir):
    files_to_remove = [
        "test_metrics_summary.txt",
        "test_batch_losses.csv",
        "test_batch_losses_warp.csv",
    ]
    for fname in files_to_remove:
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


def rotate_output(output):
    new_output = dict(output)
    # --- 旋转物体 ---
    if 'reconstructed_object' in new_output and new_output['reconstructed_object'] is not None:
        obj = new_output['reconstructed_object']
        obj = torch.rot90(obj, k=2, dims=[-2, -1])
        new_output['reconstructed_object'] = obj
    # --- 旋转光流 ---
    if 'flow_forward' in new_output and new_output['flow_forward'] is not None:
        flow = new_output['flow_forward']
        flow = torch.rot90(flow, k=2, dims=[-2, -1])
        flow[:, 0] *= -1  # x方向反向
        flow[:, 1] *= -1  # y方向反向
        new_output['flow_forward'] = flow
    if 'flow_backward' in new_output and new_output['flow_backward'] is not None:
        flow = new_output['flow_backward']
        flow = torch.rot90(flow, k=2, dims=[-2, -1])
        flow[:, 0] *= -1  # x方向反向
        flow[:, 1] *= -1  # y方向反向
        new_output['flow_backward'] = flow
    return new_output


def save_all_results(batch_idx, output, target, directories, use_speckle=False, rotate_input=False):
    """
    通用版本：整合 save_results + save_experiment_results
    参数:
        batch_idx: batch id
        output: 模型输出 dict
        target: 包含 object / flow / speckle
        directories: 各种文件夹路径
        use_speckle: 是否保存 speckle（仅实验版需要）
        rotate_input: 是否对 gt_objects 做旋转
    """
    alternating_frames2 = []
    alternating_frames1 = []

    # GT 物体序列
    gt_objects = target['object']  # [T,C,H,W]
    recon_objects = output.get('reconstructed_object', None)
    flows_fw = output.get('flow_forward', None)
    flows_bw = output.get('flow_backward', None)
    has_flow = flows_fw is not None
    T = gt_objects.shape[0]

    # --------------------------
    # 是否旋转
    # --------------------------
    if rotate_input:
        gt_objects = torch.rot90(gt_objects, k=2, dims=[-2, -1])

    # --------------------------
    # 是否有 speckle
    # --------------------------
    if use_speckle:
        gt_speckles = target['speckle']

    # --------------------------
    # 保存所有 recon_objects
    # --------------------------
    for t in range(T):
        save_image(recon_objects[t:t + 1],os.path.join(directories['recon_object'],
                   f'recon_object_img_{batch_idx}_frame_{t}.png')
        )
        save_image(gt_objects[t:t + 1],os.path.join(directories['origin_object'],
                   f'origin_object_img_{batch_idx}_frame_{t}.png'))
        # diff 图
        save_abs_diff(recon_objects[t:t + 1], gt_objects[t:t + 1],
                      os.path.join(directories['diff_recon_vs_gt'],
                                   f"diff_recon_vs_gt_batch{batch_idx}_frame{t}.png"))

    # --------------------------
    # 光流 track 叠加图
    # --------------------------
    if has_flow:
        # base = gt_objects[0][0]
        # flow_track = draw_flow_track_colorwheel(base, flows_fw, step=40)
        # cv2.imwrite(os.path.join(directories['flow_arrow_fw_overlay'],
        #                          f'flowtrack_batch{batch_idx}.png'), flow_track)
        #
        # flow_track = draw_flow_track_colorwheel(base, target['flow'], step=40)
        # cv2.imwrite(os.path.join(directories['gt_flow_arrow_fw_overlay'],
        #                          f'gt_flowtrack_batch{batch_idx}.png'), flow_track)

        # # --------------------------
        # # 光流 EPE 差异图
        # # --------------------------
        # pred = flows_fw.cpu().numpy()
        # gt = target['flow'].cpu().numpy()
        # diff_u = pred[:, 0] - gt[:, 0]
        # diff_v = pred[:, 1] - gt[:, 1]
        # epe = np.sqrt(diff_u**2 + diff_v**2)
        # epe_map = np.mean(epe, axis=0)
        # epe_norm = epe_map / (epe_map.max() + 1e-6)
        # epe_color = cv2.applyColorMap((epe_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        #
        # base = gt_objects[0][0].cpu().numpy()
        # base = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # overlay = cv2.addWeighted(base, 0, epe_color, 1, 0)
        # cv2.imwrite(os.path.join(directories['flow_diff_overlay'],
        #                          f'flow_epe_batch{batch_idx}.png'), overlay)

        # --------------------------
        # 核心修改：为4个光流物体分别计算并保存EPE差异图
        # --------------------------
        pred_flows_fw = flows_fw.cpu().numpy()  # [4,2,256,256]
        pred_flows_bw = flows_bw.cpu().numpy()  # [4,2,256,256]
        gt_flows_fw = target['flow'].cpu().numpy()  # [4,2,256,256]
        gt_flows_bw = -gt_flows_fw # [4,2,256,256]

        # 基础图像（用于叠加，和原有逻辑一致）
        base = gt_objects[0][0].cpu().numpy()
        base_black = np.zeros_like(base)       # 生成同尺寸全零数组（纯黑灰度图）
        base = cv2.cvtColor((base_black * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # 转BGR彩色

        # 遍历4个光流物体，每个物体生成1张EPE差异图
        for obj_idx in range(4):
            # 提取当前物体的预测光流和GT光流（x/y通道）
            pred_obj_fw = pred_flows_fw[obj_idx]  # [2,256,256]
            gt_obj_fw = gt_flows_fw[obj_idx]  # [2,256,256]
            pred_obj_bw = pred_flows_bw[obj_idx]  # [2,256,256]
            gt_obj_bw = gt_flows_bw[obj_idx]  # [2,256,256]

            # 计算当前物体的EPE（和原有逻辑一致：x/y通道差值的欧式距离）
            diff_u_fw = pred_obj_fw[0] - gt_obj_fw[0]  # x通道差值
            diff_v_fw = pred_obj_fw[1] - gt_obj_fw[1]  # y通道差值
            epe_fw = np.sqrt(diff_u_fw ** 2 + diff_v_fw ** 2)  # [256,256]
            diff_u_bw = pred_obj_bw[0] - gt_obj_bw[0]  # x通道差值
            diff_v_bw = pred_obj_bw[1] - gt_obj_bw[1]  # y通道差值
            epe_bw = np.sqrt(diff_u_bw ** 2 + diff_v_bw ** 2)  # [256,256]

            # EPE归一化（用于可视化）
            epe_fw = epe_fw / (epe_fw.max() + 1e-6)  # 避免除零
            epe_color_fw = cv2.applyColorMap((epe_fw * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            epe_bw = epe_bw / (epe_bw.max() + 1e-6)  # 避免除零
            epe_color_bw = cv2.applyColorMap((epe_bw * 255).astype(np.uint8), cv2.COLORMAP_HOT)

            # 叠加到基础图像（和原有逻辑一致）
            diff_fw = cv2.addWeighted(base, 0, epe_color_fw, 1, 0)
            diff_bw = cv2.addWeighted(base, 0, epe_color_bw, 1, 0)

            # 保存当前物体的EPE差异图（最终输出4张：obj0~obj3）
            cv2.imwrite(
                os.path.join(directories['flow_fw_diff_each'],
                             f'flow_epe_batch{batch_idx}_obj{obj_idx}.png'),
                diff_fw
            )
            cv2.imwrite(
                os.path.join(directories['flow_bw_diff_each'],
                             f'flow_epe_batch{batch_idx}_obj{obj_idx}.png'),
                diff_bw
            )

    # ---------------------------------------------------
    # t 循环
    # ---------------------------------------------------
    for t in range(T - 1):
        gt1 = gt_objects[t:t+1]

        # speckle 保存（可选）
        if use_speckle:
            save_image(gt_speckles[t],
                       os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{t}.png'))
            save_image(gt_speckles[t+1],
                       os.path.join(directories['speckle2'], f'speckle2_{batch_idx}_{t+1}.png'))

        # --------------------------
        # Flow 显示/保存
        # --------------------------
        if has_flow:
            if batch_idx < 10:
                save_flow_to_csv(flows_fw[t].unsqueeze(0).cpu().numpy(),
                                 directories['model_fw'], batch_idx, t)
                save_flow_to_csv(target['flow'][t].unsqueeze(0).cpu().numpy(),
                                 directories['groundtruth_fw'], batch_idx, t)

            flow_im = flow_to_image(flows_fw[t].cpu())
            save_image(flow_im.float() / 255.0,
                       os.path.join(directories['flow_colorimage_fw'],
                                    f'flow_img_{batch_idx}_frame_{t}.png'))
            flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), flows_fw[t].cpu())
            cv2.imwrite(os.path.join(directories['flow_arrow_fw'],
                                     f'flowarrow_f_img_{batch_idx}_frame_{t}.png'), flow_arrow)

            flow_im = flow_to_image(flows_bw[t].cpu())
            save_image(flow_im.float() / 255.0,
                       os.path.join(directories['flow_colorimage_bw'],
                                    f'flow_img_{batch_idx}_frame_{t}.png'))
            flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), flows_bw[t].cpu())
            cv2.imwrite(os.path.join(directories['flow_arrow_bw'],
                                     f'flowarrow_b_img_{batch_idx}_frame_{t}.png'), flow_arrow)

            # gt1: [1,1,H,W]
            gt_obj_gray = gt1[0, 0]  # [H,W]
            # binary mask
            mask = (gt_obj_gray > 0).float()
            # ---- dilation (20 px) ----
            mask_np = mask.cpu().numpy().astype(np.uint8)
            kernel = np.ones((41, 41), np.uint8)  # 2*20+1
            mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
            mask_dilated = torch.from_numpy(mask_dilated).to(mask.device).float()
            # --------------------------
            # apply to object flow
            mask_flow = mask_dilated.unsqueeze(0)  # [1,H,W]
            # obj_flow_masked = output['object_flow'][t] * mask_flow
            # obj_flow_masked = output['object_flow'][t]
            #
            # flow_im = flow_to_image(obj_flow_masked.cpu())
            # save_image(flow_im.float() / 255.0,
            #            os.path.join(directories['object_flow_colorimage_fw'],
            #                         f'flow_img_{batch_idx}_frame_{t}.png'))
            # flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), obj_flow_masked.cpu())
            # cv2.imwrite(os.path.join(directories['object_flow_arrow_fw'],
            #                          f'flowarrow_f_img_{batch_idx}_frame_{t}.png'), flow_arrow)

        # GT flow
        if 'flow' in target:
            gt_flow = target['flow'][t]
            gt_flow_color = flow_to_image(gt_flow.cpu())
            save_image(gt_flow_color.float() / 255.0,
                       os.path.join(directories['gt_flow_colorimage'],
                                    f'gt_flow_img_{batch_idx}_frame_{t}.png'))
            gt_flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), gt_flow.cpu())
            cv2.imwrite(os.path.join(directories['gt_flow_arrow_fw'],
                                     f'gt_flowarrow_img_{batch_idx}_frame_{t}.png'),
                        gt_flow_arrow)

            gt_flow_bw = -gt_flow
            gt_flow_bw_color = flow_to_image(gt_flow_bw.cpu())
            save_image(gt_flow_bw_color.float() / 255.0,
                       os.path.join(directories['gt_flow_colorimage_bw'],
                                    f'gt_flow_img_{batch_idx}_frame_{t}.png'))
            gt_flow_arrow_bw = draw_flow_quiver(gt1.cpu().squeeze(0), gt_flow_bw.cpu())
            cv2.imwrite(os.path.join(directories['gt_flow_arrow_bw'],
                                     f'gt_flowarrow_img_{batch_idx}_frame_{t}.png'),
                        gt_flow_arrow_bw)


    # ============ GT ============
    gt_legend = os.path.join(directories["overlay_results_origin_object"], f"GT_overlay_b{batch_idx}.png")
    gt_nl = os.path.join(directories["overlay_results_nl_origin_object"], f"GT_nl_b{batch_idx}.png")

    make_overlay_with_flow_legend(gt_objects.cpu(), target['flow'].cpu(), gt_legend)
    make_overlay_no_legend(gt_objects.cpu(), gt_nl)

    # ============ UNet ============
    unet_legend = os.path.join(directories["overlay_results_recon_object"], f"UNet_overlay_b{batch_idx}.png")
    unet_nl = os.path.join(directories["overlay_results_nl_recon_object"], f"UNet_nl_b{batch_idx}.png")

    make_overlay_with_flow_legend(recon_objects.cpu(), flows_fw.cpu(), unet_legend)
    make_overlay_no_legend(recon_objects.cpu(), unet_nl)

    # ============ diff ============
    make_overlay_diff(unet_nl, gt_nl,
                      os.path.join(directories["diff_results_recon_obj_origin_obj"], f"UNet_vs_GT_diff_b{batch_idx}.png"))

    all_warped_frames = []  # 收集 (source_t, warped_tensor [1,C,H,W])
    # ==================================================
    # 每一帧作为 reference 的 warp & overlay
    # ==================================================
    for t in range(T):
        # ---------- 1. 从第 t 帧恢复完整序列 ----------
        warped_seq_t = reconstruct_sequence_from_t(
            ref_frame=recon_objects[t:t + 1],
            t=t,
            flows_fw=flows_fw,
            flows_bw=flows_bw,
            T=T
        )  # [T, C, H, W]
        # ---------- 2. overlay ----------
        overlay_path_nl = os.path.join(
            directories['warp_from_each_t_nl_overlay'],
            f'overlay_from_batch{batch_idx}_t{t}.png'
        )
        overlay_path = os.path.join(
            directories['warp_from_each_t_overlay'],
            f'overlay_from_batch{batch_idx}_t{t}.png'
        )
        make_overlay_no_legend(warped_seq_t.cpu(), overlay_path_nl)
        make_overlay_with_flow_legend(warped_seq_t.cpu(), flows_fw.cpu(), overlay_path)
        # （可选）和 GT 做 diff overlay
        gt_overlay_path = os.path.join(
            directories['overlay_gt_each_t'],
            f'gt_overlay_batch{batch_idx}_t{t}.png'
        )
        make_overlay_no_legend(gt_objects.cpu(), gt_overlay_path)
        make_overlay_diff(
            overlay_path_nl,
            gt_overlay_path,
            os.path.join(
                directories['diff_overlay_each_t'],
                f'diff_batch{batch_idx}_t{t}.png'
            )
        )
        make_single_frame_overlay_with_flow_legend(
            img_tensor=gt_objects[t:t + 1],
            flows=target['flow'],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png'
            )
        )
        make_single_frame_overlay_with_flow_legend(
            img_tensor=recon_objects[t:t + 1],
            flows=flows_fw,
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png'
            )
        )
        make_single_frame_overlay_no_legend(
            img_tensor=gt_objects[t:t + 1],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_nl_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png'
            )
        )
        make_single_frame_overlay_no_legend(
            img_tensor=recon_objects[t:t + 1],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_nl_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png'
            )
        )
        make_overlay_diff(
            os.path.join(
                directories['single_overlay_nl_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png'
            ),
            os.path.join(
                directories['single_overlay_nl_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png'
            ),
            os.path.join(
                directories['diff_single_each_t'],
                f'diff_batch{batch_idx}_t{t}.png'
            )
        )

        cur = recon_objects[t:t+1].clone()  # [1,C,H,W]
        for step in range(1, T - t):
            flow = flows_bw[t + step - 1].unsqueeze(0)  # 注意：向未来用 bw
            cur = warp(cur, flow)
            target_frame_idx = t + step  # 得到的是哪一帧
            all_warped_frames.append((t, cur.clone()))
            save_path = os.path.join(
                directories['single_overlay_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            make_single_frame_overlay_with_flow_legend(
                img_tensor=cur,
                flows=flows_fw,      # legend 用全局 flow
                global_t=target_frame_idx,
                save_path=save_path
            )
            # ---------- 新增：无 legend / 无彩色 ----------
            save_path_nl = os.path.join(
                directories['single_overlay_nl_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            make_single_frame_overlay_no_legend(
                img_tensor=cur,
                global_t=target_frame_idx,
                save_path=save_path_nl
            )

            save_path_raw = os.path.join(
                directories['single_overlay_nl_nc_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            # cur: [1, C, H, W], 假设 C=1
            save_image(
                cur.clamp(0, 1),
                save_path_raw
            )
        cur = recon_objects[t:t + 1].clone()
        for step in range(1, t + 1):
            flow = flows_fw[t - step].unsqueeze(0)  # 向过去用 fw
            cur = warp(cur, flow)
            target_frame_idx = t - step
            all_warped_frames.append((t, cur.clone()))
            save_path = os.path.join(
                directories['single_overlay_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            make_single_frame_overlay_with_flow_legend(
                img_tensor=cur,
                flows=flows_fw,
                global_t=target_frame_idx,
                save_path=save_path
            )
            # ---------- 新增 ----------
            save_path_nl = os.path.join(
                directories['single_overlay_nl_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            make_single_frame_overlay_no_legend(
                img_tensor=cur,
                global_t=target_frame_idx,
                save_path=save_path_nl
            )

            # 纯结果版
            save_path_raw = os.path.join(
                directories['single_overlay_nl_nc_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png'
            )
            save_image(cur.clamp(0, 1), save_path_raw)

    # ============================================================
    # 在 for t in range(T): 循环结束之后，生成总叠加图
    # ============================================================
    if all_warped_frames:
        H, W = all_warped_frames[0][1].shape[-2:]
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        for source_t, tensor in all_warped_frames:
            arr = tensor[0, 0].cpu().numpy()
            arr = np.clip(arr, 0, 1)
            rgba = np.zeros((H, W, 4))
            rgba[..., :3] = colors[source_t]  # 颜色代表来源帧 t
            rgba[..., 3] = arr * 0.9
            ax.imshow(rgba)
        ax.set_axis_off()
        save_path_all = os.path.join(
            directories['single_overlay_nl_warp_all_t'],
            f'batch{batch_idx}_ALL_warped_overlay.png'
        )
        plt.savefig(save_path_all, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
        plt.close()

    #     # --------------------------
    #     # GIF
    #     # --------------------------
    #     gt2_frame = np.clip(gt2[0,0].cpu().numpy()*255,0,255).astype(np.uint8)
    #     recon2_frame = np.clip(recon_obj2[0,0].cpu().numpy()*255,0,255).astype(np.uint8)
    #
    #     alternating_frames2.append(add_label_to_frame(gt2_frame, f"GT Obj {batch_idx} Frame {t+1}"))
    #     alternating_frames2.append(add_label_to_frame(recon2_frame, f"Reconstructed Obj {batch_idx} Frame {t+1}"))
    #
    #     gt1_frame = np.clip(gt1[0,0].cpu().numpy()*255,0,255).astype(np.uint8)
    #     warped_frame = np.clip(warped_obj1[0,0].cpu().numpy()*255,0,255).astype(np.uint8)
    #
    #     alternating_frames1.append(add_label_to_frame(gt1_frame, f"GT Obj {batch_idx} Frame {t}"))
    #     alternating_frames1.append(add_label_to_frame(warped_frame, f"Warped Obj {batch_idx} Frame {t}"))
    #
    # # 保存 GIF
    # imageio.mimsave(os.path.join(directories['gifs_object2'],
    #                              f'batch_{batch_idx}_alternating.gif'),
    #                 alternating_frames2, fps=0.75)
    #
    # imageio.mimsave(os.path.join(directories['gifs_object1'],
    #                              f'batch_{batch_idx}_alternating.gif'),
    #                 alternating_frames1, fps=0.75)


# def save_experimental_results(batch_idx, output, speckle1, speckle2, directories, t):
#     """Save visualization results for experimental data"""
#     flow_rot = output['flow_forward']
#     flow_rot = torch.rot90(flow_rot, k=2, dims=[-2, -1])
#     flow_rot[:, 0, :, :] *= -1  # x 分量取负
#     flow_rot[:, 1, :, :] *= -1  # y 分量取负
#     if batch_idx < 10:
#         save_flow_to_csv(flow_rot.cpu().numpy(), directories['model_fw'], batch_idx, t)
#     # Save speckle patterns
#     save_image(speckle1, os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{t}.png'))
#     save_image(speckle2, os.path.join(directories['speckle2'], f'speckle2_{batch_idx}_{t+1}.png'))
#
#     # 旋转180度
#     recon_obj1 = output['reconstructed_object1']
#     recon_obj2 = output['reconstructed_object2']
#     recon_obj1 = torch.rot90(recon_obj1, k=2, dims=[-2, -1])
#     recon_obj2 = torch.rot90(recon_obj2, k=2, dims=[-2, -1])
#     warp_obj1 = warp(recon_obj2, flow_rot)
#     save_image(warp_obj1,
#                os.path.join(directories['reconstructed_object1'], f'recon_object1_{batch_idx}_{t}.png'))
#     save_image(recon_obj2,
#                os.path.join(directories['reconstructed_object2'], f'recon_object2_{batch_idx}_{t + 1}.png'))
#     # Save reconstructed objects
#     # save_image(output['reconstructed_object1'],
#     #            os.path.join(directories['reconstructed_object1'], f'recon_object1_{batch_idx}_{t}.png'))
#     # save_image(output['reconstructed_object2'],
#     #            os.path.join(directories['reconstructed_object2'], f'recon_object2_{batch_idx}_{t+1}.png'))
#
#     # Save flow visualization
#     flow_img = flow_to_image(flow_rot.cpu().squeeze(0))
#     save_image(flow_img.float() / 255,
#                os.path.join(directories['flow_colorimage'], f'flow_{batch_idx}_{t}_to_{t+1}.png'))
#
#     # Save flow arrows on reconstructed object
#     flow_arrow = draw_flow_quiver(recon_obj1.cpu().squeeze(0), flow_rot.cpu().squeeze(0))
#     cv2.imwrite(os.path.join(directories['flow_arrow'], f'flow_arrow_{batch_idx}_{t}_to_{t+1}.png'), flow_arrow)
#
#     # # Create and save GIF of reconstructed objects
#     # frames = []
#     # # Add first reconstructed object
#     # recon1_frame = np.clip((output['reconstructed_object1'][0, 0].cpu().numpy() * 255), 0, 255).astype(np.uint8)
#     # recon1_labeled = add_label_to_frame(recon1_frame, f"Reconstructed Object {file_names[0]}")
#     # frames.append(recon1_labeled)
#     #
#     # # Add second reconstructed object
#     # recon2_frame = np.clip((output['reconstructed_object2'][0, 0].cpu().numpy() * 255), 0, 255).astype(np.uint8)
#     # recon2_labeled = add_label_to_frame(recon2_frame, f"Reconstructed Object {file_names[1]}")
#     # frames.append(recon2_labeled)
#     #
#     # # Save GIF
#     # gif_path = os.path.join(directories['gifs'], f'recon_objects_{batch_idx}_{file_names[0]}_to_{file_names[1]}.gif')
#     # imageio.mimsave(gif_path, frames, fps=1)

def save_experimental_results(batch_idx, output, speckle_seq, directories):
    """
    保存实验数据的可视化结果，按时间 t 循环保存。
    参数:
        batch_idx: 当前 batch 序号
        output: 模型输出字典，包含 flow_forward、reconstructed_object 等
        speckle_seq: 输入的散斑序列 [T, C, H, W]
        directories: 各保存目录
    """
    flow_forward = output['flow_forward']      # [T-1, 2, H, W]
    recon_objects = output['reconstructed_object']  # [T, C, H, W]

    num_frames = recon_objects.shape[0]  # 序列帧数

    # 遍历每一帧 (0 到 T-2 之间有 flow，对应 recon t 和 t+1)
    for t in range(num_frames - 1):
        # ---------- 保存 flow ----------
        flow_t = flow_forward[t:t+1]  # shape [1,2,H,W]
        # 旋转180度
        flow_rot = torch.rot90(flow_t, k=2, dims=[-2, -1])
        flow_rot[:, 0, :, :] *= -1
        flow_rot[:, 1, :, :] *= -1

        if batch_idx < 10:
            save_flow_to_csv(flow_rot.cpu().numpy(), directories['model_fw'], batch_idx, t)

        # ---------- 保存 speckle ----------
        save_image(speckle_seq[t], os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{t}.png'))
        save_image(speckle_seq[t + 1], os.path.join(directories['speckle2'], f'speckle2_{batch_idx}_{t+1}.png'))

        # ---------- 保存重建结果 ----------
        recon_obj1 = recon_objects[t:t+1]      # 当前帧
        recon_obj2 = recon_objects[t+1:t+2]    # 下一帧
        # 旋转180度
        recon_obj1 = torch.rot90(recon_obj1, k=2, dims=[-2, -1])
        recon_obj2 = torch.rot90(recon_obj2, k=2, dims=[-2, -1])

        # warp recon_obj2 根据 flow_rot 对齐到 recon_obj1
        warp_obj1 = warp(recon_obj2, flow_rot)

        save_image(warp_obj1, os.path.join(directories['reconstructed_object1'], f'recon_object1_{batch_idx}_{t}.png'))
        save_image(recon_obj2, os.path.join(directories['reconstructed_object2'], f'recon_object2_{batch_idx}_{t+1}.png'))

        # ---------- 保存 flow 可视化 ----------
        flow_img = flow_to_image(flow_rot.cpu().squeeze(0))
        save_image(flow_img.float() / 255, os.path.join(directories['flow_colorimage'], f'flow_{batch_idx}_{t}_to_{t+1}.png'))

        flow_arrow = draw_flow_quiver(recon_obj1.cpu().squeeze(0), flow_rot.cpu().squeeze(0))
        cv2.imwrite(os.path.join(directories['flow_arrow'], f'flow_arrow_{batch_idx}_{t}_to_{t+1}.png'), flow_arrow)

# def save_experimental_results(batch_idx, output, speckle_seq, directories, crop_x, crop_y, seq_idx):
#     """
#     修改保存函数，添加序列索引
#     """
#     flow_forward = output['flow_forward']  # [T-1, 2, H, W]
#     recon_objects = output['reconstructed_object']  # [T, C, H, W]
#
#     num_frames = recon_objects.shape[0]
#
#     for t in range(num_frames - 1):
#         # 保存时添加序列索引
#         flow_t = flow_forward[t:t + 1]
#         flow_rot = torch.rot90(flow_t, k=2, dims=[-2, -1])
#         flow_rot[:, 0, :, :] *= -1
#         flow_rot[:, 1, :, :] *= -1
#
#         if batch_idx < 10:
#             save_flow_to_csv(flow_rot.cpu().numpy(), directories['model_fw'], batch_idx, t)
#
#         # 在文件名中添加序列索引
#         save_image(speckle_seq[t],
#                    os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{seq_idx}_{t}_x{crop_x}y{crop_y}.png'))
#         save_image(speckle_seq[t + 1], os.path.join(directories['speckle2'],
#                                                     f'speckle2_{batch_idx}_{seq_idx}_{t + 1}_x{crop_x}y{crop_y}.png'))
#
#         recon_obj1 = recon_objects[t:t + 1]
#         recon_obj2 = recon_objects[t + 1:t + 2]
#         recon_obj1 = torch.rot90(recon_obj1, k=2, dims=[-2, -1])
#         recon_obj2 = torch.rot90(recon_obj2, k=2, dims=[-2, -1])
#
#         warp_obj1 = warp(recon_obj2, flow_rot)
#
#         save_image(warp_obj1, os.path.join(directories['reconstructed_object1'],
#                                            f'recon_object1_{batch_idx}_{seq_idx}_{t}_x{crop_x}y{crop_y}.png'))
#         save_image(recon_obj2, os.path.join(directories['reconstructed_object2'],
#                                             f'recon_object2_{batch_idx}_{seq_idx}_{t + 1}_x{crop_x}y{crop_y}.png'))
#
#         flow_img = flow_to_image(flow_rot.cpu().squeeze(0))
#         save_image(flow_img.float() / 255, os.path.join(directories['flow_colorimage'],
#                                                         f'flow_{batch_idx}_{seq_idx}_{t}_to_{t + 1}_x{crop_x}y{crop_y}.png'))
#
#         flow_arrow = draw_flow_quiver(recon_obj1.cpu().squeeze(0), flow_rot.cpu().squeeze(0))
#         cv2.imwrite(os.path.join(directories['flow_arrow'],
#                                  f'flow_arrow_{batch_idx}_{seq_idx}_{t}_to_{t + 1}_x{crop_x}y{crop_y}.png'), flow_arrow)