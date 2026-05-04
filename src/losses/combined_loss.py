"""Combined loss functions, flow utilities, and evaluation metrics."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from torch import nn


# ==============================================================================
# Loss Classes
# ==============================================================================

class CombinedLoss(nn.Module):
    """Multi-loss module for training and testing.

    Training mode:
        - Charbonnier warp losses on speckle frames
        - Temporal object warp loss (forward + backward chains)
        - MSE reconstruction loss on a random reference frame

    Test mode:
        - EPE, angular error, Fl-all, N-px accuracy for flow
        - MSE, SSIM, PSNR for reconstructed objects
        - Cross-reference warp loss chains
    """

    def __init__(self, mode: str = 'train') -> None:
        super().__init__()
        self.mode = mode
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        if self.mode == 'test':
            self.seq_warp_sum: Dict[str, Dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            self.seq_warp_count: Dict[str, Dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            self.object_mse_sum: Optional[torch.Tensor] = None
            self.object_mse_count: int = 0
            self.flow_fw_epe_sum: Optional[torch.Tensor] = None
            self.flow_fw_epe_count: Optional[torch.Tensor] = None
            self.flow_bw_epe_sum: Optional[torch.Tensor] = None
            self.flow_bw_epe_count: Optional[torch.Tensor] = None

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
        t_speckle: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        losses: Dict[str, torch.Tensor] = {}
        extra_logs: Dict[str, Any] = {}

        if self.mode == 'test':
            return self._forward_test(outputs, groundtruth)
        else:
            return self._forward_train(outputs, groundtruth, epoch, t_speckle)

    def _forward_test(
        self,
        outputs: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        losses: Dict[str, torch.Tensor] = {}
        extra_logs: Dict[str, Any] = {}

        # ---- Forward flow metrics ----
        if 'flow_forward' in outputs and 'flow' in groundtruth:
            flow_pred = outputs['flow_forward']
            flow_gt = groundtruth['flow']
            eps = 1e-6

            losses['flow_fw_loss_epe'], extra_logs['each_flow_fw_epe'] = (
                compute_flow_epe(flow_pred, flow_gt)
            )

            # Per-flow FW EPE accumulation across batches
            per_flow_fw_epe = torch.tensor(
                extra_logs['each_flow_fw_epe'], device=flow_pred.device
            ).detach()
            valid_mask = torch.isfinite(per_flow_fw_epe)
            if self.flow_fw_epe_sum is None:
                self.flow_fw_epe_sum = torch.zeros_like(per_flow_fw_epe)
                self.flow_fw_epe_count = torch.zeros_like(per_flow_fw_epe)
            self.flow_fw_epe_sum[valid_mask] += per_flow_fw_epe[valid_mask]
            self.flow_fw_epe_count[valid_mask] += 1

            # Angular error
            pred_norm = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + eps)
            gt_norm = torch.sqrt(torch.sum(flow_gt ** 2, dim=1) + eps)
            dot = torch.sum(flow_pred * flow_gt, dim=1)
            cos_theta = dot / (pred_norm * gt_norm + eps)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = angle_rad * 180.0 / 3.1415926
            losses['flow_fw_loss_angle'] = angle_deg.mean() / 180.0

            losses['flow_fw_fl_all'] = compute_flow_fl_all(flow_pred, flow_gt)
            losses['flow_fw_1px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 1.0)
            losses['flow_fw_3px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 3.0)
            losses['flow_fw_5px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 5.0)

        # ---- Backward flow metrics ----
        if 'flow_backward' in outputs and 'flow' in groundtruth:
            flow_pred = outputs['flow_backward']
            flow_gt = -groundtruth['flow']
            eps = 1e-6

            losses['flow_bw_loss_epe'], extra_logs['each_flow_bw_epe'] = (
                compute_flow_epe(flow_pred, flow_gt)
            )

            per_flow_bw_epe = torch.tensor(
                extra_logs['each_flow_bw_epe'], device=flow_pred.device
            ).detach()
            valid_mask = torch.isfinite(per_flow_bw_epe)
            if self.flow_bw_epe_sum is None:
                self.flow_bw_epe_sum = torch.zeros_like(per_flow_bw_epe)
                self.flow_bw_epe_count = torch.zeros_like(per_flow_bw_epe)
            self.flow_bw_epe_sum[valid_mask] += per_flow_bw_epe[valid_mask]
            self.flow_bw_epe_count[valid_mask] += 1

            pred_norm = torch.sqrt(torch.sum(flow_pred ** 2, dim=1) + eps)
            gt_norm = torch.sqrt(torch.sum(flow_gt ** 2, dim=1) + eps)
            dot = torch.sum(flow_pred * flow_gt, dim=1)
            cos_theta = dot / (pred_norm * gt_norm + eps)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = angle_rad * 180.0 / 3.1415926
            losses['flow_bw_loss_angle'] = angle_deg.mean() / 180.0

            losses['flow_bw_fl_all'] = compute_flow_fl_all(flow_pred, flow_gt)
            losses['flow_bw_1px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 1.0)
            losses['flow_bw_3px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 3.0)
            # losses['flow_bw_5px_acc'] = compute_flow_px_accuracy(flow_pred, flow_gt, 5.0)

        # ---- Reconstruction metrics ----
        if 'reconstructed_object' in outputs and 'object' in groundtruth:
            pred = outputs['reconstructed_object']
            gt = groundtruth['object']
            losses['speckle_object_loss'] = self.mse_loss(pred, gt)

            mask = (gt > -1).float()
            losses['speckle_object_ssim'] = masked_ssim(pred, gt, mask, data_range=1.0)
            losses['speckle_object_psnr'] = masked_psnr(pred, gt, mask, data_range=1.0)

            # Per-object MSE
            diff = pred - gt
            per_obj_mse = diff.pow(2).mean(dim=(1, 2, 3))
            extra_logs['speckle_object_items'] = per_obj_mse.tolist()

            per_obj_mse_detached = per_obj_mse.detach()
            if self.object_mse_sum is None:
                self.object_mse_sum = per_obj_mse_detached.clone()
            else:
                self.object_mse_sum += per_obj_mse_detached
            self.object_mse_count += 1

        # ---- Cross-reference warp losses ----
        if 'flow_forward' in outputs:
            warp_all_refs: Dict[int, Dict[str, Any]] = {}
            T = outputs['reconstructed_object'].shape[0]
            for ref_t in range(T):
                warp_metrics = compute_warp_losses_from_ref(
                    outputs, groundtruth, ref_t
                )
                warp_all_refs[ref_t] = warp_metrics
            extra_logs['object_warp_from_each_t'] = warp_all_refs

        total = sum(losses.values())
        losses['total_loss'] = total

        # Merge extra_logs (lists, dicts) into losses
        for k, v in extra_logs.items():
            losses[k] = v

        return total, {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in losses.items()
        }

    def _forward_train(
        self,
        outputs: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
        epoch: Optional[int],
        t_speckle: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = {}

        # ---- Warp loss on warped_speckle1 (forward warp consistency) ----
        if 'warped_speckle1' in outputs and 'speckle' in groundtruth:
            losses['speckle1_warp_loss'] = Charbonnier_loss(
                outputs['warped_speckle1'], groundtruth['speckle'][:-1]
            )
            if epoch is not None and epoch >= 2:
                valid_mask = 1.0 - outputs['fwd_occ']
                diff = (outputs['warped_speckle1'] - groundtruth['speckle'][:-1]) ** 2
                losses['speckle1_valid_region_loss'] = (
                    (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                )

        # ---- Warp loss on warped_speckle2 (backward warp consistency) ----
        if 'warped_speckle2' in outputs and 'speckle' in groundtruth:
            losses['speckle2_warp_loss'] = Charbonnier_loss(
                outputs['warped_speckle2'], groundtruth['speckle'][1:]
            )
            if epoch is not None and epoch >= 2:
                valid_mask = 1.0 - outputs['bwd_occ']
                diff = (outputs['warped_speckle2'] - groundtruth['speckle'][1:]) ** 2
                losses['speckle2_valid_region_loss'] = (
                    (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)
                )

        # ---- Reconstruction loss ----
        if (
            'reconstructed_object' in outputs
            and 'object' in groundtruth
            and t_speckle is not None
        ):
            recon_object_t = outputs['reconstructed_object'][t_speckle].unsqueeze(0)
            gt_object_t = groundtruth['object'][t_speckle].unsqueeze(0)
            losses['speckle_object_mse_loss'] = self.mse_loss(
                recon_object_t, gt_object_t
            )

            # ---- Temporal object warp loss ----
            num_frames = len(outputs['reconstructed_object'])
            object_warp_loss = 0.0
            count = 0

            if epoch is not None and epoch >= 3:
                # Forward direction (warp to past)
                if t_speckle > 0:
                    warped_recon_fw = recon_object_t.clone()
                    warped_gt_fw = gt_object_t.clone()
                    for step in range(1, t_speckle + 1):
                        flow_fw = outputs['flow_forward'][
                            t_speckle - step
                        ].unsqueeze(0)
                        warped_recon_fw = warp(warped_recon_fw, flow_fw)
                        warped_gt_fw = warp(warped_gt_fw, flow_fw)
                        object_warp_loss += self.mse_loss(
                            warped_recon_fw, warped_gt_fw
                        )
                        count += 1

                # Backward direction (warp to future)
                if t_speckle < num_frames - 1:
                    warped_recon_bw = recon_object_t.clone()
                    warped_gt_bw = gt_object_t.clone()
                    for step in range(1, num_frames - t_speckle):
                        flow_bw = outputs['flow_backward'][
                            t_speckle + step - 1
                        ].unsqueeze(0)
                        warped_recon_bw = warp(warped_recon_bw, flow_bw)
                        warped_gt_bw = warp(warped_gt_bw, flow_bw)
                        object_warp_loss += self.mse_loss(
                            warped_recon_bw, warped_gt_bw
                        )
                        count += 1

                if count > 0:
                    losses['object_warp_loss'] = object_warp_loss / count

        total = sum(losses.values())
        losses['total_loss'] = total
        return total, {k: v.item() for k, v in losses.items()}


class SimpleLoss(nn.Module):
    """Simple MSE loss for U-Net-only ablation experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        groundtruth: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = {}
        losses['reconstruction_loss'] = self.mse(
            outputs['reconstructed_object'], groundtruth['object']
        )
        losses['total_loss'] = losses['reconstruction_loss']
        return losses['total_loss'], {k: v.item() for k, v in losses.items()}


# ==============================================================================
# Warp and Flow Utilities
# ==============================================================================

def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp an image tensor using an optical flow field via grid_sample.

    Args:
        image: [B, C, H, W]
        flow: [B, 2, H, W]

    Returns:
        warped_image: [B, C, H, W] with valid-region masking applied
    """
    batch_size, _, h, w = image.size()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )
    grid = (
        torch.stack((grid_x, grid_y), dim=-1)
        .float()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
        .to(image.device)
    )

    flow = flow.permute(0, 2, 3, 1)
    new_coords = grid + flow

    new_coords[..., 0] = (new_coords[..., 0] / (w - 1)) * 2 - 1
    new_coords[..., 1] = (new_coords[..., 1] / (h - 1)) * 2 - 1

    warped_image = F.grid_sample(
        image, new_coords, mode='bilinear',
        padding_mode='zeros', align_corners=True,
    )

    mask = torch.autograd.Variable(torch.ones(image.size())).cuda()
    mask = F.grid_sample(mask, new_coords, align_corners=True)
    mask[mask < 0.99] = 0
    mask[mask > 0] = 1
    warped_image = warped_image * mask

    return warped_image


def length_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared length per pixel. x: [B, 2, H, W] -> [B, 1, H, W]."""
    return torch.sum(x ** 2, dim=1, keepdim=True)


def forward_backward_consistency_check(
    fwd_flow: torch.Tensor,
    bwd_flow: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute forward/backward occlusion masks via consistency check.

    Based on UnFlow (https://arxiv.org/abs/1711.07837).

    Args:
        fwd_flow: [B, 2, H, W] forward optical flow
        bwd_flow: [B, 2, H, W] backward optical flow
        alpha, beta: threshold parameters

    Returns:
        fwd_occ: [B, 1, H, W] forward occlusion mask
        bwd_occ: [B, 1, H, W] backward occlusion mask
    """
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2

    warped_bwd_flow = warp(bwd_flow, fwd_flow)
    warped_fwd_flow = warp(fwd_flow, bwd_flow)

    flow_mag_fw = length_sq(fwd_flow) + length_sq(warped_bwd_flow)
    flow_mag_bw = length_sq(bwd_flow) + length_sq(warped_fwd_flow)

    diff_fwd = length_sq(fwd_flow + warped_bwd_flow)
    diff_bwd = length_sq(bwd_flow + warped_fwd_flow)

    threshold_fw = alpha * flow_mag_fw + beta
    threshold_bw = alpha * flow_mag_bw + beta

    fwd_occ = (diff_fwd > threshold_fw).float()
    bwd_occ = (diff_bwd > threshold_bw).float()

    return fwd_occ, bwd_occ


def inpaint_flow(
    flow: torch.Tensor, occ_mask: torch.Tensor
) -> torch.Tensor:
    """Fill in invalid flow regions with median flow value + random perturbation.

    Args:
        flow: [B, 2, H, W]
        occ_mask: [B, 1, H, W] occlusion mask (1 = invalid)

    Returns:
        inpainted_flow: [B, 2, H, W]
    """
    B, C, H, W = flow.shape
    valid_mask = 1 - occ_mask
    inpainted_flow = flow.clone()

    for b in range(B):
        for c in range(C):
            valid_flow = flow[b, c][valid_mask[b, 0] > 0]
            if valid_flow.numel() > 0:
                median_flow = torch.median(valid_flow)
                noise = (torch.rand_like(flow[b, c]) - 0.5) * 1.0
                inpainted_flow[b, c][valid_mask[b, 0] == 0] = (
                    median_flow + noise[valid_mask[b, 0] == 0]
                )
    return inpainted_flow


# ==============================================================================
# Flow Evaluation Metrics
# ==============================================================================

def compute_flow_epe(
    pred_flow: torch.Tensor, gt_flow: torch.Tensor
) -> Tuple[torch.Tensor, List[float]]:
    """Standard endpoint error (EPE) for optical flow.

    Returns:
        epe_mean: scalar tensor
        epe_per_sample: list of per-frame EPE values
    """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    epe_mean = epe_map.mean()
    epe_per_sample_tensor = epe_map.mean(dim=(1, 2))
    epe_per_sample = epe_per_sample_tensor.detach().cpu().tolist()
    return epe_mean, epe_per_sample


def compute_flow_fl_all(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    abs_thresh: float = 3.0,
    rel_thresh: float = 0.05,
) -> torch.Tensor:
    """KITTI Fl-all outlier ratio.

    Outlier if: EPE > abs_thresh AND relative error > rel_thresh.
    """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    mag = torch.sqrt((gt_flow ** 2).sum(dim=1))
    relative_err = epe_map / (mag + 1e-6)
    outlier = (epe_map > abs_thresh) & (relative_err > rel_thresh)
    return outlier.float().mean()


def compute_flow_px_accuracy(
    pred_flow: torch.Tensor, gt_flow: torch.Tensor, threshold: float = 1.0
) -> torch.Tensor:
    """Ratio of pixels where EPE < threshold."""
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    return (epe_map < threshold).float().mean()


# ==============================================================================
# Image Quality Metrics
# ==============================================================================

def masked_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per-frame SSIM using skimage, averaged over valid frames.

    Args:
        pred, gt, mask: [T, 1, H, W] tensors
        data_range: dynamic range of the data

    Returns:
        mean_ssim: scalar tensor
    """
    T = pred.shape[0]
    ssim_vals: List[float] = []
    for t in range(T):
        if mask[t].sum() < 1:
            continue
        p = pred[t, 0].detach().cpu().numpy()
        g = gt[t, 0].detach().cpu().numpy()
        p = np.clip(p, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)
        ssim_t = sk_ssim(g, p, data_range=data_range)
        ssim_vals.append(ssim_t)
    if len(ssim_vals) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.tensor(float(np.mean(ssim_vals)), device=pred.device)


def masked_psnr(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per-frame PSNR using skimage, averaged over valid frames.

    Args:
        pred, gt, mask: [T, 1, H, W] tensors
        data_range: dynamic range of the data

    Returns:
        mean_psnr: scalar tensor
    """
    T = pred.shape[0]
    psnr_vals: List[float] = []
    for t in range(T):
        if mask[t].sum() < 1:
            continue
        p = pred[t, 0].detach().cpu().numpy()
        g = gt[t, 0].detach().cpu().numpy()
        p = np.clip(p, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)
        psnr_t = sk_psnr(g, p, data_range=data_range)
        psnr_vals.append(psnr_t)
    if len(psnr_vals) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.tensor(float(np.mean(psnr_vals)), device=pred.device)


def Charbonnier_loss(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3, alpha: float = 0.5
) -> torch.Tensor:
    """Charbonnier penalty: (z^2 + eps^2)^alpha ."""
    z = x - y
    loss = (z * z + eps * eps).pow(alpha).mean()
    return loss


# ==============================================================================
# Cross-Reference Warp Loss
# ==============================================================================

def compute_warp_losses_from_ref(
    outputs: Dict[str, torch.Tensor],
    groundtruth: Dict[str, torch.Tensor],
    ref_t: int,
) -> Dict[str, List[float]]:
    """Warp from a reference frame to all other frames, compute losses.

    Returns:
        dict with keys: target_t, mse, ssim, psnr
    """
    recon_seq = outputs['reconstructed_object']
    flows_fw = outputs['flow_forward']
    flows_bw = outputs['flow_backward']
    T = recon_seq.shape[0]

    ref = recon_seq[ref_t].unsqueeze(0)
    mse_list: List[float] = []
    ssim_list: List[float] = []
    psnr_list: List[float] = []
    tgt_list: List[int] = []

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
        'psnr': psnr_list,
    }
