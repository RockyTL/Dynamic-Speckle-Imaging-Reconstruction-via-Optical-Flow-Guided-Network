from Net_Unet import UNet
from utils.Sundries import *
from RAFT.raft import *
from argparse import Namespace
from GMA.network import *


class MotionEncoder(nn.Module):
    """
    将 speckle 映射到 motion-friendly latent
    目标：削弱 speckle 随机高频，使几何变化可被 flow 捕捉
    """
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(4, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, base_ch * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: [T, 1, H, W]
        return: [T, C, H', W']
        """
        return self.net(x)


class MotionAdapter(nn.Module):
    """
    把 motion latent (C=64) 压到 RAFT 能接受的通道数（1）
    """
    def __init__(self, in_ch=64, out_ch=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class GlobalMotionHead(nn.Module):
    """
    从 latent feature 中预测全局 motion 参数
    (tx, ty, log_s, sinθ, cosθ)
    """

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5)
        )

    def forward(self, f1, f2):
        return self.net(f2 - f1)


# def global_motion_to_flow(theta, H, W, device):
#     """
#     theta: [B, 5] = (tx, ty, log_s, sinθ, cosθ)
#     return: [B, 2, H, W]
#     """
#     tx, ty, log_s, sin_t, cos_t = theta.split(1, dim=1)
#     s = torch.exp(log_s)
#     # 坐标网格
#     y, x = torch.meshgrid(
#         torch.linspace(-1, 1, H, device=device),
#         torch.linspace(-1, 1, W, device=device),
#         indexing='ij'
#     )
#     grid = torch.stack([x, y], dim=0)  # [2, H, W]
#     grid = grid.unsqueeze(0)  # [1, 2, H, W]
#     R = torch.stack([
#         torch.cat([cos_t, -sin_t], dim=1),
#         torch.cat([sin_t, cos_t], dim=1)
#     ], dim=1)  # [B, 2, 2]
#     grid_flat = grid.view(1, 2, -1).repeat(theta.size(0), 1, 1)
#     grid_new = s.view(-1, 1, 1) * torch.bmm(R, grid_flat)
#     grid_new[:, 0] += tx
#     grid_new[:, 1] += ty
#     flow = (grid_new - grid_flat).view(-1, 2, H, W)
#     return flow


def global_motion_to_flow(theta, H, W, device):
    tx, ty, log_s, sin_t, cos_t = theta.split(1, dim=1)
    s = torch.exp(log_s)

    # 像素坐标系
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0)  # [1, 2, H, W]

    # 中心点
    cx, cy = W / 2.0, H / 2.0

    # 平移到原点 -> 旋转缩放 -> 平移回来 -> 加上tx/ty
    R = torch.stack([
        torch.cat([cos_t, -sin_t], dim=1),
        torch.cat([sin_t, cos_t], dim=1)
    ], dim=1)  # [B, 2, 2]

    grid_flat = grid.view(1, 2, -1).repeat(theta.size(0), 1, 1)  # [B, 2, HW]

    # 中心化
    grid_centered = grid_flat.clone()
    grid_centered[:, 0] -= cx
    grid_centered[:, 1] -= cy

    # 旋转 + 缩放
    grid_transformed = s.view(-1, 1, 1) * torch.bmm(R, grid_centered)

    # 平移回来 + 全局平移
    grid_new = grid_transformed.clone()
    grid_new[:, 0] += cx + tx.squeeze(1).unsqueeze(1) * W  # tx 归一化到像素
    grid_new[:, 1] += cy + ty.squeeze(1).unsqueeze(1) * H

    # flow = 目标位置 - 原始位置
    flow = (grid_new - grid_flat).view(-1, 2, H, W)
    return flow


def upsample_flow(flow, target_h, target_w):
    """
    flow: [B, 2, h, w]
    return: [B, 2, target_h, target_w]
    """
    h, w = flow.shape[-2:]
    scale_y = target_h / h
    scale_x = target_w / w
    flow_up = F.interpolate(
        flow,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    flow_up[:, 0] *= scale_x
    flow_up[:, 1] *= scale_y
    return flow_up


# 组合模型
class CompleteModel(nn.Module):
    def __init__(self):
        super(CompleteModel, self).__init__()
        # args = Namespace(
        #     small = True, dropout = 0.1, alternate_corr = False,
        #     mixed_precision = False, corr_levels = 4, corr_radius = 4
        # )
        # args = Namespace(
        #     small = False, dropout = 0, alternate_corr = False,
        #     mixed_precision = False, corr_levels = 4, corr_radius = 8
        # )
        args = Namespace(
            small = True, dropout = 0, alternate_corr = False,
            mixed_precision = False, corr_levels = 4, corr_radius = 4
        )

        # 光流网络
        self.optical_flow_model = RAFT(args)

        # args = Namespace(
        #     small=False, dropout=0.1, mixed_precision=False, alternate_corr = False,
        #     corr_levels=4, corr_radius=12, upsample_learn=True,
        #     # GMA-specific
        #     position_only=False, position_and_content=True, num_heads=4,
        # )
        #
        # self.optical_flow_model = RAFTGMA(args)

        # 散斑解码器（散斑→物体）
        self.object_reconstructor = UNet()

        # self.motion_encoder = MotionEncoder(in_ch=1, base_ch=32)
        # self.motion_adapter = MotionAdapter(in_ch=64, out_ch=1)
        # self.global_motion_head = GlobalMotionHead(in_ch=64)

    def forward(self, speckle, speckle_for_unet=None, test=False):
        if speckle_for_unet is None:
            speckle_for_unet = speckle
        # # ---------- 1. Motion-friendly latent ----------
        # feat = self.motion_encoder(speckle)  # [T, C, H', W']
        # # 2. adapter → pseudo-image
        # feat_img = self.motion_adapter(feat)  # [T, 1, H', W']
        # # ---------- 2. Global motion ----------
        # global_flows = []
        # for t in range(len(feat) - 1):
        #     theta = self.global_motion_head(feat[t:t + 1], feat[t + 1:t + 2])
        #     flow_g = global_motion_to_flow(theta, feat.shape[-2], feat.shape[-1],feat.device)
        #     global_flows.append(flow_g)
        # global_flows = torch.cat(global_flows, dim=0)
        # # ---------- 3. Residual flow (RAFT) ----------
        # residual_flow = self.optical_flow_model(feat_img[:-1],feat_img[1:])
        # flow_forward = global_flows + residual_flow
        #
        # global_flows_bw = []
        # for t in range(len(feat) - 1):
        #     theta_bw = self.global_motion_head(feat[t + 1:t + 2], feat[t:t + 1])  # 反向
        #     flow_g_bw = global_motion_to_flow(theta_bw, feat.shape[-2], feat.shape[-1], feat.device)
        #     global_flows_bw.append(flow_g_bw)
        # global_flows_bw = torch.cat(global_flows_bw, dim=0)
        # residual_flow_bw = self.optical_flow_model(feat_img[1:], feat_img[:-1])  # 反向
        # flow_backward = global_flows_bw + residual_flow_bw
        #
        # # ---------- 4. Object reconstruction ----------
        # reconstructed_object = self.object_reconstructor(speckle_for_unet)
        # # ---------- 5. Warp speckle（仅用于 loss） ----------
        # # warped_speckle1 = warp(speckle[1:], flow_forward)
        # # warped_speckle2 = warp(speckle[:-1], flow_backward)
        # flow_forward_up = upsample_flow(flow_forward,speckle.shape[-2],speckle.shape[-1])
        # flow_backward_up = upsample_flow(flow_backward,speckle.shape[-2],speckle.shape[-1])
        # warped_speckle1 = warp(speckle[1:], flow_forward_up)
        # warped_speckle2 = warp(speckle[:-1], flow_backward_up)
        # fwd_occ, bwd_occ = forward_backward_consistency_check(flow_forward_up, flow_backward_up)

        # 预测物体位移
        object_displacement = self.optical_flow_model(speckle[:-1], speckle[1:])
        object_displacement_bw = self.optical_flow_model(speckle[1:], speckle[:-1])

        # 计算前向和后向遮挡
        fwd_occ, bwd_occ = forward_backward_consistency_check(object_displacement, object_displacement_bw)

        # 散斑图像重建物体
        reconstructed_object = self.object_reconstructor(speckle_for_unet)

        warped_speckle1 = warp(speckle[1:], object_displacement)
        warped_speckle2 = warp(speckle[:-1], object_displacement_bw)

        # if test:
        #     object_flow = self.optical_flow_model(reconstructed_object[:-1], reconstructed_object[1:])

        return {
            'flow_forward': object_displacement,
            'flow_backward': object_displacement_bw,
            'reconstructed_object': reconstructed_object,
            'warped_speckle1': warped_speckle1,
            'warped_speckle2': warped_speckle2,
            'fwd_occ': fwd_occ,
            'bwd_occ': bwd_occ,
            # 'object_flow': object_flow if test else None,
        }


class SimpleReconstructionModel(nn.Module):
    def __init__(self):
        super(SimpleReconstructionModel, self).__init__()
        self.object_reconstructor = UNet()

    def forward(self, speckle_seq):
        # 直接输入整个序列（比如 [T, C, H, W]）
        reconstructed_object_seq = self.object_reconstructor(speckle_seq)
        return {'reconstructed_object': reconstructed_object_seq}