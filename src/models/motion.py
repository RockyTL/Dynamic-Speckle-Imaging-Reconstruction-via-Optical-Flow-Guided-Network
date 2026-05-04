"""Motion encoding and global motion estimation modules (experimental).

These modules map speckle to motion-friendly latent features, adapt them for
RAFT-compatible input, and estimate global motion parameters. They are kept
here as experimental components for future use.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """Map speckle to motion-friendly latent.

    The goal is to attenuate random high-frequency speckle noise
    so that geometric changes can be captured by flow estimation.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, C, H, W] -> return: [T, C, H', W']"""
        return self.net(x)


class MotionAdapter(nn.Module):
    """Compress motion latent channels (C=64) to RAFT-compatible input (C=1)."""

    def __init__(self, in_ch: int = 64, out_ch: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GlobalMotionHead(nn.Module):
    """Predict global motion parameters (tx, ty, log_s, sin_theta, cos_theta)
    from latent features."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return self.net(f2 - f1)


def global_motion_to_flow(
    theta: torch.Tensor, H: int, W: int, device: torch.device
) -> torch.Tensor:
    """Convert global motion parameters to dense flow field.

    Uses pixel-coordinate system with centering for rotation/scaling
    around the image center.

    Args:
        theta: [B, 5] tensor of (tx, ty, log_s, sin_theta, cos_theta)
        H: target height
        W: target width
        device: torch device

    Returns:
        flow: [B, 2, H, W]
    """
    tx, ty, log_s, sin_t, cos_t = theta.split(1, dim=1)
    s = torch.exp(log_s)

    # Pixel coordinate grid
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    grid = torch.stack([x, y], dim=0).unsqueeze(0)  # [1, 2, H, W]

    cx, cy = W / 2.0, H / 2.0

    # Build rotation matrix
    R = torch.stack([
        torch.cat([cos_t, -sin_t], dim=1),
        torch.cat([sin_t, cos_t], dim=1),
    ], dim=1)  # [B, 2, 2]

    grid_flat = grid.view(1, 2, -1).repeat(theta.size(0), 1, 1)  # [B, 2, HW]

    # Center -> rotate+scale -> uncenter + global translation
    grid_centered = grid_flat.clone()
    grid_centered[:, 0] -= cx
    grid_centered[:, 1] -= cy

    grid_transformed = s.view(-1, 1, 1) * torch.bmm(R, grid_centered)

    grid_new = grid_transformed.clone()
    grid_new[:, 0] += cx + tx.squeeze(1).unsqueeze(1) * W
    grid_new[:, 1] += cy + ty.squeeze(1).unsqueeze(1) * H

    flow = (grid_new - grid_flat).view(-1, 2, H, W)
    return flow


def upsample_flow(
    flow: torch.Tensor, target_h: int, target_w: int
) -> torch.Tensor:
    """Upsample flow field to target resolution.

    Args:
        flow: [B, 2, h, w]
        target_h: target height
        target_w: target width

    Returns:
        upsampled: [B, 2, target_h, target_w]
    """
    h, w = flow.shape[-2:]
    scale_y = target_h / h
    scale_x = target_w / w
    flow_up = F.interpolate(
        flow, size=(target_h, target_w), mode='bilinear', align_corners=False,
    )
    flow_up[:, 0] *= scale_x
    flow_up[:, 1] *= scale_y
    return flow_up
