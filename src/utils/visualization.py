"""Visualization utilities: flow rendering, overlay plots, debug images, and result saving."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import _make_colorwheel, flow_to_image, save_image

from src.losses.combined_loss import compute_flow_epe, warp

matplotlib.use('Agg')

# ==============================================================================
# Color Palette for Overlay (up to 30 frames)
# ==============================================================================

colors = [
    (1.0, 1.0, 0.0),   # T0  Yellow
    (1.0, 0.2, 0.2),   # T1  Red
    (0.2, 0.6, 1.0),   # T2  Blue
    (1.0, 0.3, 1.0),   # T3  Purple
    (0.3, 1.0, 0.3),   # T4  Green
    (1.0, 0.6, 0.0),   # T5  Orange
    (0.0, 1.0, 1.0),   # T6  Cyan
    (0.8, 0.1, 0.8),   # T7  Dark Purple
    (0.1, 0.8, 0.1),   # T8  Dark Green
    (1.0, 0.4, 0.0),   # T9  OrangeRed
    (0.0, 0.4, 1.0),   # T10 Deep Blue
    (1.0, 0.8, 0.2),   # T11 Light Yellow
    (0.8, 0.4, 0.4),   # T12 Light Red
    (0.4, 0.8, 1.0),   # T13 Light Blue
    (0.9, 0.5, 0.9),   # T14 Light Purple
    (0.5, 0.9, 0.5),   # T15 Light Green
    (0.7, 0.3, 0.0),   # T16 Brown Orange
    (0.0, 0.7, 0.7),   # T17 Dark Cyan
    (0.6, 0.0, 0.6),   # T18 Dark PurpleRed
    (0.0, 0.5, 0.0),   # T19 Dark Green
    (1.0, 0.0, 0.5),   # T20 Rose
    (0.5, 0.5, 1.0),   # T21 Light Sky Blue
    (0.0, 1.0, 0.5),   # T22 Mint Green
    (1.0, 0.7, 0.3),   # T23 Light Orange
    (0.4, 0.2, 0.0),   # T24 Dark Brown
    (0.7, 0.7, 1.0),   # T25 Pale Blue
    (0.9, 0.9, 0.4),   # T26 Cream Yellow
    (0.8, 0.6, 0.8),   # T27 Light Pink Purple
    (0.6, 0.8, 0.6),   # T28 Soft Green
    (1.0, 0.5, 0.2),   # T29 Tangerine
]

# ==============================================================================
# Flow Visualization
# ==============================================================================

def draw_flow(
    im: torch.Tensor,
    flow: torch.Tensor,
    step: int = 20,
    norm: int = 1,
) -> np.ndarray:
    """Draw optical flow arrows on a grayscale image at sampled points.

    Args:
        im: [1, H, W] grayscale image tensor
        flow: [2, H, W] optical flow tensor
        step: sampling interval in pixels
        norm: if truthy, scale arrow length relative to max flow

    Returns:
        vis: BGR image (H, W, 3) as uint8 ndarray
    """
    flow = flow.permute(1, 2, 0)
    _, h, w = im.shape
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    if norm:
        fx, fy = flow[y, x].T / abs(flow[y, x]).max() * step // 2
    else:
        fx, fy = flow[y, x].T

    x_t = torch.tensor(x, device=flow.device, dtype=flow.dtype)
    y_t = torch.tensor(y, device=flow.device, dtype=flow.dtype)
    ex = x_t + fx
    ey = y_t + fy
    lines = torch.vstack([x_t, y_t, ex, ey]).T.reshape(-1, 2, 2)
    lines = lines.cpu().numpy().astype(np.int32)

    vis = (im.cpu().numpy() * 255).astype(np.uint8)[0]
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.1)
        cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)

    return vis


def _flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """Convert optical flow to RGB using Middlebury color wheel.

    Args:
        flow: [2, H, W] tensor (u, v)

    Returns:
        flow_rgb: [3, H, W] uint8 tensor
    """
    device = flow.device
    if flow.ndim == 4:
        flow = flow[0]

    max_norm = torch.sqrt(torch.sum(flow ** 2, dim=0)).max()
    epsilon = torch.finfo(flow.dtype).eps
    normalized_flow = flow / (max_norm + epsilon)

    u = normalized_flow[0]
    v = normalized_flow[1]
    norm = torch.sqrt(u ** 2 + v ** 2)

    a = torch.atan2(-v, -u) / torch.pi
    colorwheel = _make_colorwheel().to(device)
    num_cols = colorwheel.shape[0]
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    H, W = flow.shape[1:]
    flow_rgb = torch.zeros((3, H, W), dtype=torch.uint8, device=device)

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm.unsqueeze(0) * (1 - col.unsqueeze(0))
        flow_rgb[c] = torch.floor(255 * col)

    return flow_rgb


def draw_flow_quiver(
    im: torch.Tensor, flow: torch.Tensor, step: int = 10
) -> np.ndarray:
    """Draw colored flow arrows on an image using matplotlib quiver.

    Args:
        im: [1, H, W] grayscale image tensor
        flow: [2, H, W] flow tensor (u, v)
        step: arrow sampling interval

    Returns:
        vis: BGR image (H, W, 3) as uint8 ndarray
    """
    if not isinstance(flow, torch.Tensor):
        flow = torch.tensor(flow)

    flow_rgb = _flow_to_rgb(flow)
    flow_rgb_np = flow_rgb.detach().cpu().numpy().transpose(1, 2, 0)

    flow_np = flow.detach().cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    H, W = u.shape
    y, x = np.mgrid[0:H:step, 0:W:step]

    flow_scale = 2.5
    u_sample = u[::step, ::step] * flow_scale
    v_sample = v[::step, ::step] * flow_scale

    colors_sample = flow_rgb_np[::step, ::step, :]
    im_np = im[0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(np.zeros_like(im_np), cmap='gray', origin='upper')
    ax.quiver(
        x, y, u_sample, v_sample,
        color=colors_sample.reshape(-1, 3) / 255.0,
        angles='xy', scale_units='xy', scale=1.0, width=0.008,
    )
    ax.axis('off')
    fig.tight_layout(pad=0)

    canvas = FigureCanvas(fig)
    canvas.draw()
    vis = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return vis


def draw_flow_track_colorwheel(
    base_image: torch.Tensor,
    flow_seq: List[torch.Tensor],
    step: int = 20,
    thickness: int = 2,
) -> np.ndarray:
    """Draw continuous flow tracks with color-coded arrows.

    Args:
        base_image: (H, W) tensor or ndarray
        flow_seq: list of [2, H, W] flow tensors
        step: sampling interval
        thickness: arrow line width

    Returns:
        track_img: BGR image
    """
    if isinstance(base_image, torch.Tensor):
        base = base_image.squeeze().cpu().numpy()
    else:
        base = base_image

    if base.ndim == 2:
        base = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = (base * 255).astype(np.uint8)

    track_img = base.copy()
    H, W = base.shape[:2]

    if isinstance(flow_seq, torch.Tensor):
        flow_seq = [flow_seq[t].cpu().numpy() for t in range(flow_seq.shape[0])]
    else:
        flow_seq = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in flow_seq]

    y, x = np.mgrid[0:H:step, 0:W:step]
    pts = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)

    for t in range(len(flow_seq)):
        flow = flow_seq[t]
        u = flow[0]
        v = flow[1]
        u_s = u[y, x].flatten()
        v_s = v[y, x].flatten()
        new_pts = pts + np.stack([u_s, v_s], axis=1)

        flow_rgb = _flow_to_rgb(torch.tensor(flow_seq[t])).cpu().numpy().transpose(1, 2, 0)
        color_sample = flow_rgb[::step, ::step].reshape(-1, 3)

        for i in range(len(pts)):
            c = color_sample[i].tolist()
            x1, y1 = pts[i]
            x2, y2 = new_pts[i]
            cv2.arrowedLine(
                track_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                c, thickness, tipLength=0.3,
            )
        pts = new_pts

    return track_img


# ==============================================================================
# Overlay Functions
# ==============================================================================

def make_overlay_no_legend(
    img_tensors: torch.Tensor, save_path: str
) -> None:
    """Overlay all frames without legend.

    Args:
        img_tensors: [T, 1, H, W] image tensor
        save_path: output PNG path
    """
    H, W = img_tensors.shape[-2:]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    for t in range(len(img_tensors)):
        arr = img_tensors[t, 0].cpu().numpy()
        arr = np.clip(arr, 0, 1)
        alpha = arr * 0.9
        rgba = np.zeros((H, W, 4))
        rgba[..., :3] = colors[t]
        rgba[..., 3] = alpha
        ax.imshow(rgba)

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


def make_overlay_with_flow_legend(
    img_tensors: torch.Tensor,
    flows: torch.Tensor,
    save_path: str,
) -> None:
    """Overlay all frames with flow displacement legend.

    Args:
        img_tensors: [T, 1, H, W]
        flows: [T-1, 2, H, W]
        save_path: output PNG path
    """
    H, W = img_tensors.shape[-2:]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    for t in range(len(img_tensors)):
        arr = img_tensors[t, 0].cpu().numpy()
        arr = np.clip(arr, 0, 1)
        alpha = arr * 0.8
        rgba = np.zeros((H, W, 4))
        rgba[..., :3] = colors[t]
        rgba[..., 3] = alpha
        ax.imshow(rgba)

    legend_w = 0.22
    legend_h = 0.27
    bar_h = legend_h / len(colors)

    for t in range(len(img_tensors)):
        y0 = 1 - 0.01 - (t + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4

        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.22, y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=colors[t],
            edgecolor='none',
        ))

        if t == 0:
            txt = " "
        else:
            flow_t = flows[t - 1].cpu().numpy()
            u, v = flow_t[0], flow_t[1]
            Hf, Wf = u.shape
            h0, h1 = int(Hf * 0.1), int(Hf * 0.9)
            w0, w1 = int(Wf * 0.1), int(Wf * 0.9)
            fx = np.mean(u[h0:h1, w0:w1])
            fy = np.mean(v[h0:h1, w0:w1])
            txt = f"x:{fx:.2f}, y:{fy:.2f}"

        ax.text(
            1 - legend_w - 0.16, y0 + bar_height / 2, txt,
            color='white', fontsize=8, ha='left', va='center',
            transform=ax.transAxes,
        )

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


def make_single_frame_overlay_with_flow_legend(
    img_tensor: torch.Tensor,
    flows: torch.Tensor,
    global_t: int,
    save_path: str,
) -> None:
    """Single-frame overlay with flow legend, highlighting the active frame.

    Args:
        img_tensor: [1, C, H, W]
        flows: [T-1, 2, H, W]
        global_t: this frame's index in the sequence
        save_path: output PNG path
    """
    H, W = img_tensor.shape[-2:]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    arr = img_tensor[0, 0].cpu().numpy()
    arr = np.clip(arr, 0, 1)
    alpha = arr * 0.8
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = colors[global_t]
    rgba[..., 3] = alpha
    ax.imshow(rgba)

    legend_w = 0.22
    legend_h = 0.27
    T = 5
    bar_h = legend_h / T
    for t in range(T):
        y0 = 1 - 0.01 - (t + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4
        is_active = (t == global_t)
        alpha_legend = 1.0 if is_active else 0.6

        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.22, y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=colors[t],
            edgecolor='none',
            alpha=alpha_legend,
        ))

        if t == 0:
            txt = " "
        else:
            flow_t = flows[t - 1].cpu().numpy()
            u, v = flow_t[0], flow_t[1]
            Hf, Wf = u.shape
            h0, h1 = int(Hf * 0.1), int(Hf * 0.9)
            w0, w1 = int(Wf * 0.1), int(Wf * 0.9)
            fx = np.mean(u[h0:h1, w0:w1])
            fy = np.mean(v[h0:h1, w0:w1])
            txt = f"x:{fx:.2f}, y:{fy:.2f}"

        ax.text(
            1 - legend_w - 0.16, y0 + bar_height / 2, txt,
            color='white', fontsize=8, ha='left', va='center',
            transform=ax.transAxes, alpha=alpha_legend,
        )

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


def make_single_frame_overlay_no_legend(
    img_tensor: torch.Tensor,
    global_t: int,
    save_path: str,
) -> None:
    """Single-frame overlay without legend.

    Args:
        img_tensor: [1, C, H, W]
        global_t: this frame's index (chooses color)
        save_path: output PNG path
    """
    H, W = img_tensor.shape[-2:]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    arr = img_tensor[0, 0].cpu().numpy()
    arr = np.clip(arr, 0, 1)
    alpha = arr * 0.9
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = colors[global_t]
    rgba[..., 3] = alpha
    ax.imshow(rgba)

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


# ==============================================================================
# Diff Visualization
# ==============================================================================

def compute_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute diff image: red = pred > gt, blue = gt > pred.

    Args:
        pred: BGR image
        gt: BGR image

    Returns:
        diff_img: BGR image with red/blue channels
    """
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff_pred = np.clip(pred_gray - gt_gray, 0, 255)
    diff_gt = np.clip(gt_gray - pred_gray, 0, 255)

    diff_pred_norm = (diff_pred / diff_pred.max() * 255) if diff_pred.max() > 0 else diff_pred
    diff_gt_norm = (diff_gt / diff_gt.max() * 255) if diff_gt.max() > 0 else diff_gt

    diff_img = np.zeros((*pred_gray.shape, 3), dtype=np.uint8)
    diff_img[:, :, 2] = diff_pred_norm
    diff_img[:, :, 0] = diff_gt_norm
    return diff_img


def add_diff_legend(diff_img: np.ndarray) -> np.ndarray:
    """Add a legend to a diff image in the upper-right corner.

    Args:
        diff_img: BGR diff image

    Returns:
        BGR image with legend
    """
    h, w, _ = diff_img.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()

    legend_w = 0.22
    legend_h = 0.10
    bar_h = legend_h / 2

    legend_colors = [(1, 0, 0), (0, 0, 1)]
    legend_texts = ["Pred > GT", "GT > Pred"]

    for i, (c, text) in enumerate(zip(legend_colors, legend_texts)):
        y0 = 1 - 0.01 - (i + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4
        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.09, y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=c,
            edgecolor='none',
        ))
        ax.text(
            1 - legend_w - 0.02, y0 + bar_height / 2, text,
            color='white', fontsize=8, ha='left', va='center',
            transform=ax.transAxes,
        )

    fig.canvas.draw()
    img_legend = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_legend = img_legend.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2BGR)
    return img_legend


def make_overlay_diff(
    no_legend_pred_path: str,
    no_legend_gt_path: str,
    save_path: str,
) -> None:
    """Create a diff image from two overlay images.

    Args:
        no_legend_pred_path: path to prediction overlay (no legend)
        no_legend_gt_path: path to GT overlay (no legend)
        save_path: output path
    """
    img1 = cv2.imread(no_legend_pred_path)
    img2 = cv2.imread(no_legend_gt_path)

    if img2.shape != img1.shape:
        img2 = cv2.resize(
            img2, (img1.shape[1], img1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    diff_img = compute_diff(img1, img2)
    diff_img = add_diff_legend(diff_img)
    cv2.imwrite(save_path, diff_img)


def save_abs_diff(
    img1: torch.Tensor, img2: torch.Tensor, save_path: str
) -> None:
    """Save absolute difference heatmap between two images.

    Args:
        img1: [1, 1, H, W] tensor
        img2: [1, 1, H, W] tensor
        save_path: output PNG path
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1[0, 0].cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[0, 0].cpu().numpy()

    diff = np.abs(img1 - img2)
    if diff.max() > 0:
        diff = diff / diff.max()

    h, w = diff.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    im = ax.imshow(diff, cmap='hot')
    ax.axis('off')

    cax = inset_axes(
        ax, width="38%", height="3%", loc="upper right", borderpad=0.5,
    )
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6, colors='white')

    plt.savefig(save_path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)


# ==============================================================================
# Image Label
# ==============================================================================

def add_label_to_frame(
    frame: np.ndarray,
    label: str,
    font_path: Optional[str] = None,
    font_size: int = 20,
) -> np.ndarray:
    """Add a text label to the upper-left corner of a frame.

    Args:
        frame: uint8 image array
        label: text to overlay
        font_path: optional path to a .ttf font file
        font_size: font size in points

    Returns:
        labeled frame as ndarray
    """
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.array(image)


# ==============================================================================
# Sequence Reconstruction
# ==============================================================================

def reconstruct_sequence_from_t(
    ref_frame: torch.Tensor,
    t: int,
    flows_fw: torch.Tensor,
    flows_bw: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Reconstruct full sequence from a single reference frame using flow warping.

    Args:
        ref_frame: [1, C, H, W] reference frame at time t
        t: reference frame index
        flows_fw: [T-1, 2, H, W] forward flows
        flows_bw: [T-1, 2, H, W] backward flows
        T: total number of frames

    Returns:
        seq: [T, C, H, W] reconstructed sequence
    """
    seq: List[Optional[torch.Tensor]] = [None] * T
    seq[t] = ref_frame.clone()

    # Warp to past (use forward flow)
    cur = ref_frame.clone()
    for step in range(1, t + 1):
        flow_fw = flows_fw[t - step].unsqueeze(0)
        cur = warp(cur, flow_fw)
        seq[t - step] = cur.clone()

    # Warp to future (use backward flow)
    cur = ref_frame.clone()
    for step in range(1, T - t):
        flow_bw = flows_bw[t + step - 1].unsqueeze(0)
        cur = warp(cur, flow_bw)
        seq[t + step] = cur.clone()

    return torch.cat([s for s in seq if s is not None], dim=0)


# ==============================================================================
# Debug Visualization
# ==============================================================================

def visualize_debug_images(
    output: Dict[str, Any],
    target: Dict[str, Any],
    epoch: int,
    batch_idx: int,
    save_dir: str,
    t: int,
    step: int = 20,
    save_every: int = 500,
) -> None:
    """Create a debug figure showing UNet output, warp results, flow, and occlusion.

    Args:
        output: model output dict
        target: ground truth dict
        epoch: current epoch
        batch_idx: current batch index
        save_dir: output directory
        t: random reference frame index
        step: flow sampling interval
        save_every: save frequency (every N batches)
    """
    if batch_idx % save_every != 0:
        return

    with torch.no_grad():
        num_cols, num_rows = 5, 4
        fig, axs = plt.subplots(
            num_rows, num_cols,
            figsize=(6 * num_cols, 6 * num_rows),
        )

        def plot_pair(
            axs: np.ndarray, row: int, col_start: int,
            pred_img: np.ndarray, gt_img: np.ndarray,
            title_pred: str, title_gt: str,
        ) -> None:
            axs[row, col_start].imshow(pred_img, cmap='gray')
            axs[row, col_start].set_title(title_pred)
            axs[row, col_start + 1].imshow(gt_img, cmap='gray')
            axs[row, col_start + 1].set_title(title_gt)

            overlay = np.zeros((*pred_img.shape, 3))
            overlay[..., 0] = np.clip(pred_img, 0, 1)
            overlay[..., 1] = np.clip(gt_img, 0, 1)
            axs[row + 1, col_start].imshow(overlay)
            axs[row + 1, col_start].set_title(f"{title_pred} vs {title_gt}")

            diff = np.abs(pred_img - gt_img)
            diff = diff / np.max(diff) if np.max(diff) > 0 else diff
            axs[row + 1, col_start + 1].imshow(diff, cmap='hot')
            axs[row + 1, col_start + 1].set_title(f"Diff: {title_pred}-{title_gt}")

        # UNet frame at t
        if output.get('reconstructed_object') is not None:
            unet_img = output['reconstructed_object'][t, 0].cpu().numpy()
            gt_img = target['object'][t, 0].cpu().numpy()
            plot_pair(axs, 0, 0, unet_img, gt_img, f"UNet_t{t}", f"GT_t{t}")

            # Warp comparison
            if t != 0:
                warp_img = warp(
                    output['reconstructed_object'][t].unsqueeze(0),
                    output['flow_forward'][t - 1].unsqueeze(0),
                )[0, 0].cpu().numpy()
            else:
                warp_img = warp(
                    output['reconstructed_object'][t].unsqueeze(0),
                    output['flow_backward'][t].unsqueeze(0),
                )[0, 0].cpu().numpy()

            gt_img = target['object'][t - 1 if t != 0 else 1, 0].cpu().numpy()
            plot_pair(
                axs, 2, 0, warp_img, gt_img,
                f"Warp_t{t - 1 if t != 0 else 1}",
                f"GT_t{t - 1 if t != 0 else 1}",
            )

        # Speckle warp
        if output.get('warped_speckle1') is not None:
            if t != 0:
                speckle_warped = output['warped_speckle1'][t - 1, 0].cpu().numpy()
            else:
                speckle_warped = output['warped_speckle2'][0, 0].cpu().numpy()
            speckle_gt = target['speckle'][t - 1 if t != 0 else 1, 0].cpu().numpy()
            plot_pair(
                axs, 0, 2, speckle_warped, speckle_gt,
                f"Speckle_t{t - 1 if t != 0 else 1}",
                f"Speckle_t{t - 1 if t != 0 else 1}",
            )

        # Flow visualization
        if output.get('flow_forward') is not None:
            flow = output['flow_forward'][t - 1 if t != 0 else 0].cpu().numpy()
            u, v = flow[0], flow[1]
            Hf, Wf = u.shape
            y, x = np.mgrid[0:Hf:step, 0:Wf:step]

            axs[0, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
            axs[0, 4].quiver(
                x, y, u[::step, ::step], v[::step, ::step],
                color='red', angles='xy', scale_units='xy', scale=1.0, width=0.003,
            )
            axs[0, 4].set_title("Predicted Flow")

            if target.get('flow') is not None:
                gt_flow = target['flow'][t - 1 if t != 0 else 0].cpu().numpy()
                gt_u, gt_v = gt_flow[0], gt_flow[1]
                axs[1, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
                axs[1, 4].quiver(
                    x, y, gt_u[::step, ::step], gt_v[::step, ::step],
                    color='red', angles='xy', scale_units='xy', scale=1.0,
                    width=0.003,
                )
                axs[1, 4].set_title("GT Flow")

            flow_color = flow_to_image(
                output['flow_forward'][t - 1 if t != 0 else 0].cpu()
            )
            axs[2, 4].imshow(flow_color.permute(1, 2, 0))
            axs[2, 4].set_title("Flow Forward Color")

            flow_bw = output['flow_backward'][t - 1 if t != 0 else 0].cpu().numpy()
            u, v = flow_bw[0], flow_bw[1]
            axs[3, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
            axs[3, 4].quiver(
                x, y, u[::step, ::step], v[::step, ::step],
                color='blue', angles='xy', scale_units='xy', scale=1.0, width=0.003,
            )
            axs[3, 4].set_title("Predicted Flow Bw")

        # Occlusion maps
        if output.get('fwd_occ') is not None:
            fwd_occ = output['fwd_occ'][0, 0].cpu().detach().numpy()
            axs[2, 2].imshow(fwd_occ, cmap='gray')
            axs[2, 2].set_title("Forward Occlusion")

            bwd_occ = output['bwd_occ'][0, 0].cpu().detach().numpy()
            axs[2, 3].imshow(bwd_occ, cmap='gray')
            axs[2, 3].set_title("Backward Occlusion")

        plt.tight_layout()
        debug_dir = os.path.join(save_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        plt.savefig(os.path.join(
            debug_dir, f"debug_epoch{epoch}_batch{batch_idx}.png"
        ))
        plt.close()


# ==============================================================================
# Validation Metrics
# ==============================================================================

def validate_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Compute validation flow magnitude and EPE with one forward pass.

    Returns:
        fw_mag, bw_mag, fw_epe, bw_epe (all float)
    """
    model.eval()
    total_fw_mag, total_bw_mag = 0.0, 0.0
    total_fw_epe, total_bw_epe = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            speckle_seq = batch['speckle_seq'].to(device)
            flow_gt = batch['flow_seq'].to(device)
            output = model(speckle_seq)
            flow_fw = output['flow_forward']
            flow_bw = output['flow_backward']

            total_fw_mag += torch.sqrt(
                flow_fw[:, 0] ** 2 + flow_fw[:, 1] ** 2
            ).mean().item()
            total_bw_mag += torch.sqrt(
                flow_bw[:, 0] ** 2 + flow_bw[:, 1] ** 2
            ).mean().item()

            fw_epe, _ = compute_flow_epe(flow_fw, flow_gt)
            bw_epe, _ = compute_flow_epe(flow_bw, -flow_gt)
            total_fw_epe += fw_epe.item()
            total_bw_epe += bw_epe.item()
            count += 1

    model.train()
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_fw_mag / count, total_bw_mag / count,
        total_fw_epe / count, total_bw_epe / count,
    )


# ==============================================================================
# Training Curve Plotting
# ==============================================================================

def plot_losses(
    train_losses: Dict[str, List[float]], save_dir: str
) -> None:
    """Plot training loss curves for all tracked loss keys.

    Args:
        train_losses: {loss_name: [values_per_epoch]}
        save_dir: output directory
    """
    num_losses = len(train_losses)
    ncols = min(3, num_losses)
    nrows = math.ceil(num_losses / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
    )
    if num_losses == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (key, values) in enumerate(train_losses.items()):
        ax = axes[idx]
        ax.plot(values, label=f'Train {key}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{key.capitalize()} Loss')
        ax.legend()

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()


def plot_flow_magnitude(
    mag_history: Dict[str, List[float]], save_dir: str
) -> None:
    """Plot validation flow magnitude over epochs.

    Args:
        mag_history: {'fw': [float, ...], 'bw': [float, ...]}
        save_dir: output directory
    """
    epochs = list(range(1, len(mag_history['fw']) + 1))
    plt.figure(figsize=(8, 5))
    if mag_history['fw']:
        plt.plot(
            epochs, mag_history['fw'],
            marker='o', linewidth=1.5, color='steelblue',
            label='Val Flow Magnitude (forward)',
        )
    if mag_history['bw']:
        plt.plot(
            epochs, mag_history['bw'],
            marker='s', linewidth=1.5, color='tomato',
            label='Val Flow Magnitude (backward)',
        )
    plt.xlabel('Epoch')
    plt.ylabel('Mean Flow Magnitude (pixels)')
    plt.title('Validation Flow Magnitude per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_flow_magnitude.png'), dpi=150)
    plt.close()


def plot_val_epe(
    val_epe_history: Dict[str, List[float]], save_dir: str
) -> None:
    """Plot validation EPE over epochs.

    Args:
        val_epe_history: {'fw': [float, ...], 'bw': [float, ...]}
        save_dir: output directory
    """
    epochs = list(range(1, len(val_epe_history['fw']) + 1))
    plt.figure(figsize=(8, 5))
    if val_epe_history['fw']:
        plt.plot(
            epochs, val_epe_history['fw'],
            marker='o', linewidth=1.5, color='steelblue',
            label='Val EPE (forward)',
        )
    if val_epe_history['bw']:
        plt.plot(
            epochs, val_epe_history['bw'],
            marker='s', linewidth=1.5, color='tomato',
            label='Val EPE (backward)',
        )
    plt.xlabel('Epoch')
    plt.ylabel('EPE (pixels)')
    plt.title('Validation Flow EPE per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_flow_epe.png'), dpi=150)
    plt.close()


# ==============================================================================
# Simple Result Saving (U-Net only)
# ==============================================================================

def save_simple_results(
    batch_idx: int,
    output: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    directories: Dict[str, str],
) -> None:
    """Save U-Net-only reconstruction results.

    Args:
        batch_idx: batch index
        output: model output dict (has 'reconstructed_object')
        target: ground truth dict (has 'object')
        directories: output directory paths dict
    """
    recon_objects = output['reconstructed_object']
    object_seq = target['object']
    T = recon_objects.shape[0]
    for t in range(T):
        save_image(
            recon_objects[t:t + 1],
            os.path.join(directories['recon_object1'], f'batch{batch_idx:04d}_t{t}.png'),
        )
        save_image(
            object_seq[t:t + 1],
            os.path.join(directories['origin_object1'], f'batch{batch_idx:04d}_t{t}.png'),
        )


# ==============================================================================
# Full Result Saving (CompleteModel)
# ==============================================================================

def save_all_results(
    batch_idx: int,
    output: Dict[str, Any],
    target: Dict[str, torch.Tensor],
    directories: Dict[str, str],
    use_speckle: bool = False,
    rotate_input: bool = False,
) -> None:
    """Save all visualization results for synthetic/test data evaluation.

    Args:
        batch_idx: batch index
        output: model output dict
        target: ground truth dict with 'object', 'flow', 'speckle'
        directories: output directory paths dict
        use_speckle: whether to save speckle images (experimental mode only)
        rotate_input: whether to rotate GT objects by 180 degrees
    """
    # GT object sequence
    gt_objects = target['object']
    recon_objects = output.get('reconstructed_object', None)
    flows_fw = output.get('flow_forward', None)
    flows_bw = output.get('flow_backward', None)
    has_flow = flows_fw is not None
    T = gt_objects.shape[0]

    # Optional rotation
    if rotate_input:
        gt_objects = torch.rot90(gt_objects, k=2, dims=[-2, -1])

    # Optional speckle
    if use_speckle:
        gt_speckles = target['speckle']

    # ---- Save reconstructed and GT objects per frame ----
    for t in range(T):
        save_image(
            recon_objects[t:t + 1],
            os.path.join(
                directories['recon_object'],
                f'recon_object_img_{batch_idx}_frame_{t}.png',
            ),
        )
        save_image(
            gt_objects[t:t + 1],
            os.path.join(
                directories['origin_object'],
                f'origin_object_img_{batch_idx}_frame_{t}.png',
            ),
        )
        save_abs_diff(
            recon_objects[t:t + 1],
            gt_objects[t:t + 1],
            os.path.join(
                directories['diff_recon_vs_gt'],
                f"diff_recon_vs_gt_batch{batch_idx}_frame{t}.png",
            ),
        )

    # ---- Flow EPE difference maps ----
    if has_flow:
        pred_flows_fw = flows_fw.cpu().numpy()
        pred_flows_bw = flows_bw.cpu().numpy()
        gt_flows_fw = target['flow'].cpu().numpy()
        gt_flows_bw = -gt_flows_fw

        base = gt_objects[0][0].cpu().numpy()
        base_black = np.zeros_like(base)
        base_bgr = cv2.cvtColor(
            (base_black * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )

        for obj_idx in range(min(4, pred_flows_fw.shape[0])):
            # Forward flow EPE
            pred_obj_fw = pred_flows_fw[obj_idx]
            gt_obj_fw = gt_flows_fw[obj_idx]
            diff_u_fw = pred_obj_fw[0] - gt_obj_fw[0]
            diff_v_fw = pred_obj_fw[1] - gt_obj_fw[1]
            epe_fw = np.sqrt(diff_u_fw ** 2 + diff_v_fw ** 2)
            epe_fw = epe_fw / (epe_fw.max() + 1e-6)
            epe_color_fw = cv2.applyColorMap(
                (epe_fw * 255).astype(np.uint8), cv2.COLORMAP_HOT
            )
            diff_fw = cv2.addWeighted(base_bgr, 0, epe_color_fw, 1, 0)
            cv2.imwrite(
                os.path.join(
                    directories['flow_fw_diff_each'],
                    f'flow_epe_batch{batch_idx}_obj{obj_idx}.png',
                ),
                diff_fw,
            )

            # Backward flow EPE
            pred_obj_bw = pred_flows_bw[obj_idx]
            gt_obj_bw = gt_flows_bw[obj_idx]
            diff_u_bw = pred_obj_bw[0] - gt_obj_bw[0]
            diff_v_bw = pred_obj_bw[1] - gt_obj_bw[1]
            epe_bw = np.sqrt(diff_u_bw ** 2 + diff_v_bw ** 2)
            epe_bw = epe_bw / (epe_bw.max() + 1e-6)
            epe_color_bw = cv2.applyColorMap(
                (epe_bw * 255).astype(np.uint8), cv2.COLORMAP_HOT
            )
            diff_bw = cv2.addWeighted(base_bgr, 0, epe_color_bw, 1, 0)
            cv2.imwrite(
                os.path.join(
                    directories['flow_bw_diff_each'],
                    f'flow_epe_batch{batch_idx}_obj{obj_idx}.png',
                ),
                diff_bw,
            )

    # ---- Per-frame loop ----
    for t in range(T - 1):
        gt1 = gt_objects[t:t + 1]

        if use_speckle:
            save_image(
                gt_speckles[t],
                os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{t}.png'),
            )
            save_image(
                gt_speckles[t + 1],
                os.path.join(directories['speckle2'], f'speckle2_{batch_idx}_{t + 1}.png'),
            )

        if has_flow:
            if batch_idx < 10:
                from src.utils.io_utils import save_flow_to_csv
                save_flow_to_csv(
                    flows_fw[t].unsqueeze(0).cpu().numpy(),
                    directories['model_fw'], batch_idx, t,
                )
                save_flow_to_csv(
                    target['flow'][t].unsqueeze(0).cpu().numpy(),
                    directories['groundtruth_fw'], batch_idx, t,
                )

            # Forward flow visualization
            flow_im = flow_to_image(flows_fw[t].cpu())
            save_image(
                flow_im.float() / 255.0,
                os.path.join(
                    directories['flow_colorimage_fw'],
                    f'flow_img_{batch_idx}_frame_{t}.png',
                ),
            )
            flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), flows_fw[t].cpu())
            cv2.imwrite(
                os.path.join(
                    directories['flow_arrow_fw'],
                    f'flowarrow_f_img_{batch_idx}_frame_{t}.png',
                ),
                flow_arrow,
            )

            # Backward flow visualization
            flow_im = flow_to_image(flows_bw[t].cpu())
            save_image(
                flow_im.float() / 255.0,
                os.path.join(
                    directories['flow_colorimage_bw'],
                    f'flow_img_{batch_idx}_frame_{t}.png',
                ),
            )
            flow_arrow = draw_flow_quiver(gt1.cpu().squeeze(0), flows_bw[t].cpu())
            cv2.imwrite(
                os.path.join(
                    directories['flow_arrow_bw'],
                    f'flowarrow_b_img_{batch_idx}_frame_{t}.png',
                ),
                flow_arrow,
            )

        # GT flow
        if 'flow' in target:
            gt_flow = target['flow'][t]
            gt_flow_color = flow_to_image(gt_flow.cpu())
            save_image(
                gt_flow_color.float() / 255.0,
                os.path.join(
                    directories['gt_flow_colorimage'],
                    f'gt_flow_img_{batch_idx}_frame_{t}.png',
                ),
            )
            gt_flow_arrow = draw_flow_quiver(
                gt1.cpu().squeeze(0), gt_flow.cpu()
            )
            cv2.imwrite(
                os.path.join(
                    directories['gt_flow_arrow_fw'],
                    f'gt_flowarrow_img_{batch_idx}_frame_{t}.png',
                ),
                gt_flow_arrow,
            )

            gt_flow_bw = -gt_flow
            gt_flow_bw_color = flow_to_image(gt_flow_bw.cpu())
            save_image(
                gt_flow_bw_color.float() / 255.0,
                os.path.join(
                    directories['gt_flow_colorimage_bw'],
                    f'gt_flow_img_{batch_idx}_frame_{t}.png',
                ),
            )
            gt_flow_arrow_bw = draw_flow_quiver(
                gt1.cpu().squeeze(0), gt_flow_bw.cpu()
            )
            cv2.imwrite(
                os.path.join(
                    directories['gt_flow_arrow_bw'],
                    f'gt_flowarrow_img_{batch_idx}_frame_{t}.png',
                ),
                gt_flow_arrow_bw,
            )

    # ============ GT Overlay ============
    gt_legend = os.path.join(
        directories["overlay_results_origin_object"],
        f"GT_overlay_b{batch_idx}.png",
    )
    gt_nl = os.path.join(
        directories["overlay_results_nl_origin_object"],
        f"GT_nl_b{batch_idx}.png",
    )
    make_overlay_with_flow_legend(gt_objects.cpu(), target['flow'].cpu(), gt_legend)
    make_overlay_no_legend(gt_objects.cpu(), gt_nl)

    # ============ UNet Overlay ============
    unet_legend = os.path.join(
        directories["overlay_results_recon_object"],
        f"UNet_overlay_b{batch_idx}.png",
    )
    unet_nl = os.path.join(
        directories["overlay_results_nl_recon_object"],
        f"UNet_nl_b{batch_idx}.png",
    )
    make_overlay_with_flow_legend(recon_objects.cpu(), flows_fw.cpu(), unet_legend)
    make_overlay_no_legend(recon_objects.cpu(), unet_nl)

    # ============ Diff ============
    make_overlay_diff(
        unet_nl, gt_nl,
        os.path.join(
            directories["diff_results_recon_obj_origin_obj"],
            f"UNet_vs_GT_diff_b{batch_idx}.png",
        ),
    )

    # ============ Per-frame warp & overlay ============
    all_warped_frames: List[Tuple[int, torch.Tensor]] = []

    for t in range(T):
        # Reconstruct full sequence from frame t
        warped_seq_t = reconstruct_sequence_from_t(
            ref_frame=recon_objects[t:t + 1],
            t=t,
            flows_fw=flows_fw,
            flows_bw=flows_bw,
            T=T,
        )

        overlay_path_nl = os.path.join(
            directories['warp_from_each_t_nl_overlay'],
            f'overlay_from_batch{batch_idx}_t{t}.png',
        )
        overlay_path = os.path.join(
            directories['warp_from_each_t_overlay'],
            f'overlay_from_batch{batch_idx}_t{t}.png',
        )
        make_overlay_no_legend(warped_seq_t.cpu(), overlay_path_nl)
        make_overlay_with_flow_legend(
            warped_seq_t.cpu(), flows_fw.cpu(), overlay_path,
        )

        # GT overlay for diff comparison
        gt_overlay_path = os.path.join(
            directories['overlay_gt_each_t'],
            f'gt_overlay_batch{batch_idx}_t{t}.png',
        )
        make_overlay_no_legend(gt_objects.cpu(), gt_overlay_path)
        make_overlay_diff(
            overlay_path_nl, gt_overlay_path,
            os.path.join(
                directories['diff_overlay_each_t'],
                f'diff_batch{batch_idx}_t{t}.png',
            ),
        )

        # Single-frame overlays
        make_single_frame_overlay_with_flow_legend(
            img_tensor=gt_objects[t:t + 1],
            flows=target['flow'],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
        )
        make_single_frame_overlay_with_flow_legend(
            img_tensor=recon_objects[t:t + 1],
            flows=flows_fw,
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
        )
        make_single_frame_overlay_no_legend(
            img_tensor=gt_objects[t:t + 1],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_nl_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
        )
        make_single_frame_overlay_no_legend(
            img_tensor=recon_objects[t:t + 1],
            global_t=t,
            save_path=os.path.join(
                directories['single_overlay_nl_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
        )
        make_overlay_diff(
            os.path.join(
                directories['single_overlay_nl_gt'],
                f'gt_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
            os.path.join(
                directories['single_overlay_nl_unet'],
                f'unet_overlay_frame_batch{batch_idx}_t{t}.png',
            ),
            os.path.join(
                directories['diff_single_each_t'],
                f'diff_batch{batch_idx}_t{t}.png',
            ),
        )

        # Warp forward in time (to future frames)
        cur = recon_objects[t:t + 1].clone()
        for step in range(1, T - t):
            flow_bw_step = flows_bw[t + step - 1].unsqueeze(0)
            cur = warp(cur, flow_bw_step)
            target_frame_idx = t + step
            all_warped_frames.append((t, cur.clone()))
            save_path = os.path.join(
                directories['single_overlay_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            make_single_frame_overlay_with_flow_legend(
                img_tensor=cur,
                flows=flows_fw,
                global_t=target_frame_idx,
                save_path=save_path,
            )
            save_path_nl = os.path.join(
                directories['single_overlay_nl_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            make_single_frame_overlay_no_legend(
                img_tensor=cur,
                global_t=target_frame_idx,
                save_path=save_path_nl,
            )
            save_path_raw = os.path.join(
                directories['single_overlay_nl_nc_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            save_image(cur.clamp(0, 1), save_path_raw)

        # Warp backward in time (to past frames)
        cur = recon_objects[t:t + 1].clone()
        for step in range(1, t + 1):
            flow_fw_step = flows_fw[t - step].unsqueeze(0)
            cur = warp(cur, flow_fw_step)
            target_frame_idx = t - step
            all_warped_frames.append((t, cur.clone()))
            save_path = os.path.join(
                directories['single_overlay_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            make_single_frame_overlay_with_flow_legend(
                img_tensor=cur,
                flows=flows_fw,
                global_t=target_frame_idx,
                save_path=save_path,
            )
            save_path_nl = os.path.join(
                directories['single_overlay_nl_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            make_single_frame_overlay_no_legend(
                img_tensor=cur,
                global_t=target_frame_idx,
                save_path=save_path_nl,
            )
            save_path_raw = os.path.join(
                directories['single_overlay_nl_nc_warp'],
                f'batch{batch_idx}_t{t}_warped_{target_frame_idx}.png',
            )
            save_image(cur.clamp(0, 1), save_path_raw)

    # All-warped-frames total overlay
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
            rgba[..., :3] = colors[source_t]
            rgba[..., 3] = arr * 0.9
            ax.imshow(rgba)
        ax.set_axis_off()
        save_path_all = os.path.join(
            directories['single_overlay_nl_warp_all_t'],
            f'batch{batch_idx}_ALL_warped_overlay.png',
        )
        plt.savefig(
            save_path_all, dpi=100,
            bbox_inches=None, pad_inches=0, facecolor='black',
        )
        plt.close()


# ==============================================================================
# Experimental Result Saving
# ==============================================================================

def save_experimental_results(
    batch_idx: int,
    output: Dict[str, Any],
    speckle_seq: torch.Tensor,
    directories: Dict[str, str],
) -> None:
    """Save visualization results for experimental data (no GT flow/object).

    Args:
        batch_idx: batch index
        output: model output dict with flow_forward, reconstructed_object
        speckle_seq: input speckle sequence [T, C, H, W]
        directories: output directory paths dict
    """
    from src.utils.io_utils import save_flow_to_csv

    flow_forward = output['flow_forward']
    recon_objects = output['reconstructed_object']
    num_frames = recon_objects.shape[0]

    for t in range(num_frames - 1):
        flow_t = flow_forward[t:t + 1]
        # Rotate 180 degrees for experimental data alignment
        flow_rot = torch.rot90(flow_t, k=2, dims=[-2, -1])
        flow_rot[:, 0, :, :] *= -1
        flow_rot[:, 1, :, :] *= -1

        if batch_idx < 10:
            save_flow_to_csv(
                flow_rot.cpu().numpy(), directories['model_fw'], batch_idx, t
            )

        # Save speckle frames
        save_image(
            speckle_seq[t],
            os.path.join(directories['speckle1'], f'speckle1_{batch_idx}_{t}.png'),
        )
        save_image(
            speckle_seq[t + 1],
            os.path.join(directories['speckle2'], f'speckle2_{batch_idx}_{t + 1}.png'),
        )

        # Rotate reconstructed objects
        recon_obj1 = recon_objects[t:t + 1]
        recon_obj2 = recon_objects[t + 1:t + 2]
        recon_obj1 = torch.rot90(recon_obj1, k=2, dims=[-2, -1])
        recon_obj2 = torch.rot90(recon_obj2, k=2, dims=[-2, -1])

        # Warp obj2 to obj1 using rotated flow
        warp_obj1 = warp(recon_obj2, flow_rot)

        save_image(
            warp_obj1,
            os.path.join(
                directories['reconstructed_object1'],
                f'recon_object1_{batch_idx}_{t}.png',
            ),
        )
        save_image(
            recon_obj2,
            os.path.join(
                directories['reconstructed_object2'],
                f'recon_object2_{batch_idx}_{t + 1}.png',
            ),
        )

        # Flow visualization
        flow_img = flow_to_image(flow_rot.cpu().squeeze(0))
        save_image(
            flow_img.float() / 255,
            os.path.join(
                directories['flow_colorimage'],
                f'flow_{batch_idx}_{t}_to_{t + 1}.png',
            ),
        )

        flow_arrow = draw_flow_quiver(
            recon_obj1.cpu().squeeze(0), flow_rot.cpu().squeeze(0)
        )
        cv2.imwrite(
            os.path.join(
                directories['flow_arrow'],
                f'flow_arrow_{batch_idx}_{t}_to_{t + 1}.png',
            ),
            flow_arrow,
        )
