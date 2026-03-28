import os
import shutil
import zipfile
from zipfile import ZipFile
import h5py
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
import math

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils.Sundries import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from prettytable import PrettyTable
from torchvision.utils import _make_colorwheel, flow_to_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# 在间隔分开的像素采样点处绘制光流
def draw_flow(im, flow, step=20, norm=1):
    flow = flow.permute(1, 2, 0)

    _, h, w = im.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

    if norm:
        fx, fy = flow[y, x].T / abs(flow[y, x]).max() * step // 2
    else:
        fx, fy = flow[y, x].T
    # 将 x 和 y 转换为 PyTorch 张量
    x = torch.tensor(x, device=flow.device, dtype=flow.dtype)
    y = torch.tensor(y, device=flow.device, dtype=flow.dtype)

    # 创建线的终点
    ex = x + fx
    ey = y + fy
    lines = torch.vstack([x, y, ex, ey]).T.reshape(-1, 2, 2)
    lines = lines.cpu().numpy().astype(np.uint32)

    # 创建图像并绘制
    vis = (im.cpu().numpy()*255).astype(np.uint8)[0]
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.1)
        cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)

    return vis


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


def visualize_debug_images(output, target, epoch, batch_idx, save_dir, t, step=20, save_every=500):
    if batch_idx % save_every != 0:
        return

    with torch.no_grad():
        num_cols, num_rows = 5, 4
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))

        # 小工具函数：画预测图、GT图、叠加图、差异图
        def plot_pair(axs, row, col_start, pred_img, gt_img, title_pred, title_gt):
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

        # --- 画 unet 第 t 帧 ---
        if output['reconstructed_object'] is not None:
            unet_img = output['reconstructed_object'][t, 0].cpu().numpy()
            gt_img = target['object'][t, 0].cpu().numpy()
            plot_pair(axs, 0, 0, unet_img, gt_img, f"UNet_t{t}", f"GT_t{t}")

        # --- 画 warp 对应的帧 ---
            if t != 0:
                warp_img = warp(
                    output['reconstructed_object'][t].unsqueeze(0), output['flow_forward'][t-1].unsqueeze(0))[0,0].cpu().numpy()
            else:
                warp_img = warp(
                    output['reconstructed_object'][t].unsqueeze(0), output['flow_backward'][t].unsqueeze(0))[0,0].cpu().numpy()

            gt_img = target['object'][t-1 if t != 0 else 1, 0].cpu().numpy()
            plot_pair(axs, 2, 0, warp_img, gt_img, f"Warp_t{t - 1 if t != 0 else 1}",
                      f"GT_t{t - 1 if t != 0 else 1}")

        # --- speckle 对应帧 ---
        if output['warped_speckle1'] is not None:
            if t != 0:
                speckle_warped = output['warped_speckle1'][t-1, 0].cpu().numpy()
            else:
                speckle_warped = output['warped_speckle2'][0, 0].cpu().numpy()
            speckle_gt = target['speckle'][t-1 if t != 0 else 1, 0].cpu().numpy()
            plot_pair(axs, 0, 2, speckle_warped, speckle_gt, f"Speckle_t{t-1 if t != 0 else 1}",
                      f"Speckle_t{t-1 if t != 0 else 1}")

        # 光流、遮挡保持原逻辑
        # -------------------------------------------------
        if output.get('flow_forward') is not None:
            flow = output['flow_forward'][t-1 if t!=0 else 0].cpu().numpy()
            u, v = flow[0], flow[1]
            H, W = u.shape
            y, x = np.mgrid[0:H:step, 0:W:step]
            axs[0, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
            axs[0, 4].quiver(x, y, u[::step, ::step], v[::step, ::step], color='red', angles='xy', scale_units='xy', scale=1.0, width=0.003)
            axs[0, 4].set_title("Predicted Flow")

            if target.get('flow') is not None:
                gt_flow = target['flow'][t-1 if t!=0 else 0].cpu().numpy()
                gt_u, gt_v = gt_flow[0], gt_flow[1]
                axs[1, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
                axs[1, 4].quiver(x, y, gt_u[::step, ::step], gt_v[::step, ::step], color='red', angles='xy', scale_units='xy', scale=1.0, width=0.003)
                axs[1, 4].set_title("GT Flow")

            flow = output['flow_forward'][t-1 if t!=0 else 0].cpu()
            flow_color_image = flow_to_image(flow)
            axs[2, 4].imshow(flow_color_image.permute(1, 2, 0))
            axs[2, 4].set_title("Flow Forward Color")

            flow_bw = output['flow_backward'][t-1 if t!=0 else 0].cpu().numpy()
            u, v = flow_bw[0], flow_bw[1]
            axs[3, 4].imshow(target['object'][t, 0].cpu().numpy(), cmap='gray')
            axs[3, 4].quiver(x, y, u[::step, ::step], v[::step, ::step], color='blue', angles='xy', scale_units='xy', scale=1.0, width=0.003)
            axs[3, 4].set_title("Predicted Flow Bw")

        # 添加前向遮挡图（如果可用）
        if output['fwd_occ'] is not None:
            fwd_occ = output['fwd_occ'][0,0].cpu().detach().numpy()
            axs[2, 2].imshow(fwd_occ, cmap='gray')
            axs[2, 2].set_title("Forward Occlusion")

            bwd_occ = output['bwd_occ'][0,0].cpu().detach().numpy()
            axs[2, 3].imshow(bwd_occ, cmap='gray')
            axs[2, 3].set_title("Backward Occlusion")

        plt.tight_layout()
        debug_dir = os.path.join(save_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        plt.savefig(os.path.join(debug_dir, f"debug_epoch{epoch}_batch{batch_idx}.png"))
        plt.close()


# 使用 matplotlib 的 quiver 方式绘制光流并转换为图像（uint8 格式）
def draw_flow_quiver(im, flow, step=10):
    """
    使用 matplotlib 的 quiver 函数在灰度图上绘制光流箭头，并返回 uint8 图像。
    箭头颜色使用标准光流可视化的色彩映射（Middlebury colorwheel）。

    Args:
        im (Tensor): 输入图像，shape 为 (1, H, W)，灰度图。
        flow (Tensor): 光流张量，shape 为 (2, H, W)，第 0 维是 u，第 1 维是 v。
        step (int): 光流箭头的采样间隔。
    Returns:
        vis (np.ndarray): 可视化图像，shape 为 (H, W, 3)，类型为 np.uint8。
    """
    # 确保输入是正确的类型
    if not isinstance(flow, torch.Tensor):
        flow = torch.tensor(flow)

    # 获取完整的光流彩色表示
    flow_rgb = _flow_to_rgb(flow)
    flow_rgb_np = flow_rgb.detach().cpu().numpy().transpose(1, 2, 0)  # 转为HWC格式

    # 提取采样点的光流向量和对应颜色
    flow = flow.detach().cpu().numpy()
    u, v = flow[0], flow[1]
    H, W = u.shape
    y, x = np.mgrid[0:H:step, 0:W:step]

    flow_scale = 2.5
    u_sample = u[::step, ::step] * flow_scale
    v_sample = v[::step, ::step] * flow_scale

    # 获取采样点对应的颜色
    colors_sample = flow_rgb_np[::step, ::step, :]

    im_np = im[0].detach().cpu().numpy()

    # 创建 matplotlib 图并绘图
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    # fig.patch.set_facecolor('black')
    # ax.set_facecolor('black')

    ax.imshow(np.zeros_like(im_np), cmap='gray', origin='upper')
    # ax.imshow(np.ones_like(im_np, dtype=np.uint8)*255, cmap='gray', origin='upper', vmin=0, vmax=255)

    # 使用采样的颜色绘制箭头
    ax.quiver(x, y, u_sample, v_sample, color=colors_sample.reshape(-1, 3) / 255.0,
                  angles='xy', scale_units='xy', scale=1.0, width=0.008)

    ax.axis('off')
    fig.tight_layout(pad=0)

    # 将图像渲染为 numpy 数组
    canvas = FigureCanvas(fig)
    canvas.draw()
    vis = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # 关闭图形防止内存泄露
    return vis


def _flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """
    将光流转换为RGB彩色图像，使用标准的Middlebury色彩映射。

    Args:
        flow (Tensor): 光流张量，shape 为 (2, H, W)。

    Returns:
        flow_rgb (Tensor): RGB图像张量，shape 为 (3, H, W)，类型为torch.uint8。
    """
    device = flow.device
    if flow.ndim == 4:
        flow = flow[0]  # 取第一个batch

    # 计算光流幅度和归一化
    max_norm = torch.sqrt(torch.sum(flow ** 2, dim=0)).max()
    epsilon = torch.finfo(flow.dtype).eps
    normalized_flow = flow / (max_norm + epsilon)

    # 计算光流方向和幅度
    u = normalized_flow[0]
    v = normalized_flow[1]
    norm = torch.sqrt(u ** 2 + v ** 2)

    # 计算角度并转换为色环索引
    a = torch.atan2(-v, -u) / torch.pi
    colorwheel = _make_colorwheel().to(device)
    num_cols = colorwheel.shape[0]
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    # 生成RGB图像
    H, W = flow.shape[1:]
    flow_rgb = torch.zeros((3, H, W), dtype=torch.uint8, device=device)

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        # 根据光流幅度调整颜色亮度
        col = 1 - norm.unsqueeze(0) * (1 - col.unsqueeze(0))
        flow_rgb[c] = torch.floor(255 * col)

    return flow_rgb



# 把光流保存为csv文件
def save_flow_to_csv(flow_data, save_dir, batch_idx=None, t=None):
    _, num_channels, height, width = flow_data.shape
    for c in range(num_channels):
        filename = f'flow_img{batch_idx}_frame{t}_channel{c + 1}.csv'
        channel_data = flow_data[0, c]
        df = pd.DataFrame(channel_data)
        df.to_csv(os.path.join(save_dir, filename), index=False, header=False)


def save_flow_to_hdf5(flow_data, save_dir, batch_idx, t):
    """
    将光流数据保存到 HDF5 文件中

    Args:
        flow_data: 光流数据 (shape: [1, num_channels, height, width])
        save_dir: 保存目录
        batch_idx: batch 索引
        t: 帧索引
    """
    _, num_channels, height, width = flow_data.shape

    # 构建 HDF5 文件名
    hdf5_file = os.path.join(save_dir, f'object_{batch_idx}_flow_data.h5')

    # 如果文件不存在，创建新文件
    with h5py.File(hdf5_file, 'a') as hdf5:
        for c in range(num_channels):
            dataset_name = f'frame{t}_channel{c + 1}'
            if dataset_name in hdf5:
                # 如果数据集已存在，删除旧数据
                del hdf5[dataset_name]
            hdf5.create_dataset(dataset_name, data=flow_data[0, c], compression="gzip")


def convert_hdf5_to_excel(hdf5_dir, excel_dir):
    """
    将 HDF5 文件批量转换为 Excel 文件

    Args:
        hdf5_dir: HDF5 文件所在目录
        excel_dir: Excel 文件保存目录
    """
    os.makedirs(excel_dir, exist_ok=True)

    for file_name in os.listdir(hdf5_dir):
        if file_name.endswith('.h5'):
            hdf5_file = os.path.join(hdf5_dir, file_name)
            excel_file = os.path.join(excel_dir, file_name.replace('.h5', '.xlsx'))

            with h5py.File(hdf5_file, 'r') as hdf5, pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for dataset_name in hdf5.keys():
                    data = hdf5[dataset_name][:]
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=dataset_name, index=False, header=False)


# 左上角添加图片标签
def add_label_to_frame(frame, label, font_path=None, font_size=20):
    image = Image.fromarray(frame).convert("RGB")  # 转换为 RGB 图像
    draw = ImageDraw.Draw(image)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    # 在左上角绘制标签文字
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    return np.array(image)



def save_abs_diff(img1, img2, save_path):
    if isinstance(img1, torch.Tensor):
        img1 = img1[0, 0].cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2[0, 0].cpu().numpy()
    diff = np.abs(img1 - img2)
    if diff.max() > 0:
        diff = diff / diff.max()
    # 创建与图像等尺寸的 figure
    h, w = diff.shape
    dpi = 100
    # plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    # plt.imshow(diff, cmap='hot')
    # plt.axis('off')
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.savefig(save_path, dpi=dpi, pad_inches=0)
    # plt.close()
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])        # 填满整个画布
    im = ax.imshow(diff, cmap='hot')
    ax.axis('off')

    # ---- 在右上角添加横向 colorbar ----
    # 把高度从 5% 调整到 8%，防止刻度挤压
    cax = inset_axes(
        ax,
        width="38%",
        height="3%",
        loc="upper right",
        borderpad=0.5     # 加大padding，避免裁掉刻度文字
    )
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    # 强制让刻度字体为黑色，以确保在浅色背景也可见
    cbar.ax.tick_params(labelsize=6, colors='white')
    # 避免刻度被裁掉
    plt.savefig(save_path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)


def draw_flow_track_colorwheel(base_image, flow_seq, step=20, thickness=2):
    """
    连续光流轨迹 + Middlebury 颜色映射箭头

    Args:
        base_image: (H,W) 或 (H,W,3)
        flow_seq: Tensor [T-1,2,H,W] 或 list of [2,H,W]
        step: 采样距离
        thickness: 线宽
    """

    # --- 底图转 numpy ---
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

    # --- flow_seq 标准化成 list ---
    if isinstance(flow_seq, torch.Tensor):
        flow_seq = [flow_seq[t].cpu().numpy() for t in range(flow_seq.shape[0])]
    else:
        flow_seq = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in flow_seq]

    # --- 采样点 ---
    y, x = np.mgrid[0:H:step, 0:W:step]
    pts = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)

    # ==================================================
    # 连续轨迹 + 颜色根据方向
    # ==================================================
    for t in range(len(flow_seq)):
        flow = flow_seq[t]
        u = flow[0]
        v = flow[1]

        # 取本帧采样点光流
        u_s = u[y, x].flatten()
        v_s = v[y, x].flatten()

        # 本帧终点
        new_pts = pts + np.stack([u_s, v_s], axis=1)

        # 用你的颜色映射（方向 → RGB）
        flow_rgb = _flow_to_rgb(torch.tensor(flow_seq[t])).cpu().numpy().transpose(1, 2, 0)
        color_sample = flow_rgb[::step, ::step].reshape(-1, 3)  # Nx3

        # --- 画连续箭头 ---
        for i in range(len(pts)):
            c = color_sample[i].tolist()
            x1, y1 = pts[i]
            x2, y2 = new_pts[i]

            cv2.arrowedLine(
                track_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                c,  # 方向颜色！
                thickness,
                tipLength=0.3
            )

        pts = new_pts  # 下一帧起点

    return track_img


# 帧颜色
# colors = [
#     (1.0, 1.0, 0.0),   # T0 亮黄
#     (1.0, 0.2, 0.2),   # T1 亮红
#     (0.2, 0.6, 1.0),   # T2 亮蓝
#     (1.0, 0.3, 1.0),   # T3 亮紫
#     (0.3, 1.0, 0.3)    # T4 亮绿
# ]
colors = [
    (1.0, 1.0, 0.0),   # T0 亮黄
    (1.0, 0.2, 0.2),   # T1 亮红
    (0.2, 0.6, 1.0),   # T2 亮蓝
    (1.0, 0.3, 1.0),   # T3 亮紫
    (0.3, 1.0, 0.3),   # T4 亮绿
    (1.0, 0.6, 0.0),   # T5 亮橙
    (0.0, 1.0, 1.0),   # T6 亮青
    (0.8, 0.1, 0.8),   # T7 深紫
    (0.1, 0.8, 0.1),   # T8 深绿
    (1.0, 0.4, 0.0),   # T9 橙红
    (0.0, 0.4, 1.0),   # T10 深蓝
    (1.0, 0.8, 0.2),   # T11 浅黄
    (0.8, 0.4, 0.4),   # T12 浅红
    (0.4, 0.8, 1.0),   # T13 浅蓝
    (0.9, 0.5, 0.9),   # T14 浅紫
    (0.5, 0.9, 0.5),   # T15 浅绿
    (0.7, 0.3, 0.0),   # T16 棕橙
    (0.0, 0.7, 0.7),   # T17 深青
    (0.6, 0.0, 0.6),   # T18 深紫红
    (0.0, 0.5, 0.0),   # T19 墨绿
    (1.0, 0.0, 0.5),   # T20 玫红
    (0.5, 0.5, 1.0),   # T21 淡蓝
    (0.0, 1.0, 0.5),   # T22 薄荷绿
    (1.0, 0.7, 0.3),   # T23 浅橙
    (0.4, 0.2, 0.0),   # T24 深棕
    (0.7, 0.7, 1.0),   # T25 极浅蓝
    (0.9, 0.9, 0.4),   # T26 奶黄
    (0.8, 0.6, 0.8),   # T27 浅粉紫
    (0.6, 0.8, 0.6),   # T28 嫩绿
    (1.0, 0.5, 0.2)    # T29 橘红
]

def make_overlay_no_legend(img_tensors, save_path):
    """
    仅叠加，不加图例（供 diff 使用）
    img_tensors: [T,1,H,W]
    """
    H, W = img_tensors.shape[-2:]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    for t in range(len(img_tensors)):
        arr = img_tensors[t,0].cpu().numpy()
        arr = np.clip(arr, 0, 1)
        alpha = arr * 0.9
        rgba = np.zeros((H, W, 4))
        rgba[...,:3] = colors[t]
        rgba[...,3] = alpha
        ax.imshow(rgba)

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


def make_overlay_with_flow_legend(img_tensors, flows, save_path):
    """
    带图例（Frame + 光流位移）
    """
    H, W = img_tensors.shape[-2:]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # 先叠加
    for t in range(len(img_tensors)):
        arr = img_tensors[t,0].cpu().numpy()
        arr = np.clip(arr,0,1)
        alpha = arr * 0.8
        rgba = np.zeros((H, W, 4))
        rgba[...,:3] = colors[t]
        rgba[...,3] = alpha
        ax.imshow(rgba)

    # ---------------- 图例（Frame + 位移） ----------------
    legend_w = 0.22
    legend_h = 0.27
    bar_h    = legend_h / len(colors)

    for t in range(len(img_tensors)):
        y0 = 1 - 0.01 - (t + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4

        # 色条
        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.22, y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=colors[t],
            edgecolor='none'
        ))

        # 文本
        if t == 0:
            txt = f" "
        else:
            # fx = flows[t-1,0].mean().item()
            # fy = flows[t-1,1].mean().item()
            # txt = f"x:{fx:.2f}, y:{fy:.2f}"

            flow_t = flows[t - 1].cpu().numpy()  # [2, H, W]
            u = flow_t[0]
            v = flow_t[1]
            # # ----- 中位数 -----
            # u_med = np.median(u)
            # v_med = np.median(v)
            # ----- 中心区域均值（20% × 20%）-----
            H, W = u.shape
            h0, h1 = int(H * 0.1), int(H * 0.9)
            w0, w1 = int(W * 0.1), int(W * 0.9)
            u_center = np.mean(u[h0:h1, w0:w1])
            v_center = np.mean(v[h0:h1, w0:w1])
            fx = u_center
            fy = v_center
            txt = f"x:{fx:.2f}, y:{fy:.2f}"

        ax.text(
            1 - legend_w - 0.16,
            y0 + bar_height/2,
            txt,
            color='white',
            fontsize=8,
            ha='left', va='center',
            transform=ax.transAxes
        )

    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close()


def make_single_frame_overlay_with_flow_legend(
    img_tensor,     # [1, C, H, W]
    flows,          # [T-1, 2, H, W]
    global_t,       # 该帧在序列中的索引
    save_path
):
    """
    单帧 overlay，但：
    - 颜色使用 colors[global_t]
    - legend 显示 global_t 对应的 flow
    """
    H, W = img_tensor.shape[-2:]
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    # ---------- 只画这一帧 ----------
    arr = img_tensor[0, 0].cpu().numpy()
    arr = np.clip(arr, 0, 1)
    alpha = arr * 0.8
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = colors[global_t]
    rgba[..., 3] = alpha
    ax.imshow(rgba)
    # ---------- legend ----------
    legend_w = 0.22
    legend_h = 0.27
    T = 5
    bar_h = legend_h / T
    for t in range(T):
        y0 = 1 - 0.01 - (t + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4
        is_active = (t == global_t)
        alpha_legend = 1.0 if is_active else 0.6
        # ---------- 色条 ----------
        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.22, y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=colors[t],
            edgecolor='none',
            alpha=alpha_legend
        ))
        # ---------- flow 数值（始终显示） ----------
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
            1 - legend_w - 0.16,
            y0 + bar_height / 2,
            txt,
            color='white',
            fontsize=8,
            ha='left', va='center',
            transform=ax.transAxes,
            alpha=alpha_legend   # ⭐ 关键就在这里
        )
    ax.set_axis_off()
    plt.savefig(save_path, dpi=100, bbox_inches=None,
                pad_inches=0, facecolor='black')
    plt.close()


def make_single_frame_overlay_no_legend(
    img_tensor,   # [1, 1, H, W] 或 [1, C, H, W]
    global_t,     # 当前帧索引，用来选颜色
    save_path
):
    """
    单帧 overlay，不加图例
    用于与 make_single_frame_overlay_with_full_legend 配套
    """
    H, W = img_tensor.shape[-2:]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    # ---------- 仅画当前这一帧 ----------
    arr = img_tensor[0, 0].cpu().numpy()
    arr = np.clip(arr, 0, 1)
    alpha = arr * 0.9
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = colors[global_t]
    rgba[..., 3] = alpha
    ax.imshow(rgba)
    ax.set_axis_off()
    plt.savefig(
        save_path,
        dpi=100,
        bbox_inches=None,
        pad_inches=0,
        facecolor='black'
    )


def make_overlay_diff(no_legend_pred_path, no_legend_gt_path, save_path):
    """
    diff 输入必须使用 *无图例* 的叠加图！！
    """
    img1 = cv2.imread(no_legend_pred_path)
    img2 = cv2.imread(no_legend_gt_path)

    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

    diff_img = compute_diff(img1, img2)
    diff_img = add_diff_legend(diff_img)

    cv2.imwrite(save_path, diff_img)


# 差值图生成
def compute_diff(pred, gt):
    """ pred > gt → 红色，gt > pred → 蓝色 """
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff_pred = np.clip(pred_gray - gt_gray, 0, 255)
    diff_gt = np.clip(gt_gray - pred_gray, 0, 255)

    diff_pred_norm = (diff_pred / diff_pred.max() * 255) if diff_pred.max() > 0 else diff_pred
    diff_gt_norm = (diff_gt / diff_gt.max() * 255) if diff_gt.max() > 0 else diff_gt

    diff_img = np.zeros((*pred_gray.shape, 3), dtype=np.uint8)
    diff_img[:, :, 2] = diff_pred_norm  # 红
    diff_img[:, :, 0] = diff_gt_norm    # 蓝
    return diff_img


def add_diff_legend(diff_img):
    """
    右上角放置差值图图例：
    - 左边色块：红色 → pred > gt，蓝色 → gt > pred
    - 右边文字说明
    风格和 overlay_frames_for_one_object 类似
    """
    h, w, _ = diff_img.shape
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()

    # -------------------右上角图例设置-------------------
    legend_w = 0.22
    legend_h = 0.10  # 差值图只两行
    bar_h = legend_h / 2

    # 图例内容
    legend_colors = [(1, 0, 0), (0, 0, 1)]  # 红 / 蓝
    legend_texts  = ["Pred > GT", "GT > Pred"]

    for i, (c, text) in enumerate(zip(legend_colors, legend_texts)):
        y0 = 1 - 0.01 - (i + 1) * bar_h + bar_h * 0.15
        bar_height = bar_h * 0.4
        # 色块
        ax.add_patch(plt.Rectangle(
            (1 - legend_w - 0.09 , y0),
            0.05, bar_height,
            transform=ax.transAxes,
            facecolor=c,
            edgecolor='none'
        ))
        # 文字
        ax.text(
            1 - legend_w - 0.02,
            y0 + bar_height / 2,
            text,
            color='white',
            fontsize=8,
            ha='left', va='center',
            transform=ax.transAxes
        )

    # 渲染回 cv2 图片
    fig.canvas.draw()
    img_legend = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_legend = img_legend.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2BGR)
    return img_legend


def reconstruct_sequence_from_t(ref_frame, t, flows_fw, flows_bw, T):
    """
    返回: [T, C, H, W]
    """
    seq = [None] * T
    seq[t] = ref_frame.clone()
    # --------- 向过去（前向光流）---------
    cur = ref_frame.clone()
    for step in range(1, t + 1):
        flow_fw = flows_fw[t - step].unsqueeze(0)
        cur = warp(cur, flow_fw)
        seq[t - step] = cur.clone()
    # --------- 向未来（反向光流）---------
    cur = ref_frame.clone()
    for step in range(1, T - t):
        flow_bw = flows_bw[t + step - 1].unsqueeze(0)
        cur = warp(cur, flow_bw)
        seq[t + step] = cur.clone()
    return torch.cat(seq, dim=0)  # [T, C, H]


def write_metric_block(f, title, values, prefix):
    """
    f      : opened file handle
    title  : block title (string)
    values : list of float
    prefix : line prefix, e.g. 'Object' or 'Flow FW'
    """
    f.write(f"{'=' * 60}")
    f.write(f"{title}")
    f.write(f"{'=' * 60}")
    for i, v in enumerate(values):
        f.write(f"{prefix}{i}, {v:.6f}\n")
    f.write("\n")




def format_batch_loss_table(epoch, num_epochs, batch_idx, total_batches, batch_losses):
    header = f"\nEpoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{total_batches}]\n"
    table = PrettyTable()
    table.field_names = ["Loss Type", "Average Batch", "Each Batch"]
    for key, values in batch_losses.items():
        avg = values.get("average batch", 0.0)
        batch = values.get("each batch", 0.0)
        table.add_row([key, f"{avg:.6f}", f"{batch:.6f}"])
    return header + str(table)


def format_epoch_loss_table(epoch, num_epochs, epoch_losses, best_loss=None):
    model_info = (f"\nEpoch {epoch + 1}/{num_epochs}\n"
                  f"New best model saved with loss: {best_loss:.6f}\n"
                  if best_loss else f"\nEpoch {epoch + 1}/{num_epochs}\n")
    table = PrettyTable()
    table.field_names = ["Loss Type", "Average"]
    for key, values in epoch_losses.items():
        avg = values.get("average", 0.0)
        table.add_row([key, f"{avg:.6f}"])
    return model_info + str(table)


def format_test_loss_table(batch_idx, num_batches, batch_losses):
    table = PrettyTable()
    table.field_names = ["Loss Type", "Each Batch", "Average So Far"]
    for k, v in batch_losses.items():
        table.add_row([k, f"{v['each batch']:.6f}", f"{v['average batch']:.6f}"])
    table.title = f"Batch {batch_idx + 1}/{num_batches}"
    return str(table)


def format_final_loss_table(test_losses):
    table = PrettyTable()
    table.field_names = ["Loss Type", "Average Over Testset"]
    for k, v in test_losses.items():
        table.add_row([k, f"{v:.6f}"])
    table.title = f"Final Test Results"
    return str(table)


def batch_loss_summary(epoch_losses, batch_losses, batch_idx):
    return {
        key: {
            "average batch": epoch_losses[key] / (batch_idx + 1),
            "each batch": batch_losses[key],
        }
        for key in epoch_losses.keys()
    }


def epoch_loss_summary(epoch_losses):
    return {
        key: {
            "average": epoch_losses[key]
        }
        for key in epoch_losses.keys()
    }


def avg_loss_summary(total_loss, current_loss, batch_idx):
    return {
        key: {
            "each batch": current_loss[key],
            "average batch": total_loss[key] / (batch_idx + 1)
        }
        for key in total_loss.keys()
    }


# def validate_flow_magnitude(model, val_loader, device):
#     """
#     在验证集上跑一遍，计算所有前向光流的平均 magnitude（像素/帧）。
#     返回标量 float。
#     """
#     model.eval()
#     total_mag = 0.0
#     total_count = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             speckle_seq = batch['speckle_seq'].to(device)  # [T, 1, H, W]
#             output = model(speckle_seq)
#             flow_fw = output['flow_forward']  # [T-1, 2, H, W]
#             # magnitude: sqrt(u^2 + v^2)，对每个像素求，再对全图/全帧平均
#             mag = torch.sqrt(flow_fw[:, 0] ** 2 + flow_fw[:, 1] ** 2)  # [T-1, H, W]
#             total_mag += mag.mean().item()
#             total_count += 1
#     model.train()
#     return total_mag / total_count if total_count > 0 else 0.0


# def plot_flow_magnitude(flow_mag_history, save_dir):
#     """
#     画验证集光流平均 magnitude 随 epoch 的曲线。
#     flow_mag_history: list of float，每个元素是一个 epoch 的均值
#     """
#     import matplotlib.pyplot as plt
#     epochs = list(range(1, len(flow_mag_history) + 1))
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, flow_mag_history, marker='o', linewidth=1.5, color='steelblue', label='Val Flow Magnitude')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Flow Magnitude (pixels)')
#     plt.title('Validation Flow Magnitude per Epoch')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     save_path = os.path.join(save_dir, 'val_flow_magnitude.png')
#     plt.savefig(save_path, dpi=150)
#     plt.close()

def compute_flow_epe(pred_flow, gt_flow):
    """ 标准 EPE，光流论文通用 """
    epe_map = torch.sqrt(((pred_flow - gt_flow) ** 2).sum(dim=1))
    epe_mean = epe_map.mean()
    epe_per_sample_tensor = epe_map.mean(dim=(1, 2))
    epe_per_sample = epe_per_sample_tensor.detach().cpu().tolist()
    return epe_mean, epe_per_sample

def validate_metrics(model, val_loader, device):
    """
    一次验证集 forward，同时返回 magnitude 和 EPE（前向+后向）。
    返回: fw_mag, bw_mag, fw_epe, bw_epe  （均为 float）
    """
    model.eval()
    total_fw_mag, total_bw_mag = 0.0, 0.0
    total_fw_epe, total_bw_epe = 0.0, 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            speckle_seq = batch['speckle_seq'].to(device)
            flow_gt     = batch['flow_seq'].to(device)
            output = model(speckle_seq)
            flow_fw = output['flow_forward']
            flow_bw = output['flow_backward']
            total_fw_mag += torch.sqrt(flow_fw[:, 0]**2 + flow_fw[:, 1]**2).mean().item()
            total_bw_mag += torch.sqrt(flow_bw[:, 0]**2 + flow_bw[:, 1]**2).mean().item()
            fw_epe, _ = compute_flow_epe(flow_fw, flow_gt)
            bw_epe, _ = compute_flow_epe(flow_bw, -flow_gt)
            total_fw_epe += fw_epe.item()
            total_bw_epe += bw_epe.item()
            count += 1
    model.train()
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (total_fw_mag / count, total_bw_mag / count,
            total_fw_epe / count, total_bw_epe / count)


def plot_flow_magnitude(mag_history, save_dir):
    """
    mag_history: {'fw': [float, ...], 'bw': [float, ...]}
    """
    epochs = list(range(1, len(mag_history['fw']) + 1))
    plt.figure(figsize=(8, 5))
    if mag_history['fw']:
        plt.plot(epochs, mag_history['fw'],
                 marker='o', linewidth=1.5, color='steelblue', label='Val Flow Magnitude (forward)')
    if mag_history['bw']:
        plt.plot(epochs, mag_history['bw'],
                 marker='s', linewidth=1.5, color='tomato', label='Val Flow Magnitude (backward)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Flow Magnitude (pixels)')
    plt.title('Validation Flow Magnitude per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_flow_magnitude.png'), dpi=150)
    plt.close()


def plot_val_epe(val_epe_history, save_dir):
    """
    画验证集 EPE 随 epoch 的曲线。
    val_epe_history: {'fw': [float, ...], 'bw': [float, ...]}
    每个列表长度 = 已跑完的 epoch 数
    """
    epochs = list(range(1, len(val_epe_history['fw']) + 1))
    plt.figure(figsize=(8, 5))
    if val_epe_history['fw']:
        plt.plot(epochs, val_epe_history['fw'],
                 marker='o', linewidth=1.5, color='steelblue', label='Val EPE (forward)')
    if val_epe_history['bw']:
        plt.plot(epochs, val_epe_history['bw'],
                 marker='s', linewidth=1.5, color='tomato',  label='Val EPE (backward)')
    plt.xlabel('Epoch')
    plt.ylabel('EPE (pixels)')
    plt.title('Validation Flow EPE per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'val_flow_epe.png')
    plt.savefig(save_path, dpi=150)
    plt.close()


# 制作train_loss图
def plot_losses(train_losses, save_dir):
    num_losses = len(train_losses)
    ncols = min(3, num_losses)  # 最多每行3个图
    nrows = math.ceil(num_losses / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if num_losses > 1 else [axes]
    for idx, (key, values) in enumerate(train_losses.items()):
        ax = axes[idx]
        ax.plot(values, label=f'Train {key}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{key.capitalize()} Loss')
        ax.legend()
    # 清除多余的子图
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()


def create_backup_zip(base_path, specific_code_files, pos):
    """
    创建备份压缩文件，包括data/results文件夹和指定的代码文件
    参数:
    base_path (str): 项目的基础路径
    specific_code_files (list): 要备份的特定代码文件名列表
    """
    # 设置输出路径为data文件夹
    data_path = os.path.join(base_path, 'data')
    output_filename = os.path.join(data_path, f'{pos}.zip')
    # 创建一个临时目录来组织文件
    temp_dir = os.path.join(base_path, 'temp_backup')
    os.makedirs(temp_dir, exist_ok=True)
    try:
        # 复制results文件夹到临时目录的results文件夹下
        results_src = os.path.join(base_path, 'data', 'results')
        results_dest = os.path.join(temp_dir, 'results')
        if os.path.exists(results_src):
            shutil.copytree(results_src, results_dest)
        # 创建code文件夹用于存放指定的代码文件
        code_dest = os.path.join(temp_dir, 'code')
        os.makedirs(code_dest, exist_ok=True)

        # 复制RAFT文件夹到code文件夹中
        raft_src = os.path.join(base_path, 'RAFT')
        raft_dest = os.path.join(code_dest, 'RAFT')
        if os.path.exists(raft_src):
            shutil.copytree(raft_src, raft_dest)
        else:
            print(f"警告：RAFT文件夹未找到")

        # 复制utils文件夹到code文件夹中
        utils_src = os.path.join(base_path, 'utils')
        utils_dest = os.path.join(code_dest, 'utils')
        if os.path.exists(utils_src):
            shutil.copytree(utils_src, utils_dest)
        else:
            print(f"警告：utils文件夹未找到")

        # 复制指定的代码文件
        for filename in specific_code_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                shutil.copy2(file_path, os.path.join(code_dest, filename))
            else:
                print(f"警告：文件 {filename} 未找到")

        # 创建压缩文件
        with ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname=arcname)
        print(f"备份压缩文件 {output_filename} 创建成功！")
    except Exception as e:
        print(f"创建备份时发生错误: {e}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)