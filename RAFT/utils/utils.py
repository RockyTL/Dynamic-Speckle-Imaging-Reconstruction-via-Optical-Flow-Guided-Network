import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


# 对输入图像进行填充，使得图像的宽高都能被8整除
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8       # pad_ht 和 pad_wd 分别计算高度和宽度需要填充的像素数
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        # 如果模式是 sintel（来自 Sintel 数据集的一种处理方式），填充会在上下、左右两边平均分布。否则，按另一种方式填充
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    # 使用 F.pad 函数，mode='replicate' 表示使用边缘值填充。
    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    # unpad 方法用于移除之前的填充，恢复到原始大小
    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    # detach() 从计算图中移除张量，cpu() 移到 CPU，numpy() 转为 NumPy 数组
    flow = flow.detach().cpu().numpy()
    # dx 和 dy 分别是水平和垂直的光流分量
    dx, dy = flow[0], flow[1]

    # 生成一个网格坐标 x0, y0，大小为光流的宽度和高度，分别表示每个像素点的 x 和 y 坐标
    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    # 根据光流的变化量，计算像素点新的位置 x1, y1
    x1 = x0 + dx
    y1 = y0 + dy

    # 将 x1, y1 和 dx, dy 展平为一维数组，方便后续插值处理
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    # 通过条件 valid 过滤出有效的像素点，即新位置没有超出图像边界的点。只保留这些有效点的坐标和光流值
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    # 使用 scipy.interpolate.griddata 进行插值操作。
    # 将有效点的 dx, dy 值映射到原始坐标 x0, y0 上，使用最近邻插值法。如果某些像素点没有对应值，用 0 填充
    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    # 最后，将插值后的 flow_x 和 flow_y 堆叠回原来的 [2, H, W] 形状的光流张量
    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    # 使用 split 将其分割为 xgrid（x 坐标）和 ygrid（y 坐标）
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # 将坐标网格的范围从像素坐标（0 到 W, 0 到 H）转换到标准化的网格坐标（-1 到 1）
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    # 将 xgrid 和 ygrid 沿最后一维拼接，形成 [batch, H, W, 2] 大小的坐标网格
    grid = torch.cat([xgrid, ygrid], dim=-1)
    # 使用 F.grid_sample 根据网格采样图像。align_corners=True 确保对齐时考虑边界像素
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)

    # 如果 mask=True，还会返回一个掩码，标记哪些采样点在图像范围内（即 xgrid 和 ygrid 都在 -1 到 1 之间）
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# 生成批次的二维坐标网格
def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# 将光流张量放大8倍
def upflow8(flow, mode='bilinear'):
    # new_size 是原光流尺寸的8倍。使用 F.interpolate 对光流进行插值操作，插值模式默认为双线性（bilinear）
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    # 插值后的光流乘以8，以适应新的空间分辨率
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
