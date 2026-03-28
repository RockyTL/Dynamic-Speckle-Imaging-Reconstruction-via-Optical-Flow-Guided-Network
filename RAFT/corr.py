import torch
import torch.nn.functional as F
from RAFT.utils.utils import bilinear_sampler, coords_grid

try:
    import RAFT.alt_cuda_corr as alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

# CorrBlock 类用于计算特征图的全局相关性，生成一个相关性金字塔，并且能够在不同层级上通过双线性插值采样相应的相关性
# AlternateCorrBlock 类则使用一个替代的CUDA加速方法（alt_cuda_corr），通过对每一层的特征图进行不同尺度的相关性计算，并返回最终的相关性堆叠


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels    # 设置相关性金字塔的层数
        self.radius = radius    # 设置采样半径
        self.corr_pyramid = []      # 初始化相关性金字塔的列表，用于存储不同尺度下的相关性图

        # all pairs correlation     调用静态方法corr计算特征图fmap1和fmap2之间的相关性
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)       # 将相关性张量重新调整形状以便后续处理
        
        self.corr_pyramid.append(corr)      # 将最初的相关性图添加到corr_pyramid中
        for i in range(self.num_levels-1):      # 遍历层数，构建相关性金字塔
            corr = F.avg_pool2d(corr, 2, stride=2)      # 使用平均池化操作将相关性图在每个层中下采样
            self.corr_pyramid.append(corr)      # 将下采样后的相关性图添加到金字塔中

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)     # 调整坐标的维度顺序，从(B, C, H, W)调整为(B, H, W, C)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []        # 初始化输出相关性金字塔的列表
        for i in range(self.num_levels):        # 遍历相关性金字塔的每一层
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)      # 创建一个从-r到r的等间距向量，用于定义采样窗口
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)        # 生成一个2D网格，用于表示坐标偏移

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i      # 根据当前层的缩放比例计算采样窗口的中心点
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl       # 计算采样点的实际坐标

            corr = bilinear_sampler(corr, coords_lvl)       # 使用双线性采样获取相关性图中的值
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)        # 将每层采样后的相关性图添加到输出金字塔中

        out = torch.cat(out_pyramid, dim=-1)        # 将不同尺度的相关性图拼接在一起
        return out.permute(0, 3, 1, 2).contiguous().float()     # 调整输出的维度顺序为(B, C, H, W)

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)       # 将特征图调整为 (B, D, H*W) 的形状，展平空间维度
        
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)        # 计算两个特征图的矩阵乘法，得到它们的相关性矩阵。
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())        # 对相关性矩阵进行归一化处理


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]     # 初始化特征金字塔，将输入的特征图fmap1和fmap2作为第一层
        for i in range(self.num_levels):        # 遍历金字塔层数
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)        # 对fmap1和fmap2进行平均池化，下采样每一层特征图，并将它们添加到金字塔中
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)     # 将输入坐标的维度从 (B, C, H, W) 调整为 (B, H, W, C)
        B, H, W, _ = coords.shape       # 获取输入坐标的批量大小、高度和宽度
        dim = self.pyramid[0][0].shape[1]       # 获取特征图的通道维度

        corr_list = []      # 初始化一个空列表，用于存储各层的相关性结果
        for i in range(self.num_levels):        # 遍历金字塔的每一层
            r = self.radius
            # 从特征金字塔中获取当前层的fmap1和fmap2，并将它们的维度从 (B, C, H, W) 调整为 (B, H, W, C)
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            # 对坐标进行缩放以适应当前层的尺度（每层的特征图尺寸减半，因此坐标要缩放）。
            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)  # 用alt_cuda_corr的CUDA加速相关性计算，计算当前层的相关性
            corr_list.append(corr.squeeze(1))  # 将计算得到的相关性图压缩并添加到corr_list中

        corr = torch.stack(corr_list, dim=1)        # 将所有层的相关性结果沿着维度1堆叠，形成一个新的张量
        corr = corr.reshape(B, -1, H, W)        # 将相关性结果调整为 (B, num_levels * radius^2, H, W) 的形状，表示在不同尺度下的相关性堆叠
        return corr / torch.sqrt(torch.tensor(dim).float())     # 归一化相关性张量以避免数值过大，返回最终结果
