import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RAFT.update import BasicUpdateBlock, SmallUpdateBlock
from RAFT.extractor import BasicEncoder, SmallEncoder
from RAFT.corr import CorrBlock, AlternateCorrBlock
from RAFT.utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        # 如果 args.small 为真，使用小模型，设置隐藏维度 hidden_dim 为 96，context_dim 为 64，并设置相关层数和相关半径。
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        # 否则，使用大模型，hidden_dim 和 context_dim 都为 128，相关层数和半径也相应调整
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        # 如果参数中没有 dropout，默认将其设置为 0，避免过拟合
        if 'dropout' not in self.args:
            self.args.dropout = 0

        # 如果参数中没有 alternate_corr，默认设置为 False，决定是否使用替代的相关块。
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        # 对于小模型，定义特征网络 fnet、上下文网络 cnet，以及更新块 update_block。这些网络用于光流估计中不同阶段的处理
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        # 对于大模型，使用更大的 BasicEncoder，特征网络输出维度为 256，并采用批归一化和更大的隐藏维度。
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    # 定义 freeze_bn 方法，用于冻结批归一化层，切换到评估模式 eval()，避免在训练时更新其参数
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # initialize_flow 方法初始化光流，创建两个坐标网格 coords0 和 coords1，这些网格后续用于计算光流。
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    # upsample_flow 方法用于上采样光流，将低分辨率光流字段从 [H/8, W/8, 2] 上采样为 [H, W, 2]，通过使用卷积掩码和 torch.softmax 进行组合计算。
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    # forward 函数是模型前向传播的主要逻辑，输入两张图片 image1 和 image2，迭代次数 iters，初始光流 flow_init，
    # 是否进行上采样 upsample，以及是否处于测试模式 test_mode。函数的任务是估计两个图像帧之间的光流。
    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # 将输入的图像像素值从 [0, 255] 归一化到 [-1, 1] 范围内，以便更好地适应神经网络的输入要求
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        # 将图像数据转换为内存连续的格式，以提高后续操作的效率
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # 提取隐藏维度 hdim 和上下文维度 cdim 以供后续使用
        hdim = self.hidden_dim
        cdim = self.context_dim

        # 使用特征网络 fnet 提取两个图像帧的特征图 fmap1 和 fmap2，并在混合精度模式下进行计算，以提高性能
        # run the feature network
        with autocast('cuda', enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        # 根据参数选择使用 AlternateCorrBlock 或 CorrBlock 进行特征图的相关性计算，这个操作用于估计图像之间的位移（光流）。
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # 通过上下文网络 cnet 计算上下文特征，将其拆分为 net 和 inp，并分别通过 tanh 和 relu 激活函数进行处理
        # run the context network
        with autocast('cuda', enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # 初始化光流坐标，coords0 和 coords1 是图像帧的初始坐标网格
        coords0, coords1 = self.initialize_flow(image1)

        # 如果提供了初始光流 flow_init，将其加到 coords1 上，用于初始化当前光流
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            # 进入光流迭代计算。首先，将 coords1 从计算图中分离出来，然后计算当前坐标的相关性
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            # 计算光流 flow 作为坐标网格的差值。通过更新块 update_block，计算新的隐藏状态 net、上采样掩码 up_mask 和光流增量 delta_flow
            flow = coords1 - coords0
            with autocast('cuda', enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            # 更新光流，下一时刻的坐标为当前坐标加上光流增量 delta_flow
            coords1 = coords1 + delta_flow

            # upsample predictions
            # 对预测的光流进行上采样。如果没有上采样掩码，使用默认的 upflow8 函数进行上采样；否则，使用 upsample_flow 方法对光流进行上采样
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        #     # 将当前迭代生成的上采样光流添加到 flow_predictions 列表中
        #     flow_predictions.append(flow_up)
        #
        # if test_mode:
        #     return coords1 - coords0, flow_up
            
        return flow_up
