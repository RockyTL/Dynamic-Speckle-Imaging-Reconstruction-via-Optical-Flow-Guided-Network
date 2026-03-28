# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


# 生成颜色轮，将不同的方向和幅度编码为颜色。颜色轮由多个区段组成，每个区段有不同的颜色过渡
# 包括了红-黄（RY）、黄-绿（YG）、绿-青（GC）、青-蓝（CB）、蓝-品红（BM）、品红-红（MR）六个部分
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


# 将光流的水平分量 u 和垂直分量 v 转换为彩色图像
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    # 创建一个空的三通道图像 flow_image，大小与输入光流 u 和 v 相同，用来存储最终的颜色图像
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    # 调用 make_colorwheel() 函数生成颜色轮
    colorwheel = make_colorwheel()  # shape [55x3]
    # 获取颜色轮的颜色数量 ncols
    ncols = colorwheel.shape[0]
    # 计算每个像素点的光流幅度（或速率）rad，即光流的大小
    rad = np.sqrt(np.square(u) + np.square(v))
    # 计算光流的方向角 a，使用反正切函数 np.arctan2()，范围在 [-1, 1] 之间
    a = np.arctan2(-v, -u)/np.pi
    # 将方向角 a 映射到 [0, ncols-1] 的区间，得到 fk，这是一个浮点数，表示该方向在颜色轮中的位置
    fk = (a+1) / 2*(ncols-1)
    # k0 是 fk 向下取整的整数部分，表示当前方向对应的颜色索引
    k0 = np.floor(fk).astype(np.int32)
    # k1 是 k0 + 1，表示下一个颜色索引
    k1 = k0 + 1
    # 如果 k1 超过颜色轮最大值 ncols，则将其设置为 0（循环颜色轮）
    k1[k1 == ncols] = 0
    # f 是 fk 的小数部分，用于在颜色轮的两个相邻颜色之间进行线性插值
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        # tmp 是颜色轮中的第 i 个通道（红、绿、蓝中的一个）
        tmp = colorwheel[:,i]
        # col0 是 k0 索引处的颜色值，col1 是 k1 索引处的颜色值，这些值被归一化到 [0, 1]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        # 对颜色值进行线性插值：col 是两个相邻颜色的混合，权重由 f 决定
        col = (1-f)*col0 + f*col1
        # 如果光流幅度 rad 小于或等于 1，则通过 1 - rad 对颜色进行调整，使颜色值越接近 1（表示小流速的亮度更高）
        # 如果光流幅度大于 1，则将颜色值乘以 0.75，减小亮度
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        # 将插值后的颜色值乘以 255，并存入 flow_image 的第 ch_idx 通道
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    # 如果提供了 clip_flow，则将光流的值裁剪到 [0, clip_flow] 之间，以避免极端的光流值影响可视化效果
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    # 计算每个像素的光流幅度 rad，并找到整个图像中的最大幅度 rad_max
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    # 对 u 和 v 进行归一化处理
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)