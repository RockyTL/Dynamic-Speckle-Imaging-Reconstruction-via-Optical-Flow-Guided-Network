# Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network

## Introduction / 项目简介

This repository implements an optical-flow-guided dynamic speckle imaging reconstruction framework for dynamic object recovery under speckle imaging conditions.

The method combines motion estimation and image reconstruction to improve temporal consistency and reconstruction quality for dynamic scenes.

本项目实现了一种基于光流引导的动态散斑成像重建方法，用于动态目标在散斑成像条件下的重建。

该方法结合运动估计与图像重建，以提高动态场景中的时间一致性与重建质量。

---

## Method Overview / 方法概述

The framework mainly consists of the following modules:

* **Motion Encoder**: extracts motion-sensitive latent features from speckle sequences
* **RAFT Optical Flow Module**: estimates forward and backward optical flow
* **U-Net Reconstruction Network**: reconstructs object images

该框架主要包含以下模块：

* **Motion Encoder**：提取散斑序列中的运动敏感特征
* **RAFT 光流模块**：估计前向与后向光流
* **U-Net 重建网络**：完成目标图像重建

---

## Project Structure / 项目结构

```text
project/
├── Main.py                  # training / testing pipeline
├── Net.py                   # main reconstruction network
├── Net_Unet.py              # U-Net backbone
├── MOD.py                   # simulation data generation
├── Dataset.py               # dataset loading
├── RAFT/                    # RAFT optical flow related modules
├── utils/
│   ├── Sundries.py          # loss functions and evaluation metrics
│   ├── Visual_utils.py      # visualization tools
│   ├── Mainloss_manage.py   # training loss management
│   └── ...
└── README.md
```

---

## Training and Testing / 训练与测试

Run:

```bash
python Main.py
```

Please manually switch the running mode inside `Main.py`:

```python
train_model(...)
test_model(...)
```

在 `Main.py` 中手动切换：

```python
train_model(...)
test_model(...)
```

---

## Loss Functions / 损失函数设计

Training loss includes:

* Speckle warping loss
* Object reconstruction loss
* Temporal consistency loss

训练损失包括：

* 散斑 warping loss
* 目标重建 loss
* 时间一致性 loss

Evaluation metrics include:

* End-Point Error (EPE)
* Angular Error
* Flow Accuracy (1px / 3px / 5px)
* SSIM
* PSNR

测试指标包括：

* EPE（终点误差）
* 角度误差
* Flow Accuracy（1px / 3px / 5px）
* SSIM
* PSNR

---

## Output Results / 输出结果

Testing outputs include:

* Forward / backward optical flow visualization
* Reconstructed object frames
* Ground truth comparison
* Difference maps
* Metric logs

测试输出包括：

* 前向 / 后向光流可视化
* 重建目标图像
* Ground truth 对比
* 差异图
* 指标日志

---

## Notes / 说明

* RAFT module is used for optical flow estimation.
* Simulation data can be generated using `MOD.py`.
* Additional utility functions are stored in `utils/`.

说明：

* RAFT 模块用于光流估计
* 可通过 `MOD.py` 生成仿真数据
* 杂项功能存放于 `utils/` 文件夹

---

## Citation / 引用

If you use this code, please cite:

**Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network**

(Optics Express, First Author)

如果使用本代码，请引用对应论文。
