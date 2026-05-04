# Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network

[![DOI](https://img.shields.io/badge/DOI-10.1364/OE.591608-blue)](https://doi.org/10.1364/OE.591608)

## Introduction / 项目简介

This repository implements an optical-flow-guided dynamic speckle imaging reconstruction framework for dynamic object recovery under speckle imaging conditions.

The method combines motion estimation and image reconstruction to improve temporal consistency and reconstruction quality for dynamic scenes, with potential applications in optical imaging, remote sensing, and medical imaging.

本项目实现了一种基于光流引导的运动散斑成像重建方法，用于运动目标在散斑成像条件下的高质量重建。该方法结合运动估计与图像重建技术，在动态场景下提升时间一致性与空间重建质量，可应用于光学成像、遥感探测、医学影像以及动态目标观测等领域。

---

## Method Overview / 方法概述

The framework consists of the following core modules:

* **RAFT Optical Flow Module**: Estimates forward and backward optical flow between consecutive speckle frames, providing accurate motion cues for temporal constraint.
* **U-Net Reconstruction Network**: Encoder-decoder structure with SE attention and skip connections to reconstruct high-fidelity object images from speckle frames.
* **Multi-Loss Fusion Module**: Integrates reconstruction loss, flow consistency loss, and temporal stability loss to balance multiple optimization targets.
* **Dynamic Data Simulation Module**: Generates controllable synthetic datasets for model training and validation.

该框架主要包含以下核心模块：

* **RAFT 光流模块**：估计连续散斑帧之间的前向/后向光流，为时序约束提供精确运动信息。
* **U-Net 重建网络**：采用带SE注意力与跳跃连接的编码器-解码器结构，从散斑图像中恢复目标图像。
* **多损失融合模块**：联合优化重建误差、光流一致性误差及时间稳定性误差。
* **动态数据仿真模块**：生成可控动态散斑数据集，用于训练、验证与测试。

---

## Representative Result / 结果示意图

![Result Demo](Image/result_simulation.jpg)

---

## Project Structure / 项目结构

```
project/
├── configs/
│   ├── train.yaml              # Training hyperparameters and paths
│   └── test.yaml               # Testing parameters and paths
├── src/
│   ├── models/
│   │   ├── complete_model.py   # CompleteModel, SimpleReconstructionModel
│   │   ├── unet.py             # U-Net with SE attention blocks
│   │   └── motion.py           # MotionEncoder, MotionAdapter, GlobalMotionHead (experimental)
│   ├── data/
│   │   └── datasets.py         # Dataset classes for synthetic and experimental data
│   ├── losses/
│   │   └── combined_loss.py    # CombinedLoss, SimpleLoss, warp, flow utilities, metrics
│   ├── engine/
│   │   ├── trainer.py          # Training and fine-tuning loops
│   │   └── evaluator.py        # Model evaluation functions
│   └── utils/
│       ├── visualization.py    # Flow rendering, overlays, result saving
│       ├── metrics.py          # CSV logging, table formatting
│       └── io_utils.py         # File I/O, directory creation, checkpoint scanning
├── scripts/
│   ├── train.py                # Training entry point
│   ├── test.py                 # Testing entry point
│   └── run_checkpoints.py      # Batch checkpoint evaluation
├── RAFT/                       # Third-party optical flow module (unchanged)
├── MOD.py                      # Synthetic data generation
├── requirements.txt
└── README.md
```

---

## Environment Requirements / 环境依赖

- Python >= 3.8
- PyTorch >= 1.10.0
- torchvision >= 0.11.0

### Installation

```bash
pip install -r requirements.txt
```

---

## Data Preparation / 数据准备

### 1. Synthetic Data Generation / 合成数据生成

```bash
python MOD.py
```

Key configurable parameters in `MOD.py`:
- `train_size / val_size / test_size`: dataset scale
- `num_frames`: frames per sequence
- `obj_size`, `bg_size`, `move_range`: object/background/motion settings
- `base_path`: output directory

### 2. Expected Dataset Directory Structure / 数据集目录结构

```
data/datasets/obj_128_bg_256_move_16/
├── train_speckle_images/
├── train_object_images/
├── train_flow/
├── val_speckle_images/
├── val_object_images/
├── val_flow/
├── test_speckle_images/
├── test_object_images/
└── test_flow/
```

### 3. Real Experimental Data / 真实实验数据

Organize `.bmp` speckle images in a flat directory named as `Image{N}_frame{M}.bmp`.

---

## Training / 训练

### Full Model (RAFT + U-Net)

```bash
python scripts/train.py --config configs/train.yaml
```

### U-Net Only (Ablation)

```bash
python scripts/train.py --config configs/train.yaml --mode unet_only
```

### Fine-Tuning on Experimental Data

```bash
python scripts/train.py --config configs/train.yaml --mode finetune
```

### Resume from Checkpoint

```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/best_model.pth
```

---

## Testing / 测试

### Synthetic Data

```bash
python scripts/test.py --config configs/test.yaml --checkpoint path/to/model.pth
```

### Experimental Data (no GT)

```bash
python scripts/test.py --config configs/test.yaml --mode experiment
```

### Experimental Data (with GT)

```bash
python scripts/test.py --config configs/test.yaml --mode experiment_withobj
```

### Batch Checkpoint Evaluation

```bash
python scripts/run_checkpoints.py --config configs/test.yaml
python scripts/run_checkpoints.py --config configs/test.yaml --mode experiment --start_epoch 15 --end_epoch 25
```

---

## Loss Functions & Evaluation Metrics / 损失函数与评估指标

### Training Loss

```
L_total = L_recon + L_warp + L_temporal
```

- `L_recon`: Charbonnier + MSE reconstruction loss
- `L_warp`: speckle warp consistency loss (forward + backward)
- `L_temporal`: object warp chain loss across time

### Evaluation Metrics

| Metric        | Definition                              |
| ------------- | --------------------------------------- |
| EPE           | Endpoint Error (average L2 distance)    |
| Angular Error | Flow direction angular difference       |
| Fl-all        | KITTI outlier ratio                     |
| N-px Accuracy | Ratio of pixels with EPE < N pixels     |
| SSIM          | Structural Similarity                   |
| PSNR          | Peak Signal-to-Noise Ratio              |
| MSE           | Mean Squared Error                      |

---

## Output Results / 输出结果

```
results/
├── checkpoints/
├── flowdata/
│   ├── flow_arrow_fw/     flow_colorimage_fw/
│   ├── flow_arrow_bw/     flow_colorimage_bw/
│   └── gt_flow_*/
├── origin_object/
├── recon_object/
├── diff_recon_vs_gt/
├── overlay_results_*/
├── test_metrics_summary.txt
└── test_batch_losses.csv
```

---

## Key Notes / 核心说明

- RAFT folder should remain in project root — do not modify
- GPU is strongly recommended
- Supports fine-tuning for real experimental data
- All configuration is centralized in `configs/*.yaml`

---

## Common Issues / 常见问题

| Issue              | Solution                           |
| ------------------ | ---------------------------------- |
| RAFT import error  | Install RAFT dependencies          |
| Out of Memory      | Reduce batch size / frame count    |
| Low SSIM           | Increase epochs / adjust LR        |
| Flow error         | Use pretrained RAFT weights        |
| YAML parse error   | Check config file indentation      |

---

## Citation / 引用

If you use this code in your research, please cite:

**Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network**  
Optics Express, Vol. 34, No. 8, pp. 14534-14548 (2026)  
DOI: https://doi.org/10.1364/OE.591608

```bibtex
@article{speckle_flow_guided_2026,
  title   = {Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network},
  journal = {Optics Express},
  volume  = {34}, number = {8},
  pages   = {14534--14548},
  year    = {2026},
  doi     = {10.1364/OE.591608}
}
```
