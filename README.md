# Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network

## Introduction / 项目简介

This repository implements an optical-flow-guided dynamic speckle imaging reconstruction framework for dynamic object recovery under speckle imaging conditions.

The method combines motion estimation and image reconstruction to improve temporal consistency and reconstruction quality for dynamic scenes, with potential applications in optical imaging, remote sensing, and medical imaging.

本项目实现了一种基于光流引导的动态散斑成像重建方法，用于运动目标在散斑成像条件下的高质量重建。

该方法结合运动估计与图像重建技术，在动态场景下提升时间一致性与空间重建质量，可应用于光学成像、遥感探测、医学影像以及动态目标观测等领域。

---

## Method Overview / 方法概述

The framework mainly consists of the following core modules:

* **RAFT Optical Flow Module**: Estimates forward and backward optical flow between consecutive speckle frames, providing accurate motion cues for temporal constraint.
* **U-Net Reconstruction Network**: Adopts encoder-decoder structure with skip connections to reconstruct high-fidelity object images from speckle frames.
* **Multi-Loss Fusion Module**: Integrates reconstruction loss, flow consistency loss, and temporal stability loss to balance multiple optimization targets.
* **Dynamic Data Simulation Module**: Generates controllable synthetic datasets for model training and validation.

该框架主要包含以下核心模块：

* **RAFT 光流模块**：估计连续散斑帧之间的前向/后向光流，为时序约束提供精确运动信息。
* **U-Net 重建网络**：采用带跳跃连接的编码器-解码器结构，从散斑图像中恢复目标图像。
* **多损失融合模块**：联合优化重建误差、光流一致性误差及时间稳定性误差。
* **动态数据仿真模块**：生成可控动态散斑数据集，用于训练、验证与测试。

---

## Project Structure / 项目结构

```text
project/
├── Main.py                  # Core entry (training/testing pipeline, mode switch)
├── Net.py                   # Main network (combines optical flow and reconstruction modules)
├── Net_Unet.py              # U-Net backbone (encoder-decoder with skip connections)
├── MOD.py                   # Synthetic data generation (speckle simulation + motion modeling)
├── Dataset.py               # Dataset loader (preprocessing, sampling, augmentation)
├── Mainloss_manage.py       # Loss management (multi-loss fusion, weighting)
├── RAFT/                    # Optical flow dependency (RAFT-related modules)
├── utils/
│   ├── Sundries.py          # Loss functions & evaluation metrics
│   ├── Visual_utils.py      # Visualization tools
│   └── ...                  # Auxiliary functions
└── README.md
```

---

## Environment Requirements / 环境依赖

### Required Versions

```bash
Python >= 3.8
PyTorch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
matplotlib >= 3.4.0
opencv-python >= 4.5.0
scipy >= 1.7.0
tqdm >= 4.62.0
pillow >= 8.3.0
pandas >= 1.3.0
```

### Installation Command

```bash
pip install torch torchvision numpy matplotlib opencv-python scipy tqdm pillow pandas
```

If RAFT requires additional dependencies:

```bash
cd RAFT
pip install -r requirements.txt
```

如 RAFT 子模块包含额外依赖，请进入 RAFT 文件夹单独安装。

---

## Data Preparation / 数据准备

### 1. Synthetic Data Generation / 合成数据生成（推荐）

```bash
python MOD.py
```

Key configurable parameters in `MOD.py`:

* `object_size`: target object resolution
* `bg_size`: background resolution
* `move_range`: motion range
* `num_frames`: frames per sequence
* `train_size / test_size / val_size`: dataset scale
* `save_path`: output path

主要可调参数包括：

* 目标尺寸
* 背景尺寸
* 运动范围
* 序列帧数
* 训练/测试/验证样本数量
* 数据保存路径

---

### 2. Real Dataset Adaptation / 真实数据适配

```text
real_data/
├── train/
│   ├── speckle/
│   ├── object/
│   └── flow/
└── test/
```

For real data:

* Modify `Dataset.py`
* Keep normalization consistent with synthetic data

真实数据使用时需：

* 修改 `Dataset.py` 中的数据读取路径
* 保持与仿真数据一致的归一化方式

---

## Training and Testing / 训练与测试

### Training / 训练

```bash
python Main.py
```

Enable:

```python
train_model(...)
```

---

### Testing / 测试

```bash
python Main.py
```

Enable:

```python
test_model(...)
```

---

### Quick Mode Switch / 快速模式切换

```python
if __name__ == "__main__":
    main()

    # test_model(...)
```

建议在 `Main.py` 中手动切换训练与测试入口。

---

## Loss Functions & Evaluation Metrics / 损失函数与评估指标

### Training Loss / 训练损失

```text
L_total = L_recon + λ1 L_flow + λ2 L_temporal
```

* `L_recon`: reconstruction loss
* `L_flow`: flow estimation loss
* `L_temporal`: temporal consistency loss

---

### Evaluation Metrics / 评估指标

| Metric        | English Definition                                       | 中文说明   |
| ------------- | -------------------------------------------------------- | ------ |
| EPE           | Average Euclidean distance between predicted and GT flow | 光流终点误差 |
| Angular Error | Angular difference of flow direction                     | 光流角度误差 |
| Flow Accuracy | Ratio under 1px / 3px / 5px                              | 光流精度   |
| SSIM          | Structural Similarity                                    | 结构相似性  |
| PSNR          | Peak Signal-to-Noise Ratio                               | 峰值信噪比  |
| MSE           | Mean Squared Error                                       | 均方误差   |

---

## Output Results / 输出结果

```text
results/
├── checkpoints/
├── flowdata/
│   ├── flow_arrow_fw/
│   ├── flow_colorimage_fw/
│   └── gt_flow_*/
├── origin_object/
├── recon_object/
├── diff_recon_vs_gt/
├── overlay_results_*/
├── test_metrics_summary.txt
└── test_batch_losses.csv
```

测试结果默认保存在 `save_dir` 下，包括：

* 光流可视化结果
* 重建图像
* Ground Truth 对比
* 差异图
* 定量指标日志

---

## Key Notes / 核心说明

* RAFT folder should remain in project root
* GPU is strongly recommended
* Supports finetuning for real data
* Data augmentation is integrated

核心说明：

* RAFT 文件夹建议保持在项目根目录
* 推荐使用 GPU 加速训练
* 支持真实数据微调
* 已集成基础数据增强策略

---

## Common Issues / 常见问题

| Issue              | Solution                      |
| ------------------ | ----------------------------- |
| RAFT import error  | install RAFT dependencies     |
| Out of Memory      | reduce batch size             |
| Low SSIM           | increase epochs / adjust loss |
| Flow error         | use pretrained RAFT           |
| Real data mismatch | modify Dataset.py             |

---

## Citation / 引用

If you use this code in your research, please cite:

Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network  
Optics Express  
DOI: https://doi.org/10.1364/OE.591608

```bibtex
@article{speckle_flow_guided_2025,
  title={Dynamic Speckle Imaging Reconstruction via Optical-Flow-Guided Network},
  journal={Optics Express},
  doi={10.1364/OE.591608}
}
```

如在研究中使用本代码，请引用对应论文。


