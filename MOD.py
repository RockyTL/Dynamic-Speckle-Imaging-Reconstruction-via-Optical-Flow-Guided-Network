import os
import re
import time
import cv2
import pandas as pd
import torch
import shutil
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Dict
from torchvision.datasets import ImageFolder


def parse_dataset_name(name: str) -> Dict[str, int]:
    """解析数据集名称，并提取参数"""
    pattern = r"obj_(\d+)_bg_(\d+)_move_(\d+)_psf_(\d+)"
    match = re.search(pattern, name)
    if not match:
        raise ValueError(f"Invalid dataset name format: {name}")
    obj_size, bg_size, move_range, psf_size = map(int, match.groups())
    return {"obj_size": obj_size, "bg_size": bg_size, "move_range": move_range, "psf_size": psf_size}


@dataclass
class MovementConfig:
    move_range: int  # 物体移动范围 (-move_range, move_range)
    initial_offset: Tuple[int, int] = (0, 0)  # (vertical_offset, horizontal_offset)

    def get_random_shift(self) -> Tuple[int, int]:
        return random.randint(-self.move_range, self.move_range), random.randint(-self.move_range, self.move_range)


class DatasetGenerator:
    def __init__(self, base_path: str, dataset_name: str, object_shifts: Dict[int, List[Tuple[int, int]]], object_initial_positions):
        self.base_path = base_path
        self.params = parse_dataset_name(dataset_name)
        self.movement_config = MovementConfig(move_range=self.params["move_range"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用预计算的物体移动，按物体ID索引
        self.object_shifts = object_shifts
        self.object_initial_positions = object_initial_positions

        # 生成分组目录 (相同 obj_size, move_range, bg_size)
        self.group_key = f"{self.params['obj_size']}_{self.params['move_range']}_{self.params['bg_size']}"
        self.grouped_dataset_dir = os.path.join(self.base_path, "grouped", self.group_key, dataset_name)

    def _get_directories(self, mode: str) -> dict:
        """获取数据集保存路径"""
        dirs = {
            'speckle': os.path.join(self.grouped_dataset_dir, f'{mode}_speckle_images'),
            'object': os.path.join(self.grouped_dataset_dir, f'{mode}_object_images'),
            'flow': os.path.join(self.grouped_dataset_dir, f'{mode}_flow'),
            'object_log': os.path.join(self.grouped_dataset_dir, f'{mode}_object_image_log'),
            'labels_speckle': os.path.join(self.grouped_dataset_dir, f'{mode}_labels_speckle'),
            'labels_object': os.path.join(self.grouped_dataset_dir, f'{mode}_labels_object'),
        }
        return dirs

    def convolution_speckle(self, image: np.ndarray, psf_normal: str, psf_dark: str) -> np.ndarray:
        """
        image: np.ndarray, 单通道物体图片 (H x W)
        psf_normal_path: 正常PSF的文件路径
        psf_dark_path: 挡光暗场PSF的文件路径
        返回值: 归一化的散斑图像
        """
        # 加载并转换物体图片
        transform = transforms.ToTensor()
        object_image = transform(image).unsqueeze(0).to(self.device).float()  # [1, 1, H, W]

        # 定义对PSF的处理流程：先裁剪到self.params['psf_size'], 再转为Tensor
        preprocess_psf = transforms.Compose([
                transforms.CenterCrop(self.params['psf_size']),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32)
        ])

        # 分别处理正常PSF和暗场PSF
        psf_normal = preprocess_psf(psf_normal).to(self.device).unsqueeze(0)
        psf_dark = preprocess_psf(psf_dark).to(self.device).unsqueeze(0)

        # 先裁剪后相减
        psf_subtracted = psf_normal - psf_dark
        psf_subtracted = torch.clamp(psf_subtracted, min=0.0)  # 防止出现负值

        # 标准化到总和=1
        psf_subtracted /= psf_subtracted.sum() + 1e-8

        # 卷积
        speckle = F.conv2d(object_image, psf_subtracted, padding='same').squeeze().cpu().numpy()

        # 归一化到0-255
        speckle = (speckle - speckle.min()) / (speckle.max() - speckle.min() + 1e-8) * 255
        return speckle

    def expand_image(self, image, mean, std, target_size, intermediate_size, offset):
        # 将输入图像调整到 intermediate_size
        image_resized = np.array(Image.fromarray(image).resize((intermediate_size, intermediate_size)))
        # 创建一个全黑的背景图像
        background = np.zeros((target_size, target_size), dtype=np.float32)
        # 获取偏移量
        vertical_offset, horizontal_offset = (
            (target_size - intermediate_size) // 2 + offset[0], (target_size - intermediate_size) // 2 + offset[1])
        # 将调整过的图像放入背景中
        background[vertical_offset: vertical_offset + intermediate_size,
        horizontal_offset: horizontal_offset + intermediate_size] = image_resized
        # 创建背景掩膜：背景区域设为1，图像区域设为0
        mask = (background == 0).astype(np.float32)
        # 生成高斯噪声
        noise = np.random.normal(mean, std, background.shape)
        # 仅在背景区域添加噪声
        background = background + noise * mask
        background = np.clip(background, 0, 255)
        return background

    def image_resize_crop_resize(self, image: np.ndarray, crop_size: int = 256, target_size: int = 512) -> np.ndarray:
        # Step 1: 从中心裁剪 crop_size x crop_size
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        half_crop = crop_size // 2
        crop_image = image[center_y - half_crop:center_y + half_crop, center_x - half_crop:center_x + half_crop]
        # Step 2: 将裁剪的图像缩放回 target_size x target_size
        resized_image = np.array(Image.fromarray(crop_image).resize((target_size, target_size), Image.BICUBIC))
        return resized_image

    def process_frame(self, image: np.ndarray, total_shift: Tuple[int, int], shift: Tuple[int, int], psf: Image.Image, psf_dark: Image.Image) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理单帧图像"""
        expanded_image = self.expand_image(
            image, 0, 0.1, self.params['bg_size'], self.params['obj_size'], offset=self.movement_config.initial_offset)
        # moved_image = np.roll(expanded_image, shift=total_shift, axis=(0, 1))
        # 定义平移矩阵
        M = np.float32([[1, 0, total_shift[1]], [0, 1, total_shift[0]]])  # (dx, dy)
        # 使用 warpAffine 进行平移，并填充超出边界的部分为 0
        moved_image = cv2.warpAffine(expanded_image, M, (expanded_image.shape[1], expanded_image.shape[0]),
                                     borderValue=0)
        speckle_image = self.convolution_speckle(moved_image, psf, psf_dark)
        # 提取物体掩膜
        prev_mask_position = np.roll(moved_image, shift=(total_shift[0] - shift[0], total_shift[1] - shift[1]),
                                     axis=(0, 1))
        object_mask = (prev_mask_position > 5).astype(np.float32)

        return speckle_image, moved_image, object_mask

    def generate_flow(self, shift: Tuple[int, int]) -> np.ndarray:
        """生成光流数据，仅限于物体图的部分"""
        flow = np.zeros((2, self.params['bg_size'], self.params['bg_size']), dtype=np.float32)
        # 设置移动的值
        flow[0, :, :] = shift[1]  # 水平位移
        flow[1, :, :] = shift[0]  # 垂直位移
        return flow

    def process_dataset(self, dataset: torch.utils.data.Dataset, psf: Image.Image, psf_dark: Image.Image, mode: str, num_frames: int):
        """处理数据集"""
        dirs = self._get_directories(mode)

        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {mode} set"):
            # 使用物体ID获取预计算的移动序列
            object_shifts = self.object_shifts[idx]

            total_shift = (0, 0)
            shift = (0, 0)
            flows = np.zeros((num_frames - 1, 2, self.params['bg_size'], self.params['bg_size']), dtype=np.float32)

            # # 创建记录位置信息的文件
            # movement_log_path = os.path.join(dirs['object_log'], f'{mode}_movement_log_{idx}.txt')
            # with open(movement_log_path, 'w') as log_file:
            #     log_file.write(f"Object {idx} Movement Log\n")
            #
            # for t in range(num_frames):
            #     image_np = (image.numpy().squeeze() * 255).astype(np.uint8)
            #
            #     if t > 0:
            #         # 使用物体特定的移动序列
            #         shift = object_shifts[t - 1]
            #         total_shift = (total_shift[0] + shift[0], total_shift[1] + shift[1])
            #
            #     speckle, moved, object_mask = self.process_frame(image_np, total_shift, shift, psf, psf_dark)
            #
            #     # 保存图像
            #     for img_type, img in [('speckle', speckle), ('object', moved)]:
            #         path = os.path.join(dirs[img_type], f'{mode}_image_{idx}_frame_{t}.png')
            #         Image.fromarray(img.astype(np.uint8), 'L').save(path)
            #
            #     # 记录位置和移动信息
            #     if t == 0:
            #         initial_position = self.object_initial_positions[idx]
            #         self.movement_config.initial_offset = initial_position
            #
            #         # 保存标签
            #         speckle_label_path = os.path.join(dirs['labels_speckle'], f'{mode}_label_speckle_{idx}.txt')
            #         object_label_path = os.path.join(dirs['labels_object'], f'{mode}_label_object_{idx}.txt')
            #         with open(speckle_label_path, 'w') as f:
            #             f.write(str(label))
            #         with open(object_label_path, 'w') as f:
            #             f.write(str(label))
            #
            #     with open(movement_log_path, 'a') as log_file:
            #         if t == 0:
            #             log_file.write(f"Initial position: {initial_position}\n")
            #         log_file.write(f"Frame {t}: Shift {shift}, Total Shift {total_shift}\n")
            #
            #     if t > 0:
            #         flows[t - 1] = self.generate_flow(shift)

            # 保存光流数据
            # 创建记录位置信息的文件
            movement_log_path = os.path.join(dirs['object_log'], f'{mode}_movement_log_{idx}.txt')
            with open(movement_log_path, 'w') as log_file:
                log_file.write(f"Object {idx} Movement Log\n")
                # --- BEGIN PATCH: set initial position BEFORE processing frames ---
            # 先把该物体的初始位置设置好（非常重要：必须在处理第0帧前设置）
            initial_position = self.object_initial_positions[idx]
            self.movement_config.initial_offset = initial_position
            # 记录初始位置到 log
            with open(movement_log_path, 'a') as log_file:
                log_file.write(f"Initial position: {initial_position}\n")
            # --- END PATCH ---

            for t in range(num_frames):
                image_np = (image.numpy().squeeze() * 255).astype(np.uint8)
                if t > 0:
                    # 使用物体特定的移动序列
                    shift = object_shifts[t - 1]
                    total_shift = (total_shift[0] + shift[0], total_shift[1] + shift[1])
                speckle, moved, object_mask = self.process_frame(image_np, total_shift, shift, psf, psf_dark)
                # 保存图像
                for img_type, img in [('speckle', speckle), ('object', moved)]:
                    path = os.path.join(dirs[img_type], f'{mode}_image_{idx}_frame_{t}.png')
                    Image.fromarray(img.astype(np.uint8), 'L').save(path)
                # 记录位置和移动信息
                # （initial_position 已在外面设置并写过一次）
                with open(movement_log_path, 'a') as log_file:
                    log_file.write(f"Frame {t}: Shift {shift}, Total Shift {total_shift}\n")
                    if t > 0:
                        flows[t - 1] = self.generate_flow(shift)
            np.save(os.path.join(dirs['flow'], f'{mode}_flow_{idx}.npy'), flows)


def load_dataset_names_from_csv(filename):
    df = pd.read_csv(filename)
    dataset_names = [f"random_movement_obj_{row.obj_size}_bg_{row.bg_size}_move_{row.move_range}_psf_{row.psf_size}" for
                     _, row in df.iterrows()]
    return dataset_names


def generate_random_initial_position(obj_size, bg_size):
    half_range = (bg_size - obj_size) // 2  # 256-128 = 128 → ±64
    init_y = random.randint(-half_range, half_range)
    init_x = random.randint(-half_range, half_range)
    return (init_y, init_x)


# def generate_fixed_shifts_per_object(num_objects: int, num_frames: int, move_range: int) -> Dict[int, List[Tuple[int, int]]]:
#     """为每个物体生成固定的移动序列，确保包含极限情况"""
#     random.seed(42)  # 确保每次运行相同
#     # 为每个物体生成移动序列
#     object_shifts = {}
#
#     # 定义移动类型的分布
#     # 10% 极限情况，10% 接近极限，80% 完全随机
#     extreme_ratio = 0.1
#     near_extreme_ratio = 0.1
#
#     for obj_idx in range(num_objects):
#         shifts = []
#         for frame in range(num_frames - 1):
#             movement_type = random.random()  # 0-1之间的随机数
#
#             if movement_type < extreme_ratio:
#                 # 极限情况：使用边界值或接近边界值
#                 x_shift = random.choice([-move_range, move_range])
#                 y_shift = random.choice([-move_range, move_range])
#             elif movement_type < extreme_ratio + near_extreme_ratio:
#                 # 接近极限的情况：使用接近边界的值
#                 edge_range = int(move_range * 0.2)  # 边缘区域的范围，例如20%
#                 x_sign = random.choice([-1, 1])
#                 y_sign = random.choice([-1, 1])
#                 x_shift = x_sign * random.randint(move_range - edge_range, move_range)
#                 y_shift = y_sign * random.randint(move_range - edge_range, move_range)
#             else:
#                 # 普通随机情况
#                 x_shift = random.randint(-move_range, move_range)
#                 y_shift = random.randint(-move_range, move_range)
#
#             shifts.append((y_shift, x_shift))  # 注意：y轴对应的是垂直方向
#
#         object_shifts[obj_idx] = shifts
#
#     return object_shifts

def generate_fixed_shifts_per_object(num_objects, num_frames, move_range, obj_size, bg_size):
    object_initial_positions = {}
    object_shifts = {}
    half_range = (bg_size - obj_size) // 2  # = 64
    for obj_idx in range(num_objects):
        # ① 生成随机初始位置（保证物体不会出界）
        init_pos = generate_random_initial_position(obj_size, bg_size)
        object_initial_positions[obj_idx] = init_pos
        # ② 生成移动序列
        shifts = []
        total_y, total_x = 0, 0  # 累积移动
        for t in range(num_frames - 1):
            while True:
                # 随机移动（你的三类策略依然可以放进来）
                dy = random.randint(-move_range, move_range)
                dx = random.randint(-move_range, move_range)
                new_y = total_y + dy
                new_x = total_x + dx
                # 检查移动后是否越界
                if (-half_range <= init_pos[0] + new_y <= half_range and
                        -half_range <= init_pos[1] + new_x <= half_range):
                    # 合法 → 接受移动
                    total_y = new_y
                    total_x = new_x
                    shifts.append((dy, dx))
                    break
            # 如果不合法则继续 while True 重抽
        object_shifts[obj_idx] = shifts
    return object_initial_positions, object_shifts

def load_custom_digit_dataset(root_dir, train_per_class=50, test_per_class=10):
    """
    root_dir: 数据集根目录（包含 train/ 和 test/ 文件夹）
    train_per_class: 每个数字类别从训练集中抽取多少张
    test_per_class: 每个数字类别从测试集中抽取多少张
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 加载 train / test 文件夹（ImageFolder 会自动按文件夹名作为类别标签）
    full_train_dataset = ImageFolder(os.path.join(root_dir, "train"), transform=transform)
    full_test_dataset = ImageFolder(os.path.join(root_dir, "test"), transform=transform)

    def sample_per_class(dataset, num_per_class):
        indices = []
        targets = np.array(dataset.targets)
        for label in range(10):
            class_indices = np.where(targets == label)[0]
            sampled = random.sample(list(class_indices), min(num_per_class, len(class_indices)))
            indices.extend(sampled)
        return Subset(dataset, indices)

    train_dataset = sample_per_class(full_train_dataset, train_per_class)
    test_dataset = sample_per_class(full_test_dataset, test_per_class)

    return train_dataset, test_dataset

def main():
    start_time = time.time()  # 记录开始时间
    # 基础配置
    base_path = 'data/datasets'
    dataset_names = load_dataset_names_from_csv("datasets.csv")
    # train_size, test_size, num_frames = 6, 6, 5
    train_size, val_size, test_size, num_frames = 500, 15, 100, 5

    # 加载MNIST数据集和PSF
    mnist_train_dataset = MNIST(root='./data/datasets', train=True, download=False, transform=transforms.ToTensor())
    mnist_test_dataset = MNIST(root='./data/datasets', train=False, download=False, transform=transforms.ToTensor())

    # 从训练集和测试集中分别随机抽取指定数量的样本
    train_dataset, val_dataset,  _ = torch.utils.data.random_split(mnist_train_dataset,
                                                     [train_size, val_size, len(mnist_train_dataset) - train_size - val_size])
    test_dataset, _ = torch.utils.data.random_split(mnist_test_dataset,
                                                    [test_size, len(mnist_test_dataset) - test_size])

    # # 使用自定义文件夹数据集
    # custom_root = "data/datasets/printnumber"  # 改成你自己的数据路径
    # train_dataset, test_dataset = load_custom_digit_dataset(custom_root, train_per_class=50, test_per_class=10)
    #
    # # 如果你还想保留验证集，可以自己拆，比如：
    # val_size = 50
    # train_size = len(train_dataset) - val_size
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # from torchvision.datasets.folder import default_loader
    # class CustomImageDataset(torch.utils.data.Dataset):
    #     def __init__(self, image_dir, transform=None):
    #         self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
    #                                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    #         self.transform = transform or transforms.ToTensor()
    #
    #     def __len__(self):
    #         return len(self.image_paths)
    #
    #     def __getitem__(self, idx):
    #         image = default_loader(self.image_paths[idx])  # 加载为PIL图像
    #         image = image.convert('L')  # 转为灰度图
    #         return self.transform(image), 0  # label暂时为0
    #
    # # 加载你的自定义图像数据
    # custom_train_dataset = CustomImageDataset('data/USAFdigits/')
    # # train_dataset, _ = torch.utils.data.random_split(custom_train_dataset,
    # #                                                  [train_size, len(custom_train_dataset) - train_size])
    # # test_dataset = Subset(train_dataset.dataset, train_dataset.indices[:test_size])
    # train_dataset = torch.utils.data.Subset(custom_train_dataset, list(range(train_size)))
    # test_dataset = torch.utils.data.Subset(custom_train_dataset, list(range(test_size)))

    # 为每个物体生成固定的移动序列（对于每个PSF都是相同的）
    # 这里按照物体ID索引的移动序列
    object_shifts = {}
    init_pos = {}
    for dataset_name in dataset_names:
        params = parse_dataset_name(dataset_name)
        group_key = f"{params['obj_size']}_{params['move_range']}_{params['bg_size']}"

        if group_key not in object_shifts:
            # 为训练集和测试集分别生成物体移动
            init_pos_train, train_object_shifts = generate_fixed_shifts_per_object(
                train_size, num_frames, params['move_range'], params['obj_size'], params['bg_size'])
            init_pos_val, val_object_shifts = generate_fixed_shifts_per_object(
                val_size, num_frames, params['move_range'], params['obj_size'], params['bg_size'])
            init_pos_test, test_object_shifts = generate_fixed_shifts_per_object(
                test_size, num_frames, params['move_range'], params['obj_size'], params['bg_size'])
            object_shifts[group_key] = {
                'train': train_object_shifts,
                'val': val_object_shifts,
                'test': test_object_shifts
            }
            init_pos[group_key] = {
                'train': init_pos_train,
                'val': init_pos_val,
                'test': init_pos_test
            }

    # 处理每个数据集
    for dataset_name in dataset_names:
        params = parse_dataset_name(dataset_name)
        group_key = f"{params['obj_size']}_{params['move_range']}_{params['bg_size']}"

        # 替换原来的 psf = Image.open(...) 调用：
        psf = Image.open('data/datasets/PSF/psf_1202.bmp').convert('L')
        psf_dark = Image.open('data/datasets/PSF/psf-拿书挡住-曝光3883.bmp').convert('L')

        # 创建数据生成器
        generator = DatasetGenerator(base_path, dataset_name, object_shifts[group_key]['train'],init_pos[group_key]['train'])
        generator.process_dataset(train_dataset, psf, psf_dark, 'train', num_frames)

        generator = DatasetGenerator(base_path, dataset_name, object_shifts[group_key]['val'],init_pos[group_key]['val'])
        generator.process_dataset(val_dataset, psf, psf_dark, 'val', num_frames)

        generator = DatasetGenerator(base_path, dataset_name, object_shifts[group_key]['test'], init_pos[group_key]['test'])
        generator.process_dataset(test_dataset, psf, psf_dark, 'test', num_frames)

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"代码运行总时间: {total_time / 60:.2f}分钟, {total_time / 3600:.2f}小时, 数据集处理完成")


if __name__ == "__main__":
    main()
