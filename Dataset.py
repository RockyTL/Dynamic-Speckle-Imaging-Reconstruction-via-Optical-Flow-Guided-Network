import os
import pywt
import cv2
import torch
import torch.nn.functional as F
from PIL import ImageOps
from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from PIL import ImageEnhance, ImageFilter
from scipy.ndimage import zoom

class SpeckleDataset_New(Dataset):
    def __init__(self, base_path, mode='train', pos='left_to_right'):
        self.base_path = os.path.join(base_path, pos)
        self.mode = mode
        self.speckle_dir = os.path.join(self.base_path, f'{mode}_speckle_images')
        self.object_dir = os.path.join(self.base_path, f'{mode}_object_images')
        self.flow_dir = os.path.join(self.base_path, f'{mode}_flow')

        # 获取物体的帧序列
        self.object_ids = self._get_object_ids()

    def _get_object_ids(self):
        # 假设文件命名格式一致，提取所有物体的编号
        flow_files = sorted(os.listdir(self.flow_dir))
        object_ids = list(set([int(os.path.splitext(f)[0].split('_')[2]) for f in flow_files]))
        return object_ids

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, index):
        obj_id = self.object_ids[index]

        # 加载该物体的所有帧
        speckle_seq = []
        object_seq = []
        flow_seq = np.load(os.path.join(self.flow_dir, f'{self.mode}_flow_{obj_id}.npy'))

        for t in range(flow_seq.shape[0] + 1):
            speckle_path = os.path.join(self.speckle_dir, f'{self.mode}_image_{obj_id}_frame_{t}.png')
            object_path = os.path.join(self.object_dir, f'{self.mode}_image_{obj_id}_frame_{t}.png')
            speckle_seq.append(transforms.ToTensor()(Image.open(speckle_path)))
            object_seq.append(transforms.ToTensor()(Image.open(object_path)))

        return {
            'speckle_seq': torch.stack(speckle_seq),
            'object_seq': torch.stack(object_seq),
            'flow_seq': torch.tensor(flow_seq)
        }


def normalization(img_pil):
    img_np = np.array(img_pil).astype(np.float32)  # 转为 float32 numpy
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)  # Min-Max 归一化
    img_tensor = torch.from_numpy(img_np)  # 加上通道维度 [1, H, W]
    return img_tensor

class SpeckleOnlySequenceDataset(Dataset):
    def __init__(self, image_dir, sequence_length=5):
        """
        Args:
            image_dir (str): 包含 Image1.bmp 到 Image500.bmp 的目录路径
            sequence_length (int): 每个物体对应的帧数
            crop_size (int): 中心裁剪的大小，例如 512 表示 512x512
        """
        self.image_dir = image_dir
        self.sequence_length = sequence_length

        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]
        # 解析出 Image编号 和 frame编号
        groups = {}
        for f in all_files:
            name, _ = os.path.splitext(f)
            # 例如 name = "Image1_frame3"
            if "_frame_" in name:
                base, frame = name.split("_frame_")
                if base not in groups:
                    groups[base] = []
                groups[base].append(f)
        # 每个 group 内按 frameX 的数字排序
        self.grouped_files = []
        for base, files in sorted(groups.items(), key=lambda x: int(x[0].split("_")[1])):
            sorted_files = sorted(files, key=lambda x: int(x.split("_frame_")[-1].split(".")[0]))
            self.grouped_files.append(sorted_files)
        self.num_sequences = len(self.grouped_files)

        # 定义 transforms：中心裁剪 + 旋转180度 + 转tensor
        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.RandomRotation((180, 180)),  # 固定角度旋转
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),
        ])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        speckle_seq = []
        speckle_raw_seq = []

        frames = self.grouped_files[index]

        for img_name in frames:
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path).convert('L')

            # 加载参考PSF图并转为 numpy array
            ref_img_path = os.path.join('data/datasets/psf-blockedbybook-exposure3883.bmp')
            ref_img = Image.open(ref_img_path).convert('L')
            # 当前帧 numpy
            img_np = np.array(img).astype(np.float32)
            # 参考图 numpy
            ref_np = np.array(ref_img).astype(np.float32)
            # 如果尺寸不一致，做中心裁剪
            H_img, W_img = img_np.shape
            H_ref, W_ref = ref_np.shape
            if (H_img != H_ref) or (W_img != W_ref):
                top = (H_ref - H_img) // 2
                left = (W_ref - W_img) // 2
                ref_np_cropped = ref_np[top:top + H_img, left:left + W_img]
            else:
                ref_np_cropped = ref_np
            # 做减法并截断
            img_np = img_np - ref_np_cropped
            # 转回 PIL 图像
            img = Image.fromarray(img_np.astype(np.uint8))
            img = self.transform(img)

            img1 = (img - img.min()) / (img.max() - img.min() + 1e-8)
            speckle_raw_seq.append(img1)

            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(192, 192), mode='bilinear')
            img = F.interpolate(img, size=(128, 128), mode='bilinear')
            img = F.interpolate(img, size=(96, 96), mode='bilinear')
            img = F.interpolate(img, size=(64, 64), mode='bilinear')
            img = F.interpolate(img, size=(32, 32), mode='bilinear')
            img = F.interpolate(img, size=(64, 64), mode='bicubic')
            img = F.interpolate(img, size=(96, 96), mode='bicubic')
            img = F.interpolate(img, size=(128, 128), mode='bicubic')
            img = F.interpolate(img, size=(192, 192), mode='bicubic')
            img = F.interpolate(img, size=(256, 256), mode='bicubic')
            img = img.squeeze(0)

            img = normalization(img)

            speckle_seq.append(img)

        return {
            'speckle_seq': torch.stack(speckle_seq),  # [T, 1, H, W]
            'speckle_raw_seq': torch.stack(speckle_raw_seq)
            # 'index': index
        }


# class SpeckleOnlySequenceDataset(Dataset):
#     def __init__(self, image_dir, crop_size=896, target_size=256, step_size=64):
#         """
#         Args:
#             image_dir (str): 包含散斑图像的目录路径
#             crop_size (int): 原始裁剪大小
#             target_size (int): 目标裁剪大小
#             step_size (int): 滑动窗口步长
#         """
#         self.image_dir = image_dir
#         self.crop_size = crop_size
#         self.target_size = target_size
#         self.step_size = step_size
#
#         # 获取所有图像文件并分组
#         all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
#         groups = {}
#         for f in all_files:
#             name, _ = os.path.splitext(f)
#             if "_frame" in name:
#                 base, frame = name.split("_frame")
#                 if base not in groups:
#                     groups[base] = []
#                 groups[base].append(f)
#
#         # 每个序列内按帧号排序
#         self.grouped_files = []
#         for base, files in sorted(groups.items(), key=lambda x: int(x[0].replace("Image", ""))):
#             sorted_files = sorted(files, key=lambda x: int(x.split("_frame")[-1].split(".")[0]))
#             self.grouped_files.append(sorted_files)
#         self.num_sequences = len(self.grouped_files)
#         self.frames_per_sequence = len(self.grouped_files[0]) if self.grouped_files else 0
#
#         print(f"总序列数: {self.num_sequences}")
#         print(f"每序列帧数: {self.frames_per_sequence}")
#         print(f"总图像数: {self.num_sequences * self.frames_per_sequence}")
#
#         # 计算所有可能的裁剪位置
#         self.crop_positions = []
#         max_pos = crop_size - target_size
#         for y in range(0, max_pos + 1, step_size):
#             for x in range(0, max_pos + 1, step_size):
#                 self.crop_positions.append((x, y))
#
#         print(f"总裁剪位置: {len(self.crop_positions)}")
#         print(f"总样本数: {len(self.crop_positions)}")  # 每个位置处理所有序列
#
#         # 定义变换：首先中心裁剪到896x896，然后旋转180度
#         self.transform = transforms.Compose([
#             transforms.CenterCrop(crop_size),
#             transforms.RandomRotation((180, 180)),
#             transforms.ToTensor(),
#         ])
#
#     def __len__(self):
#         return len(self.crop_positions)  # 只按裁剪位置数
#
#     def __getitem__(self, index):
#         # 只按裁剪位置索引
#         crop_x, crop_y = self.crop_positions[index]
#
#         # 存储所有序列的所有帧
#         all_speckle_seqs = []  # 形状: [num_sequences, frames_per_sequence, 1, H, W]
#         all_speckle_raw_seqs = []  # 形状: [num_sequences, frames_per_sequence, 1, H, W]
#
#         # 对每个序列进行处理
#         for seq_idx, frames in enumerate(self.grouped_files):
#             speckle_seq = []
#             speckle_raw_seq = []
#
#             for img_name in frames:
#                 img_path = os.path.join(self.image_dir, img_name)
#                 img = Image.open(img_path).convert('L')
#
#                 # 参考PSF处理
#                 ref_img_path = os.path.join('data/datasets/psf-blockedbybook-exposure3883.bmp')
#                 ref_img = Image.open(ref_img_path).convert('L')
#
#                 img_np = np.array(img).astype(np.float32)
#                 ref_np = np.array(ref_img).astype(np.float32)
#
#                 H_img, W_img = img_np.shape
#                 H_ref, W_ref = ref_np.shape
#                 if (H_img != H_ref) or (W_img != W_ref):
#                     top = (H_ref - H_img) // 2
#                     left = (W_ref - W_img) // 2
#                     ref_np_cropped = ref_np[top:top + H_img, left:left + W_img]
#                 else:
#                     ref_np_cropped = ref_np
#                 # img_np = img_np - ref_np_cropped
#
#                 # 转回PIL图像
#                 img = Image.fromarray(img_np.astype(np.uint8))
#
#                 # 应用变换（中心裁剪到896x896 + 旋转）
#                 img_tensor = self.transform(img)
#
#                 # 在相同位置裁剪
#                 img_cropped = img_tensor[:, crop_y:crop_y + self.target_size, crop_x:crop_x + self.target_size]
#
#                 # 原始散斑序列
#                 img1 = (img_cropped - img_cropped.min()) / (img_cropped.max() - img_cropped.min() + 1e-8)
#                 speckle_raw_seq.append(img1)
#
#                 # 处理后的散斑序列（下采样+上采样）
#                 img_processed = img_cropped.unsqueeze(0)
#                 img_processed = F.interpolate(img_processed, size=(192, 192), mode='bilinear')
#                 img_processed = F.interpolate(img_processed, size=(128, 128), mode='bilinear')
#                 img_processed = F.interpolate(img_processed, size=(96, 96), mode='bilinear')
#                 img_processed = F.interpolate(img_processed, size=(64, 64), mode='bilinear')
#                 img_processed = F.interpolate(img_processed, size=(32, 32), mode='bilinear')
#                 img_processed = F.interpolate(img_processed, size=(64, 64), mode='bicubic')
#                 img_processed = F.interpolate(img_processed, size=(96, 96), mode='bicubic')
#                 img_processed = F.interpolate(img_processed, size=(128, 128), mode='bicubic')
#                 img_processed = F.interpolate(img_processed, size=(256, 256), mode='bicubic')
#                 img_processed = img_processed.squeeze(0)
#
#                 img_processed = normalization(img_processed)
#                 speckle_seq.append(img_processed)
#
#             all_speckle_seqs.append(torch.stack(speckle_seq))  # [frames_per_sequence, 1, H, W]
#             all_speckle_raw_seqs.append(torch.stack(speckle_raw_seq))  # [frames_per_sequence, 1, H, W]
#
#         return {
#             'speckle_seq': torch.stack(all_speckle_seqs),  # [num_sequences, frames_per_sequence, 1, H, W]
#             'speckle_raw_seq': torch.stack(all_speckle_raw_seqs),
#             'crop_x': crop_x,
#             'crop_y': crop_y,
#             'crop_position_idx': index
#


class SpeckleOnlySequenceDatasetWithObjectAndFlow(Dataset):
    def __init__(self, speckle_dir, object_dir, flow_dir=None, sequence_length=5, crop_size=256):
        """
        Args:
            speckle_dir (str): 散斑图像所在目录，包含 Image1.bmp 到 Image500.bmp
            object_dir (str): 对应物体图像所在目录，命名为 image_0_frame_0.png 等
            sequence_length (int): 每个物体的帧数
            crop_size (int): 中心裁剪大小
        """
        self.speckle_dir = speckle_dir
        self.object_dir = object_dir
        self.flow_dir = flow_dir
        self.sequence_length = sequence_length
        self.crop_size = crop_size

        all_files = [f for f in os.listdir(speckle_dir) if f.lower().endswith('.bmp')]
        # 解析出 Image编号 和 frame编号
        groups = {}
        for f in all_files:
            name, _ = os.path.splitext(f)
            if "_frame_" in name:
                base, frame = name.split("_frame_")
                if base not in groups:
                    groups[base] = []
                groups[base].append(f)
        # 每个 group 内按 frameX 的数字排序
        self.grouped_files = []
        for base, files in sorted(groups.items(), key=lambda x: int(x[0].split("_")[1])):
            sorted_files = sorted(files, key=lambda x: int(x.split("_frame_")[-1].split(".")[0]))
            self.grouped_files.append(sorted_files)
        self.num_sequences = len(self.grouped_files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-8)),
        ])

        self.object_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((180, 180)),
        ])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        speckle_seq = []
        speckle_raw_seq = []
        object_seq = []

        frames = self.grouped_files[index]

        for img_name in frames:
            # Speckle image
            img_path = os.path.join(self.speckle_dir, img_name)
            img = Image.open(img_path).convert('L')

            # 加载参考PSF图并转为 numpy array
            ref_img_path = os.path.join('data/datasets/psf-blockedbybook-exposure3883.bmp')
            ref_img = Image.open(ref_img_path).convert('L')
            # 当前帧 numpy
            img_np = np.array(img).astype(np.float32)
            # 参考图 numpy
            ref_np = np.array(ref_img).astype(np.float32)
            # 如果尺寸不一致，做中心裁剪
            H_img, W_img = img_np.shape
            H_ref, W_ref = ref_np.shape
            if (H_img != H_ref) or (W_img != W_ref):
                top = (H_ref - H_img) // 2
                left = (W_ref - W_img) // 2
                ref_np_cropped = ref_np[top:top + H_img, left:left + W_img]
            else:
                ref_np_cropped = ref_np
            # 做减法并截断
            img_np = img_np - ref_np_cropped
            # 转回 PIL 图像
            img = Image.fromarray(img_np.astype(np.uint8))
            img = self.transform(img)

            img1 = (img - img.min()) / (img.max() - img.min() + 1e-8)
            speckle_raw_seq.append(img1)

            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(192, 192), mode='bilinear')
            img = F.interpolate(img, size=(128, 128), mode='bilinear')
            img = F.interpolate(img, size=(96, 96), mode='bilinear')
            img = F.interpolate(img, size=(64, 64), mode='bilinear')
            img = F.interpolate(img, size=(32, 32), mode='bilinear')
            img = F.interpolate(img, size=(64, 64), mode='bicubic')
            img = F.interpolate(img, size=(96, 96), mode='bicubic')
            img = F.interpolate(img, size=(128, 128), mode='bicubic')
            img = F.interpolate(img, size=(192, 192), mode='bicubic')
            img = F.interpolate(img, size=(256, 256), mode='bicubic')
            img = img.squeeze(0)

            img = normalization(img)

            speckle_seq.append(img)

            # Object image
            # === 读取物体 ===
            base_name = os.path.splitext(img_name)[0]  # 去掉.bmp扩展名
            object_name = base_name + '.png'  # 改成.png
            object_path = os.path.join(self.object_dir, object_name)
            object_img = Image.open(object_path).convert('L')
            object_seq.append(self.object_transform(object_img))

        if self.flow_dir:
            flow_path = os.path.join(self.flow_dir, f'flow_image_{index}.npy')
            flow = torch.from_numpy(np.load(flow_path)).float()  # [4, 2, 512, 512]
        else:
            flow = None

        return {
            'speckle_seq': torch.stack(speckle_seq),  # [T, 1, H, W]
            'speckle_raw_seq': torch.stack(speckle_raw_seq),
            'object_seq': torch.stack(object_seq),     # [T, 1, H, W]
            'flow_seq': flow
        }
