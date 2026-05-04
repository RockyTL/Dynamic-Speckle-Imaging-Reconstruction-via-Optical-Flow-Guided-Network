"""Dataset classes for speckle imaging with object and optical flow ground truth."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SpeckleDataset_New(Dataset):
    """Synthetic speckle datasets with object and flow ground truth.

    Expects directory structure:
        base_path/pos/
            train_speckle_images/
            train_object_images/
            train_flow/
            val_speckle_images/ ...
            test_speckle_images/ ...
    """

    def __init__(self, base_path: str, mode: str = 'train', pos: str = 'left_to_right') -> None:
        self.base_path = os.path.join(base_path, pos)
        self.mode = mode
        self.speckle_dir = os.path.join(self.base_path, f'{mode}_speckle_images')
        self.object_dir = os.path.join(self.base_path, f'{mode}_object_images')
        self.flow_dir = os.path.join(self.base_path, f'{mode}_flow')
        self.object_ids = self._get_object_ids()

    def _get_object_ids(self) -> List[int]:
        flow_files = sorted(os.listdir(self.flow_dir))
        object_ids = list(set(
            int(os.path.splitext(f)[0].split('_')[2]) for f in flow_files
        ))
        return object_ids

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        obj_id = self.object_ids[index]

        flow_seq = np.load(
            os.path.join(self.flow_dir, f'{self.mode}_flow_{obj_id}.npy')
        )

        speckle_seq = []
        object_seq = []
        for t in range(flow_seq.shape[0] + 1):
            speckle_path = os.path.join(
                self.speckle_dir, f'{self.mode}_image_{obj_id}_frame_{t}.png'
            )
            object_path = os.path.join(
                self.object_dir, f'{self.mode}_image_{obj_id}_frame_{t}.png'
            )
            speckle_seq.append(transforms.ToTensor()(Image.open(speckle_path)))
            object_seq.append(transforms.ToTensor()(Image.open(object_path)))

        return {
            'speckle_seq': torch.stack(speckle_seq),
            'object_seq': torch.stack(object_seq),
            'flow_seq': torch.tensor(flow_seq),
        }


def normalization(img_pil: Image.Image) -> torch.Tensor:
    """Min-max normalize a PIL image and return a tensor [H, W]."""
    img_np = np.array(img_pil).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_tensor = torch.from_numpy(img_np)
    return img_tensor


class SpeckleOnlySequenceDataset(Dataset):
    """Dataset for experimental speckle data (no object/flow ground truth).

    Loads .bmp speckle images, groups them into sequences, applies
    PSF subtraction, center-crop, rotation, and multi-scale processing.

    Args:
        image_dir: Directory containing Image1_frame0.bmp ... ImageN_frame4.bmp
        sequence_length: Number of frames per sequence
    """

    def __init__(self, image_dir: str, sequence_length: int = 5) -> None:
        self.image_dir = image_dir
        self.sequence_length = sequence_length

        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.bmp')]
        groups: Dict[str, List[str]] = {}
        for f in all_files:
            name, _ = os.path.splitext(f)
            if "_frame_" in name:
                base, frame = name.split("_frame_")
                if base not in groups:
                    groups[base] = []
                groups[base].append(f)

        self.grouped_files: List[List[str]] = []
        for base, files in sorted(
            groups.items(), key=lambda x: int(x[0].split("_")[1])
        ):
            sorted_files = sorted(
                files, key=lambda x: int(x.split("_frame_")[-1].split(".")[0])
            )
            self.grouped_files.append(sorted_files)
        self.num_sequences = len(self.grouped_files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        speckle_seq = []
        speckle_raw_seq = []

        frames = self.grouped_files[index]

        for img_name in frames:
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path).convert('L')

            # Load reference PSF and subtract
            ref_img_path = os.path.join(
                'data/datasets/psf-blockedbybook-exposure3883.bmp'
            )
            ref_img = Image.open(ref_img_path).convert('L')

            img_np = np.array(img).astype(np.float32)
            ref_np = np.array(ref_img).astype(np.float32)

            H_img, W_img = img_np.shape
            H_ref, W_ref = ref_np.shape
            if (H_img != H_ref) or (W_img != W_ref):
                top = (H_ref - H_img) // 2
                left = (W_ref - W_img) // 2
                ref_np_cropped = ref_np[top:top + H_img, left:left + W_img]
            else:
                ref_np_cropped = ref_np

            img_np = img_np - ref_np_cropped
            img = Image.fromarray(img_np.astype(np.uint8))
            img = self.transform(img)

            img1 = (img - img.min()) / (img.max() - img.min() + 1e-8)
            speckle_raw_seq.append(img1)

            # Multi-scale processing: down-sample then up-sample
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
            'speckle_seq': torch.stack(speckle_seq),
            'speckle_raw_seq': torch.stack(speckle_raw_seq),
        }


class SpeckleOnlySequenceDatasetWithObjectAndFlow(Dataset):
    """Dataset for experimental speckle data with object and flow ground truth.

    Loads speckle .bmp images, object .png images, and optional flow .npy files.

    Args:
        speckle_dir: Directory containing Image1_frame0.bmp ... ImageN_frame4.bmp
        object_dir: Directory containing matching object .png images
        flow_dir: Optional directory containing flow_image_*.npy files
        sequence_length: Number of frames per sequence
        crop_size: Center crop size
    """

    def __init__(
        self,
        speckle_dir: str,
        object_dir: str,
        flow_dir: Optional[str] = None,
        sequence_length: int = 5,
        crop_size: int = 256,
    ) -> None:
        self.speckle_dir = speckle_dir
        self.object_dir = object_dir
        self.flow_dir = flow_dir
        self.sequence_length = sequence_length
        self.crop_size = crop_size

        all_files = [
            f for f in os.listdir(speckle_dir) if f.lower().endswith('.bmp')
        ]
        groups: Dict[str, List[str]] = {}
        for f in all_files:
            name, _ = os.path.splitext(f)
            if "_frame_" in name:
                base, frame = name.split("_frame_")
                if base not in groups:
                    groups[base] = []
                groups[base].append(f)

        self.grouped_files: List[List[str]] = []
        for base, files in sorted(
            groups.items(), key=lambda x: int(x[0].split("_")[1])
        ):
            sorted_files = sorted(
                files, key=lambda x: int(x.split("_frame_")[-1].split(".")[0])
            )
            self.grouped_files.append(sorted_files)
        self.num_sequences = len(self.grouped_files)

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor(),
        ])

        self.object_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((180, 180)),
        ])

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        speckle_seq = []
        speckle_raw_seq = []
        object_seq = []

        frames = self.grouped_files[index]

        for img_name in frames:
            # Speckle image
            img_path = os.path.join(self.speckle_dir, img_name)
            img = Image.open(img_path).convert('L')

            # Reference PSF subtraction
            ref_img_path = os.path.join(
                'data/datasets/psf-blockedbybook-exposure3883.bmp'
            )
            ref_img = Image.open(ref_img_path).convert('L')

            img_np = np.array(img).astype(np.float32)
            ref_np = np.array(ref_img).astype(np.float32)

            H_img, W_img = img_np.shape
            H_ref, W_ref = ref_np.shape
            if (H_img != H_ref) or (W_img != W_ref):
                top = (H_ref - H_img) // 2
                left = (W_ref - W_img) // 2
                ref_np_cropped = ref_np[top:top + H_img, left:left + W_img]
            else:
                ref_np_cropped = ref_np

            img_np = img_np - ref_np_cropped
            img = Image.fromarray(img_np.astype(np.uint8))
            img = self.transform(img)

            img1 = (img - img.min()) / (img.max() - img.min() + 1e-8)
            speckle_raw_seq.append(img1)

            # Multi-scale processing
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
            base_name = os.path.splitext(img_name)[0]
            object_name = base_name + '.png'
            object_path = os.path.join(self.object_dir, object_name)
            object_img = Image.open(object_path).convert('L')
            object_seq.append(self.object_transform(object_img))

        if self.flow_dir:
            flow_path = os.path.join(self.flow_dir, f'flow_image_{index}.npy')
            flow = torch.from_numpy(np.load(flow_path)).float()
        else:
            flow = None

        return {
            'speckle_seq': torch.stack(speckle_seq),
            'speckle_raw_seq': torch.stack(speckle_raw_seq),
            'object_seq': torch.stack(object_seq),
            'flow_seq': flow,
        }
