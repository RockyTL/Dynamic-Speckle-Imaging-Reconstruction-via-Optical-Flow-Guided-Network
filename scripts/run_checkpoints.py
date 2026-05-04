#!/usr/bin/env python
"""Batch checkpoint evaluation for Dynamic Speckle Imaging Reconstruction.

Evaluates multiple model checkpoints sequentially and saves results
to per-epoch output directories.

Usage:
    python scripts/run_checkpoints.py --config configs/test.yaml
    python scripts/run_checkpoints.py --config configs/test.yaml --mode experiment
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.data.datasets import (
    SpeckleDataset_New,
    SpeckleOnlySequenceDatasetWithObjectAndFlow,
)
from src.engine.evaluator import test_experimental_data_withobj, test_model
from src.models.complete_model import CompleteModel
from src.utils.io_utils import scan_checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate model checkpoints."
    )
    parser.add_argument(
        '--config', type=str, default='configs/test.yaml',
        help='Path to test config YAML file.',
    )
    parser.add_argument(
        '--mode', type=str, default='synthetic',
        choices=['synthetic', 'experiment'],
        help='Evaluation mode.',
    )
    parser.add_argument(
        '--start_epoch', type=int, default=None,
        help='Minimum epoch to evaluate (overrides config).',
    )
    parser.add_argument(
        '--end_epoch', type=int, default=None,
        help='Maximum epoch to evaluate (overrides config).',
    )
    return parser.parse_args()


def run_all_checkpoints(
    checkpoint_dir: str,
    save_root: str,
    start_epoch: int,
    end_epoch: int,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """Evaluate all checkpoints in a range on synthetic test data."""
    os.makedirs(save_root, exist_ok=True)
    epoch_files = scan_checkpoints(checkpoint_dir, start_epoch, end_epoch)

    print(f"Found checkpoint files: {epoch_files}")

    for ep, file in epoch_files:
        print(f"\n========== Testing epoch {ep} ==========")
        save_dir = os.path.join(save_root, f"results_epoch_{ep}")
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, file)
        print(f"Loading model: {model_path}")
        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(model_path))

        test_model(model, test_loader, device, save_dir=save_dir)
        print(f"Epoch {ep} done. Results saved to: {save_dir}")


def run_all_experiment_checkpoints(
    checkpoint_dir: str,
    save_root: str,
    start_epoch: int,
    end_epoch: int,
    test_loader: DataLoader,
    device: torch.device,
    num_frames: int = 5,
) -> None:
    """Evaluate all checkpoints in a range on experimental data with GT."""
    os.makedirs(save_root, exist_ok=True)
    epoch_files = scan_checkpoints(checkpoint_dir, start_epoch, end_epoch)

    print(f"Found experiment checkpoint files: {epoch_files}")

    for ep, file in epoch_files:
        print(f"\n========== Testing experiment epoch {ep} ==========")
        save_dir = os.path.join(save_root, f"experiment_results_epoch_{ep}")
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, file)
        print(f"Loading model: {model_path}")
        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(model_path))

        test_experimental_data_withobj(
            model, test_loader, device,
            save_dir=save_dir, num_frames=num_frames,
        )
        print(f"Epoch {ep} experiment done. Results saved to: {save_dir}")


def main() -> None:
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_cfg = cfg['batch_eval']
    checkpoint_dir = batch_cfg['checkpoint_dir']
    start_epoch = args.start_epoch or batch_cfg['start_epoch']
    end_epoch = args.end_epoch or batch_cfg['end_epoch']

    data_cfg = cfg['data']

    if args.mode == 'experiment':
        exp_cfg = cfg['experiment']
        dataset = SpeckleOnlySequenceDatasetWithObjectAndFlow(
            exp_cfg['speckle_data_dir_v2'],
            exp_cfg['object_data_dir_v2'],
            exp_cfg['flow_data_dir'],
        )
        test_loader = DataLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )
        run_all_experiment_checkpoints(
            checkpoint_dir=checkpoint_dir,
            save_root=batch_cfg['experiment_save_root'],
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            test_loader=test_loader,
            device=device,
            num_frames=data_cfg['num_frames'],
        )
    else:
        test_dataset = SpeckleDataset_New(
            data_cfg['base_path'], mode='test', pos=data_cfg['pos']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )
        run_all_checkpoints(
            checkpoint_dir=checkpoint_dir,
            save_root=batch_cfg['save_root'],
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            test_loader=test_loader,
            device=device,
        )


if __name__ == "__main__":
    main()
