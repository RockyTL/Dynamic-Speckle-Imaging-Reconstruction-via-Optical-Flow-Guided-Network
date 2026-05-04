#!/usr/bin/env python
"""Training entry point for Dynamic Speckle Imaging Reconstruction.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --mode finetune
    python scripts/train.py --config configs/train.yaml --mode unet_only
    python scripts/train.py --config configs/train.yaml --resume checkpoints/best_model.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.data.datasets import SpeckleDataset_New
from src.engine.trainer import finetune_model, train_model, train_simple
from src.models.complete_model import CompleteModel, SimpleReconstructionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train dynamic speckle imaging reconstruction model."
    )
    parser.add_argument(
        '--config', type=str, default='configs/train.yaml',
        help='Path to training config YAML file.',
    )
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['full', 'unet_only', 'finetune'],
        help='Training mode: full (RAFT+UNet), unet_only, or finetune.',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint for resuming training.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    data_cfg = cfg['data']
    train_cfg = cfg['training']

    train_dataset = SpeckleDataset_New(
        data_cfg['base_path'], mode='train', pos=data_cfg['pos']
    )
    test_dataset = SpeckleDataset_New(
        data_cfg['base_path'], mode='test', pos=data_cfg['pos']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True,
        num_workers=data_cfg['num_workers'],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=None, shuffle=False,
        num_workers=data_cfg['num_workers'],
    )

    # ---- Mode dispatch ----
    if args.mode == 'unet_only':
        cfg_unet = cfg['unet_only']
        model = SimpleReconstructionModel().to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_simple(
            model, train_loader, device,
            num_epochs=cfg_unet['num_epochs'],
            save_dir=cfg_unet['save_dir'],
        )

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"Peak GPU memory: {peak_mem:.2f} MB")

    elif args.mode == 'finetune':
        from src.data.datasets import SpeckleOnlySequenceDatasetWithObjectAndFlow

        ft_cfg = cfg['finetune']
        experiment_speckle_data_dir = 'data/datasets/LCDMnist'
        experiment_object_data_dir = 'data/datasets/obj_128obj_movetimes4'

        dataset = SpeckleOnlySequenceDatasetWithObjectAndFlow(
            experiment_speckle_data_dir, experiment_object_data_dir
        )
        finetune_loader = DataLoader(
            dataset, batch_size=None, shuffle=True, num_workers=4,
        )

        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(ft_cfg['pretrained_checkpoint']))

        finetune_model(
            model, finetune_loader, device,
            num_epochs=ft_cfg['num_epochs'],
            save_dir=ft_cfg['save_dir'],
            num_frames=data_cfg['num_frames'],
        )

    else:  # full training
        val_dataset = SpeckleDataset_New(
            data_cfg['base_path'], mode='val', pos=data_cfg['pos']
        )
        val_loader = DataLoader(
            val_dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )

        model = CompleteModel().to(device)
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        start_time = time.time()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        val_loader_arg = val_loader if cfg.get('validation', {}).get('enabled', True) else None
        train_model(
            model, train_loader, device,
            num_epochs=train_cfg['num_epochs'],
            save_dir=train_cfg['save_dir'],
            num_frames=data_cfg['num_frames'],
            val_loader=val_loader_arg,
        )

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"Peak GPU memory: {peak_mem:.2f} MB")

        elapsed = time.time() - start_time
        print(f"Total time: {elapsed / 60:.2f} min ({elapsed / 3600:.2f} hr)")

        # Write summary log
        log_file = os.path.join(train_cfg['save_dir'], 'result_log.csv')
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"\nConfig: {data_cfg['pos']}")
            log.write(f"\nTotal params: {total_params:,}")
            log.write(f"\nTrainable params: {trainable_params:,}")
            if device.type == "cuda":
                log.write(f"\nPeak GPU memory: {peak_mem:.2f} MB")
            log.write(f"\nTotal time: {elapsed / 60:.2f} min ({elapsed / 3600:.2f} hr)\n")


if __name__ == "__main__":
    main()
