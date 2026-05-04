#!/usr/bin/env python
"""Testing entry point for Dynamic Speckle Imaging Reconstruction.

Usage:
    python scripts/test.py --config configs/test.yaml --checkpoint path/to/model.pth
    python scripts/test.py --config configs/test.yaml --mode experiment
    python scripts/test.py --config configs/test.yaml --mode experiment_withobj
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from src.data.datasets import (
    SpeckleDataset_New,
    SpeckleOnlySequenceDataset,
    SpeckleOnlySequenceDatasetWithObjectAndFlow,
)
from src.engine.evaluator import (
    test_experimental_data,
    test_experimental_data_withobj,
    test_model,
    test_simple,
)
from src.models.complete_model import CompleteModel, SimpleReconstructionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test dynamic speckle imaging reconstruction model."
    )
    parser.add_argument(
        '--config', type=str, default='configs/test.yaml',
        help='Path to test config YAML file.',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint. Overrides config value.',
    )
    parser.add_argument(
        '--mode', type=str, default='test',
        choices=['test', 'test_simple', 'experiment', 'experiment_withobj'],
        help='Testing mode.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg['data']
    chkpt_path = args.checkpoint or cfg['checkpoint']['path']
    output_cfg = cfg['output']

    if args.mode == 'test_simple':
        test_dataset = SpeckleDataset_New(
            data_cfg['base_path'], mode='test', pos=data_cfg['pos']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )

        model = SimpleReconstructionModel().to(device)
        model.load_state_dict(torch.load(chkpt_path))
        print(f"Loaded checkpoint: {chkpt_path}")

        test_simple(model, test_loader, device, save_dir=output_cfg['save_dir'])

    elif args.mode == 'experiment':
        exp_cfg = cfg['experiment']
        dataset = SpeckleOnlySequenceDataset(
            exp_cfg['speckle_data_dir']
        )
        test_loader = DataLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )

        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(chkpt_path))
        print(f"Loaded checkpoint: {chkpt_path}")

        test_experimental_data(
            model, test_loader, device,
            save_dir=output_cfg['save_dir'],
        )

    elif args.mode == 'experiment_withobj':
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

        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(chkpt_path))
        print(f"Loaded checkpoint: {chkpt_path}")

        test_experimental_data_withobj(
            model, test_loader, device,
            save_dir=output_cfg['save_dir'],
            num_frames=data_cfg['num_frames'],
        )

    else:  # default: full test
        test_dataset = SpeckleDataset_New(
            data_cfg['base_path'], mode='test', pos=data_cfg['pos']
        )
        test_loader = DataLoader(
            test_dataset, batch_size=None, shuffle=False,
            num_workers=data_cfg['num_workers'],
        )

        model = CompleteModel().to(device)
        model.load_state_dict(torch.load(chkpt_path))
        print(f"Loaded checkpoint: {chkpt_path}")

        test_model(model, test_loader, device, save_dir=output_cfg['save_dir'])


if __name__ == "__main__":
    main()
