"""File I/O utilities: flow saving, CSV management, directory creation, and backups."""

import os
import re
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
import torch


# ==============================================================================
# Directory Utilities
# ==============================================================================

def build_output_dirs(
    save_dir: str, mode: str = 'test'
) -> Dict[str, str]:
    """Create and return a dict of output subdirectory paths.

    Args:
        save_dir: root output directory
        mode: 'test' (synthetic data) or 'experiment' (real data)

    Returns:
        dict mapping directory keys to absolute paths
    """
    if mode == 'test':
        directories = {
            'flow_arrow_fw': os.path.join(save_dir, 'flowdata/flow_arrow_fw'),
            'flow_colorimage_fw': os.path.join(save_dir, 'flowdata/flow_colorimage_fw'),
            'flow_arrow_bw': os.path.join(save_dir, 'flowdata/flow_arrow_bw'),
            'flow_colorimage_bw': os.path.join(save_dir, 'flowdata/flow_colorimage_bw'),
            'gt_flow_arrow_fw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_fw'),
            'gt_flow_colorimage': os.path.join(save_dir, 'flowdata/gt_flow_colorimage'),
            'gt_flow_arrow_bw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_bw'),
            'gt_flow_colorimage_bw': os.path.join(save_dir, 'flowdata/gt_flow_colorimage_bw'),
            'object_flow_arrow_fw': os.path.join(save_dir, 'flowdata/object_flow_arrow_fw'),
            'object_flow_colorimage_fw': os.path.join(save_dir, 'flowdata/object_flow_colorimage_fw'),
            'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
            'groundtruth_fw': os.path.join(save_dir, 'flowdata/groundtruth_fw'),
            'origin_object': os.path.join(save_dir, 'origin_object'),
            'recon_object': os.path.join(save_dir, 'recon_object'),
            'diff_recon_vs_gt': os.path.join(save_dir, 'diff_recon_vs_gt'),
            'overlay_results_origin_object': os.path.join(save_dir, 'overlay_results_origin_object'),
            'overlay_results_nl_origin_object': os.path.join(save_dir, 'overlay_results_nl_origin_object'),
            'overlay_results_recon_object': os.path.join(save_dir, 'overlay_results_recon_object'),
            'overlay_results_nl_recon_object': os.path.join(save_dir, 'overlay_results_nl_recon_object'),
            'diff_results_recon_obj_origin_obj': os.path.join(save_dir, 'diff_results_recon_obj_origin_obj'),
            'diff_results_warp_origin_obj': os.path.join(save_dir, 'diff_results_warp_origin_obj'),
            'warp_from_each_t_nl_overlay': os.path.join('data/results/newcode', 'warp_from_each_t_nl_overlay'),
            'warp_from_each_t_overlay': os.path.join('data/results/newcode', 'warp_from_each_t_overlay'),
            'overlay_gt_each_t': os.path.join('data/results/newcode', 'overlay_gt_each_t'),
            'diff_overlay_each_t': os.path.join('data/results/newcode', 'diff_overlay_each_t'),
            'single_overlay_gt': os.path.join('data/results/newcode', 'single_overlay_gt'),
            'single_overlay_unet': os.path.join('data/results/newcode', 'single_overlay_unet'),
            'single_overlay_warp': os.path.join('data/results/newcode', 'single_overlay_warp'),
            'single_overlay_nl_warp': os.path.join('data/results/newcode', 'single_overlay_nl_warp'),
            'single_overlay_nl_nc_warp': os.path.join('data/results/newcode', 'single_overlay_nl_nc_warp'),
            'single_overlay_nl_gt': os.path.join('data/results/newcode', 'single_overlay_nl_gt'),
            'single_overlay_nl_unet': os.path.join('data/results/newcode', 'single_overlay_nl_unet'),
            'single_overlay_nl_warp_all_t': os.path.join('data/results/newcode', 'single_overlay_nl_warp_all_t'),
            'diff_single_each_t': os.path.join('data/results/newcode', 'diff_single_each_t'),
            'flow_fw_diff_each': os.path.join('data/results/newcode', 'flow_fw_diff_each'),
            'flow_bw_diff_each': os.path.join('data/results/newcode', 'flow_bw_diff_each'),
        }
    elif mode == 'experiment':
        directories = {
            'speckle1': os.path.join(save_dir, 'speckle1'),
            'speckle2': os.path.join(save_dir, 'speckle2'),
            'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
            'flow_colorimage': os.path.join(save_dir, 'flowdata/flow_colorimage'),
            'flow_arrow': os.path.join(save_dir, 'flowdata/flow_arrow'),
            'reconstructed_object1': os.path.join(save_dir, 'reconstructed_object1'),
            'reconstructed_object2': os.path.join(save_dir, 'reconstructed_object2'),
        }
    elif mode == 'experiment_withobj':
        directories = {
            'speckle1': os.path.join(save_dir, 'speckle1'),
            'speckle2': os.path.join(save_dir, 'speckle2'),
            'flow_arrow_fw': os.path.join(save_dir, 'flowdata/flow_arrow_fw'),
            'flow_colorimage_fw': os.path.join(save_dir, 'flowdata/flow_colorimage_fw'),
            'flow_arrow_bw': os.path.join(save_dir, 'flowdata/flow_arrow_bw'),
            'flow_colorimage_bw': os.path.join(save_dir, 'flowdata/flow_colorimage_bw'),
            'gt_flow_arrow_fw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_fw'),
            'gt_flow_colorimage': os.path.join(save_dir, 'flowdata/gt_flow_colorimage'),
            'gt_flow_arrow_bw': os.path.join(save_dir, 'flowdata/gt_flow_arrow_bw'),
            'gt_flow_colorimage_bw': os.path.join(save_dir, 'flowdata/gt_flow_colorimage_bw'),
            'object_flow_arrow_fw': os.path.join(save_dir, 'flowdata/object_flow_arrow_fw'),
            'object_flow_colorimage_fw': os.path.join(save_dir, 'flowdata/object_flow_colorimage_fw'),
            'model_fw': os.path.join(save_dir, 'flowdata/model_fw'),
            'groundtruth_fw': os.path.join(save_dir, 'flowdata/groundtruth_fw'),
            'origin_object': os.path.join(save_dir, 'origin_object'),
            'recon_object': os.path.join(save_dir, 'recon_object'),
            'diff_recon_vs_gt': os.path.join(save_dir, 'diff_recon_vs_gt'),
            'overlay_results_origin_object': os.path.join(save_dir, 'overlay_results_origin_object'),
            'overlay_results_nl_origin_object': os.path.join(save_dir, 'overlay_results_nl_origin_object'),
            'overlay_results_recon_object': os.path.join(save_dir, 'overlay_results_recon_object'),
            'overlay_results_nl_recon_object': os.path.join(save_dir, 'overlay_results_nl_recon_object'),
            'diff_results_recon_obj_origin_obj': os.path.join(save_dir, 'diff_results_recon_obj_origin_obj'),
            'diff_results_warp_origin_obj': os.path.join(save_dir, 'diff_results_warp_origin_obj'),
            'warp_from_each_t_nl_overlay': os.path.join(save_dir, 'newcode', 'warp_from_each_t_nl_overlay'),
            'warp_from_each_t_overlay': os.path.join(save_dir, 'newcode', 'warp_from_each_t_overlay'),
            'overlay_gt_each_t': os.path.join(save_dir, 'newcode', 'overlay_gt_each_t'),
            'diff_overlay_each_t': os.path.join(save_dir, 'newcode', 'diff_overlay_each_t'),
            'single_overlay_gt': os.path.join(save_dir, 'newcode', 'single_overlay_gt'),
            'single_overlay_unet': os.path.join(save_dir, 'newcode', 'single_overlay_unet'),
            'single_overlay_warp': os.path.join(save_dir, 'newcode', 'single_overlay_warp'),
            'single_overlay_nl_warp': os.path.join(save_dir, 'newcode', 'single_overlay_nl_warp'),
            'single_overlay_nl_nc_warp': os.path.join(save_dir, 'newcode', 'single_overlay_nl_nc_warp'),
            'single_overlay_nl_gt': os.path.join(save_dir, 'newcode', 'single_overlay_nl_gt'),
            'single_overlay_nl_unet': os.path.join(save_dir, 'newcode', 'single_overlay_nl_unet'),
            'diff_single_each_t': os.path.join(save_dir, 'newcode', 'diff_single_each_t'),
            'flow_fw_diff_each': os.path.join(save_dir, 'newcode', 'flow_fw_diff_each'),
            'flow_bw_diff_each': os.path.join(save_dir, 'newcode', 'flow_bw_diff_each'),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories


def scan_checkpoints(
    checkpoint_dir: str,
    start_epoch: int = 1,
    end_epoch: int = 999,
) -> List[Tuple[int, str]]:
    """Scan a directory for checkpoint files matching 'checkpoint_epoch_N.pth'.

    Args:
        checkpoint_dir: path to directory containing .pth files
        start_epoch: minimum epoch number to include
        end_epoch: maximum epoch number to include

    Returns:
        list of (epoch_number, filename) tuples, sorted by epoch
    """
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    all_files = os.listdir(checkpoint_dir)
    epoch_files: List[Tuple[int, str]] = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            ep = int(m.group(1))
            if start_epoch <= ep <= end_epoch:
                epoch_files.append((ep, f))
    epoch_files.sort()
    return epoch_files


# ==============================================================================
# CSV Management
# ==============================================================================

def clear_csv_files(file_paths: List[str]) -> None:
    """Truncate specified CSV files (create empty files).

    Args:
        file_paths: list of CSV file paths to clear
    """
    for file_path in file_paths:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            pass


def remove_existing_test_logs(save_dir: str) -> None:
    """Remove existing test log files from the output directory."""
    files_to_remove = [
        "test_metrics_summary.txt",
        "test_batch_losses.csv",
        "test_batch_losses_warp.csv",
    ]
    for fname in files_to_remove:
        fpath = os.path.join(save_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


# ==============================================================================
# Flow I/O
# ==============================================================================

def save_flow_to_csv(
    flow_data: np.ndarray,
    save_dir: str,
    batch_idx: Optional[int] = None,
    t: Optional[int] = None,
) -> None:
    """Save flow data as CSV files (one per channel).

    Args:
        flow_data: [1, num_channels, H, W]
        save_dir: output directory
        batch_idx: batch index for filename
        t: frame index for filename
    """
    _, num_channels, height, width = flow_data.shape
    for c in range(num_channels):
        filename = f'flow_img{batch_idx}_frame{t}_channel{c + 1}.csv'
        channel_data = flow_data[0, c]
        df = pd.DataFrame(channel_data)
        df.to_csv(os.path.join(save_dir, filename), index=False, header=False)


def save_flow_to_hdf5(
    flow_data: np.ndarray,
    save_dir: str,
    batch_idx: int,
    t: int,
) -> None:
    """Save flow data to HDF5 file with per-frame, per-channel datasets.

    Args:
        flow_data: [1, num_channels, H, W]
        save_dir: output directory
        batch_idx: batch index
        t: frame index
    """
    _, num_channels, height, width = flow_data.shape
    hdf5_file = os.path.join(save_dir, f'object_{batch_idx}_flow_data.h5')

    with h5py.File(hdf5_file, 'a') as hdf5:
        for c in range(num_channels):
            dataset_name = f'frame{t}_channel{c + 1}'
            if dataset_name in hdf5:
                del hdf5[dataset_name]
            hdf5.create_dataset(
                dataset_name, data=flow_data[0, c], compression="gzip"
            )


def convert_hdf5_to_excel(hdf5_dir: str, excel_dir: str) -> None:
    """Batch convert HDF5 flow files to Excel format.

    Args:
        hdf5_dir: directory containing .h5 files
        excel_dir: output directory for .xlsx files
    """
    os.makedirs(excel_dir, exist_ok=True)

    for file_name in os.listdir(hdf5_dir):
        if file_name.endswith('.h5'):
            hdf5_file = os.path.join(hdf5_dir, file_name)
            excel_file = os.path.join(
                excel_dir, file_name.replace('.h5', '.xlsx')
            )

            with h5py.File(hdf5_file, 'r') as hdf5, \
                    pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for dataset_name in hdf5.keys():
                    data = hdf5[dataset_name][:]
                    df = pd.DataFrame(data)
                    df.to_excel(
                        writer, sheet_name=dataset_name,
                        index=False, header=False,
                    )


# ==============================================================================
# Data Transforms
# ==============================================================================

def rotate_output(
    output: Dict[str, Optional[torch.Tensor]]
) -> Dict[str, Optional[torch.Tensor]]:
    """Rotate model output by 180 degrees for experimental data alignment.

    Applies 180-degree rotation to reconstructed_object, flow_forward,
    and flow_backward tensors, inverting flow direction components.

    Args:
        output: model output dict

    Returns:
        new dict with rotated tensors
    """
    new_output = dict(output)

    if (
        'reconstructed_object' in new_output
        and new_output['reconstructed_object'] is not None
    ):
        obj = new_output['reconstructed_object']
        obj = torch.rot90(obj, k=2, dims=[-2, -1])
        new_output['reconstructed_object'] = obj

    if 'flow_forward' in new_output and new_output['flow_forward'] is not None:
        flow = new_output['flow_forward']
        flow = torch.rot90(flow, k=2, dims=[-2, -1])
        flow[:, 0] *= -1
        flow[:, 1] *= -1
        new_output['flow_forward'] = flow

    if 'flow_backward' in new_output and new_output['flow_backward'] is not None:
        flow = new_output['flow_backward']
        flow = torch.rot90(flow, k=2, dims=[-2, -1])
        flow[:, 0] *= -1
        flow[:, 1] *= -1
        new_output['flow_backward'] = flow

    return new_output


# ==============================================================================
# Backup
# ==============================================================================

def create_backup_zip(
    base_path: str, specific_code_files: List[str], pos: str
) -> None:
    """Create a backup zip archive of results and code files.

    Args:
        base_path: project root path
        specific_code_files: list of source file names to include
        pos: datasets position identifier (used in zip filename)
    """
    data_path = os.path.join(base_path, 'data')
    output_filename = os.path.join(data_path, f'{pos}.zip')
    temp_dir = os.path.join(base_path, 'temp_backup')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Copy results folder
        results_src = os.path.join(base_path, 'data', 'results')
        results_dest = os.path.join(temp_dir, 'results')
        if os.path.exists(results_src):
            shutil.copytree(results_src, results_dest)

        # Copy code files
        code_dest = os.path.join(temp_dir, 'code')
        os.makedirs(code_dest, exist_ok=True)

        raft_src = os.path.join(base_path, 'RAFT')
        raft_dest = os.path.join(code_dest, 'RAFT')
        if os.path.exists(raft_src):
            shutil.copytree(raft_src, raft_dest)

        utils_src = os.path.join(base_path, 'utils')
        utils_dest = os.path.join(code_dest, 'utils')
        if os.path.exists(utils_src):
            shutil.copytree(utils_src, utils_dest)

        for filename in specific_code_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                shutil.copy2(file_path, os.path.join(code_dest, filename))

        # Create zip
        with ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname=arcname)

        print(f"Backup zip {output_filename} created successfully!")
    except Exception as e:
        print(f"Error creating backup: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
