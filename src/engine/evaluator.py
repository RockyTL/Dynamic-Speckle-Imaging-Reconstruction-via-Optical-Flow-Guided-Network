"""Model evaluation and testing functions."""

import os
from collections import defaultdict
from typing import Any, Dict

import torch

from src.losses.combined_loss import CombinedLoss
from src.utils.io_utils import (
    build_output_dirs,
    clear_csv_files,
    remove_existing_test_logs,
    rotate_output,
)
from src.utils.metrics import (
    append_test_csv,
    append_warp_csv,
    append_warp_csv1,
    append_warp_csv2,
    avg_loss_summary,
    format_final_loss_table,
    format_test_loss_table,
    write_metric_block,
)
from src.utils.visualization import save_all_results, save_experimental_results


# ==============================================================================
# Testing Helpers (from Mainloss_manage.py)
# ==============================================================================

def _process_test_frame_losses(
    test_losses: Dict[str, float],
    loss_dict: Dict[str, Any],
) -> Dict[str, float]:
    """Accumulate scalar losses from a test batch, skipping list/dict values."""
    for key, val in loss_dict.items():
        if isinstance(val, (float, int)):
            test_losses[key] += val
    return test_losses


def _process_test_results(
    test_losses: Dict[str, float],
    test_loader_len: int,
) -> Dict[str, float]:
    """Normalize test losses by the number of batches."""
    for key in test_losses.keys():
        test_losses[key] /= test_loader_len
    return test_losses


def _log_test_results(
    test_losses: Dict[str, float],
    save_dir: str,
) -> None:
    """Log final test results to console and CSV file."""
    final_loss_table = format_final_loss_table(test_losses)
    print(final_loss_table)

    log_file = os.path.join(save_dir, 'result_log.csv')
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + final_loss_table)


def _write_metrics_summary(
    save_dir: str,
    criterion: CombinedLoss,
) -> None:
    """Write per-object MSE and per-flow EPE summary to a text file.

    Reads accumulated metrics from the CombinedLoss criterion (test mode).
    """
    save_path = os.path.join(save_dir, "test_metrics_summary.txt")
    with open(save_path, "w") as f:
        if criterion.object_mse_sum is not None:
            mean_object_mse = (
                criterion.object_mse_sum / criterion.object_mse_count
            ).tolist()
            print("=== Mean per-object MSE over ALL test batches ===")
            for i, v in enumerate(mean_object_mse):
                print(f"Object {i}: {v:.6f}")
            write_metric_block(
                f,
                title="Mean per-object MSE over ALL test batches",
                values=mean_object_mse,
                prefix="Object",
            )

        if criterion.flow_fw_epe_sum is not None:
            mean_flow_fw_epe = (
                criterion.flow_fw_epe_sum
                / torch.clamp(criterion.flow_fw_epe_count, min=1)
            ).tolist()
            print("=== Mean per-flow FORWARD EPE over ALL test batches ===")
            for i, v in enumerate(mean_flow_fw_epe):
                print(f"Flow FW {i}: {v:.6f}")
            write_metric_block(
                f,
                title="Mean per-flow FORWARD EPE over ALL test batches",
                values=mean_flow_fw_epe,
                prefix="Flow FW",
            )

        if criterion.flow_bw_epe_sum is not None:
            mean_flow_bw_epe = (
                criterion.flow_bw_epe_sum
                / torch.clamp(criterion.flow_bw_epe_count, min=1)
            ).tolist()
            print("=== Mean per-flow BACKWARD EPE over ALL test batches ===")
            for i, v in enumerate(mean_flow_bw_epe):
                print(f"Flow BW {i}: {v:.6f}")
            write_metric_block(
                f,
                title="Mean per-flow BACKWARD EPE over ALL test batches",
                values=mean_flow_bw_epe,
                prefix="Flow BW",
            )


# ==============================================================================
# Testing Functions
# ==============================================================================

def test_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = 'data/results',
) -> None:
    """Evaluate CompleteModel on synthetic test data with full metrics.

    Saves flow visualizations, reconstructed objects, overlays, and
    writes per-batch CSV logs and a summary metrics file.

    Args:
        model: CompleteModel instance
        test_loader: test data loader
        device: torch device
        save_dir: output directory for results
    """
    directories = build_output_dirs(save_dir, mode='test')

    csv_files = [
        os.path.join(save_dir, "test_batch_losses.csv"),
        os.path.join(save_dir, "test_batch_losses_warp.csv"),
        os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
        os.path.join(save_dir, "test_batch_losses_warp_psnr.csv"),
    ]
    clear_csv_files(csv_files)

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses: Dict[str, float] = defaultdict(float)
    remove_existing_test_logs(save_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)
            flow_seq = batch['flow_seq'].to(device)

            target = {
                'flow': flow_seq,
                'object': object_seq,
                'speckle': speckle_seq,
            }

            output = model(speckle_seq, speckle_for_unet=None, test=True)

            # Save visualizations
            save_all_results(batch_idx, output, target, directories)

            # Compute loss
            _, loss_dict = criterion(output, target)
            test_losses = _process_test_frame_losses(test_losses, loss_dict)

            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

            # Write CSV logs
            append_test_csv(
                os.path.join(save_dir, "test_batch_losses.csv"),
                batch_idx,
                loss_dict,
                loss_dict.get("speckle_object_items", []),
                loss_dict.get("object_warp_items", []),
            )
            append_warp_csv(
                os.path.join(save_dir, "test_batch_losses_warp.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )
            append_warp_csv1(
                os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )
            append_warp_csv2(
                os.path.join(save_dir, "test_batch_losses_warp_psnr.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )

    normalized_test_losses = _process_test_results(test_losses, len(test_loader))
    _log_test_results(normalized_test_losses, save_dir)
    _write_metrics_summary(save_dir, criterion)


def test_simple(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = 'data/results',
) -> None:
    """Evaluate SimpleReconstructionModel (U-Net only) on test data.

    Args:
        model: SimpleReconstructionModel instance
        test_loader: test data loader
        device: torch device
        save_dir: output directory for results
    """
    directories = {
        'origin_object1': os.path.join(save_dir, 'origin_object1'),
        'origin_object2': os.path.join(save_dir, 'origin_object2'),
        'recon_object2': os.path.join(save_dir, 'recon_object2'),
        'recon_object1': os.path.join(save_dir, 'recon_object1'),
        'gifs_object1': os.path.join(save_dir, 'gifs_object1'),
        'gifs_object2': os.path.join(save_dir, 'gifs_object2'),
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses: Dict[str, float] = defaultdict(float)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)

            output = model(speckle_seq)
            target = {
                'object': object_seq,
                'speckle': speckle_seq,
            }

            save_all_results(batch_idx, output, target, directories)
            _, loss_dict = criterion(output, target)
            test_losses = _process_test_frame_losses(test_losses, loss_dict)

            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

    normalized_test_losses = _process_test_results(test_losses, len(test_loader))
    _log_test_results(normalized_test_losses, save_dir)


def test_experimental_data(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = 'data/experimental_results',
) -> None:
    """Evaluate CompleteModel on experimental speckle data (no GT).

    Args:
        model: CompleteModel instance
        data_loader: experimental data loader
        device: torch device
        save_dir: output directory for results
    """
    directories = build_output_dirs(save_dir, mode='experiment')

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            speckle_seq = batch['speckle_seq'].to(device)
            speckle_raw_seq = batch['speckle_raw_seq'].to(device)

            output = model(speckle_raw_seq, speckle_seq)

            save_experimental_results(batch_idx, output, speckle_seq, directories)
            print(f"Processed batch {batch_idx + 1}/{len(data_loader)}")


def test_experimental_data_withobj(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = 'data/experiment_results_withobjandflow',
    num_frames: int = 5,
) -> None:
    """Evaluate CompleteModel on experimental data with object and flow GT.

    Saves the same comprehensive outputs as test_model, with 180-degree
    rotation applied to model outputs for experimental data alignment.

    Args:
        model: CompleteModel instance
        test_loader: experimental data loader with object/flow GT
        device: torch device
        save_dir: output directory for results
        num_frames: number of frames per sequence
    """
    directories = build_output_dirs(save_dir, mode='experiment_withobj')

    model.eval()
    criterion = CombinedLoss(mode='test')
    test_losses: Dict[str, float] = defaultdict(float)
    remove_existing_test_logs(save_dir)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            speckle_seq = batch['speckle_seq'].to(device)
            speckle_raw_seq = batch['speckle_raw_seq'].to(device)
            object_seq = batch['object_seq'].to(device)
            flow_seq = batch['flow_seq'].to(device)

            output = model(speckle_raw_seq, speckle_seq)
            output = rotate_output(output)

            target = {
                'flow': flow_seq,
                'object': object_seq,
                'speckle': speckle_seq,
            }

            save_all_results(
                batch_idx, output, target, directories,
                use_speckle=True, rotate_input=True,
            )

            _, loss_dict = criterion(output, target)
            test_losses = _process_test_frame_losses(test_losses, loss_dict)

            batch_losses = avg_loss_summary(test_losses, loss_dict, batch_idx)
            print(format_test_loss_table(batch_idx, len(test_loader), batch_losses))

            # Write CSV logs
            append_test_csv(
                os.path.join(save_dir, "test_batch_losses.csv"),
                batch_idx,
                loss_dict,
                loss_dict.get("speckle_object_items", []),
                loss_dict.get("object_warp_items", []),
            )
            append_warp_csv(
                os.path.join(save_dir, "test_batch_losses_warp.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )
            append_warp_csv1(
                os.path.join(save_dir, "test_batch_losses_warp_ssim.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )
            append_warp_csv2(
                os.path.join(save_dir, "test_batch_losses_warp_psnr.csv"),
                batch_idx,
                loss_dict['object_warp_from_each_t'],
            )

    normalized_test_losses = _process_test_results(test_losses, len(test_loader))
    _log_test_results(normalized_test_losses, save_dir)
    _write_metrics_summary(save_dir, criterion)
