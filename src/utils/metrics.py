"""Metrics formatting, CSV logging, and table display utilities."""

import csv
import os
from typing import Any, Dict, List, Optional, Union

from prettytable import PrettyTable


# ==============================================================================
# CSV Logging Helpers
# ==============================================================================

def append_test_csv(
    csv_path: str,
    batch_idx: int,
    loss_dict: Dict[str, Any],
    speckle_item_losses: List[float],
    warp_item_losses: List[Any],
    digits: int = 6,
) -> None:
    """Append a test batch result row to a CSV file.

    Writes a header on first call based on dict keys.
    """

    def _round_value(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, digits)
        elif isinstance(v, list):
            return [round(float(x), digits) for x in v]
        else:
            try:
                return round(float(v), digits)
            except (TypeError, ValueError):
                return v

    rounded_loss_dict = {k: _round_value(v) for k, v in loss_dict.items()}
    speckle_item_losses = [_round_value(v) for v in speckle_item_losses]
    warp_item_losses = [_round_value(v) for v in warp_item_losses]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["batch"] + list(rounded_loss_dict.keys())
            header += [
                f"speckle_object_item_{i}"
                for i in range(len(speckle_item_losses))
            ]
            header += [f"warp_item_{i}" for i in range(len(warp_item_losses))]
            writer.writerow(header)

        row = [batch_idx] + list(rounded_loss_dict.values())
        row += speckle_item_losses
        row += warp_item_losses
        writer.writerow(row)


def append_warp_csv(
    csv_path: str,
    batch_idx: int,
    warp_dict: Dict[int, Dict[str, List[float]]],
) -> None:
    """Append warp MSE entries to a CSV file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, mse in zip(item['target_t'], item['mse']):
                writer.writerow([batch_idx, ref_t, tgt_t, mse])


def append_warp_csv1(
    csv_path: str,
    batch_idx: int,
    warp_dict: Dict[int, Dict[str, List[float]]],
) -> None:
    """Append warp SSIM entries to a CSV file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, ssim in zip(item['target_t'], item['ssim']):
                writer.writerow([batch_idx, ref_t, tgt_t, ssim])


def append_warp_csv2(
    csv_path: str,
    batch_idx: int,
    warp_dict: Dict[int, Dict[str, List[float]]],
) -> None:
    """Append warp PSNR entries to a CSV file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for ref_t, item in warp_dict.items():
            for tgt_t, psnr_val in zip(item['target_t'], item['psnr']):
                writer.writerow([batch_idx, ref_t, tgt_t, psnr_val])


def write_metric_block(
    f: Any, title: str, values: List[float], prefix: str
) -> None:
    """Write a labeled metric block to an open file handle.

    Args:
        f: file handle opened for writing
        title: block title string
        values: list of float values
        prefix: line prefix, e.g. 'Object' or 'Flow FW'
    """
    f.write(f"{'=' * 60}\n")
    f.write(f"{title}\n")
    f.write(f"{'=' * 60}\n")
    for i, v in enumerate(values):
        f.write(f"{prefix}{i}, {v:.6f}\n")
    f.write("\n")


# ==============================================================================
# Table Formatting Functions
# ==============================================================================

def format_batch_loss_table(
    epoch: int,
    num_epochs: int,
    batch_idx: int,
    total_batches: int,
    batch_losses: Dict[str, Dict[str, float]],
) -> str:
    """Format per-batch loss summary as a PrettyTable string."""
    header = (
        f"\nEpoch [{epoch + 1}/{num_epochs}], "
        f"Batch [{batch_idx + 1}/{total_batches}]\n"
    )
    table = PrettyTable()
    table.field_names = ["Loss Type", "Average Batch", "Each Batch"]
    for key, values in batch_losses.items():
        avg = values.get("average batch", 0.0)
        batch = values.get("each batch", 0.0)
        table.add_row([key, f"{avg:.6f}", f"{batch:.6f}"])
    return header + str(table)


def format_epoch_loss_table(
    epoch: int,
    num_epochs: int,
    epoch_losses: Dict[str, Dict[str, float]],
    best_loss: Optional[float] = None,
) -> str:
    """Format per-epoch loss summary as a PrettyTable string."""
    if best_loss:
        model_info = (
            f"\nEpoch {epoch + 1}/{num_epochs}\n"
            f"New best model saved with loss: {best_loss:.6f}\n"
        )
    else:
        model_info = f"\nEpoch {epoch + 1}/{num_epochs}\n"

    table = PrettyTable()
    table.field_names = ["Loss Type", "Average"]
    for key, values in epoch_losses.items():
        avg = values.get("average", 0.0)
        table.add_row([key, f"{avg:.6f}"])
    return model_info + str(table)


def format_test_loss_table(
    batch_idx: int, num_batches: int, batch_losses: Dict[str, Dict[str, float]]
) -> str:
    """Format per-batch test loss summary as a PrettyTable string."""
    table = PrettyTable()
    table.field_names = ["Loss Type", "Each Batch", "Average So Far"]
    for k, v in batch_losses.items():
        table.add_row([
            k,
            f"{v['each batch']:.6f}",
            f"{v['average batch']:.6f}",
        ])
    table.title = f"Batch {batch_idx + 1}/{num_batches}"
    return str(table)


def format_final_loss_table(
    test_losses: Dict[str, float]
) -> str:
    """Format final test loss summary as a PrettyTable string."""
    table = PrettyTable()
    table.field_names = ["Loss Type", "Average Over Testset"]
    for k, v in test_losses.items():
        table.add_row([k, f"{v:.6f}"])
    table.title = "Final Test Results"
    return str(table)


def batch_loss_summary(
    epoch_losses: Dict[str, float],
    batch_losses: Dict[str, float],
    batch_idx: int,
) -> Dict[str, Dict[str, float]]:
    """Build per-batch summary with running average and current values."""
    return {
        key: {
            "average batch": epoch_losses[key] / (batch_idx + 1),
            "each batch": batch_losses[key],
        }
        for key in epoch_losses.keys()
    }


def epoch_loss_summary(
    epoch_losses: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Build per-epoch summary dict."""
    return {key: {"average": epoch_losses[key]} for key in epoch_losses.keys()}


def avg_loss_summary(
    total_loss: Dict[str, float],
    current_loss: Dict[str, float],
    batch_idx: int,
) -> Dict[str, Dict[str, float]]:
    """Build test batch summary with running average and current values."""
    return {
        key: {
            "each batch": current_loss[key],
            "average batch": total_loss[key] / (batch_idx + 1),
        }
        for key in total_loss.keys()
    }
