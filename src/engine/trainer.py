"""Training and fine-tuning loop functions."""

import os
import random
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.losses.combined_loss import CombinedLoss, SimpleLoss
from src.utils.metrics import (
    batch_loss_summary,
    epoch_loss_summary,
    format_batch_loss_table,
    format_epoch_loss_table,
)
from src.utils.visualization import (
    plot_flow_magnitude,
    plot_losses,
    plot_val_epe,
    validate_metrics,
    visualize_debug_images,
)


# ==============================================================================
# Training Helpers (from Mainloss_manage.py)
# ==============================================================================

def _process_batch_losses(
    object_losses: list,
    epoch_losses: Dict[str, float],
) -> Dict[str, float]:
    """Accumulate per-object losses into batch and epoch totals."""
    batch_losses: Dict[str, float] = defaultdict(float)
    for obj_loss in object_losses:
        for key, value in obj_loss.items():
            epoch_losses[key] += value
            batch_losses[key] += value
    return batch_losses


def _update_epoch_losses(
    epoch_losses: Dict[str, float],
    train_loader_len: int,
) -> Dict[str, float]:
    """Normalize epoch losses by the number of batches."""
    for key in epoch_losses.keys():
        epoch_losses[key] /= train_loader_len
    return epoch_losses


def _save_best_model(
    model: nn.Module,
    epoch_losses: Dict[str, float],
    best_loss: float,
    save_dir: str,
    epoch: int,
) -> float:
    """Save model checkpoint if current loss is best, plus periodic saves."""
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if epoch_losses['total_loss'] < best_loss:
        best_loss = epoch_losses['total_loss']
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, 'best_model.pth'),
        )
    else:
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, 'other_best_model.pth'),
        )

    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), ckpt_path)

    return best_loss


def _log_epoch_results(
    epoch: int,
    num_epochs: int,
    epoch_losses: Dict[str, float],
    best_loss: float,
    save_dir: str,
) -> None:
    """Log epoch results to console and CSV file."""
    epoch_losses_summary = epoch_loss_summary(epoch_losses)
    loss_table = format_epoch_loss_table(
        epoch, num_epochs, epoch_losses_summary, best_loss
    )
    print(loss_table)

    log_file = os.path.join(save_dir, 'result_log.csv')
    if epoch == 0:
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write("")

    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(loss_table)


# ==============================================================================
# Training Loop Functions
# ==============================================================================

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = 'data/results',
    num_frames: int = 5,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
) -> None:
    """Train CompleteModel with RAFT + U-Net.

    Args:
        model: CompleteModel instance
        train_loader: training data loader
        device: torch device
        num_epochs: number of training epochs
        save_dir: output directory for checkpoints and logs
        num_frames: number of frames per sequence
        val_loader: optional validation data loader
    """
    optimizer = optim.Adam([
        {'params': model.optical_flow_model.parameters(), 'lr': 1e-4},
        {'params': model.object_reconstructor.parameters(), 'lr': 1e-4},
    ])
    criterion = CombinedLoss(mode='train')

    train_losses: Dict[str, list] = defaultdict(list)
    best_loss = float('inf')
    val_flow_mag_history: Dict[str, list] = {'fw': [], 'bw': []}
    val_epe_history: Dict[str, list] = {'fw': [], 'bw': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: Dict[str, float] = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            object_losses = []

            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)
            flow_seq = batch['flow_seq'].to(device)
            t_random = random.randint(0, num_frames - 1)

            optimizer.zero_grad()

            output = model(speckle_seq)
            target = {
                'flow': flow_seq.to(device),
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }

            loss, loss_dict = criterion(output, target, epoch, t_random)
            object_losses.append(loss_dict)

            visualize_debug_images(
                output, target, epoch, batch_idx, save_dir,
                t_random, step=15, save_every=250,
            )

            loss.backward()
            optimizer.step()

            batch_losses = _process_batch_losses(object_losses, epoch_losses)
            batch_losses_summary = batch_loss_summary(
                epoch_losses, batch_losses, batch_idx
            )
            print(format_batch_loss_table(
                epoch, num_epochs, batch_idx, len(train_loader),
                batch_losses_summary,
            ))

        # Normalize and log epoch results
        normalized_epoch_losses = _update_epoch_losses(
            epoch_losses, len(train_loader)
        )
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])

        best_loss = _save_best_model(
            model, normalized_epoch_losses, best_loss, save_dir, epoch
        )
        _log_epoch_results(
            epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir
        )
        plot_losses(train_losses, save_dir)

        # Validation metrics
        if val_loader is not None:
            fw_mag, bw_mag, fw_epe, bw_epe = validate_metrics(
                model, val_loader, device
            )
            val_flow_mag_history['fw'].append(fw_mag)
            val_flow_mag_history['bw'].append(bw_mag)
            val_epe_history['fw'].append(fw_epe)
            val_epe_history['bw'].append(bw_epe)
            print(
                f"[Epoch {epoch + 1}/{num_epochs}] "
                f"Magnitude FW: {fw_mag:.4f}  BW: {bw_mag:.4f} | "
                f"EPE FW: {fw_epe:.4f}  BW: {bw_epe:.4f}"
            )
            plot_flow_magnitude(val_flow_mag_history, save_dir)
            plot_val_epe(val_epe_history, save_dir)


def train_simple(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = 'data/results',
) -> None:
    """Train SimpleReconstructionModel (U-Net only, no flow).

    Args:
        model: SimpleReconstructionModel instance
        train_loader: training data loader
        device: torch device
        num_epochs: number of training epochs
        save_dir: output directory for checkpoints and logs
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SimpleLoss()

    train_losses: Dict[str, list] = defaultdict(list)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: Dict[str, float] = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            object_losses = []
            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)

            optimizer.zero_grad()
            outputs = model(speckle_seq)
            target = {
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }
            loss, loss_dict = criterion(outputs, target)
            object_losses.append(loss_dict)

            loss.backward()
            optimizer.step()

            batch_losses = _process_batch_losses(object_losses, epoch_losses)
            batch_losses_summary = batch_loss_summary(
                epoch_losses, batch_losses, batch_idx
            )
            print(format_batch_loss_table(
                epoch, num_epochs, batch_idx, len(train_loader),
                batch_losses_summary,
            ))

        normalized_epoch_losses = _update_epoch_losses(
            epoch_losses, len(train_loader)
        )
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])

        best_loss = _save_best_model(
            model, normalized_epoch_losses, best_loss, save_dir, epoch
        )
        _log_epoch_results(
            epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir
        )
        plot_losses(train_losses, save_dir)


def finetune_model(
    model: nn.Module,
    finetune_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    save_dir: str = 'data/finetune_results',
    num_frames: int = 5,
) -> None:
    """Fine-tune a pretrained CompleteModel on experimental data.

    Uses lower learning rates for stable adaptation.

    Args:
        model: CompleteModel instance with pretrained weights
        finetune_loader: fine-tuning data loader
        device: torch device
        num_epochs: number of fine-tuning epochs
        save_dir: output directory for checkpoints and logs
        num_frames: number of frames per sequence
    """
    optimizer = optim.Adam([
        {'params': model.optical_flow_model.parameters(), 'lr': 1e-6},
        {'params': model.object_reconstructor.parameters(), 'lr': 1e-5},
    ])
    criterion = CombinedLoss(mode='train')

    train_losses: Dict[str, list] = defaultdict(list)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: Dict[str, float] = defaultdict(float)

        for batch_idx, batch in enumerate(finetune_loader):
            object_losses = []

            speckle_seq = batch['speckle_seq'].to(device)
            object_seq = batch['object_seq'].to(device)
            t_random = random.randint(0, num_frames - 1)

            optimizer.zero_grad()

            output = model(speckle_seq)
            target = {
                'object': object_seq.to(device),
                'speckle': speckle_seq.to(device),
            }

            loss, loss_dict = criterion(output, target, epoch, t_random)
            object_losses.append(loss_dict)

            visualize_debug_images(
                output, target, epoch, batch_idx, save_dir,
                t_random, step=15, save_every=500,
            )

            loss.backward()
            optimizer.step()

            batch_losses = _process_batch_losses(object_losses, epoch_losses)
            batch_losses_summary = batch_loss_summary(
                epoch_losses, batch_losses, batch_idx
            )
            print(format_batch_loss_table(
                epoch, num_epochs, batch_idx, len(finetune_loader),
                batch_losses_summary,
            ))

        normalized_epoch_losses = _update_epoch_losses(
            epoch_losses, len(finetune_loader)
        )
        for key in normalized_epoch_losses.keys():
            train_losses[key].append(normalized_epoch_losses[key])

        best_loss = _save_best_model(
            model, normalized_epoch_losses, best_loss, save_dir, epoch
        )
        _log_epoch_results(
            epoch, num_epochs, normalized_epoch_losses, best_loss, save_dir
        )
        plot_losses(train_losses, save_dir)
