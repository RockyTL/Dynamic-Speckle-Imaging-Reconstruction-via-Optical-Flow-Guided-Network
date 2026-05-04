"""Complete model combining RAFT optical flow and U-Net reconstruction."""

from argparse import Namespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from RAFT.raft import RAFT
from src.losses.combined_loss import forward_backward_consistency_check, warp
from src.models.unet import UNet


class CompleteModel(nn.Module):
    """Complete model: RAFT optical flow + U-Net object reconstruction.

    Pipeline:
        1. RAFT estimates forward/backward flow between speckle frames
        2. Forward-backward consistency check computes occlusion masks
        3. U-Net reconstructs object sequence from speckle frames
        4. Speckle frames are warped using predicted flow for consistency loss
    """

    def __init__(self) -> None:
        super().__init__()

        args = Namespace(
            small=True,
            dropout=0,
            alternate_corr=False,
            mixed_precision=False,
            corr_levels=4,
            corr_radius=4,
        )

        self.optical_flow_model: RAFT = RAFT(args)
        self.object_reconstructor: UNet = UNet()

    def forward(
        self,
        speckle: torch.Tensor,
        speckle_for_unet: Optional[torch.Tensor] = None,
        test: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
            speckle: [T, 1, H, W] speckle sequence for flow estimation
            speckle_for_unet: optional separate input for U-Net
            test: if True, also compute object flow (currently disabled)

        Returns:
            dict with keys:
                flow_forward, flow_backward, reconstructed_object,
                warped_speckle1, warped_speckle2, fwd_occ, bwd_occ
        """
        if speckle_for_unet is None:
            speckle_for_unet = speckle

        # Predict object displacement (forward and backward flow)
        object_displacement = self.optical_flow_model(speckle[:-1], speckle[1:])
        object_displacement_bw = self.optical_flow_model(speckle[1:], speckle[:-1])

        # Compute forward and backward occlusion masks
        fwd_occ, bwd_occ = forward_backward_consistency_check(
            object_displacement, object_displacement_bw
        )

        # Reconstruct object from speckle images
        reconstructed_object = self.object_reconstructor(speckle_for_unet)

        # Warp speckle frames for consistency loss
        warped_speckle1 = warp(speckle[1:], object_displacement)
        warped_speckle2 = warp(speckle[:-1], object_displacement_bw)

        return {
            'flow_forward': object_displacement,
            'flow_backward': object_displacement_bw,
            'reconstructed_object': reconstructed_object,
            'warped_speckle1': warped_speckle1,
            'warped_speckle2': warped_speckle2,
            'fwd_occ': fwd_occ,
            'bwd_occ': bwd_occ,
        }


class SimpleReconstructionModel(nn.Module):
    """U-Net-only reconstruction model for ablation studies.

    Directly reconstructs object sequence from speckle without optical flow.
    """

    def __init__(self) -> None:
        super().__init__()
        self.object_reconstructor: UNet = UNet()

    def forward(
        self, speckle_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            speckle_seq: [T, C, H, W] input speckle sequence

        Returns:
            dict with key 'reconstructed_object': [T, C, H, W]
        """
        reconstructed_object_seq = self.object_reconstructor(speckle_seq)
        return {'reconstructed_object': reconstructed_object_seq}
