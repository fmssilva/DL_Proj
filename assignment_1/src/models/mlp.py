# MLP builder for Task 1.  Flat pixel vector -> FC stack -> 9 class logits.
# One class replaces the old 7 (VanillaMLP, MLP, NarrowMLP, WiderMLP, …).
# No softmax — CrossEntropyLoss handles that.
#
# Usage in notebook (architecture visible at the call site):
#   MLP(layers=[512, 256, 128], dropout=0.3)                     # was "MLP"
#   MLP(layers=[128, 64], dropout=0.0, use_bn=False)             # was "VanillaMLP"
#   MLP(layers=[512, 512, 512], dropout=0.3, use_residual=True)  # residual MLP

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class _ResidualBlock(nn.Module):
    """One FC residual block: two linear layers with a skip connection. Width stays fixed."""

    def __init__(self, width: int, dropout: float, use_bn: bool):
        super().__init__()
        # two linears per block so there's something non-trivial to skip over
        layers: list[nn.Module] = [nn.Linear(width, width)]
        if use_bn:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, width))
        if use_bn:
            layers.append(nn.BatchNorm1d(width))
        self.block = nn.Sequential(*layers)
        self.act   = nn.ReLU()
        # dropout after the residual add (post-activation)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classic residual: activate AFTER adding the skip, then optionally drop
        return self.drop(self.act(x + self.block(x)))


class MLP(nn.Module):
    """
    Configurable fully-connected classifier.

    Parameters
    ----------
    layers : list[int]
        Hidden-layer widths between the input and the final 9-class head.
        Example: [512, 256, 128] builds  input -> 512 -> 256 -> 128 -> 9.
        For residual mode all widths must be equal (e.g. [512, 512, 512]).
    img_size : int
        Spatial size of the square input image (default 64).
    dropout : float
        Dropout probability after every hidden layer (0.0 = disabled).
    use_bn : bool
        If True each hidden layer gets BatchNorm1d before ReLU.
    in_channels : int
        Number of input channels (3 = RGB, 1 = grayscale).
    use_residual : bool
        If True, each entry in `layers` becomes a residual block instead of a
        plain linear layer. Requires all widths in `layers` to be equal so the
        skip addition needs no projection matrix.
    """

    def __init__(
        self,
        layers: Sequence[int] = (512, 256, 128),
        img_size: int = 64,
        in_channels: int = 3,
        dropout: float = 0.4,
        use_bn: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        input_dim = img_size * img_size * in_channels

        if use_residual:
            # residual blocks require a fixed width — fail loudly if the caller forgets
            if len(set(layers)) > 1:
                raise ValueError(
                    f"use_residual=True requires all layer widths to be equal, got {list(layers)}"
                )
            width = layers[0]
            # project from flat input into the residual width, then stack blocks
            blocks: list[nn.Module] = [nn.Linear(input_dim, width)]
            if use_bn:
                blocks.append(nn.BatchNorm1d(width))
            blocks.append(nn.ReLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            for _ in layers:
                blocks.append(_ResidualBlock(width, dropout, use_bn))
            blocks.append(nn.Linear(width, NUM_CLASSES))
        else:
            # original plain sequential stack — unchanged
            blocks = []
            prev = input_dim
            for width in layers:
                blocks.append(nn.Linear(prev, width))
                if use_bn:
                    blocks.append(nn.BatchNorm1d(width))
                blocks.append(nn.ReLU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
                prev = width
            blocks.append(nn.Linear(prev, NUM_CLASSES))

        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))
