# MLP builder for Task 1.  Flat pixel vector -> FC stack -> 9 class logits.
# One class replaces the old 7 (VanillaMLP, MLP, NarrowMLP, WiderMLP, …).
# No softmax — CrossEntropyLoss handles that.
#
# Usage in notebook (architecture visible at the call site):
#   MLP(layers=[512, 256, 128], dropout=0.3)          # was "MLP"
#   MLP(layers=[128, 64], dropout=0.0, use_bn=False)  # was "VanillaMLP"

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class MLP(nn.Module):
    """
    Configurable fully-connected classifier.

    Parameters
    ----------
    layers : list[int]
        Hidden-layer widths between the input and the final 9-class head.
        Example: [512, 256, 128] builds  input → 512 → 256 → 128 → 9.
    img_size : int
        Spatial size of the square input image (default 64).
    dropout : float
        Dropout probability after every hidden layer (0.0 = disabled).
    use_bn : bool
        If True each hidden layer gets BatchNorm1d before ReLU.
    in_channels : int
        Number of input channels (3 = RGB, 1 = grayscale).
    """

    def __init__(
        self,
        layers: Sequence[int] = (512, 256, 128),
        img_size: int = 64,
        in_channels: int = 3,
        dropout: float = 0.4,
        use_bn: bool = True,
    ):
        super().__init__()
        input_dim = img_size * img_size * in_channels

        blocks: list[nn.Module] = []
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
