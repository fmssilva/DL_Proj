# MLP model for Task 1. Flat pixel vector -> FC stack -> 9 class logits.

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class MLP(nn.Module):
    """
    Flatten -> [input_dim -> 512 -> 256 -> 128 -> 9]
    Each hidden layer: Linear -> BatchNorm1d -> ReLU -> Dropout(0.4)
    No softmax — CrossEntropyLoss handles that.
    """

    def __init__(self, img_size: int = 64):
        super().__init__()

        input_dim = img_size * img_size * 3  # default 64*64*3 = 12288

        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            # layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            # layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # output
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten (B, C, H, W) -> (B, C*H*W) before passing through FC stack
        x = x.view(x.size(0), -1)
        return self.net(x)
