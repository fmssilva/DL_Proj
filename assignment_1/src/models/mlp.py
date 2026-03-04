# MLP model for Task 1. Flat pixel vector -> FC stack -> 9 class logits.

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class MLP(nn.Module):
    """
    Flatten -> [input_dim -> 512 -> 256 -> 128 -> 9]
    Each hidden layer: Linear -> BatchNorm1d -> ReLU -> Dropout(p)
    No softmax — CrossEntropyLoss handles that.
    dropout: tune this per experiment (0.3 = less regularisation, 0.5 = more)
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4):
        super().__init__()

        input_dim = img_size * img_size * 3  # default 64*64*3 = 12288

        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # output
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten (B, C, H, W) -> (B, C*H*W) before passing through FC stack
        x = x.view(x.size(0), -1)
        return self.net(x)


class NarrowMLP(nn.Module):
    """
    Deeper and narrower than MLP: 4 hidden layers, max width 256.
    Hypothesis: fewer params per layer forces more compact representations,
    which might reduce overfitting on the small dataset (2880 train images).
    Architecture: input -> 256 -> 128 -> 64 -> 32 -> 9
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4):
        super().__init__()

        input_dim = img_size * img_size * 3

        def _block(in_f: int, out_f: int) -> list:
            return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(
            *_block(input_dim, 256),
            *_block(256, 128),
            *_block(128, 64),
            *_block(64, 32),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class BottleneckMLP(nn.Module):
    """
    Expand-then-compress (bottleneck) shape: 512 -> 1024 -> 256 -> 128 -> 9.
    The wide middle layer captures more feature combinations before compressing.
    Hypothesis: the expansion step might catch cross-pixel patterns that the
    standard funnel misses — though still no spatial awareness.
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4):
        super().__init__()

        input_dim = img_size * img_size * 3

        def _block(in_f: int, out_f: int) -> list:
            return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(
            *_block(input_dim, 512),
            *_block(512, 1024),   # expand
            *_block(1024, 256),   # compress
            *_block(256, 128),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))
