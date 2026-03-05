# MLP models for Task 1. Flat pixel vector -> FC stack -> 9 class logits.
# All classes accept in_channels (default 3 for RGB, set to 1 for grayscale).
# No softmax — CrossEntropyLoss handles that.

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class MLP(nn.Module):
    """
    Standard 3-layer funnel: input -> 512 -> 256 -> 128 -> 9.
    Each hidden layer: Linear -> BatchNorm1d -> ReLU -> Dropout(p).
    in_channels=1 for grayscale input (input_dim = img_size*img_size).
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4, in_channels: int = 3):
        super().__init__()
        input_dim = img_size * img_size * in_channels

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten (B, C, H, W) -> (B, C*H*W) before FC stack
        return self.net(x.view(x.size(0), -1))


class VanillaMLP(nn.Module):
    """
    Tiny 2-layer baseline: input -> 128 -> 64 -> 9. No BN, no Dropout.
    Winning architecture from the first Colab run — simpler often beats
    over-regularised models on small datasets.
    in_channels=1 for grayscale input.
    """

    def __init__(self, img_size: int = 64, in_channels: int = 3):
        super().__init__()
        input_dim = img_size * img_size * in_channels

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class VanillaMLP_v2(nn.Module):
    """
    Slightly wider vanilla baseline: input -> 256 -> 128 -> 9. No BN, no Dropout.
    One more unit in first layer vs VanillaMLP — tests whether 256 vs 128 capacity
    is worth the extra params (still ~3x fewer than MLP).
    in_channels=1 for grayscale input.
    """

    def __init__(self, img_size: int = 64, in_channels: int = 3):
        super().__init__()
        input_dim = img_size * img_size * in_channels

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class NarrowMLP(nn.Module):
    """
    Deeper and narrower: input -> 256 -> 128 -> 64 -> 32 -> 9, with BN+Dropout.
    Hypothesis: fewer params per layer forces compact representations.
    Result: worst in full run (0.1343) — never converged in 30 epochs.
    in_channels=1 for grayscale input.
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4, in_channels: int = 3):
        super().__init__()
        input_dim = img_size * img_size * in_channels

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
    Expand-then-compress: input -> 512 -> 1024 -> 256 -> 128 -> 9, with BN+Dropout.
    Wide middle layer captures more feature combinations before compressing.
    2nd best in full run (0.2072) — close to VanillaMLP winner (0.2104).
    in_channels=1 for grayscale input.
    """

    def __init__(self, img_size: int = 64, dropout: float = 0.4, in_channels: int = 3):
        super().__init__()
        input_dim = img_size * img_size * in_channels

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
