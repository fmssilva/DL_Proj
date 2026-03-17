# CNN architectures for Task 2: BaseCNN, DeepCNN, WideCNN, ResidualCNN, MultiScaleCNN.
# All use GlobalAvgPool head (not Flatten) to minimise params and overfitting.
# Input: (B, 3, 64, 64). Output: (B, NUM_CLASSES) raw logits — no softmax.

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


# ── building blocks ───────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, use_bn: bool = True) -> nn.Sequential:
    """Conv(3x3, pad=1) -> [BN] -> ReLU -> MaxPool(2). Halves spatial dims."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
    return nn.Sequential(*layers)


def _gap_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """GlobalAvgPool -> Dropout -> Linear. Minimal FC head after conv blocks."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) -> (B, C, 1, 1)
        nn.Flatten(),              # (B, C)
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BaseCNN — 3-block baseline
# ══════════════════════════════════════════════════════════════════════════════

class BaseCNN(nn.Module):
    """
    3 conv blocks + GlobalAvgPool head. Clean baseline for all comparisons.
    use_bn=False gives the NoBN ablation variant (experiment L).

    (B,3,64,64) -> Conv32 -> Pool -> Conv64 -> Pool -> Conv128 -> Pool
                -> GAP -> [128] -> Dropout -> Linear -> (B,9)
    ~120 K params (with BN).
    """
    def __init__(self, dropout: float = 0.5, use_bn: bool = True):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, use_bn=use_bn),     # 64 -> 32
            _conv_block(32, 64, use_bn=use_bn),     # 32 -> 16
            _conv_block(64, 128, use_bn=use_bn),    # 16 -> 8
        )
        self.head = _gap_head(128, NUM_CLASSES, dropout)

    def forward(self, x):
        return self.head(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# DeepCNN — 4 blocks, more depth
# ══════════════════════════════════════════════════════════════════════════════

class DeepCNN(nn.Module):
    """
    4 conv blocks — tests maximum useful depth on 64px input.
    After 4 pools: 64 -> 32 -> 16 -> 8 -> 4. GAP on 4x4 is still fine.

    (B,3,64,64) -> Conv32 -> Conv64 -> Conv128 -> Conv256
                -> GAP -> [256] -> Dropout -> Linear -> (B,9)
    ~480 K params.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),       # 64 -> 32
            _conv_block(32, 64),      # 32 -> 16
            _conv_block(64, 128),     # 16 -> 8
            _conv_block(128, 256),    # 8 -> 4
        )
        self.head = _gap_head(256, NUM_CLASSES, dropout)

    def forward(self, x):
        return self.head(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# WideCNN — 3 blocks, wider channels (2x BaseCNN)
# ══════════════════════════════════════════════════════════════════════════════

class WideCNN(nn.Module):
    """
    3 conv blocks with 2x the channels of BaseCNN. Same depth, richer features.

    (B,3,64,64) -> Conv64 -> Conv128 -> Conv256
                -> GAP -> [256] -> Dropout -> Linear -> (B,9)
    ~420 K params.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64),       # 64 -> 32
            _conv_block(64, 128),     # 32 -> 16
            _conv_block(128, 256),    # 16 -> 8
        )
        self.head = _gap_head(256, NUM_CLASSES, dropout)

    def forward(self, x):
        return self.head(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# SEBlock — channel attention (Squeeze-and-Excitation)
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    squeeze: GAP -> [C]
    excite:  FC(C -> C//r) -> ReLU -> FC(C//r -> C) -> Sigmoid
    scale:   element-wise multiply on original feature map
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # scale shape: (B, C) -> (B, C, 1, 1) for broadcasting
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ══════════════════════════════════════════════════════════════════════════════
# ResidualCNN — 3 residual blocks + optional SE attention
# ══════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    """
    Single residual block:
      input -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> (+skip) -> ReLU -> [SE] -> MaxPool
    1x1 projection on skip only when in_ch != out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)

        # 1x1 projection only when channels change
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        out = self.se(out)
        return self.pool(out)


class ResidualCNN(nn.Module):
    """
    3 residual blocks + GlobalAvgPool head. Optional SE channel attention.

    (B,3,64,64) -> ResBlock(3->64) -> ResBlock(64->128) -> ResBlock(128->256)
                -> GAP -> [256] -> Dropout -> Linear -> (B,9)

    use_se=False: plain residual (~780 K params)
    use_se=True:  residual + SE attention (~790 K params)
    """
    def __init__(self, dropout: float = 0.5, use_se: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            _ResBlock(3, 64, use_se=use_se),       # 64 -> 32
            _ResBlock(64, 128, use_se=use_se),      # 32 -> 16
            _ResBlock(128, 256, use_se=use_se),     # 16 -> 8
        )
        self.head = _gap_head(256, NUM_CLASSES, dropout)

    def forward(self, x):
        return self.head(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# MultiScaleCNN — parallel k=3 and k=5 branches per block
# ══════════════════════════════════════════════════════════════════════════════

class _MultiScaleBlock(nn.Module):
    """Parallel Conv(k=3,p=1) and Conv(k=5,p=2) branches, concatenated, then BN+ReLU+Pool."""
    def __init__(self, in_ch: int, out_ch_per_branch: int):
        super().__init__()
        self.branch3 = nn.Conv2d(in_ch, out_ch_per_branch, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_ch, out_ch_per_branch, kernel_size=5, padding=2)
        self.bn   = nn.BatchNorm2d(out_ch_per_branch * 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        return self.pool(self.relu(self.bn(out)))


class MultiScaleCNN(nn.Module):
    """
    3 multi-scale blocks + GlobalAvgPool head. Captures both fine (k=3) and
    coarse (k=5) texture patterns at each spatial resolution.

    Block 1: [Conv3(3->16) || Conv5(3->16)] -> cat=32 -> BN -> ReLU -> Pool  # 64->32
    Block 2: [Conv3(32->32)|| Conv5(32->32)]-> cat=64 -> BN -> ReLU -> Pool  # 32->16
    Block 3: [Conv3(64->64)|| Conv5(64->64)]-> cat=128-> BN -> ReLU -> Pool  # 16->8
    GAP -> [128] -> Dropout -> Linear -> (B,9)
    ~160 K params.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _MultiScaleBlock(3, 16),     # out: 32 channels, 32x32
            _MultiScaleBlock(32, 32),    # out: 64 channels, 16x16
            _MultiScaleBlock(64, 64),    # out: 128 channels, 8x8
        )
        self.head = _gap_head(128, NUM_CLASSES, dropout)

    def forward(self, x):
        return self.head(self.features(x))
