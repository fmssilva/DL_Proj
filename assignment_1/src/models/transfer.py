import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from ..config import NUM_CLASSES

class EfficientNetB0Transfer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

      
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        return self.backbone(x)