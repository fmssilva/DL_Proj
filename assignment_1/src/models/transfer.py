import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights,vgg16, VGG16_Weights, swin_v2_t, Swin_V2_T_Weights, resnet34, ResNet34_Weights,convnext_tiny,ConvNeXt_Tiny_Weights

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
                kernel_size=(3, 3), 
                stride=(2, 2), 
                padding=(1, 1), 
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
    
class VGG_16_Transfer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1
        self.backbone = vgg16(weights=weights)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3,3), 
                stride=(1, 1), 
                padding=(1, 1)
            )

      
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(in_features=4096, out_feature=NUM_CLASSES, bias=True)
  )

    def forward(self, x):
        return self.backbone(x)
    
class Swin_V2_t_Transfer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        weights = Swin_V2_T_Weights.IMAGENET1K_V1
        self.backbone = swin_v2_t(weights=weights)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 
                96, 
                kernel_size=(4, 4), 
                stride=(4, 4)
            )

      
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features=in_features, 
                                       out_features=NUM_CLASSES, 
                                       bias=True)

    def forward(self, x):
        return self.backbone(x)
    
class ResNet34_Transfer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        weights = ResNet34_Weights.IMAGENET1K_V1
        self.backbone = resnet34(weights=weights)

        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                64, 
                kernel_size=(7, 7), 
                stride=(2, 2), 
                padding=(3, 3), 
                bias=False
            )

      
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, 
                                     out_features=NUM_CLASSES, 
                                     bias=True)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
    
class ConvNext_tiny_Transfer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 
                96, 
                kernel_size=(4, 4), 
                stride=(4, 4)
            )

      
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.LayerNorm2d((in_features,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=in_features, out_features=NUM_CLASSES, bias=True)
            )

    def forward(self, x):
        return self.backbone(x)