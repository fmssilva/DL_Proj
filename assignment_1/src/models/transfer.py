# Transfer learning model wrappers: 5 pretrained backbones, each with a new 9-class head.
# All freeze the backbone by default (Stage 1 feature extraction).
# Stage 2 fine-tuning uses unfreeze_backbone() + get_backbone_lr_groups() below.

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    vgg16, VGG16_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    resnet34, ResNet34_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    googlenet, GoogLeNet_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    alexnet, AlexNet_Weights,
)


from .mlp import MLP

from ..config import NUM_CLASSES


# ── backbone wrappers ──────────────────────────────────────────────────────────

class EfficientNetB0Transfer(nn.Module):
    """EfficientNet-B0: backbone frozen, head = Dropout -> Linear(1280, 9)."""
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str = "BASE"):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        
        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        
        for p in self.backbone.features.parameters():
            p.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        if(head == "BASE"):
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, NUM_CLASSES),
            )
        elif (head == "SIMPLE"):
            self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )

    def forward(self, x):
        return self.backbone(x)


class VGG_16_Transfer(nn.Module):
    """VGG-16: backbone frozen, keeps the original 3-layer head (25088->4096->4096->9).
    This multi-layer head is VGG's original design -- kept to reflect the architecture.
    Note: heavier than other models; may need batch_size=16 on low-VRAM setups."""
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str = "BASE"):
        super().__init__()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            self.backbone.features[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1
            )

        for p in self.backbone.features.parameters():
            p.requires_grad = False

        in_features = self.backbone.classifier[0].in_features

        if(head == "BASE"):
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, NUM_CLASSES),
            )
        elif (head == "SIMPLE"):
            self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )

    def forward(self, x):
        return self.backbone(x)


class Swin_V2_t_Transfer(nn.Module):
    """Swin-V2-t transformer: features frozen, head = Dropout -> Linear(768, 9).
    Bug fix: original code had no Dropout despite accepting the param."""
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str = "BASE"):
        super().__init__()
        self.backbone = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=4
            )

        for p in self.backbone.features.parameters():
            p.requires_grad = False
        
        for p in self.backbone.norm.parameters():
            p.requires_grad = False

        in_features = self.backbone.head.in_features
        if(head=="BASE"):
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, NUM_CLASSES),
            )
        elif (head == "MLP"):
            self.backbone.head = MLP(layers=[512, 256, 128], input_dim=in_features, dropout=dropout, use_bn=False)
        elif (head == "SIMPLE"):
            self.backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )

    def forward(self, x):
        return self.backbone(x)


class ResNet34_Transfer(nn.Module):
    """ResNet-34: all params frozen, then fc replaced with Dropout -> Linear(512, 9).
    Bug fix: original code had no Dropout despite accepting the param."""
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str ="BASE"):
        super().__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        
        for p in self.backbone.parameters():
            p.requires_grad = False

        in_features = self.backbone.fc.in_features
       
        if(head == "BASE"):
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, NUM_CLASSES),
            )
        elif (head == "SIMPLE"):
            self.backbone.fc = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )
        
        

    def forward(self, x):
        return self.backbone(x)


class ConvNext_tiny_Transfer(nn.Module):
    """ConvNeXt-Tiny: features frozen, head = LayerNorm -> Flatten -> Dropout -> Linear(768, 9).
    Bug fix: original used nn.BatchNorm2d(768) on a 1D tensor after GlobalAvgPool -> shape crash."""
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str ="BASE"):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 96, kernel_size=4, stride=4
            )

        for p in self.backbone.features.parameters():
            p.requires_grad = False

        in_features = self.backbone.classifier[2].in_features
        
        if (head =="BASE"):
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),     
                nn.LayerNorm(in_features),  
                nn.Dropout(dropout),
                nn.Linear(in_features, NUM_CLASSES),
            )
        elif (head == "SIMPLE"):
            self.backbone.classifier = nn.Sequential(
            nn.Flatten(start_dim=1), 
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )

    def forward(self, x):
        return self.backbone(x)

class Efficientnet_v2_s_Transfer(nn.Module):
    
    def __init__(self, in_channels: int = 3, dropout: float = 0.4, head:str="BASE"):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )

        for p in self.backbone.features.parameters():
            p.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        if(head == "BASE"):
            self.backbone.classifier = nn.Sequential( 
                nn.Dropout(dropout, inplace=True),
                nn.Linear(in_features, NUM_CLASSES),
            )
        elif (head == "MLP"):
            self.backbone.classifier = MLP(layers=[512, 256, 128], input_dim=in_features, dropout=dropout, use_bn=False)

        elif (head == "SIMPLE"):
            self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
            )


    def forward(self, x):
        return self.backbone(x)



# ── fine-tuning utilities ──────────────────────────────────────────────────────

# head attribute names per backbone (used by both helpers below)
_HEAD_NAMES = {"classifier", "fc", "head"}


def unfreeze_backbone(model: nn.Module, n_layers: int) -> None:
    """Unfreeze the last n_layers top-level children of model.backbone.
    n_layers=0  -> keep everything frozen (same as __init__)
    n_layers=1  -> unfreeze last 1 block (partial fine-tune, Stage 2F)
    n_layers=-1 -> unfreeze all backbone layers (full fine-tune, Stage 2G)
    The head (classifier/fc/head) is always kept trainable regardless.
    Prints trainable/total param counts so the notebook output is self-documenting."""
    assert hasattr(model, "backbone"), "unfreeze_backbone: model must have a .backbone attribute"

    # re-freeze everything first so n_layers=0 is a clean no-op
    for p in model.backbone.parameters():
        p.requires_grad = False

    if "ConvNeXt" in model.backbone.__class__.__name__:
        stages = [
            model.backbone.features[1],  
            model.backbone.features[2],  
            model.backbone.features[3],  
            model.backbone.features[4],  
        ]

        

        if n_layers == -1:
            to_unfreeze = stages
        elif n_layers > 0:
            to_unfreeze = stages[-n_layers:]
        else:
            to_unfreeze = []

        for stage in to_unfreeze:
            for p in stage.parameters():
                p.requires_grad = True
                
        for name, child in model.backbone.named_children():
            if name in _HEAD_NAMES:
                for p in child.parameters():
                    p.requires_grad = True

        _print_param_counts(model, n_layers)
        return

    if n_layers == 0:
        # just make sure the head stays trainable
        for name, child in model.backbone.named_children():
            if name in _HEAD_NAMES:
                for p in child.parameters():
                    p.requires_grad = True
        _print_param_counts(model, n_layers)
        return

    children = list(model.backbone.named_children())
    # skip head and param-free children (e.g. avgpool) when counting "backbone blocks"
    backbone_children = [
        name for name, child in children
        if name not in _HEAD_NAMES and sum(p.numel() for p in child.parameters()) > 0
    ]

    # n_layers=-1 -> all backbone blocks; n_layers>0 -> last n blocks
    to_unfreeze = set(backbone_children if n_layers == -1 else backbone_children[-n_layers:])

    for name, child in children:
        if name in to_unfreeze or name in _HEAD_NAMES:
            for p in child.parameters():
                p.requires_grad = True

    _print_param_counts(model, n_layers)


def _print_param_counts(model: nn.Module, n_layers: int) -> None:
    """Print trainable vs total params after an unfreeze call."""
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"  unfreeze_backbone(n_layers={n_layers}): "
        f"{n_train:,}/{n_total:,} params trainable ({100 * n_train / n_total:.1f}%)"
    )


def get_backbone_lr_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list:
    """Return two param groups for differential-LR optimizers.
    backbone_lr: tiny (e.g. 1e-5) -- nudges pretrained weights without destroying them
    head_lr:     normal (e.g. 1e-3) -- trains the new classification layer at full speed
    Only includes backbone params that are unfrozen (requires_grad=True).
    Usage: optimizer = Adam(get_backbone_lr_groups(model, 1e-5, 1e-3))"""
    assert hasattr(model, "backbone"), "get_backbone_lr_groups: model must have a .backbone attribute"

    backbone_params, head_params = [], []
    for name, child in model.backbone.named_children():
        params = list(child.parameters())
        if name in _HEAD_NAMES:
            head_params.extend(params)
        else:
            # only include params that were unfrozen by unfreeze_backbone
            backbone_params.extend(p for p in params if p.requires_grad)

    assert head_params,     "get_backbone_lr_groups: no head params found -- check _HEAD_NAMES"
    assert backbone_params, (
        "get_backbone_lr_groups: no unfrozen backbone params -- call unfreeze_backbone first"
    )
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]