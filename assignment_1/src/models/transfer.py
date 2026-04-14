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
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    resnet18, ResNet18_Weights
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
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
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES)
            )


    def forward(self, x):
        return self.backbone(x)
    

class ResNet18Transfer(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES)
        )


    def forward(self, x):
        return self.backbone(x)



# ── fine-tuning utilities ──────────────────────────────────────────────────────

# head attribute names per backbone (used by both helpers below)
_HEAD_NAMES = {"classifier", "fc", "head"}


def unfreeze_backbone(model: nn.Module, n_layers: int) -> None:
    """Universal unfreeze for ConvNeXt, ResNet, EfficientNet, Swin, VGG.

    n_layers=1  -> unfreeze last real block (+ stem/norm extras on full unfreeze).
    n_layers=-1 -> unfreeze entire backbone (ALL non-head params become trainable).

    The function first re-freezes every backbone param, re-enables the head,
    then selectively unfreezes the requested number of backbone blocks.

    Architecture-specific notes
    ---------------------------
    ConvNeXt : features[0]=stem, features[1..7]=blocks+downsamples. All 8 children covered.
    ResNet   : stem = (conv1, bn1). blocks = [layer1, layer2, layer3, layer4].
               Stem is included in full unfreeze (-1) and excluded from partial.
    Swin     : backbone.norm (LayerNorm) sits between features and head.
               Always unfrozen together with the selected blocks so gradients flow.
    VGG      : features split into 5 conv groups matching the 5 pooling stages.
    EfficientNet: features[0]=stem, features[1..7]=MBConv stages, features[8]=top conv+BN.
               For n_layers=1, unfreeze the last MBConv *stage* (features[-2]),
               NOT the tiny top-conv (features[-1]) which is just a projection layer.
    """
    assert hasattr(model, "backbone"), "model must have .backbone"

    # Step 1: freeze everything, then restore head
    for p in model.backbone.parameters():
        p.requires_grad = False
    for child_name, child in model.backbone.named_children():
        if child_name in _HEAD_NAMES:
            for p in child.parameters():
                p.requires_grad = True

    arch = model.backbone.__class__.__name__

    # ── collect blocks (ordered shallow → deep) ──────────────────────────
    # "stem_extras" are early layers (stem, input-norm) → only on full unfreeze (-1)
    # "bridge_extras" sit between last block and head → always unfrozen with blocks
    blocks = []
    stem_extras = []     # unfrozen only on n_layers == -1
    bridge_extras = []   # unfrozen on any partial or full unfreeze

    if "Swin" in arch:
        # features[0..1]=patch embed + norm, features[2..5]=transformer stages, [6..7]=down+norm
        blocks = list(model.backbone.features.children())
        # backbone.norm is a separate child that MUST be unfrozen for gradient flow
        if hasattr(model.backbone, "norm"):
            bridge_extras = [model.backbone.norm]

    elif "ConvNeXt" in arch:
        # features has 8 children: [0]=stem, [1..7]=stages+downsamples
        blocks = list(model.backbone.features.children())

    elif "VGG" in arch:
        f = model.backbone.features
        blocks = [
            f[0:5],     # conv block 1
            f[5:10],    # conv block 2
            f[10:17],   # conv block 3
            f[17:24],   # conv block 4
            f[24:31],   # conv block 5
        ]

    elif "ResNet" in arch or hasattr(model.backbone, "layer4"):
        # stem = conv1 + bn1 (not inside any layerN)
        for sname in ("conv1", "bn1"):
            if hasattr(model.backbone, sname):
                stem_extras.append(getattr(model.backbone, sname))
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            if hasattr(model.backbone, lname):
                blocks.append(getattr(model.backbone, lname))

    elif "EfficientNet" in arch or hasattr(model.backbone, "features"):
        # features children: [0]=stem Conv2d+BN, [1..N-2]=MBConv stages, [N-1]=top Conv2d+BN
        # The top projection (features[-1]) is a bridge between last MBConv and classifier.
        all_children = list(model.backbone.features.children())
        blocks = all_children[:-1]           # stem + MBConv stages
        bridge_extras = [all_children[-1]]   # top Conv2d+BN: always unfrozen with blocks

    else:
        # generic fallback: every non-head child with parameters
        blocks = [
            child for child_name, child in model.backbone.named_children()
            if child_name not in _HEAD_NAMES
            and sum(p.numel() for p in child.parameters()) > 0
        ]

    # ── select which blocks to unfreeze ──────────────────────────────────
    if n_layers == -1:
        to_unfreeze = blocks + stem_extras + bridge_extras   # everything
    elif n_layers > 0:
        to_unfreeze = blocks[-n_layers:] + bridge_extras
    else:
        to_unfreeze = []

    for block in to_unfreeze:
        for p in block.parameters():
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
            backbone_params.extend(p for p in params if p.requires_grad)

    assert head_params,     "get_backbone_lr_groups: no head params found -- check _HEAD_NAMES"
    assert backbone_params, (
        "get_backbone_lr_groups: no unfrozen backbone params -- call unfreeze_backbone first"
    )
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]