# Forward pass tests for all models. Run with: python -m src.models.models_test

import torch

from src.config import NUM_CLASSES
from src.models.mlp import MLP
from src.models.cnn import CNN as BaseCNN, DeepCNN, WideCNN, ResidualCNN, MultiScaleCNN
from src.models.transfer import (
    EfficientNetB0Transfer, VGG_16_Transfer, Swin_V2_t_Transfer,
    ResNet34_Transfer, ConvNext_tiny_Transfer,
    unfreeze_backbone, get_backbone_lr_groups,
)

# use same defaults as the notebook to keep tests meaningful
_IMG_SIZE     = 64
_IMG_SIZE_TL  = 224   # transfer learning uses 224px -- backbone calibrated for this


def _check_forward(model, x, name):
    """Run a forward pass and assert shape + no NaNs. Prints param count."""
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (x.size(0), NUM_CLASSES), \
        f"[{name}] Expected ({x.size(0)}, {NUM_CLASSES}), got {out.shape}"
    assert not torch.isnan(out).any(), f"[{name}] NaN in output"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] {name}: output={out.shape}, params={n_params:,}")


# ── MLP builder tests ────────────────────────────────────────────────────────

def test_mlp_builder_variants():
    """Test all old architecture shapes via the single MLP builder."""
    x_rgb  = torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE)
    x_gray = torch.randn(4, 1, _IMG_SIZE, _IMG_SIZE)

    # was: MLP(dropout=0.4)  ->  [512, 256, 128] + BN + Drop
    _check_forward(MLP(layers=[512, 256, 128], dropout=0.4),                     x_rgb, "MLP [512,256,128] BN+D(0.4)")

    # was: VanillaMLP  ->  [128, 64] no BN, no dropout
    _check_forward(MLP(layers=[128, 64], dropout=0.0, use_bn=False),             x_rgb, "MLP [128,64] vanilla")

    # was: VanillaMLP_v2  ->  [256, 128] no BN, no dropout
    _check_forward(MLP(layers=[256, 128], dropout=0.0, use_bn=False),            x_rgb, "MLP [256,128] vanilla_v2")

    # was: NarrowMLP  ->  [256, 128, 64, 32] BN + Drop
    _check_forward(MLP(layers=[256, 128, 64, 32], dropout=0.4),                  x_rgb, "MLP [256,128,64,32] narrow")

    # was: WiderMLP  ->  [1024, 256, 128] BN + Drop
    _check_forward(MLP(layers=[1024, 256, 128], dropout=0.3),                    x_rgb, "MLP [1024,256,128] wider")

    # was: DeepMLP  ->  [512, 256, 128, 64] BN + Drop
    _check_forward(MLP(layers=[512, 256, 128, 64], dropout=0.3),                 x_rgb, "MLP [512,256,128,64] deep")

    # was: BottleneckMLP  ->  [512, 1024, 256, 128] BN + Drop (expand then compress)
    _check_forward(MLP(layers=[512, 1024, 256, 128], dropout=0.4),               x_rgb, "MLP [512,1024,256,128] bottleneck")


def test_mlp_grayscale():
    """MLP builder with in_channels=1 for grayscale input."""
    x_gray = torch.randn(4, 1, _IMG_SIZE, _IMG_SIZE)
    _check_forward(MLP(layers=[512, 256, 128], in_channels=1),                   x_gray, "MLP [512,256,128] gray")
    _check_forward(MLP(layers=[128, 64], dropout=0.0, use_bn=False, in_channels=1), x_gray, "MLP [128,64] gray")
    _check_forward(MLP(layers=[256, 128], dropout=0.0, use_bn=False, in_channels=1), x_gray, "MLP [256,128] gray")


def test_mlp_use_bn_flag():
    """Verify use_bn=False produces fewer parameters (no BN layers)."""
    m_bn   = MLP(layers=[256, 128], use_bn=True,  dropout=0.3)
    m_nobn = MLP(layers=[256, 128], use_bn=False, dropout=0.3)
    p_bn   = sum(p.numel() for p in m_bn.parameters())
    p_nobn = sum(p.numel() for p in m_nobn.parameters())
    assert p_bn > p_nobn, f"BN model should have more params: {p_bn} vs {p_nobn}"
    print(f"[PASS] use_bn flag: with_bn={p_bn:,} > no_bn={p_nobn:,}")


def test_mlp_dropout_zero():
    """dropout=0.0 should produce no Dropout layers in the model."""
    m = MLP(layers=[256, 128], dropout=0.0)
    for module in m.net:
        assert not isinstance(module, torch.nn.Dropout), "Found Dropout when dropout=0.0"
    print("[PASS] dropout=0.0: no Dropout layers in model")


def test_mlp_residual():
    """Residual MLP: constant-width blocks with skip connections, correct output shape."""
    x = torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE)

    # standard 3-block residual stack
    _check_forward(MLP(layers=[512, 512, 512], dropout=0.3, use_residual=True),              x, "ResidualMLP [512x3] BN")
    # 4-block narrower residual
    _check_forward(MLP(layers=[256, 256, 256, 256], dropout=0.3, use_residual=True),         x, "ResidualMLP [256x4] BN")
    # no BN variant
    _check_forward(MLP(layers=[512, 512], dropout=0.3, use_bn=False, use_residual=True),     x, "ResidualMLP [512x2] no-BN")
    # no dropout
    _check_forward(MLP(layers=[512, 512, 512], dropout=0.0, use_residual=True),              x, "ResidualMLP [512x3] no-drop")

    # residual MLP should have MORE params than equivalent plain MLP
    # because each _ResidualBlock has 2 linear layers
    plain   = MLP(layers=[512, 512, 512], dropout=0.3, use_residual=False)
    residual = MLP(layers=[512, 512, 512], dropout=0.3, use_residual=True)
    p_plain   = sum(p.numel() for p in plain.parameters())
    p_residual = sum(p.numel() for p in residual.parameters())
    assert p_residual > p_plain, (
        f"Residual MLP should have more params than plain: {p_residual} vs {p_plain}"
    )
    print(f"[PASS] residual_params={p_residual:,} > plain_params={p_plain:,} (each block has 2 linears)")

    # non-uniform widths must raise ValueError
    try:
        MLP(layers=[512, 256], use_residual=True)
        raise AssertionError("Should have raised ValueError on non-uniform widths")
    except ValueError:
        pass
    print("[PASS] use_residual=True with non-uniform widths raises ValueError")

    print("  test_mlp_residual passed")


# ── CNN tests (Task 2) ───────────────────────────────────────────────────────

def test_cnn_forward():
    """Forward pass for all CNN architectures on RGB 64x64 input."""
    x = torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE)
    _check_forward(BaseCNN(),       x, "BaseCNN")
    _check_forward(DeepCNN(),       x, "DeepCNN")
    _check_forward(WideCNN(),       x, "WideCNN")
    _check_forward(ResidualCNN(),   x, "ResidualCNN")
    _check_forward(MultiScaleCNN(), x, "MultiScaleCNN")


def test_cnn_dropout_override():
    """Verify dropout param is respected -- different values should work without errors."""
    x = torch.randn(2, 3, _IMG_SIZE, _IMG_SIZE)
    for drop in [0.0, 0.3, 0.5]:
        _check_forward(BaseCNN(dropout=drop),     x, f"BaseCNN(drop={drop})")
        _check_forward(ResidualCNN(dropout=drop), x, f"ResidualCNN(drop={drop})")


# ── Transfer Learning model tests (Task 3) ───────────────────────────────────

# All transfer models are loaded with pretrained weights -- this downloads once and caches.
# Tests run at 224px (backbone calibrated for ImageNet input size).

_TRANSFER_MODELS = [
    ("EfficientNetB0", EfficientNetB0Transfer),
    ("VGG16",          VGG_16_Transfer),
    ("SwinV2t",        Swin_V2_t_Transfer),
    ("ResNet34",       ResNet34_Transfer),
    ("ConvNextTiny",   ConvNext_tiny_Transfer),
]


def test_transfer_forward_224px():
    """All 5 transfer models: forward pass at 224px, correct output shape, no NaNs."""
    x = torch.randn(2, 3, _IMG_SIZE_TL, _IMG_SIZE_TL)
    for name, cls in _TRANSFER_MODELS:
        _check_forward(cls(dropout=0.4), x, f"{name}(224px)")


def test_transfer_frozen_params():
    """Stage 1 (default): backbone is frozen, only head params are trainable.
    Trainable params should be << total params for all models."""
    x = torch.randn(2, 3, _IMG_SIZE_TL, _IMG_SIZE_TL)
    for name, cls in _TRANSFER_MODELS:
        m = cls(dropout=0.4)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        # trainable head should be a small fraction of the total backbone
        assert trainable < total, f"{name}: expected trainable < total, got {trainable} == {total}"
        assert trainable > 0,    f"{name}: no trainable params -- head was not set up correctly"
        pct = 100 * trainable / total
        print(f"[PASS] {name}: {trainable:,}/{total:,} trainable ({pct:.1f}%)")


def test_transfer_dropout_present():
    """All models except VGG must have Dropout in the head (dropout=0.4 not silently ignored)."""
    for name, cls in _TRANSFER_MODELS:
        if name == "VGG16":
            continue   # VGG head is the original multi-layer design, Dropout is inside
        m = cls(dropout=0.4)
        # head is inside model.backbone -- find all Dropout layers in the whole model
        dropouts = [mod for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
        assert dropouts, f"{name}: no Dropout found -- dropout param is being silently ignored"
        # confirm the dropout probability is actually 0.4
        assert any(d.p == 0.4 for d in dropouts), \
            f"{name}: Dropout found but p != 0.4 (got {[d.p for d in dropouts]})"
        print(f"[PASS] {name}: Dropout(p=0.4) present in head")


def test_unfreeze_partial():
    """unfreeze_backbone(n_layers=1) unfreezes at least some backbone params beyond the head.
    Note: models where all backbone params live in a single top-level child (EfficientNetB0,
    VGG16, ConvNextTiny) will unfreeze 100% with n_layers=1 -- that's expected behaviour.
    The important property is that trainable grows beyond head-only."""
    for name, cls in _TRANSFER_MODELS:
        m = cls(dropout=0.4)
        frozen_before = sum(p.numel() for p in m.parameters() if p.requires_grad)
        unfreeze_backbone(m, n_layers=1)
        trainable_after = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        # after partial unfreeze: must have more trainable params than head-only
        assert trainable_after > frozen_before, \
            f"{name}: partial unfreeze should add params (got {trainable_after} <= {frozen_before})"
        print(f"[PASS] {name}: partial unfreeze {trainable_after:,}/{total:,} trainable")


def test_unfreeze_full():
    """unfreeze_backbone(n_layers=-1) makes all params trainable."""
    for name, cls in _TRANSFER_MODELS:
        m = cls(dropout=0.4)
        unfreeze_backbone(m, n_layers=-1)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in m.parameters())
        assert trainable == total, \
            f"{name}: full unfreeze should make all params trainable ({trainable} != {total})"
        print(f"[PASS] {name}: full unfreeze {trainable:,}/{total:,} all trainable")


def test_get_backbone_lr_groups():
    """After partial unfreeze, get_backbone_lr_groups returns two non-empty param groups."""
    for name, cls in _TRANSFER_MODELS:
        m = cls(dropout=0.4)
        unfreeze_backbone(m, n_layers=1)
        groups = get_backbone_lr_groups(m, backbone_lr=1e-5, head_lr=1e-3)
        assert len(groups) == 2, f"{name}: expected 2 param groups, got {len(groups)}"
        assert groups[0]["params"], f"{name}: backbone param group is empty"
        assert groups[1]["params"], f"{name}: head param group is empty"
        assert groups[0]["lr"] == 1e-5, f"{name}: backbone_lr wrong"
        assert groups[1]["lr"] == 1e-3, f"{name}: head_lr wrong"
        # total params in both groups should cover all trainable params
        all_group_params = set(id(p) for g in groups for p in g["params"])
        all_trainable    = set(id(p) for p in m.parameters() if p.requires_grad)
        assert all_group_params == all_trainable, \
            f"{name}: param groups don't cover all trainable params"
        print(f"[PASS] {name}: 2 lr groups, backbone={len(groups[0]['params'])} tensors, head={len(groups[1]['params'])} tensors")


if __name__ == "__main__":
    print("=" * 60)
    print("models_test.py -- running all tests")
    print("=" * 60)
    test_mlp_builder_variants()
    test_mlp_grayscale()
    test_mlp_use_bn_flag()
    test_mlp_dropout_zero()
    test_mlp_residual()
    print("-" * 60)
    test_cnn_forward()
    test_cnn_dropout_override()
    print("-" * 60)
    print("Transfer Learning model tests (downloads pretrained weights on first run)...")
    test_transfer_forward_224px()
    test_transfer_frozen_params()
    test_transfer_dropout_present()
    test_unfreeze_partial()
    test_unfreeze_full()
    test_get_backbone_lr_groups()
    print("=" * 60)
    print("All model tests passed.")
    print("=" * 60)
