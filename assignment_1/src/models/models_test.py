# Forward pass tests for all models. Run with: python -m src.models.models_test

import torch

from src.config import NUM_CLASSES
from src.models.mlp import MLP
from src.models.cnn import BaseCNN, DeepCNN, WideCNN, ResidualCNN, MultiScaleCNN

# use same defaults as the notebook to keep tests meaningful
_IMG_SIZE = 64


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
    _check_forward(BaseCNN(),                        x, "BaseCNN")
    _check_forward(BaseCNN(use_bn=False),            x, "BaseCNN (NoBN)")
    _check_forward(DeepCNN(),                        x, "DeepCNN")
    _check_forward(WideCNN(),                        x, "WideCNN")
    _check_forward(ResidualCNN(use_se=False),        x, "ResidualCNN")
    _check_forward(ResidualCNN(use_se=True),         x, "ResidualCNN+SE")
    _check_forward(MultiScaleCNN(),                  x, "MultiScaleCNN")


def test_cnn_dropout_override():
    """Verify dropout param is respected — different values should work without errors."""
    x = torch.randn(2, 3, _IMG_SIZE, _IMG_SIZE)
    for drop in [0.0, 0.3, 0.5]:
        _check_forward(BaseCNN(dropout=drop), x, f"BaseCNN(drop={drop})")
        _check_forward(ResidualCNN(dropout=drop), x, f"ResidualCNN(drop={drop})")


if __name__ == "__main__":
    print("=" * 60)
    print("models_test.py — running all tests")
    print("=" * 60)
    test_mlp_builder_variants()
    test_mlp_grayscale()
    test_mlp_use_bn_flag()
    test_mlp_dropout_zero()
    test_mlp_residual()
    print("-" * 60)
    test_cnn_forward()
    test_cnn_dropout_override()
    print("=" * 60)
    print("All model tests passed.")
    print("=" * 60)
