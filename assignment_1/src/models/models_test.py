# Forward pass tests for all models. Run with: python -m src.models.models_test
# Extended for CNN (Task 2) and Transfer (Task 3) once those files exist.

import torch

from src.config import NUM_CLASSES
from src.models.mlp import MLP, VanillaMLP, VanillaMLP_v2, NarrowMLP, WiderMLP, BottleneckMLP, DeepMLP

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


def test_mlp_forward():
    """RGB 3-channel MLP forward pass."""
    _check_forward(MLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "MLP (RGB)")


def test_vanilla_mlp_forward():
    """VanillaMLP RGB forward pass."""
    _check_forward(VanillaMLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "VanillaMLP (RGB)")


def test_vanilla_mlp_v2_forward():
    """VanillaMLP_v2 RGB forward pass."""
    _check_forward(VanillaMLP_v2(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "VanillaMLP_v2 (RGB)")


def test_narrow_mlp_forward():
    """NarrowMLP RGB forward pass."""
    _check_forward(NarrowMLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "NarrowMLP (RGB)")


def test_bottleneck_mlp_forward():
    """BottleneckMLP RGB forward pass."""
    _check_forward(BottleneckMLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "BottleneckMLP (RGB)")


def test_wider_mlp_forward():
    """WiderMLP RGB forward pass (1024-wide first layer extension variant)."""
    _check_forward(WiderMLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "WiderMLP (RGB)")


def test_deep_mlp_forward():
    """DeepMLP RGB forward pass (4-layer funnel: 512->256->128->64)."""
    _check_forward(DeepMLP(img_size=_IMG_SIZE), torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE), "DeepMLP (RGB)")


def test_gray_mlp_forward():
    """All models must also accept grayscale (1-channel) input when in_channels=1."""
    gray_input = torch.randn(4, 1, _IMG_SIZE, _IMG_SIZE)
    _check_forward(MLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "MLP (gray)")
    _check_forward(VanillaMLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "VanillaMLP (gray)")
    _check_forward(VanillaMLP_v2(img_size=_IMG_SIZE, in_channels=1), gray_input, "VanillaMLP_v2 (gray)")
    _check_forward(NarrowMLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "NarrowMLP (gray)")
    _check_forward(WiderMLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "WiderMLP (gray)")
    _check_forward(BottleneckMLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "BottleneckMLP (gray)")
    _check_forward(DeepMLP(img_size=_IMG_SIZE, in_channels=1), gray_input, "DeepMLP (gray)")


if __name__ == "__main__":
    print("=" * 60)
    print("models_test.py — running all tests")
    print("=" * 60)
    test_mlp_forward()
    test_vanilla_mlp_forward()
    test_vanilla_mlp_v2_forward()
    test_narrow_mlp_forward()
    test_bottleneck_mlp_forward()
    test_wider_mlp_forward()
    test_deep_mlp_forward()
    test_gray_mlp_forward()
    print("=" * 60)
    print("All model tests passed.")
    print("=" * 60)
