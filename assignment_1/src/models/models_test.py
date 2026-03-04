# Forward pass tests for all models. Run with: python -m src.models.models_test
# Extended for CNN (Task 2) and Transfer (Task 3) once those files exist.

import torch

from src.config import NUM_CLASSES
from src.models.mlp import MLP

# use same defaults as the notebook to keep tests meaningful
_IMG_SIZE = 64


def test_mlp_forward():
    """Dummy batch through MLP: output shape must be (B, 9), no NaNs."""
    model = MLP(img_size=_IMG_SIZE).eval()
    # (B, C, H, W) — forward() flattens it internally
    x = torch.randn(4, 3, _IMG_SIZE, _IMG_SIZE)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, NUM_CLASSES), \
        f"Expected (4, {NUM_CLASSES}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN values in MLP output"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] MLP forward: output shape={out.shape}, params={n_params:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("models_test.py — running all tests")
    print("=" * 60)
    test_mlp_forward()
    print("=" * 60)
    print("All model tests passed.")
    print("=" * 60)
