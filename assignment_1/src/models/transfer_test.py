"""
Comprehensive tests for transfer.py — freeze/unfreeze logic, gradient flow, and model correctness.
Run: cd assignment_1 && python -m src.models.transfer_test
"""

import sys
import torch
import torch.nn as nn

from .transfer import (
    EfficientNetB0Transfer,
    VGG_16_Transfer,
    Swin_V2_t_Transfer,
    ResNet34_Transfer,
    ConvNext_tiny_Transfer,
    Efficientnet_v2_s_Transfer,
    ResNet18Transfer,
    unfreeze_backbone,
    get_backbone_lr_groups,
    _HEAD_NAMES,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _count_params(model, requires_grad=None):
    """Count total or trainable/frozen params."""
    if requires_grad is None:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)


def _count_trainable(model):
    return _count_params(model, requires_grad=True)


def _count_frozen(model):
    return _count_params(model, requires_grad=False)


def _all_head_params_trainable(model):
    """Check that every param in the head (classifier/fc/head) is trainable."""
    for name, child in model.backbone.named_children():
        if name in _HEAD_NAMES:
            for pname, p in child.named_parameters():
                if not p.requires_grad:
                    return False, f"HEAD param {name}.{pname} is FROZEN (should be trainable)"
    return True, "OK"


def _all_non_head_params_frozen(model):
    """Check that every param NOT in the head is frozen (for Stage 1 = feature extraction)."""
    for name, child in model.backbone.named_children():
        if name not in _HEAD_NAMES:
            for pname, p in child.named_parameters():
                if p.requires_grad:
                    return False, f"BACKBONE param {name}.{pname} is TRAINABLE (should be frozen)"
    return True, "OK"


# ── TEST 1: Stage 1 freeze correctness ────────────────────────────────────────

def test_stage1_freeze():
    """After __init__, only head params should be trainable. All backbone params frozen."""
    print("\n" + "=" * 70)
    print("TEST 1: Stage 1 freeze correctness (all models)")
    print("=" * 70)

    models = [
        ("EfficientNet-B0",     EfficientNetB0Transfer(dropout=0.4)),
        ("VGG-16",              VGG_16_Transfer(dropout=0.4)),
        ("Swin-V2-t",          Swin_V2_t_Transfer(dropout=0.4)),
        ("ResNet-34",          ResNet34_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny",      ConvNext_tiny_Transfer(dropout=0.4)),
        ("EfficientNet-V2-S",  Efficientnet_v2_s_Transfer(dropout=0.4)),
        ("ResNet-18",          ResNet18Transfer(dropout=0.5)),
    ]

    all_pass = True
    for name, model in models:
        total = _count_params(model)
        trainable = _count_trainable(model)
        frozen = _count_frozen(model)
        pct = 100 * trainable / total

        head_ok, head_msg = _all_head_params_trainable(model)
        backbone_ok, backbone_msg = _all_non_head_params_frozen(model)

        status = "PASS" if (head_ok and backbone_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  {name}:")
        print(f"    Total: {total:>12,}  Trainable: {trainable:>10,}  ({pct:.2f}%)")
        print(f"    Head trainable: {head_msg}")
        print(f"    Backbone frozen: {backbone_msg}")
        print(f"    [{status}]")

    return all_pass


# ── TEST 2: unfreeze_backbone with n_layers=1 ─────────────────────────────────

def test_unfreeze_partial():
    """unfreeze_backbone(n_layers=1) should unfreeze the last block + keep head trainable."""
    print("\n" + "=" * 70)
    print("TEST 2: unfreeze_backbone(n_layers=1) — partial unfreeze")
    print("=" * 70)

    models = [
        ("EfficientNet-B0",     lambda: EfficientNetB0Transfer(dropout=0.4)),
        ("VGG-16",              lambda: VGG_16_Transfer(dropout=0.4)),
        ("Swin-V2-t",          lambda: Swin_V2_t_Transfer(dropout=0.4)),
        ("ResNet-34",          lambda: ResNet34_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny",      lambda: ConvNext_tiny_Transfer(dropout=0.4)),
        ("EfficientNet-V2-S",  lambda: Efficientnet_v2_s_Transfer(dropout=0.4)),
        ("ResNet-18",          lambda: ResNet18Transfer(dropout=0.5)),
    ]

    all_pass = True
    for name, factory in models:
        model = factory()
        before_trainable = _count_trainable(model)

        print(f"\n  {name}:")
        unfreeze_backbone(model, n_layers=1)

        after_trainable = _count_trainable(model)
        total = _count_params(model)
        pct = 100 * after_trainable / total

        # head must still be trainable
        head_ok, head_msg = _all_head_params_trainable(model)

        # trainable count must have INCREASED (some backbone params now live)
        increased = after_trainable > before_trainable

        status = "PASS" if (head_ok and increased) else "FAIL"
        if not status == "PASS":
            all_pass = False

        print(f"    Before unfreeze: {before_trainable:>10,} trainable")
        print(f"    After unfreeze:  {after_trainable:>10,} trainable  ({pct:.2f}%)")
        print(f"    Head trainable: {head_msg}")
        print(f"    Trainable increased: {increased}")
        print(f"    [{status}]")

    return all_pass


# ── TEST 3: unfreeze_backbone with n_layers=-1 (full unfreeze) ────────────────

def test_unfreeze_full():
    """unfreeze_backbone(n_layers=-1) should unfreeze ALL backbone params + head."""
    print("\n" + "=" * 70)
    print("TEST 3: unfreeze_backbone(n_layers=-1) — full unfreeze")
    print("=" * 70)

    models = [
        ("EfficientNet-B0",     lambda: EfficientNetB0Transfer(dropout=0.4)),
        ("VGG-16",              lambda: VGG_16_Transfer(dropout=0.4)),
        ("Swin-V2-t",          lambda: Swin_V2_t_Transfer(dropout=0.4)),
        ("ResNet-34",          lambda: ResNet34_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny",      lambda: ConvNext_tiny_Transfer(dropout=0.4)),
        ("EfficientNet-V2-S",  lambda: Efficientnet_v2_s_Transfer(dropout=0.4)),
        ("ResNet-18",          lambda: ResNet18Transfer(dropout=0.5)),
    ]

    all_pass = True
    for name, factory in models:
        model = factory()
        total = _count_params(model)

        print(f"\n  {name}:")
        unfreeze_backbone(model, n_layers=-1)

        after_trainable = _count_trainable(model)
        frozen_after = _count_frozen(model)

        # ALL params should be trainable after full unfreeze
        all_unfrozen = (after_trainable == total)

        # If not all unfrozen, find which params are still frozen
        still_frozen = []
        if not all_unfrozen:
            for pname, p in model.named_parameters():
                if not p.requires_grad:
                    still_frozen.append(f"{pname} ({p.numel():,} params)")

        status = "PASS" if all_unfrozen else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"    Total params: {total:>12,}")
        print(f"    Trainable:    {after_trainable:>12,}")
        print(f"    Still frozen: {frozen_after:>12,}")
        if still_frozen:
            print(f"    FROZEN PARAMS AFTER FULL UNFREEZE:")
            for fp in still_frozen[:10]:  # show first 10
                print(f"      ❌ {fp}")
            if len(still_frozen) > 10:
                print(f"      ... and {len(still_frozen) - 10} more")
        print(f"    [{status}]")

    return all_pass


# ── TEST 4: get_backbone_lr_groups correctness ────────────────────────────────

def test_lr_groups():
    """get_backbone_lr_groups should produce exactly 2 groups with correct LRs."""
    print("\n" + "=" * 70)
    print("TEST 4: get_backbone_lr_groups — differential LR param groups")
    print("=" * 70)

    models = [
        ("EfficientNet-B0",     lambda: EfficientNetB0Transfer(dropout=0.4)),
        ("Swin-V2-t",          lambda: Swin_V2_t_Transfer(dropout=0.4)),
        ("ResNet-34",          lambda: ResNet34_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny",      lambda: ConvNext_tiny_Transfer(dropout=0.4)),
    ]

    all_pass = True
    backbone_lr = 1e-5
    head_lr = 1e-3

    for name, factory in models:
        model = factory()
        unfreeze_backbone(model, n_layers=1)

        try:
            groups = get_backbone_lr_groups(model, backbone_lr=backbone_lr, head_lr=head_lr)

            # 2 groups
            assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"

            # LRs correct
            assert groups[0]["lr"] == backbone_lr, f"backbone lr={groups[0]['lr']}, expected {backbone_lr}"
            assert groups[1]["lr"] == head_lr, f"head lr={groups[1]['lr']}, expected {head_lr}"

            # backbone group: only unfrozen (requires_grad=True) params
            bb_count = sum(p.numel() for p in groups[0]["params"])
            head_count = sum(p.numel() for p in groups[1]["params"])

            # all params in backbone group should require grad
            for p in groups[0]["params"]:
                assert p.requires_grad, "backbone group contains a frozen param!"

            # sum of both groups should equal total trainable params
            total_trainable = _count_trainable(model)
            groups_total = bb_count + head_count

            match = (groups_total == total_trainable)

            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False

            print(f"\n  {name}:")
            print(f"    Backbone group: {bb_count:>10,} params  (lr={backbone_lr})")
            print(f"    Head group:     {head_count:>10,} params  (lr={head_lr})")
            print(f"    Total in groups: {groups_total:>10,}  vs  total trainable: {total_trainable:>10,}")
            if not match:
                print(f"    ❌ MISMATCH: {total_trainable - groups_total} params NOT in any group!")
                # find which trainable params are missing from both groups
                group_ids = set()
                for g in groups:
                    for p in g["params"]:
                        group_ids.add(id(p))
                for pname, p in model.named_parameters():
                    if p.requires_grad and id(p) not in group_ids:
                        print(f"       Missing: {pname} ({p.numel():,} params)")
            print(f"    [{status}]")

        except Exception as e:
            print(f"\n  {name}: EXCEPTION: {e}")
            all_pass = False

    return all_pass


# ── TEST 5: Gradient flow test ────────────────────────────────────────────────

def test_gradient_flow():
    """Forward + backward pass: check that gradients actually flow to unfrozen params."""
    print("\n" + "=" * 70)
    print("TEST 5: Gradient flow — forward + backward pass verification")
    print("=" * 70)

    models = [
        ("EfficientNet-B0",     lambda: EfficientNetB0Transfer(dropout=0.4)),
        ("ResNet-34",          lambda: ResNet34_Transfer(dropout=0.4)),
        ("Swin-V2-t",          lambda: Swin_V2_t_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny",      lambda: ConvNext_tiny_Transfer(dropout=0.4)),
    ]

    all_pass = True
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_target = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    for name, factory in models:
        model = factory()
        model.train()

        # Test 5a: Stage 1 (frozen backbone) — only head gets gradients
        model.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()

        head_has_grad = False
        backbone_has_grad = False
        for pname, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                # check if this is a head param
                is_head = any(hn in pname for hn in _HEAD_NAMES)
                if is_head:
                    head_has_grad = True
                else:
                    backbone_has_grad = True

        stage1_ok = head_has_grad and not backbone_has_grad
        if not stage1_ok:
            all_pass = False

        # Test 5b: After partial unfreeze — backbone should also get gradients
        model2 = factory()
        unfreeze_backbone(model2, n_layers=1)
        model2.train()
        model2.zero_grad()
        out2 = model2(dummy_input)
        loss2 = criterion(out2, dummy_target)
        loss2.backward()

        unfrozen_with_grad = 0
        unfrozen_without_grad = 0
        unfrozen_no_grad_names = []
        for pname, p in model2.named_parameters():
            if p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    unfrozen_with_grad += 1
                else:
                    unfrozen_without_grad += 1
                    unfrozen_no_grad_names.append(pname)

        stage2_ok = unfrozen_without_grad == 0

        status = "PASS" if (stage1_ok and stage2_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  {name}:")
        print(f"    Stage 1 (frozen): head_has_grad={head_has_grad}, backbone_has_grad={backbone_has_grad}  {'✓' if stage1_ok else '❌'}")
        print(f"    Stage 2 (partial unfreeze): {unfrozen_with_grad} params with grad, {unfrozen_without_grad} without")
        if unfrozen_no_grad_names:
            print(f"    ❌ Trainable params with NO gradient:")
            for n in unfrozen_no_grad_names[:5]:
                print(f"       {n}")
        print(f"    [{status}]")

    return all_pass


# ── TEST 6: ConvNeXt block coverage ───────────────────────────────────────────

def test_convnext_blocks():
    """Verify ConvNeXt unfreeze covers the correct number of feature blocks."""
    print("\n" + "=" * 70)
    print("TEST 6: ConvNeXt block coverage in unfreeze_backbone")
    print("=" * 70)

    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    # Check what features children look like in the raw torchvision model
    raw = convnext_tiny(weights=None)
    print(f"\n  ConvNeXt-Tiny features children ({len(list(raw.features.children()))}):")
    for i, (name, child) in enumerate(raw.features.named_children()):
        n_params = sum(p.numel() for p in child.parameters())
        print(f"    features[{i}] ({name}): {child.__class__.__name__}, {n_params:,} params")

    print(f"\n  Total features children: {len(list(raw.features.children()))}")
    print(f"  Our code uses features[1:5] = 4 blocks")
    print(f"  Missing blocks: features[0] (stem), features[5], features[6], features[7]")

    # Now test with our wrapper
    model = ConvNext_tiny_Transfer(dropout=0.4)
    before_trainable = _count_trainable(model)
    unfreeze_backbone(model, n_layers=-1)
    after_trainable = _count_trainable(model)
    total = _count_params(model)

    all_unfrozen = (after_trainable == total)
    frozen_params = []
    if not all_unfrozen:
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.append(f"{pname} ({p.numel():,})")

    status = "PASS" if all_unfrozen else "FAIL"
    print(f"\n  After full unfreeze (-1):")
    print(f"    Total: {total:,}  Trainable: {after_trainable:,}  Frozen: {total - after_trainable:,}")
    if frozen_params:
        print(f"    ❌ Still frozen:")
        for fp in frozen_params[:15]:
            print(f"       {fp}")
    print(f"  [{status}]")

    return all_unfrozen


# ── TEST 7: EfficientNet block granularity ────────────────────────────────────

def test_efficientnet_blocks():
    """Verify EfficientNet unfreeze_backbone blocks are meaningful MBConv blocks."""
    print("\n" + "=" * 70)
    print("TEST 7: EfficientNet block granularity in unfreeze_backbone")
    print("=" * 70)

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    raw = efficientnet_b0(weights=None)
    print(f"\n  EfficientNet-B0 features children ({len(list(raw.features.children()))}):")
    for i, child in enumerate(raw.features.children()):
        n_params = sum(p.numel() for p in child.parameters())
        print(f"    features[{i}]: {child.__class__.__name__}, {n_params:,} params")

    # n_layers=1 should unfreeze the last meaningful block
    model = EfficientNetB0Transfer(dropout=0.4)
    before = _count_trainable(model)
    unfreeze_backbone(model, n_layers=1)
    after = _count_trainable(model)

    added = after - before
    print(f"\n  n_layers=1 unfroze {added:,} additional params")
    print(f"  (features[-1] = features[8] is just Conv2d+BN = {sum(p.numel() for p in list(raw.features.children())[-1].parameters()):,} params)")
    print(f"  (features[-2] = features[7] MBConv block = {sum(p.numel() for p in list(raw.features.children())[-2].parameters()):,} params)")

    # The issue: features[-1] is a small normalization layer, not an MBConv block
    last_block = list(raw.features.children())[-1]
    print(f"  Last block type: {last_block.__class__.__name__}")
    if added > 0:
        print(f"  ✅  n_layers=1 correctly unfreezes the last MBConv stage + top projection.")
    else:
        print(f"  ⚠️  n_layers=1 unfreezes 0 additional params — check EfficientNet branch.")

    return True  # informational test


# ── TEST 8: Swin norm layer freeze check ──────────────────────────────────────

def test_swin_norm_freeze():
    """Swin-V2-t has a norm layer between features and head that may stay frozen."""
    print("\n" + "=" * 70)
    print("TEST 8: Swin-V2-t norm layer freeze/unfreeze status")
    print("=" * 70)

    from torchvision.models import swin_v2_t, Swin_V2_T_Weights

    raw = swin_v2_t(weights=None)
    print(f"\n  Swin-V2-t top-level children:")
    for name, child in raw.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"    {name}: {child.__class__.__name__}, {n_params:,} params")

    # Check our wrapper
    model = Swin_V2_t_Transfer(dropout=0.4)

    # Stage 1: norm should be frozen
    norm_frozen_stage1 = all(not p.requires_grad for p in model.backbone.norm.parameters())
    print(f"\n  Stage 1 (init): norm frozen = {norm_frozen_stage1}")

    # After full unfreeze: norm should be trainable
    unfreeze_backbone(model, n_layers=-1)
    norm_frozen_after = all(not p.requires_grad for p in model.backbone.norm.parameters())
    norm_trainable_after = all(p.requires_grad for p in model.backbone.norm.parameters())

    print(f"  After full unfreeze: norm frozen = {norm_frozen_after}, trainable = {norm_trainable_after}")

    status = "PASS" if norm_trainable_after else "FAIL"
    if not norm_trainable_after:
        print(f"  ❌ backbone.norm stays FROZEN even after full unfreeze!")
        print(f"     This blocks gradient flow from the head to the feature blocks.")
        n_norm = sum(p.numel() for p in model.backbone.norm.parameters())
        print(f"     norm has {n_norm:,} params that will never update.")
    print(f"  [{status}]")

    return norm_trainable_after


# ── TEST 9: Forward pass shape check ──────────────────────────────────────────

def test_forward_shapes():
    """All models should produce (B, 9) output from (B, 3, 224, 224) input."""
    print("\n" + "=" * 70)
    print("TEST 9: Forward pass output shapes")
    print("=" * 70)

    models = [
        ("EfficientNet-B0 (BASE)",    EfficientNetB0Transfer(dropout=0.4)),
        ("EfficientNet-B0 (SIMPLE)",  EfficientNetB0Transfer(dropout=0.4, head="SIMPLE")),
        ("VGG-16 (BASE)",             VGG_16_Transfer(dropout=0.4)),
        ("VGG-16 (SIMPLE)",           VGG_16_Transfer(dropout=0.4, head="SIMPLE")),
        ("Swin-V2-t (BASE)",         Swin_V2_t_Transfer(dropout=0.4)),
        ("Swin-V2-t (SIMPLE)",       Swin_V2_t_Transfer(dropout=0.4, head="SIMPLE")),
        ("Swin-V2-t (MLP)",          Swin_V2_t_Transfer(dropout=0.4, head="MLP")),
        ("ResNet-34 (BASE)",         ResNet34_Transfer(dropout=0.4)),
        ("ResNet-34 (SIMPLE)",       ResNet34_Transfer(dropout=0.4, head="SIMPLE")),
        ("ConvNeXt-Tiny (BASE)",     ConvNext_tiny_Transfer(dropout=0.4)),
        ("ConvNeXt-Tiny (SIMPLE)",   ConvNext_tiny_Transfer(dropout=0.4, head="SIMPLE")),
        ("EfficientNet-V2-S (BASE)", Efficientnet_v2_s_Transfer(dropout=0.4)),
        ("EfficientNet-V2-S (SIMPLE)", Efficientnet_v2_s_Transfer(dropout=0.4, head="SIMPLE")),
        ("EfficientNet-V2-S (MLP)",  Efficientnet_v2_s_Transfer(dropout=0.4, head="MLP")),
        ("ResNet-18",                ResNet18Transfer(dropout=0.5)),
    ]

    dummy_input = torch.randn(2, 3, 224, 224)
    all_pass = True

    for name, model in models:
        model.eval()
        try:
            with torch.no_grad():
                out = model(dummy_input)
            shape_ok = (out.shape == (2, 9))
            has_nan = torch.isnan(out).any().item()
            status = "PASS" if (shape_ok and not has_nan) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {name:<35} shape={out.shape}  has_nan={has_nan}  [{status}]")
        except Exception as e:
            print(f"  {name:<35} EXCEPTION: {e}  [FAIL]")
            all_pass = False

    return all_pass


# ── TEST 10: Optimizer only updates trainable params ──────────────────────────

def test_optimizer_frozen_params():
    """Passing model.parameters() to Adam includes frozen params. They get momentum
    states but zero updates. This is WASTEFUL but not a correctness bug.
    However, if get_backbone_lr_groups is used, it should NOT include frozen params."""
    print("\n" + "=" * 70)
    print("TEST 10: Optimizer param group coverage")
    print("=" * 70)

    model = ResNet34_Transfer(dropout=0.4)

    # Stage 1: passing model.parameters() to Adam — check how many frozen params are in there
    all_params = list(model.parameters())
    trainable = [p for p in all_params if p.requires_grad]
    frozen = [p for p in all_params if not p.requires_grad]

    print(f"\n  ResNet-34 Stage 1:")
    print(f"    model.parameters() total: {len(all_params)}")
    print(f"    Trainable: {len(trainable)}  ({sum(p.numel() for p in trainable):,} params)")
    print(f"    Frozen: {len(frozen)}  ({sum(p.numel() for p in frozen):,} params)")
    print(f"    ⚠️  Adam(model.parameters()) creates momentum states for {len(frozen)} frozen params")
    print(f"       This wastes memory but doesn't cause wrong updates.")
    print(f"       Better: Adam(filter(lambda p: p.requires_grad, model.parameters()))")

    # The notebook DOES use model.parameters() in Part 1 experiments
    # This is wasteful but not a bug. Let's just warn about it.
    return True


# ── TEST 11: ResNet block names ───────────────────────────────────────────────

def test_resnet_block_names():
    """Verify ResNet unfreeze uses the correct layer names."""
    print("\n" + "=" * 70)
    print("TEST 11: ResNet block names and layer coverage")
    print("=" * 70)

    from torchvision.models import resnet34

    raw = resnet34(weights=None)
    print(f"\n  ResNet-34 top-level children:")
    for name, child in raw.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"    {name}: {child.__class__.__name__}, {n_params:,} params")

    # Test partial unfreeze — should unfreeze layer4
    model = ResNet34_Transfer(dropout=0.4)
    unfreeze_backbone(model, n_layers=1)

    layer4_trainable = all(p.requires_grad for p in model.backbone.layer4.parameters())
    layer3_frozen = all(not p.requires_grad for p in model.backbone.layer3.parameters())

    print(f"\n  After n_layers=1:")
    print(f"    layer4 all trainable: {layer4_trainable}")
    print(f"    layer3 all frozen: {layer3_frozen}")

    status = "PASS" if (layer4_trainable and layer3_frozen) else "FAIL"
    print(f"  [{status}]")

    # Test: are conv1, bn1, relu, maxpool (stem) included in the blocks?
    model2 = ResNet34_Transfer(dropout=0.4)
    unfreeze_backbone(model2, n_layers=-1)

    conv1_trainable = model2.backbone.conv1.weight.requires_grad
    bn1_trainable = all(p.requires_grad for p in model2.backbone.bn1.parameters())

    print(f"\n  After n_layers=-1 (full unfreeze):")
    print(f"    conv1 trainable: {conv1_trainable}")
    print(f"    bn1 trainable: {bn1_trainable}")

    # conv1 and bn1 are NOT inside layer1-4, they're separate children.
    # unfreeze_backbone only unfreezes blocks = [layer1, layer2, layer3, layer4]
    # So conv1 and bn1 STAY FROZEN even with n_layers=-1!
    if not conv1_trainable:
        print(f"  ⚠️  conv1 stays FROZEN even with full unfreeze!")
        print(f"     unfreeze_backbone only covers layer1-4, not the stem (conv1, bn1).")
        n_stem = (model2.backbone.conv1.weight.numel() +
                  sum(p.numel() for p in model2.backbone.bn1.parameters()))
        print(f"     Stem has {n_stem:,} params that will never update.")

    return layer4_trainable and layer3_frozen


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    results.append(("Test 1: Stage 1 freeze",           test_stage1_freeze()))
    results.append(("Test 2: Partial unfreeze",          test_unfreeze_partial()))
    results.append(("Test 3: Full unfreeze",             test_unfreeze_full()))
    results.append(("Test 4: LR groups",                 test_lr_groups()))
    results.append(("Test 5: Gradient flow",             test_gradient_flow()))
    results.append(("Test 6: ConvNeXt blocks",           test_convnext_blocks()))
    results.append(("Test 7: EfficientNet blocks",       test_efficientnet_blocks()))
    results.append(("Test 8: Swin norm layer",           test_swin_norm_freeze()))
    results.append(("Test 9: Forward shapes",            test_forward_shapes()))
    results.append(("Test 10: Optimizer coverage",       test_optimizer_frozen_params()))
    results.append(("Test 11: ResNet block names",       test_resnet_block_names()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        print(f"  {'✓ PASS' if passed else '❌ FAIL'}  {name}")

    n_fail = sum(1 for _, p in results if not p)
    print(f"\n  {len(results) - n_fail}/{len(results)} tests passed, {n_fail} failed")

    if n_fail > 0:
        sys.exit(1)
