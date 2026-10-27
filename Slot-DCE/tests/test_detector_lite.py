"""Regression tests for the only supported SlotDCE architecture."""

from __future__ import annotations

from pathlib import Path

import torch

from config_utils import (
    ARCHITECTURE,
    build_loss,
    build_model,
    load_config,
    load_model_checkpoint,
)
from models import SlotDCEDetectionAdapter
from models.slot_dce import SpaceToDepthRecoveryHead, SpaceToDepthStem


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "slotDCE_v9_8_detector_lite.yaml"


def components():
    settings = load_config(CONFIG)
    model = build_model(settings["model"])
    criterion = build_loss(settings["loss"], settings["model"])
    return settings, model, criterion


def convolution_macs(model: torch.nn.Module, image: torch.Tensor) -> int:
    total = 0
    handles = []

    def count(module, _inputs, output):
        nonlocal total
        batch, channels, height, width = output.shape
        kh, kw = module.kernel_size
        total += (
            batch
            * channels
            * height
            * width
            * (module.in_channels // module.groups)
            * kh
            * kw
        )

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(count))
    with torch.no_grad():
        model(image)
    for handle in handles:
        handle.remove()
    return total


def test_architecture_and_shapes() -> None:
    settings, model, _ = components()
    model.eval()
    assert settings["architecture"] == ARCHITECTURE
    assert isinstance(model.encoder[0], SpaceToDepthStem)
    assert model.encoder[0].factor == 4
    assert isinstance(model.recovery_head, SpaceToDepthRecoveryHead)
    assert model.recovery_head.factor == 4

    for height, width in ((64, 80), (65, 81)):
        image = torch.rand(1, 3, height, width)
        with torch.no_grad():
            outputs = model(image)
            enhanced, guidance = model.enhance_for_detection(
                image,
                return_guidance=True,
            )
            pyramid_enhanced, guidance_pyramid = model.enhance_for_detection(
                image,
                return_guidance=True,
                guidance_sizes=((8, 10), (4, 5)),
            )
            prepared = model.prepare_detection_context(image)
            prepared_enhanced, prepared_guidance = model.enhance_for_detection(
                image,
                return_guidance=True,
                guidance_sizes=((8, 10), (4, 5)),
                _prepared_context=prepared,
            )
        assert outputs["enhanced"].shape == image.shape
        assert torch.equal(enhanced, outputs["enhanced"])
        assert torch.equal(pyramid_enhanced, enhanced)
        assert torch.allclose(prepared_enhanced, enhanced, atol=1e-6, rtol=1e-5)
        for actual, expected in zip(prepared_guidance, guidance_pyramid):
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        assert isinstance(guidance_pyramid, tuple)
        assert len(guidance_pyramid) == 2
        for target_size, compact_guidance in zip(
            ((8, 10), (4, 5)), guidance_pyramid
        ):
            expected_guidance = torch.nn.functional.interpolate(
                guidance,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            assert torch.allclose(
                compact_guidance,
                expected_guidance,
                atol=1e-7,
                rtol=1e-6,
            )
        assert guidance.shape == (
            image.shape[0],
            model.detection_guidance_channels,
            height,
            width,
        )
        assert bool(torch.isfinite(guidance).all())
        assert outputs["attn_map_low"].shape[-2:] == (
            (height + 3) // 4,
            (width + 3) // 4,
        )
        assert bool(torch.isfinite(outputs["enhanced"]).all())


def test_size_and_compute_budget() -> None:
    _, model, _ = components()
    model.eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 68_819
    image = torch.rand(1, 3, 96, 96)
    assert convolution_macs(model, image) < 40_000_000


def test_forward_recovery_loss_and_backward() -> None:
    _, model, criterion = components()
    image = torch.rand(1, 3, 64, 80)
    recovery_input = (1.8 * image + 0.12).clamp(0.0, 1.0)
    outputs = model(image)
    recovery_outputs = model.forward_recovery(recovery_input)
    losses = criterion(
        image,
        outputs,
        recovery_target=image,
        recovery_outputs=recovery_outputs,
        recovery_mask=torch.ones_like(image[:, :1]),
    )
    assert tuple(losses) == ("total", "tone", "texture", "recovery", "smoothness")
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses["total"].backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_checkpoint_and_detection_adapter(tmp_path: Path) -> None:
    _, source, _ = components()
    checkpoint = tmp_path / "model.pth"
    torch.save({"architecture": ARCHITECTURE, "model": source.state_dict()}, checkpoint)
    _, restored, _ = components()
    load_model_checkpoint(restored, checkpoint, torch.device("cpu"))
    source.eval()
    restored.eval()
    image = torch.rand(1, 3, 65, 81)
    with torch.no_grad():
        expected = source(image)["enhanced"]
        actual = SlotDCEDetectionAdapter(restored)(image)
    assert torch.equal(expected, actual)
