"""Slot-YOLO with independently propagated raw and enhanced branches."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ultralytics.nn.modules.head import OBB, RTDETRDecoder
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss, v8OBBLoss

from .myconfigv2 import SlotDCE, SlotDCEConfig
from .myconfigv2.feature_pipeline import (
    ConfiguredFeatureStage,
    build_feature_stages,
    build_neck,
    build_sgdc,
)


@contextmanager
def _restore_batch_norm_running_stats(*modules: nn.Module):
    """Restore BN buffers after checkpoint recomputation updates them."""

    batch_norms = []
    seen: set[int] = set()
    for module in modules:
        for child in module.modules():
            if (
                isinstance(child, nn.modules.batchnorm._BatchNorm)
                and child.track_running_stats
                and id(child) not in seen
            ):
                seen.add(id(child))
                batch_norms.append(child)
    snapshots = [
        (
            child.running_mean.detach().clone(),
            child.running_var.detach().clone(),
            child.num_batches_tracked.detach().clone(),
        )
        for child in batch_norms
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for child, (mean, variance, batches) in zip(batch_norms, snapshots):
                child.running_mean.copy_(mean)
                child.running_var.copy_(variance)
                child.num_batches_tracked.copy_(batches)


def _read_slot_dce_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Slot-DCE config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Slot-DCE config must contain a 'model' mapping: {config_path}")
    return config


def _neck_output_channels(base_layers: list[nn.Module]) -> tuple[int, int, int]:
    """Read the parsed Detect input contract, independently of YAML layer numbers."""
    head = base_layers[-1]
    channels = tuple(
        next(module.in_channels for module in branch.modules() if isinstance(module, nn.Conv2d))
        for branch in (head.input_proj if isinstance(head, RTDETRDecoder) else head.cv2)
    )
    if len(channels) != 3:
        raise ValueError("Slot-YOLO requires a three-scale Detect input contract")
    return channels


def _load_enhancer_weights(
    enhancer: SlotDCE,
    weights: str | Path,
    expected_architecture: str,
) -> dict[str, Any]:
    """Load a standalone Slot-DCE checkpoint into the embedded enhancer."""
    checkpoint_path = Path(weights).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Slot-DCE checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Slot-DCE checkpoint must contain a mapping")

    checkpoint_architecture = checkpoint.get("architecture")
    if (
        checkpoint_architecture is not None
        and str(checkpoint_architecture) != expected_architecture
    ):
        LOGGER.warning(
            "Slot-DCE checkpoint label is %r while the configured label is %r; "
            "strict state-dict validation will determine compatibility.",
            checkpoint_architecture,
            expected_architecture,
        )

    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, Mapping):
        raise ValueError("Slot-DCE checkpoint does not contain a model state dictionary")
    state = dict(state)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    try:
        enhancer.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Slot-DCE checkpoint parameters do not exactly match the configured "
            f"{expected_architecture} enhancer"
        ) from error
    return dict(checkpoint)


def build_enhancer(
    config: Mapping[str, Any],
    module_version: str = "v2",
) -> tuple[SlotDCE, dict[str, Any], str]:
    """Build the current Slot-DCE architecture and optionally load its weights."""
    config = dict(config)
    pretrained = bool(config.get("pretrained", False))
    weights = config.get("weights")
    if pretrained != bool(weights):
        raise ValueError(
            "enhancer.pretrained and enhancer.weights must be enabled together"
        )

    architecture = config.get("architecture")
    if architecture is not None and not isinstance(architecture, Mapping):
        raise TypeError("enhancer.architecture must be a mapping when provided")
    architecture = dict(architecture or {})

    if config.get("architecture_config"):
        standalone = _read_slot_dce_config(config["architecture_config"])
        architecture = dict(standalone["model"])
        architecture_name = str(standalone.get("architecture", "Slot-DCE"))
        source = str(config["architecture_config"])
    elif architecture:
        architecture_name = "inline_slot_dce"
        source = "inline enhancer.architecture"
    else:
        raise ValueError(
            "Set enhancer.architecture_config or provide enhancer.architecture"
        )

    if module_version != "v2":
        raise ValueError("The retained Slot-YOLO architecture requires module_version='v2'")
    Enhancer, EnhancerConfig = SlotDCE, SlotDCEConfig
    enhancer = Enhancer(EnhancerConfig.from_mapping(architecture))
    if pretrained:
        metadata = _load_enhancer_weights(
            enhancer,
            weights,
            expected_architecture=architecture_name,
        )
        LOGGER.info(
            "%s initialized from %s (checkpoint epoch %s).",
            architecture_name,
            weights,
            metadata.get("epoch", "unknown"),
        )
    else:
        LOGGER.info(
            "%s is randomly initialized for joint end-to-end training (%s).",
            architecture_name,
            source,
        )
    return enhancer, architecture, architecture_name


def _backbone_stages(layers, stage_ends=None):
    """Split a sequential parsed backbone at its P2/P3/P4/P5 endpoints."""
    if stage_ends is None:
        if len(layers) != 11:
            raise ValueError(f"YOLO26 backbone must contain 11 layers, got {len(layers)}")
        stage_ends = (3, 5, 7, 11)
    if len(stage_ends) != 4 or stage_ends[-1] != len(layers):
        raise ValueError("Backbone requires four stage endpoints covering all layers")
    starts = (0, *stage_ends[:-1])
    return nn.ModuleList(nn.Sequential(*layers[start:end]) for start, end in zip(starts, stage_ends))


class SharedYOLO26Backbone(nn.Module):
    """Share weights with separate BN; retain the legacy name for old checkpoints.

    Explicit stage endpoints support YOLOv8, YOLO12 and ResNet-50 pyramids.
    """

    def __init__(self, layers: list[nn.Module], stage_ends=None) -> None:
        super().__init__()
        self.stages = _backbone_stages(layers, stage_ends)
        self.enhanced_stages = deepcopy(self.stages)
        self._share_non_norm_parameters(self.stages, self.enhanced_stages)

    @staticmethod
    def _share_non_norm_parameters(
        raw_stages: nn.Module,
        enhanced_stages: nn.Module,
    ) -> None:
        """Tie learned weights except normalization parameters between branches."""
        raw_modules = dict(raw_stages.named_modules())
        enhanced_modules = dict(enhanced_stages.named_modules())
        if raw_modules.keys() != enhanced_modules.keys():
            raise RuntimeError("Raw and enhanced backbone structures do not match")
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        for name, raw_module in raw_modules.items():
            enhanced_module = enhanced_modules[name]
            if type(raw_module) is not type(enhanced_module):
                raise RuntimeError(f"Backbone module mismatch at {name!r}")
            if isinstance(raw_module, norm_types):
                continue
            for parameter_name, parameter in raw_module.named_parameters(recurse=False):
                setattr(enhanced_module, parameter_name, parameter)

    def infer_metadata(
        self, image_size: int = 64
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Infer P3/P4/P5 channels and strides without changing BN statistics."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feature = torch.zeros(1, 3, image_size, image_size)
            feature = self.stages[0](feature)
            p3 = self.stages[1](feature)
            p4 = self.stages[2](p3)
            p5 = self.stages[3](p4)
        self.train(was_training)
        features = (p3, p4, p5)
        channels = tuple(int(feature.shape[1]) for feature in features)
        strides = tuple(int(image_size // feature.shape[-1]) for feature in features)
        return channels, strides

    def infer_channels(self, image_size: int = 64) -> tuple[int, int, int]:
        """Infer P3/P4/P5 channels without changing BatchNorm statistics."""
        return self.infer_metadata(image_size)[0]

    def forward(
        self,
        raw_image: Tensor,
        enhanced_image: Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        """Return independent raw and enhanced P3/P4/P5 pyramids."""
        if raw_image.shape[0] != enhanced_image.shape[0]:
            raise ValueError("raw and enhanced branches must use the same batch size")

        def run_stage(index: int, raw: Tensor, enhanced: Tensor) -> tuple[Tensor, Tensor]:
            raw_stage = self.stages[index]
            enhanced_stage = self.enhanced_stages[index]

            def stage_forward(raw_value: Tensor, enhanced_value: Tensor):
                return raw_stage(raw_value), enhanced_stage(enhanced_value)

            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(
                    stage_forward,
                    raw,
                    enhanced,
                    use_reentrant=False,
                    # The original forward updates each branch-specific BN once.
                    # Recomputation uses the same batch statistics but must not
                    # update running mean/variance for a second time.
                    context_fn=lambda: (
                        nullcontext(),
                        _restore_batch_norm_running_stats(raw_stage, enhanced_stage),
                    ),
                )
            return stage_forward(raw, enhanced)

        raw, enhanced = run_stage(0, raw_image, enhanced_image)
        raw_p3, enhanced_p3 = run_stage(1, raw, enhanced)
        raw_p4, enhanced_p4 = run_stage(2, raw_p3, enhanced_p3)
        raw_p5, enhanced_p5 = run_stage(3, raw_p4, enhanced_p4)
        return (raw_p3, raw_p4, raw_p5), (enhanced_p3, enhanced_p4, enhanced_p5)


class SingleYOLO26Backbone(nn.Module):
    """Run the selected original-image backbone; retain the legacy pickle name."""

    def __init__(self, layers: list[nn.Module], stage_ends=None) -> None:
        super().__init__()
        self.stages = _backbone_stages(layers, stage_ends)

    def infer_metadata(
        self, image_size: int = 64
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Infer P3/P4/P5 channels and strides without changing BN statistics."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            p3, p4, p5 = self(torch.zeros(1, 3, image_size, image_size))
        self.train(was_training)
        features = (p3, p4, p5)
        channels = tuple(int(feature.shape[1]) for feature in features)
        strides = tuple(int(image_size // feature.shape[-1]) for feature in features)
        return channels, strides

    def infer_channels(self, image_size: int = 64) -> tuple[int, int, int]:
        """Infer P3/P4/P5 channels without changing BatchNorm statistics."""
        return self.infer_metadata(image_size)[0]

    def forward(
        self,
        image: Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the raw-image P3/P4/P5 pyramid without a second branch."""

        def run_stage(stage: nn.Module, feature: Tensor) -> Tensor:
            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(
                    stage,
                    feature,
                    use_reentrant=False,
                    context_fn=lambda: (
                        nullcontext(),
                        _restore_batch_norm_running_stats(stage),
                    ),
                )
            return stage(feature)

        feature = run_stage(self.stages[0], image)
        p3 = run_stage(self.stages[1], feature)
        p4 = run_stage(self.stages[2], p3)
        p5 = run_stage(self.stages[3], p4)
        return p3, p4, p5


class SlotYOLODetectionModel(DetectionModel):
    """Paired YOLO/RT-DETR heads with Slot-DCE and a dual-branch feature pipeline."""

    def __init__(
        self,
        cfg: str | Path | dict = "model_config/yolov8s-obb.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        slot_config: Mapping[str, Any] | None = None,
    ) -> None:
        if ch != 3:
            raise ValueError("Slot-YOLO requires 3-channel RGB input")
        if slot_config is None:
            raise ValueError("slot_config is required to configure Slot-YOLO")
        slot_config = deepcopy(dict(slot_config))
        from backbone_config import backbone_model_path, selected_backbone, validate_backbone_runtime
        selection = selected_backbone(slot_config)
        validate_backbone_runtime(slot_config)
        cfg = backbone_model_path(slot_config) or cfg
        from rtdetrv2_hye_config import (
            normalize_rtdetrv2_hye_yolo_config,
            validate_rtdetrv2_hye_yolo_selection,
        )
        use_rtdetrv2_hye_yolo = validate_rtdetrv2_hye_yolo_selection(slot_config)
        if use_rtdetrv2_hye_yolo:
            slot_config["rtdetrv2_hye_yolo"] = normalize_rtdetrv2_hye_yolo_config(
                slot_config.get("rtdetrv2_hye_yolo", {})
            )
        from module_version import module_version
        version = module_version(slot_config)
        if version != "v2":
            raise ValueError("The retained Slot-YOLO architecture requires module_version='v2'")
        from .myconfigv2 import feature_pipeline as pipeline
        from ig_config import validate_ig_config, validate_p5_config
        ig_flags = validate_ig_config(slot_config)
        p5_dual_branch = validate_p5_config(slot_config)

        # DetectionModel performs a dummy forward while it initializes stride.
        object.__setattr__(self, "_building_base", True)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        self._building_base = False

        base_layers = list(self.model.children())
        backbone_length = len(self.yaml["backbone"])
        self.is_rtdetr = isinstance(base_layers[-1], RTDETRDecoder)
        self.backbone_name = selection or "yolo26s"
        self.nc = base_layers[-1].nc
        stage_ends = {
            "yolov8s": (3, 5, 7, 10),
            "yolo12s": (3, 5, 7, 9),
            "rtdetr_r50": (2, 3, 4, 5),
            "rtdetr_r18": (2, 3, 4, 5),
        }.get(selection)
        parsed_neck_layers = base_layers[backbone_length:-1]
        if not parsed_neck_layers:
            raise ValueError("Slot-YOLO requires a parsed YOLO neck before Detect")
        head_task = "obb" if isinstance(base_layers[-1], OBB) else "detect"
        requested_task = str(slot_config.get("task", head_task)).lower()
        requested_task = "detect" if requested_task == "hbb" else requested_task
        if requested_task not in {"detect", "obb"}:
            raise ValueError("Slot-YOLO task must be detect/hbb or obb")
        if requested_task != head_task:
            raise ValueError(
                f"Slot-YOLO task={requested_task!r} requires a matching model "
                f"head, but {Path(str(cfg)).name!r} contains a {head_task!r} head"
            )

        use_slotdce_isaf = slot_config.get("use_slotdce_isaf", True)
        if not isinstance(use_slotdce_isaf, bool):
            raise TypeError("use_slotdce_isaf must be true or false")
        enhancer_config = dict(slot_config.get("enhancer", {}))
        if use_slotdce_isaf:
            enhancer, enhancer_architecture, architecture_name = build_enhancer(
                enhancer_config, module_version=version
            )
            backbone = SharedYOLO26Backbone(base_layers[:backbone_length], stage_ends)
            guidance_channels = enhancer.detection_guidance_channels
        else:
            enhancer = nn.Identity()
            enhancer_architecture = {}
            architecture_name = "disabled"
            backbone = SingleYOLO26Backbone(base_layers[:backbone_length], stage_ends)
            guidance_channels = 0
        (
            (p3_channels, p4_channels, p5_channels),
            neck_input_strides,
        ) = backbone.infer_metadata()
        pan_channels = _neck_output_channels(base_layers)
        if self.is_rtdetr and p5_dual_branch:
            if p5_channels != 512:
                raise ValueError(
                    f"P5 dual branch needs a 512-channel backbone P5, got {p5_channels}. "
                    "Select backbone: rtdetr_r18 to keep the P5 dual branch."
                )
            # Keep cat(CCFF, semantic refinement) at 512 channels. The decoder's
            # own input projection maps it to its 256-dimensional attention space.
            head = base_layers[-1]
            old_projection = head.input_proj[-1][0]
            projection = nn.Conv2d(512, old_projection.out_channels, 1, bias=False)
            nn.init.xavier_uniform_(projection.weight)
            head.input_proj[-1][0] = projection
            pan_channels = _neck_output_channels(base_layers)

        (
            (fuse_p3, fuse_p4, fuse_p5),
            uses_detector_guidance,
            feature_pipeline_description,
        ) = pipeline.build_feature_stages(
            (p3_channels, p4_channels, p5_channels),
            slot_config,
            guidance_channels=guidance_channels,
        )
        neck, custom_neck_enabled, neck_description = pipeline.build_neck(
            (p3_channels, p4_channels, p5_channels),
            pan_channels,
            slot_config,
            input_strides=neck_input_strides,
        )
        if (any(ig_flags.values()) or getattr(neck, "use_ig_lsf_aifi", False)) and guidance_channels != 10:
            raise ValueError("IG neck requires SlotDCE with exactly five slots (10 detector guidance channels)")
        sgdc, sgdc_enabled, sgdc_description = pipeline.build_sgdc(
            (p3_channels, p4_channels, p5_channels),
            slot_config,
            calibration_channels=pan_channels,
        )

        # Preserve the original PAN for controlled ablations. In custom-neck mode it
        # is replaced at construction time rather than merely bypassed, so
        # unused PAN parameters do not enter the optimizer or model-size count.
        # Keep the original Slot-YOLO state-dict positions in every mode.
        neck_layers = (
            (neck, *(nn.Identity() for _ in parsed_neck_layers[1:]))
            if custom_neck_enabled
            else tuple(parsed_neck_layers)
        )
        self.model = nn.ModuleList(
            (
                enhancer,
                backbone,
                fuse_p3,
                fuse_p4,
                fuse_p5,
                *neck_layers,
                base_layers[-1],
            )
        )
        # Keep SGDC outside the YOLO layer list so disabling it leaves every
        # original PAN/Detect index and operation untouched.
        self.sgdc = sgdc
        self.sgdc_enabled = sgdc_enabled
        self.use_slotdce_isaf = use_slotdce_isaf
        self.sgdc_placement = str(getattr(sgdc, "placement", "post_neck"))
        self.neck_name = getattr(neck, "neck_name", "efficient_hybrid_encoder") if custom_neck_enabled else "yolo"
        self.use_efficient_hybrid_encoder = self.neck_name == "efficient_hybrid_encoder"
        self.use_rtdetrv2_hye_yolo = self.neck_name == "rtdetrv2_hye_yolo"
        self.use_ig_lsf_aifi = getattr(neck, "use_ig_lsf_aifi", False)
        self.use_ig_aifi = ig_flags["use_ig_aifi"]
        self.use_ig_fs_ccff = ig_flags["use_ig_fs_ccff"]
        self.use_p5_dual_branch = p5_dual_branch
        self.neck_input_channels = (p3_channels, p4_channels, p5_channels)
        self.neck_input_strides = neck_input_strides
        self.neck_output_channels = pan_channels
        self._neck_start = len(self.model) - len(parsed_neck_layers) - 1
        self.neck_description = neck_description
        self.save = []
        self.slot_config = deepcopy(dict(slot_config))
        self.module_version = version
        self.slot_task = head_task
        self.task = head_task
        self.enhancer_architecture = deepcopy(enhancer_architecture)
        self.enhancer_architecture_name = architecture_name
        self.uses_detector_guidance = uses_detector_guidance
        self.feature_pipeline_description = feature_pipeline_description
        self.sgdc_description = sgdc_description
        self.enhancer_frozen = use_slotdce_isaf and bool(enhancer_config.get("freeze", False))
        self.enhancer_eval_when_frozen = bool(
            enhancer_config.get("eval_when_frozen", True)
        )
        self.configure_runtime(slot_config.get("memory", {}))
        if self.enhancer_frozen:
            for parameter in self.enhancer.parameters():
                parameter.requires_grad_(False)
            if self.enhancer_eval_when_frozen:
                self.enhancer.eval()

        if verbose:
            self.info()
            loss_name = (
                "OBB detection + geometry auxiliary loss"
                if self.slot_task == "obb" and (
                    getattr(neck, "use_geometry_ccff", False)
                    or getattr(neck, "use_geometry_ccff_v1_1", False)
                )
                else "OBB detection loss"
            )
            LOGGER.info(
                "Slot-YOLO: shared convolution weights with branch-specific BN; %s; "
                "%s; %s; %s.",
                self.feature_pipeline_description,
                self.neck_description,
                self.sgdc_description,
                loss_name,
            )

    def slot_task_name(self) -> str:
        """Return the task, also for checkpoints saved before ``slot_task`` existed."""
        task = getattr(self, "slot_task", None)
        if task in {"detect", "obb"}:
            return task
        return "obb" if isinstance(self.model[-1], OBB) else "detect"

    def validate_neck_selection(self, config: Mapping[str, Any]) -> None:
        """Reject checkpoint/config mismatches for architecture switches."""
        from rtdetrv2_hye_config import validate_rtdetrv2_hye_yolo_selection
        validate_rtdetrv2_hye_yolo_selection(config)
        from backbone_config import selected_backbone
        expected_backbone = selected_backbone(config) or "yolo26s"
        actual_backbone = getattr(self, "backbone_name", "yolo26s")
        if expected_backbone != actual_backbone:
            raise ValueError(f"Checkpoint backbone={actual_backbone}, config backbone={expected_backbone}; use train.resume=false for a new architecture")
        from module_version import module_version
        expected_version = module_version(config)
        actual_version = getattr(self, "module_version", "v1")
        if expected_version != actual_version:
            raise ValueError(
                f"Checkpoint module_version={actual_version}, but config selects {expected_version}. "
                "Use a matching version; a new architecture requires train.resume=false."
            )
        if getattr(self, "use_efficient_hybrid_encoder", False):
            self.efficient_hybrid_neck.validate_architecture()
        if getattr(self, "use_rtdetrv2_hye_yolo", False):
            self.rtdetrv2_hye_yolo_neck.validate_architecture(
                config.get("rtdetrv2_hye_yolo", {})
            )
        if config.get("use_sgdc", False) and getattr(self, "sgdc_enabled", False):
            settings = config.get("feature_modules", {}).get("sgdc", {})
            expected = str(settings.get("name", "sgdc")).lower()
            actual = getattr(self.sgdc, "variant", "sgdc")
            if expected != actual:
                raise ValueError(
                    f"Checkpoint SGDC variant={actual}, config selects {expected}; "
                    "a new architecture requires train.resume=false"
                )
        for key, actual in (
            ("use_slotdce_isaf", getattr(self, "use_slotdce_isaf", True)),
            ("use_sgdc", getattr(self, "sgdc_enabled", False)),
            ("use_rtdetrv2_hye_yolo", getattr(self, "use_rtdetrv2_hye_yolo", False)),
            ("use_ig_aifi", getattr(self, "use_ig_aifi", False)),
            ("use_ig_fs_ccff", getattr(self, "use_ig_fs_ccff", False)),
            ("use_p5_dual_branch", getattr(self.efficient_hybrid_neck, "use_p5_dual_branch", False)),
        ):
            if key not in config:
                continue
            expected = config[key]
            if not isinstance(expected, bool):
                raise TypeError(f"{key} must be true or false")
            if actual != expected:
                raise ValueError(
                    f"Checkpoint has {key}={actual}, but the config selects {expected}. "
                    "Use a matching checkpoint/config; changing model modules requires "
                    "a fresh run (train.resume=false)."
                )
        if (
            "use_efficient_hybrid_encoder" not in config
            and "use_rtdetrv2_hye_yolo" not in config
        ):
            return
        if config.get("use_rtdetrv2_hye_yolo", False):
            expected = "rtdetrv2_hye_yolo"
        elif config.get("use_efficient_hybrid_encoder", False):
            expected = "efficient_hybrid_encoder"
        else:
            expected = "yolo"
        actual = getattr(self, "neck_name", "yolo")
        if actual != expected:
            raise ValueError(
                f"Checkpoint neck is {actual!r}, but the config selects {expected!r}. "
                "Use a matching checkpoint/config; changing neck architecture requires a fresh run (train.resume=false)."
            )

    def init_criterion(self):
        """Select the loss that matches the configured HBB or OBB head."""
        if getattr(self, "is_rtdetr", False):
            return RTDETRDetectionModel.init_criterion(self)
        if self.slot_task_name() == "obb":
            return E2ELoss(self, v8OBBLoss) if self.end2end else v8OBBLoss(self)
        return E2ELoss(self) if self.end2end else v8DetectionLoss(self)

    def loss(self, batch, preds=None):
        if getattr(self, "is_rtdetr", False):
            if getattr(self, "criterion", None) is None:
                self.criterion = self.init_criterion()
            return RTDETRDetectionModel.loss(self, batch, preds)
        neck = self.rtdetrv2_hye_yolo_neck if getattr(self, "use_rtdetrv2_hye_yolo", False) else None
        if neck is not None and (getattr(neck, "use_geometry_ccff", False)
                                or getattr(neck, "use_geometry_ccff_v1_1", False)):
            geometry_p3 = (neck.ccff.geometry_p3 if getattr(neck, "use_geometry_ccff_v1_1", False)
                           else neck.geometry_p3)
            if preds is None:
                preds = self.forward(batch["img"])
            try:
                detection_loss, detection_items = super().loss(batch, preds)
                geometry_loss = (
                    geometry_p3.auxiliary_loss(batch)
                    if self.training else detection_loss.new_zeros(())
                )
                total = detection_loss + geometry_loss * batch["img"].shape[0]
                items = torch.cat((detection_items.reshape(-1), geometry_loss.detach().reshape(1)))
                return total, items
            finally:
                geometry_p3.clear_cache()
        return super().loss(batch, preds)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None, batch=None):
        if getattr(self, "is_rtdetr", False):
            return self._predict_once(x, profile=profile, visualize=visualize, embed=embed, batch=batch)
        return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)

    def _apply(self, fn):
        super()._apply(fn)
        if getattr(self, "is_rtdetr", False):
            self.model[-1].anchors = fn(self.model[-1].anchors)
            self.model[-1].valid_mask = fn(self.model[-1].valid_mask)
        return self

    @property
    def enhancer(self) -> nn.Module:
        return self.model[0]

    @property
    def dual_backbone(self) -> nn.Module:
        return self.model[1]

    @property
    def fuse_p3(self) -> nn.Module:
        return self.model[2]

    @property
    def fuse_p4(self) -> nn.Module:
        return self.model[3]

    @property
    def fuse_p5(self) -> nn.Module:
        return self.model[4]

    @property
    def sgdc_adapter(self) -> nn.Module:
        """Return the optional SGDC module for inspection and ablations."""
        return getattr(self, "sgdc", nn.Identity())


    @property
    def neck_start(self) -> int:
        """Locate the neck after enhancer, backbone and the three fusion stages."""
        # Older full-object checkpoints predate the stored boundary.
        return getattr(self, "_neck_start", 2 + len((self.fuse_p3, self.fuse_p4, self.fuse_p5)))

    @property
    def efficient_hybrid_neck(self) -> nn.Module:
        if getattr(self, "use_efficient_hybrid_encoder", False):
            return self.model[self.neck_start]
        return nn.Identity()

    @property
    def rtdetrv2_hye_yolo_neck(self) -> nn.Module:
        if getattr(self, "use_rtdetrv2_hye_yolo", False):
            return self.model[self.neck_start]
        return nn.Identity()

    def configure_runtime(
        self,
        memory_config: Mapping[str, Any] | None = None,
    ) -> "SlotYOLODetectionModel":
        """Apply non-parameter runtime controls, including to legacy checkpoints.

        Full Ultralytics checkpoints pickle the model object and therefore do
        not rerun ``__init__`` when loaded. Hydrating these attributes here
        keeps checkpoints created before a runtime option was introduced fully
        compatible with the current forward implementation.
        """

        memory = dict(memory_config or {})
        self.checkpoint_enhancer = bool(
            memory.get("checkpoint_enhancer", True)
        ) and getattr(self, "use_slotdce_isaf", True)
        self.checkpoint_backbone = bool(
            memory.get("checkpoint_backbone", False)
        )
        self.enhancer_amp = bool(memory.get("enhancer_amp", False))
        amp_name = str(memory.get("enhancer_amp_dtype", "bfloat16")).lower()
        amp_dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if amp_name not in amp_dtypes:
            raise ValueError(
                "enhancer_amp_dtype must be bfloat16/bf16 or float16/fp16"
            )
        self.enhancer_amp_dtype = amp_dtypes[amp_name]
        self.guidance_stage_indices = tuple(
            index
            for index, module in enumerate(
                (self.fuse_p3, self.fuse_p4, self.fuse_p5)
            )
            if isinstance(module, ConfiguredFeatureStage)
        )
        self.guidance_stage_indices = self._runtime_guidance_stage_indices()
        self.uses_detector_guidance = bool(self.guidance_stage_indices)

        if hasattr(self, "slot_config"):
            runtime_config = deepcopy(dict(self.slot_config))
            runtime_config["memory"] = deepcopy(memory)
            self.slot_config = runtime_config
        return self

    def _runtime_guidance_stage_indices(self) -> tuple[int, ...]:
        indices = getattr(self, "guidance_stage_indices", None)
        if indices is None:
            indices = tuple(
                index
                for index, module in enumerate(
                    (self.fuse_p3, self.fuse_p4, self.fuse_p5)
                )
                if isinstance(module, ConfiguredFeatureStage)
            )
        neck = self.model[self.neck_start] if (
            getattr(self, "use_efficient_hybrid_encoder", False)
            or getattr(self, "use_rtdetrv2_hye_yolo", False)
        ) else None
        required = getattr(neck, "required_guidance_stages", ())
        return tuple(sorted(set(int(index) for index in indices).union(required)))

    def train(self, mode: bool = True):
        """Keep a frozen enhancer in eval mode while training the detector."""
        super().train(mode)
        if (
            getattr(self, "use_slotdce_isaf", True)
            and getattr(self, "enhancer_frozen", False)
            and getattr(self, "enhancer_eval_when_frozen", True)
        ):
            self.enhancer.eval()
        return self

    def fuse(self, verbose: bool = True):
        """Fuse non-shared layers while retaining branch-specific backbone BN."""
        if getattr(self, "_slot_inference_fused", False):
            return self

        # Folding two different BN states into one tied convolution either
        # corrupts its weight or duplicates the complete backbone. Temporarily
        # exclude the dual backbone so the standard Ultralytics implementation
        # can still fuse the independent enhancer, selected neck and Detect head safely.
        backbone = self.dual_backbone
        self.model[1] = nn.Identity()
        try:
            super().fuse(verbose=False)
        finally:
            self.model[1] = backbone
        self._slot_inference_fused = True

        if verbose:
            LOGGER.info(
                "Slot-YOLO inference fusion kept the shared dual-backbone "
                "BatchNorm layers explicit and fused all independent layers."
            )
            self.info(verbose=True)
        return self

    def _enhance(
        self,
        image: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        if not getattr(self, "use_slotdce_isaf", True):
            raise RuntimeError("Slot-DCE enhancement was requested while use_slotdce_isaf=false")
        gradient_context = torch.no_grad() if self.enhancer_frozen else nullcontext()
        parameter = next(self.enhancer.parameters())
        force_fp32 = parameter.dtype == torch.float32 and image.device.type in {"cpu", "cuda"}
        guidance_stage_indices = self._runtime_guidance_stage_indices()
        uses_detector_guidance = bool(guidance_stage_indices)
        use_enhancer_amp = (
            getattr(self, "enhancer_amp", False)
            and force_fp32
            and image.device.type == "cuda"
        )
        if use_enhancer_amp:
            precision_context = torch.autocast(
                device_type="cuda",
                dtype=getattr(self, "enhancer_amp_dtype", torch.bfloat16),
            )
        elif force_fp32:
            precision_context = torch.autocast(
                device_type=image.device.type,
                enabled=False,
            )
        else:
            precision_context = nullcontext()
        enhancer_input = image.float() if force_fp32 else image
        height, width = image.shape[-2:]
        pyramid_sizes = tuple(
            (
                (height + stride - 1) // stride,
                (width + stride - 1) // stride,
            )
            for stride in (8, 16, 32)
        )
        guidance_sizes = tuple(
            pyramid_sizes[index] for index in guidance_stage_indices
        )
        with gradient_context, precision_context:
            use_checkpoint = bool(
                getattr(self, "checkpoint_enhancer", True)
                and self.training
                and not self.enhancer_frozen
                and torch.is_grad_enabled()
            )
            prepared_context = (
                self.enhancer.prepare_detection_context(enhancer_input)
                if use_checkpoint
                else None
            )

            def enhancer_forward(
                value: Tensor,
                _gradient_anchor: Tensor | None = None,
            ):
                return self.enhancer.enhance_for_detection(
                    value,
                    return_guidance=uses_detector_guidance,
                    guidance_sizes=(
                        guidance_sizes if uses_detector_guidance else None
                    ),
                    _prepared_context=prepared_context,
                )

            if use_checkpoint:
                # Reentrant checkpointing executes its first pass without
                # building an autograd graph and is materially faster for the
                # long Slot-DCE renderer. Detector images do not require
                # gradients, so a zero-cost scalar anchor satisfies the API;
                # all enhancer parameter gradients are produced normally by
                # the recomputed pass.
                gradient_anchor = enhancer_input.new_ones((), requires_grad=True)
                if uses_detector_guidance:
                    def flat_enhancer_forward(value: Tensor, anchor: Tensor):
                        enhanced, guidance = enhancer_forward(value, anchor)
                        return (enhanced, *guidance)

                    flat_result = checkpoint(
                        flat_enhancer_forward,
                        enhancer_input,
                        gradient_anchor,
                        use_reentrant=True,
                    )
                    result = (flat_result[0], tuple(flat_result[1:]))
                else:
                    result = checkpoint(
                        enhancer_forward,
                        enhancer_input,
                        gradient_anchor,
                        use_reentrant=True,
                    )
            else:
                result = enhancer_forward(enhancer_input)
        if not uses_detector_guidance:
            if not isinstance(result, Tensor):
                raise TypeError("Slot-DCE detection path must return enhanced RGB")
            return result.to(dtype=image.dtype), None
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "Slot-DCE detection path must return (enhanced, guidance pyramid)"
            )
        enhanced, detector_guidance = result
        if not isinstance(enhanced, Tensor) or not (
            isinstance(detector_guidance, tuple)
            and len(detector_guidance) == len(guidance_stage_indices)
            and all(isinstance(value, Tensor) for value in detector_guidance)
        ):
            raise TypeError("Slot-DCE returned an invalid detector guidance pyramid")
        return enhanced.to(dtype=image.dtype), tuple(
            value.to(dtype=image.dtype) for value in detector_guidance
        )

    def _forward_yolo_neck(self, p3: Tensor, p4: Tensor, p5: Tensor):
        """Execute the original PAN modules in their parsed YAML graph order."""
        features = {
            stage[-1].i: feature
            for stage, feature in zip(self.dual_backbone.stages[1:], (p3, p4, p5))
        }
        for index in range(self.neck_start, len(self.model) - 1):
            layer = self.model[index]
            sources = [layer.f] if isinstance(layer.f, int) else layer.f
            inputs = [features[source if source >= 0 else layer.i + source] for source in sources]
            features[layer.i] = layer(inputs[0] if isinstance(layer.f, int) else inputs)
        head = self.model[-1]
        return tuple(features[source if source >= 0 else head.i + source] for source in head.f)

    def _forward_neck(self, p3: Tensor, p4: Tensor, p5: Tensor, guidance=None):
        """Run the selected neck with the unchanged SGDC/Detect tensor interface."""

        if getattr(self, "use_ig_aifi", False) or getattr(self, "use_ig_fs_ccff", False):
            return tuple(self.model[self.neck_start](p3, p4, p5, guidance=guidance))
        if getattr(self, "use_efficient_hybrid_encoder", False):
            return tuple(self.model[self.neck_start](p3, p4, p5))
        if getattr(self, "use_rtdetrv2_hye_yolo", False):
            return tuple(self.model[self.neck_start](p3, p4, p5, guidance=guidance))
        return self._forward_yolo_neck(p3, p4, p5)

    def _forward_head(self, features: tuple[Tensor, Tensor, Tensor], batch=None):
        """Run the unchanged Detect head on one three-level pyramid."""

        if getattr(self, "is_rtdetr", False):
            return self.model[-1](list(features), batch)
        return self.model[-1](list(features))

    def _forward_neck_head(self, p3: Tensor, p4: Tensor, p5: Tensor, guidance=None):
        """Run the configured neck and unchanged Detect head."""

        return self._forward_head(self._forward_neck(p3, p4, p5, guidance=guidance))

    def _apply_sgdc(
        self,
        features: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply context-guided detail calibration to the selected neck pyramid."""

        return tuple(self.sgdc(*features))

    @staticmethod
    def _process_feature_stage(
        module: nn.Module,
        raw_feature: Tensor,
        enhanced_feature: Tensor,
        illumination_prior: Tensor | None,
    ) -> Tensor:
        if isinstance(module, ConfiguredFeatureStage):
            if illumination_prior is None:
                raise RuntimeError("The configured feature stage requires an illumination prior")
            return module(raw_feature, enhanced_feature, illumination_prior)
        return module(raw_feature, enhanced_feature)

    def _predict_once(self, x, profile=False, visualize=False, embed=None, batch=None):
        if getattr(self, "_building_base", False):
            return super()._predict_once(
                x,
                profile=profile,
                visualize=visualize,
                embed=embed,
            )

        fusion_guidance: list[Tensor | None] = [None, None, None]
        if getattr(self, "use_slotdce_isaf", True):
            enhanced_image, detector_guidance = self._enhance(x)
            raw_features, enhanced_features = self.dual_backbone(
                x,
                enhanced_image,
                gradient_checkpointing=getattr(self, "checkpoint_backbone", False),
            )
            guidance_stage_indices = self._runtime_guidance_stage_indices()
            for index, guidance in zip(
                guidance_stage_indices,
                detector_guidance or (),
            ):
                fusion_guidance[index] = guidance
            p3 = self._process_feature_stage(
                self.fuse_p3,
                raw_features[0],
                enhanced_features[0],
                fusion_guidance[0],
            )
            p4 = self._process_feature_stage(
                self.fuse_p4,
                raw_features[1],
                enhanced_features[1],
                fusion_guidance[1],
            )
            p5 = self._process_feature_stage(
                self.fuse_p5,
                raw_features[2],
                enhanced_features[2],
                fusion_guidance[2],
            )
        else:
            p3, p4, p5 = self.dual_backbone(
                x,
                gradient_checkpointing=getattr(self, "checkpoint_backbone", False),
            )
        if (getattr(self, "use_ig_aifi", False) or getattr(self, "use_ig_fs_ccff", False)
                or getattr(self, "use_ig_lsf_aifi", False)):
            neck_features = self._forward_neck(p3, p4, p5, guidance=fusion_guidance)
        else:
            neck_features = self._forward_neck(p3, p4, p5)
        if getattr(self, "sgdc_enabled", False):
            neck_features = self._apply_sgdc(neck_features)
        return self._forward_head(neck_features, batch=batch)
