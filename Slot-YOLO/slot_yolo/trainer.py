"""Ultralytics trainer adapter for Slot-YOLO."""

from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
from typing import Any, Mapping

import torch
from torch import nn, optim

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.optim import MuSGD
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

from .complexity import log_model_complexity
from .model import SlotYOLODetectionModel


def split_enhancer_optimizer_groups(
    optimizer: torch.optim.Optimizer,
    enhancer_parameters,
    learning_rate_scale: float,
    weight_decay: float | None = None,
) -> int:
    """Move enhancer parameters into groups with independent LR and decay."""

    enhancer_ids = {id(parameter) for parameter in enhancer_parameters}
    enhancer_parameter_count = 0
    original_groups = list(optimizer.param_groups)
    for group in original_groups:
        enhancer_group_parameters = [
            parameter
            for parameter in group["params"]
            if id(parameter) in enhancer_ids
        ]
        if not enhancer_group_parameters:
            continue
        group["params"] = [
            parameter
            for parameter in group["params"]
            if id(parameter) not in enhancer_ids
        ]
        enhancer_group = {
            key: value for key, value in group.items() if key != "params"
        }
        enhancer_group["params"] = enhancer_group_parameters
        enhancer_group["lr"] = float(group["lr"]) * learning_rate_scale
        if weight_decay is not None and float(group.get("weight_decay", 0.0)) > 0.0:
            enhancer_group["weight_decay"] = weight_decay
        enhancer_group["param_group"] = (
            f"enhancer_{group.get('param_group', 'parameters')}"
        )
        optimizer.add_param_group(enhancer_group)
        enhancer_parameter_count += sum(
            parameter.numel() for parameter in enhancer_group_parameters
        )

    optimizer.param_groups[:] = [
        group for group in optimizer.param_groups if group["params"]
    ]
    if enhancer_parameter_count == 0:
        raise RuntimeError("The optimizer contains no Slot-DCE enhancer parameters.")
    return enhancer_parameter_count


def collect_unique_optimizer_parameters(
    model: nn.Module,
    use_muon: bool = False,
) -> tuple[list[dict[str, nn.Parameter]], int]:
    """Collect optimizer parameters once, even when modules share a Parameter."""
    groups: list[dict[str, nn.Parameter]] = [{}, {}, {}, {}]
    normalizations = tuple(
        value for name, value in nn.__dict__.items() if "Norm" in name
    )
    seen: set[int] = set()
    duplicate_references = 0
    no_decay = set()
    for module_name, module in unwrap_model(model).named_modules():
        declaration = getattr(module, "no_weight_decay", None)
        if callable(declaration):
            no_decay.update(
                f"{module_name}.{name}" if module_name else name for name in declaration()
            )

    for module_name, module in unwrap_model(model).named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            parameter_id = id(parameter)
            if parameter_id in seen:
                duplicate_references += 1
                continue
            seen.add(parameter_id)

            fullname = (
                f"{module_name}.{parameter_name}" if module_name else parameter_name
            )
            # The bundled MuSGD orthogonalizes matrices and flattens only
            # 4D conv kernels. Other ranks (e.g. IG-AIFI's [heads,5,5] R)
            # must stay in its ordinary SGD groups, with the same LR/decay.
            muon_candidate = parameter.ndim in (2, 4)
            if getattr(unwrap_model(model), "module_version", "v1") == "v2" and parameter.ndim == 4:
                # Spatially broadcast gates (e.g. ISAF base_logits) are not
                # convolution matrices and must not be orthogonalized by Muon.
                muon_candidate = parameter_name == "weight"
            if fullname in no_decay:
                groups[1][fullname] = parameter
            elif muon_candidate and use_muon:
                groups[3][fullname] = parameter
            elif "bias" in fullname:
                groups[2][fullname] = parameter
            elif isinstance(module, normalizations) or "logit_scale" in fullname:
                groups[1][fullname] = parameter
            else:
                groups[0][fullname] = parameter
    return groups, duplicate_references


class SlotYOLOTrainer(DetectionTrainer):
    """Train Slot-YOLO with a task-matched HBB or OBB validation pipeline."""

    def __init__(
        self,
        *args,
        slot_config: Mapping[str, Any],
        **kwargs,
    ) -> None:
        self.slot_config = dict(slot_config)
        self.slot_task = str(self.slot_config.get("task", "detect")).lower()
        self.slot_task = "detect" if self.slot_task == "hbb" else self.slot_task
        if self.slot_task not in {"detect", "obb"}:
            raise ValueError("Slot-YOLO task must be detect/hbb or obb")
        self.use_slotdce_isaf = self.slot_config.get("use_slotdce_isaf", True)
        if not isinstance(self.use_slotdce_isaf, bool):
            raise TypeError("use_slotdce_isaf must be true or false")
        overrides = dict(kwargs.get("overrides") or {})
        overrides["task"] = self.slot_task
        kwargs["overrides"] = overrides
        enhancer_config = dict(self.slot_config.get("enhancer", {}))
        self.enhancer_permanently_frozen = bool(
            enhancer_config.get("freeze", False)
        )
        self.enhancer_freeze_epochs = int(
            enhancer_config.get("freeze_epochs", 0)
        )
        self.enhancer_lr_scale = float(
            enhancer_config.get("lr_scale", 0.05)
        )
        enhancer_decay = enhancer_config.get("weight_decay")
        self.enhancer_weight_decay = (
            None if enhancer_decay is None else float(enhancer_decay)
        )
        self._enhancer_is_frozen = None
        self._complexity_logged = False
        super().__init__(*args, **kwargs)
        self.add_callback(
            "on_train_epoch_start", self._update_enhancer_training_stage
        )
        self.add_callback("on_train_epoch_end", self._log_sgdc_parameters)
        self.add_callback(
            "on_pretrain_routine_end", self._log_model_complexity
        )

    def _configure_versioned_ema(self) -> None:
        """Keep legacy EMA behavior for v1 and deduplicate tied weights in v2."""
        if getattr(self, "slot_config", {}).get("module_version", "v1") == "v2" and self.ema is not None:
            from .myconfigv2.training import UniqueModelEMA
            if not isinstance(self.ema, UniqueModelEMA):
                self.ema = UniqueModelEMA.from_existing(self.ema)

    def _setup_train(self):
        super()._setup_train()
        self._configure_versioned_ema()

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        """Build an official-compatible optimizer without duplicate aliases."""
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and "
                f"'momentum={self.args.momentum}' and determining them "
                "automatically..."
            )
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = (
                ("MuSGD", 0.01, 0.9)
                if iterations > 10000
                else ("AdamW", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0

        optimizers = {
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "SGD",
            "MuSGD",
            "auto",
        }
        name = {item.lower(): item for item in optimizers}.get(
            str(name).lower(), str(name)
        )
        use_muon = name == "MuSGD"
        groups, duplicate_references = collect_unique_optimizer_parameters(
            model,
            use_muon=use_muon,
        )

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer_args = {
                "lr": lr,
                "betas": (momentum, 0.999),
                "weight_decay": 0.0,
            }
        elif name == "RMSProp":
            optimizer_args = {"lr": lr, "momentum": momentum}
        elif name in {"SGD", "MuSGD"}:
            optimizer_args = {"lr": lr, "momentum": momentum, "nesterov": True}
        else:
            raise NotImplementedError(
                f"Optimizer {name!r} is not available; choose one of "
                f"{sorted(optimizers)}"
            )

        parameter_group_sizes = [
            len(groups[0]),
            len(groups[1]),
            len(groups[2]),
        ]
        if not use_muon:
            groups = [dict(group) for group in groups[:3]]
        groups[2] = {
            "params": groups[2],
            **optimizer_args,
            "param_group": "bias",
        }
        groups[0] = {
            "params": groups[0],
            **optimizer_args,
            "weight_decay": decay,
            "param_group": "weight",
        }
        groups[1] = {
            "params": groups[1],
            **optimizer_args,
            "weight_decay": 0.0,
            "param_group": "bn",
        }

        muon_fraction, sgd_fraction = 0.2, 1.0
        if use_muon:
            import re

            parameter_group_sizes[0] = len(groups[3])
            groups[3] = {
                "params": groups[3],
                **optimizer_args,
                "weight_decay": decay,
                "use_muon": True,
                "param_group": "muon",
            }
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment")
            if getattr(unwrap_model(model), "module_version", "v1") == "v2":
                from .myconfigv2.training import classification_parameter_ids
                classification_ids = classification_parameter_ids(model)
                def uses_high_lr(fullname, parameter):
                    return id(parameter) in classification_ids
            else:
                def uses_high_lr(fullname, parameter):
                    return bool(pattern.search(fullname))
            split_groups = []
            for group in groups:
                parameters = group.pop("params")
                high_lr = [
                    parameter
                    for fullname, parameter in parameters.items()
                    if uses_high_lr(fullname, parameter)
                ]
                standard_lr = [
                    parameter
                    for fullname, parameter in parameters.items()
                    if not uses_high_lr(fullname, parameter)
                ]
                split_groups.extend(
                    (
                        {"params": high_lr, **group, "lr": lr * 3},
                        {"params": standard_lr, **group},
                    )
                )
            groups = split_groups
        else:
            for group in groups:
                group["params"] = list(group["params"].values())

        optimizer_factory = getattr(
            optim,
            name,
            partial(MuSGD, muon=muon_fraction, sgd=sgd_fraction),
        )
        optimizer = optimizer_factory(params=groups)
        slot_model = unwrap_model(model)
        enhancer_parameter_count = 0
        if getattr(self, "use_slotdce_isaf", True):
            enhancer_parameter_count = split_enhancer_optimizer_groups(
                optimizer,
                slot_model.enhancer.parameters(),
                self.enhancer_lr_scale,
                self.enhancer_weight_decay,
            )
        LOGGER.info(
            "%s %s(lr=%g, momentum=%g) with %d BN/no-decay, %d decay, "
            "%d bias parameter tensors; removed %d shared aliases.",
            colorstr("optimizer:"),
            type(optimizer).__name__,
            lr,
            momentum,
            parameter_group_sizes[1],
            parameter_group_sizes[0],
            parameter_group_sizes[2],
            duplicate_references,
        )
        if getattr(self, "use_slotdce_isaf", True):
            LOGGER.info(
                "Slot-DCE optimizer groups: %s parameters, lr=%g x %g = %g, weight_decay=%s",
                f"{enhancer_parameter_count:,}",
                lr,
                self.enhancer_lr_scale,
                lr * self.enhancer_lr_scale,
                self.enhancer_weight_decay,
            )
        else:
            LOGGER.info("Slot-DCE + ISAF disabled; using the raw-image backbone path.")
        return optimizer

    def _load_checkpoint_state(self, checkpoint) -> None:
        """Restore a checkpoint, tolerating optimizer groups from the old alias bug."""
        optimizer_restored = False
        if checkpoint.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                optimizer_restored = True
            except (ValueError, RuntimeError) as error:
                LOGGER.warning(
                    "Legacy Slot-YOLO optimizer state is incompatible with the "
                    "deduplicated parameter groups and will be reinitialized. "
                    "Model, EMA, epoch and best fitness are still restored. (%s)",
                    error,
                )
        if optimizer_restored and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.ema and checkpoint.get("ema"):
            self.ema = ModelEMA(self.model)
            self.ema.ema.load_state_dict(
                checkpoint["ema"].float().state_dict(),
                strict=True,
            )
            self.ema.updates = checkpoint["updates"]
        self.best_fitness = checkpoint.get("best_fitness", 0.0)
        self._configure_versioned_ema()

    def _set_enhancer_frozen(self, frozen: bool) -> None:
        """Switch Slot-DCE between fixed preprocessing and low-LR fine-tuning."""

        slot_model = unwrap_model(self.model)
        if not getattr(self, "use_slotdce_isaf", True):
            slot_model.enhancer_frozen = False
            self._enhancer_is_frozen = False
            return
        slot_model.enhancer_frozen = bool(frozen)
        for parameter in slot_model.enhancer.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            slot_model.enhancer.eval()
        else:
            slot_model.enhancer.train()
        if self._enhancer_is_frozen != frozen:
            if frozen:
                if self.enhancer_permanently_frozen:
                    LOGGER.info("Slot-DCE permanently frozen.")
                else:
                    LOGGER.info(
                        "Slot-DCE frozen for detector warm-up (epochs 1-%d).",
                        self.enhancer_freeze_epochs,
                    )
            else:
                LOGGER.info(
                    "Slot-DCE unfrozen at epoch %d with lr scale %g.",
                    self.epoch + 1 if hasattr(self, "epoch") else self.start_epoch + 1,
                    self.enhancer_lr_scale,
                )
            self._enhancer_is_frozen = frozen

    def _update_enhancer_training_stage(self, _trainer) -> None:
        """Unfreeze Slot-DCE when the configured detector warm-up is complete."""

        if not getattr(self, "use_slotdce_isaf", True):
            return
        should_freeze = self.enhancer_permanently_frozen or (
            self.epoch < self.enhancer_freeze_epochs
        )
        self._set_enhancer_frozen(should_freeze)

    def _log_sgdc_parameters(self, _trainer) -> None:
        """Record SGDC residual, gate and directional branch parameters."""

        model = unwrap_model(self.model)
        if not getattr(model, "sgdc_enabled", False) or RANK not in {-1, 0}:
            return
        state = model.sgdc.learnable_state()
        if getattr(model.sgdc, "variant", "sgdc") == "sgdcv2":
            LOGGER.info("SGDCv2 epoch %d: %s", self.epoch + 1, state)
            return
        LOGGER.info(
            "SGDC epoch %d: gamma(p3/p4)=%.4f/%.4f, "
            "gate(p3/p4)=%.3f/%.3f, "
            "dir_p3(local/h/v/d)=%.3f/%.3f/%.3f/%.3f, "
            "dir_p4(local/h/v/d)=%.3f/%.3f/%.3f/%.3f, "
            "adapter(p3/p4)=%.5f/%.5f",
            self.epoch + 1,
            state["gamma_p3"],
            state["gamma_p4"],
            state["gate_p3"],
            state["gate_p4"],
            state["dir_p3_local"],
            state["dir_p3_horizontal"],
            state["dir_p3_vertical"],
            state["dir_p3_dilated"],
            state["dir_p4_local"],
            state["dir_p4_horizontal"],
            state["dir_p4_vertical"],
            state["dir_p4_dilated"],
            state["adapter_p3"],
            state["adapter_p4"],
        )

    def _log_model_complexity(self, _trainer) -> None:
        """Write total and module-level complexity once before training starts."""

        if self._complexity_logged or RANK not in {-1, 0}:
            return
        try:
            log_model_complexity(self.model, imgsz=self.args.imgsz)
        except Exception as error:  # profiling must never stop a training run
            LOGGER.warning("Slot-YOLO complexity profiling skipped: %s", error)
        self._complexity_logged = True

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Build a fresh detector, or restore the complete model when resuming."""
        current_task = getattr(
            self,
            "slot_task",
            str(self.slot_config.get("task", "detect")).lower(),
        )
        current_task = "detect" if current_task == "hbb" else current_task
        if isinstance(weights, SlotYOLODetectionModel):
            weights.validate_neck_selection(self.slot_config)
            checkpoint_task = weights.slot_task_name()
            if checkpoint_task != current_task:
                raise ValueError(
                    f"Cannot resume a {checkpoint_task!r} Slot-YOLO checkpoint "
                    f"with task={current_task!r}. Use a checkpoint trained "
                    "with the same HBB/OBB head."
                )
            resume_config = deepcopy(
                getattr(weights, "slot_config", self.slot_config)
            )
            resume_config["task"] = current_task
            # Memory controls are runtime behavior rather than learned model
            # structure. Always honor the current experiment config so an old
            # checkpoint can opt into newly added activation checkpointing.
            resume_config["memory"] = deepcopy(
                self.slot_config.get("memory", {})
            )
            enhancer_config = dict(resume_config.get("enhancer", {}))
            enhancer_architecture = getattr(
                weights,
                "enhancer_architecture",
                None,
            )
            if enhancer_architecture is not None:
                enhancer_config.pop("architecture_config", None)
                enhancer_config["architecture"] = deepcopy(enhancer_architecture)
            enhancer_config["pretrained"] = False
            enhancer_config["weights"] = None
            resume_config["enhancer"] = enhancer_config

            model = SlotYOLODetectionModel(
                cfg=cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                slot_config=resume_config,
            )
            model.load_state_dict(weights.state_dict(), strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.criterion = None
            LOGGER.info(
                "Rebuilt a trainable Slot-YOLO graph and restored checkpoint "
                "weights without reusing the frozen EMA object."
            )
            return self.set_model_names_for_load(model)
        if weights is not None:
            raise ValueError(
                "Slot-YOLO detection weights must start from scratch. "
                "Use train.resume for a complete Slot-YOLO checkpoint."
            )
        return self.set_model_names_for_load(
            SlotYOLODetectionModel(
                cfg=cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                slot_config=self.slot_config,
            )
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        from backbone_config import is_rtdetr_config
        if is_rtdetr_config(self.slot_config):
            from ultralytics.models.rtdetr.train import RTDETRTrainer
            return RTDETRTrainer.build_dataset(self, img_path, mode, batch)
        return super().build_dataset(img_path, mode, batch)

    def get_validator(self):
        """Use the validator and loss labels matching the selected box type."""
        from backbone_config import is_rtdetr_config
        if is_rtdetr_config(getattr(self, "slot_config", {})):
            from ultralytics.models.rtdetr.val import RTDETRValidator
            self.loss_names = ("giou_loss", "cls_loss", "l1_loss")
            return RTDETRValidator(self.test_loader, save_dir=self.save_dir,
                                   args=copy(self.args), _callbacks=self.callbacks)
        if self.slot_task == "obb":
            self.loss_names = (
                "box_loss",
                "cls_loss",
                "dfl_loss",
                "angle_loss",
            )
            neck = self.slot_config.get("rtdetrv2_hye_yolo", {})
            if isinstance(neck, dict) and (
                neck.get("use_geometry_ccff", False)
                or neck.get("use_geometry_ccff_v1_1", False)
            ):
                self.loss_names += ("geometry_loss",)
            return yolo.obb.OBBValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
        )
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def final_eval(self):
        """Validate best.pt in FP32 so Slot-DCE keeps its configured precision."""

        previous = getattr(self.validator.args, "quantize", None)
        self.validator.args.quantize = None
        try:
            return super().final_eval()
        finally:
            self.validator.args.quantize = previous
