"""Train the retained Slot-YOLO OCM OBB model."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from common import ROOT, load_config, ultralytics_args, validate_data_yaml  # noqa: E402
from backbone_config import is_rtdetr_config, validate_backbone_runtime  # noqa: E402
from slot_yolo import SlotYOLOTrainer  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg, get_save_dir  # noqa: E402
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER  # noqa: E402


class _TeeStream:
    """Mirror a console stream to a UTF-8 training log without hiding live output."""

    def __init__(self, console, logfile) -> None:
        self.console = console
        self.logfile = logfile

    def write(self, value: str) -> int:
        if value:
            self.console.write(value)
            self.logfile.write(value)
            self.console.flush()
            self.logfile.flush()
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def _quiet_known_cuda_determinism_warnings():
    """Filter known CUDA reproducibility notices without changing computation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^(?:cumsum_cuda_kernel|adaptive_avg_pool2d_backward_cuda|"
                r"reflection_pad2d_backward_cuda) does not have a deterministic "
                r"implementation, but you set 'torch\.use_deterministic_algorithms"
                r"\(True, warn_only=True\)'\."
            ),
            category=UserWarning,
        )
        yield


@contextmanager
def _capture_training_log(log_path: Path):
    """Capture stdout, stderr and Ultralytics LOGGER output in ``train.log``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logfile = log_path.open("a", encoding="utf-8", buffering=1)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(file_handler)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, logfile)
    sys.stderr = _TeeStream(original_stderr, logfile)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        LOGGER.removeHandler(file_handler)
        file_handler.close()
        logfile.close()


def _reserve_save_dir(train_args: dict) -> Path:
    """Resolve the same output directory Ultralytics will use before creating the trainer."""
    effective_args = get_cfg(DEFAULT_CFG_DICT, overrides=dict(train_args))
    save_dir = get_save_dir(effective_args)
    train_args["save_dir"] = str(save_dir)
    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "ex_config" / "config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml"),
        help="experiment YAML path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, config = load_config(args.config)
    validate_backbone_runtime(config)
    validate_data_yaml(config["data"])

    train_args = ultralytics_args(config["train"])
    train_args.update(
        data=config["data"],
        task=config["task"],
    )
    is_slot_yolo = config["model_type"] == "slot_yolo"
    if is_slot_yolo:
        train_args["model"] = config["model"]
    if is_slot_yolo and train_args.get("pretrained") not in (None, False):
        raise ValueError("train.pretrained must be false: Slot-YOLO detector training starts from scratch")
    device = train_args.get("device")
    if is_slot_yolo and isinstance(device, (list, tuple)) and len(device) > 1:
        raise NotImplementedError("The first Slot-YOLO integration supports one training process/GPU")

    save_dir = _reserve_save_dir(train_args)
    with _capture_training_log(save_dir / "train.log"), _quiet_known_cuda_determinism_warnings():
        print(f"Config          : {config_path}")
        print(f"Model type      : {config['model_type']}")
        print(f"Task            : {config['task']} ({'OBB' if config['task'] == 'obb' else 'HBB'})")
        print(f"Model           : {config['model']}")
        if config.get("backbone"):
            print(f"Backbone / head : {config['backbone']} (paired)")
        print(f"Dataset         : {config['data']}")
        if is_slot_yolo:
            print(f"Module version  : {config.get('module_version', 'v1')}")
            slotdce_isaf_enabled = bool(config.get("use_slotdce_isaf", True))
            enhancer = config.get("enhancer", {})
            print(f"Slot-DCE + ISAF : {'ON' if slotdce_isaf_enabled else 'OFF'}")
            if slotdce_isaf_enabled:
                print(f"Slot-DCE weights: {enhancer.get('weights')}")
                print(f"Enhancer init   : {'pretrained' if enhancer.get('pretrained', True) else 'random'}")
                print(f"Enhancer frozen : {enhancer.get('freeze', False)}")
                print(f"Freeze epochs   : {enhancer.get('freeze_epochs', 0)}")
                print(f"Enhancer LR     : x{enhancer.get('lr_scale', 0.05)}")
                print(f"Enhancer decay  : {enhancer.get('weight_decay', 'detector default')}")
            if config.get("use_rtdetrv2_hye_yolo", False):
                neck = "RTDETRV2_hyE_yolo"
            elif config.get("use_efficient_hybrid_encoder", False):
                neck = "efficient_hybrid_encoder"
            else:
                neck = "yolo"
            print(f"Neck            : {str(neck).upper()}")
            if config.get("use_rtdetrv2_hye_yolo", False):
                from rtdetrv2_hye_config import ccff_variant_name
                neck_settings = config["rtdetrv2_hye_yolo"]
                print(f"AIFI variant    : {'IGLSFAIFI' if neck_settings['use_ig_lsf_aifi'] else 'native AIFI'}")
                print(f"CCFF variant    : {ccff_variant_name(neck_settings)}")
            print(f"IG-AIFI         : {'ON' if config.get('use_ig_aifi', False) else 'OFF'}")
            print(f"IG-FS-CCFF      : {'ON' if config.get('use_ig_fs_ccff', False) else 'OFF'}")
            print(f"P5 dual branch  : {'ON' if config.get('use_p5_dual_branch', False) else 'OFF'}")
            modules = config.get("feature_modules", {})
            sgdc = modules.get("sgdc", {}) if isinstance(modules, dict) else {}
            sgdc_enabled = bool(config.get("use_sgdc", False))
            sgdc_name = str(sgdc.get("name", "sgdc")).upper() if isinstance(sgdc, dict) else "SGDC"
            sgdc_placement = (
                str(sgdc.get("placement", "post_neck")).replace("_", "-")
                if isinstance(sgdc, dict)
                else "post-neck"
            )
            print(
                "SGDC            : "
                f"{sgdc_name + ' ON (' + sgdc_placement + ')' if sgdc_enabled else 'OFF'}"
            )
            memory = config.get("memory", {})
            dce_checkpoint = bool(memory.get("checkpoint_enhancer", True))
            print(
                "Checkpointing    : "
                f"DCE={'selective' if dce_checkpoint else 'off'}, "
                f"backbone={memory.get('checkpoint_backbone', False)}"
            )
            enhancer_amp = bool(memory.get("enhancer_amp", False))
            enhancer_precision = (
                str(memory.get("enhancer_amp_dtype", "bfloat16")).upper()
                if enhancer_amp
                else "FP32"
            )
            precision = (
                f"{'AMP detector' if train_args.get('amp') else 'FP32 detector'} + "
                f"{enhancer_precision} Slot-DCE"
            )
        else:
            precision = "AMP" if train_args.get("amp") else "FP32"
        loss_description = (
            "RT-DETR matching + denoising (GIoU / classification / L1)"
            if is_rtdetr_config(config)
            else ("oriented-box" if config["task"] == "obb" else "HBB") + " detection only"
        )
        print(f"Training loss   : {loss_description}")
        print(f"Precision       : {precision}")
        print(f"Batch / epochs  : {train_args.get('batch')} / {train_args.get('epochs')}")
        print(
            "Data pipeline   : "
            f"cache={train_args.get('cache', False)}, "
            f"workers={train_args.get('workers', 8)}"
        )
        print(f"torch.compile   : {train_args.get('compile', False)}")
        print("CUDA warnings   : known determinism notices filtered; reproducibility limitations still apply")
        print(f"Output          : {save_dir}")
        print(f"Training log    : {save_dir / 'train.log'}")

        if is_slot_yolo:
            trainer = SlotYOLOTrainer(overrides=train_args, slot_config=config)
            trainer.train()
        else:
            if is_rtdetr_config(config):
                from ultralytics.models.rtdetr import RTDETR
                model = RTDETR(config["model"])
            else:
                model = YOLO(config["model"], task=config["task"])
            from ultralytics.nn.modules.head import OBB

            head_task = "obb" if isinstance(model.model.model[-1], OBB) else "detect"
            if head_task != config["task"]:
                raise ValueError(
                    f"Baseline task={config['task']!r} does not match the "
                    f"{head_task!r} head in {config['model']!r}"
                )
            model.train(**train_args)


if __name__ == "__main__":
    main()
