"""Validate the retained v2 Slot-YOLO implementation."""


def bind_config_version(config, path):
    """Persist the single supported module version in loaded configs."""
    config["module_version"] = module_version(config)
    return config["module_version"]


def module_version(config):
    version = config.get("module_version", "v2")
    if version != "v2":
        raise ValueError("The retained model requires module_version: v2")
    if config.get("model_type", "slot_yolo") != "slot_yolo":
        raise ValueError("module_version=v2 requires model_type=slot_yolo")
    return version
