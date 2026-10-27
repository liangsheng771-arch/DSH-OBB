"""YOLOv8s OBB with SlotDCE, ISAF and the selected RT-DETRv2 neck."""

from .model import SlotYOLODetectionModel
from .rtdetrv2_hye_yolo import RTDETRV2_hyE_yolo
from .trainer import SlotYOLOTrainer

__all__ = ("RTDETRV2_hyE_yolo", "SlotYOLODetectionModel", "SlotYOLOTrainer")
