# Slot-YOLO（OCM OBB）

当前保留两种颈部：Geometry CCFF v1 和 v1.1。共同结构为 **YOLOv8s + SlotDCE + ISAF + RTDETRV2_hyE_yolo + IG-LSF-AIFI + Geometry CCFF**，检测任务固定为 OCM 旋转框（OBB）。

| 版本 | 配置 | 颈部实现 |
| --- | --- | --- |
| v1 | `ex_config/config_RTDETRV2_hyE_yolo_geometry.yaml` | 原生 CCFF 的 P3 后接 OBB 几何引导融合 |
| v1.1 | `ex_config/config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml` | P5→P4→P3→P4→P5 双向几何融合 |

两版的 SlotDCE、ISAF、IG-LSF-AIFI、YOLOv8s、数据划分和训练参数相同。颈部结构和 checkpoint 元数据不同，不能跨版本续训或加载权重。

## 运行

在 Ultralytics 项目根目录执行：

```powershell
python Slot-YOLO/train.py --config config_RTDETRV2_hyE_yolo_geometry.yaml
python Slot-YOLO/train.py --config config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml
python Slot-YOLO/test.py --config config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml --weights D:\path\to\best.pt
python Slot-YOLO/check_model.py --config config_RTDETRV2_hyE_yolo_geometry_v1_1.yaml
```

配置中 `train.device` 等参数应按实际设备调整。`datasets_config/OCM.yaml` 指向 OCM 数据集；`slot_pt/best_loss.pth` 是 SlotDCE 初始化权重。训练脚本默认采用 v1.1 配置，切换到 v1 时显式传入对应配置。

## 主要文件

- `slot_yolo/myconfigv2/slotdce/`：SlotDCE 实现与结构配置。
- `slot_yolo/myconfigv2/isaf.py`、`feature_pipeline.py`：双分支特征融合。
- `slot_yolo/ig_lsf_aifi.py`：P5 频域与局部增强注意力。
- `slot_yolo/geometry_ccff.py`、`geometry_ccff_v1_1.py`：两版几何融合。
- `slot_yolo/model.py`、`trainer.py`：检测模型与训练接口。
- `rtdetrv2_hye_config.py`：颈部配置校验和 checkpoint 元数据归一化。

Geometry CCFF v1 的设计记录见 `docs/GEOMETRY_CCFF.md`；v1.1 的数据流见 `docs/GEOMETRY_CCFF_V1_1.md`。
