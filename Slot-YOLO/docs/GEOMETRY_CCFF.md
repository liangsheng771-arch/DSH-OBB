# Geometry CCFF v1

入口为 `slot_yolo/geometry_ccff.py` 的 `GeometryGuidedP3Fusion`。在 `rtdetrv2_hye_yolo.py` 中，IG-LSF-AIFI 先处理 P5，原生 CCFF 生成 P3/P4/P5；几何融合只修正 P3，P4/P5 继续沿原生 PAN 路径传递。

## P3 处理

1. 从原生 P3 预测六通道几何场：前景热度、双角度主轴向量、长轴、短轴及路由量。
2. 将 P4 语义降到 P3 通道数，沿预测主轴和横轴分别采样一对正负位置。每对取均值，保持 θ 与 θ+π 等价。
3. 拼接原生 P3、上采样 P4 语义、沿轴证据及沿轴与横轴的差异，经卷积生成残差。
4. 由前景、路由、长短轴差异和面积计算门控，只在相关区域向原生 P3 注入残差。最后一层零初始化，训练起点保持原生 P3。

训练时用 OBB 标签生成热度、主轴及尺度目标，辅助损失监督几何场；推理时不需要标签，也不计算辅助损失。完整配置为 `ex_config/config_RTDETRV2_hyE_yolo_geometry.yaml`。v1.1 的双向 P3/P4/P5 反馈见 `GEOMETRY_CCFF_V1_1.md`。
