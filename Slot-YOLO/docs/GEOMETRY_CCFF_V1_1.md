# Geometry CCFF v1.1

入口为 `slot_yolo/geometry_ccff_v1_1.py` 的 `GeometryAwareBiCCFF`，由 `rtdetrv2_hye_yolo.py` 在 `use_geometry_ccff_v1_1: true` 时构建。

1. IG-LSF-AIFI 处理 P5。`SemanticStructureP5P4` 将 P5 语义与 P4 结构做选择性融合，得到 `P4_td`。
2. `P4_td` 上采样后与 P3 拼接，经原生 CSPRep 得到 `P3_native`。
3. 几何场预测前景、无方向主轴、长短轴和路由量；从 `P4_td` 沿轴及横轴对称采样，修正 `P3_native`。正负采样点求均值，因此主轴的 θ 与 θ+π 等价。
4. P3 完整细节及几何门控细节共同反馈到 P4；P4 中选择的细节再更新 P5。
5. 训练时用 OBB 标签监督几何场；验证/推理时不计算该辅助损失。

输出为 P3/P4/P5，通道数保持 128/256/512。v1.1 与 v1 的结构不同，应使用各自的配置和 checkpoint。
