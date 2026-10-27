# SlotDCE Detector-Lite

当前项目只保留 `slotdce_v9_8_detector_lite`。模型面向矿井非均匀照明与检测器前置增强，使用无配对矿井图像训练；训练和推理执行完全相同的 `model(image)` 前向路径，不包含手工推理后处理、快速重建旁路或旧版本兼容逻辑。

## 数据流

1. 输入 RGB 计算亮度、低频照明、白光/灯芯/光晕连续权重。
2. `PixelUnshuffle(4)` 无损转换到 H/4，深度可分离编码器提取共享特征。
3. 自适应曝光先验根据输入亮度直方图生成五个有序曝光区间；Slot 路由聚合特征并预测五个方向受限的强度。
4. 共享预测干路产生暗部、亮场、白光、光晕和多尺度细节控制量。
5. 低频色调渲染负责暗部提亮和高亮压暗，三尺度 Retinex 细节在全分辨率重建。
6. 训练期使用同一批无配对图像合成局部过曝，恢复头学习被截断区域的颜色与纹理置信度。

模型始终约束 S1/S2 只能提亮、S3 中性、S4/S5 只能压暗，避免可学习残差翻转曝光方向。

## 损失

总损失只包含四项：

```text
L = L_tone + 0.25 L_texture + 0.50 L_recovery + 0.05 L_smoothness
```

- `tone`：监督暗部提亮、普通亮场/白光/灯芯压暗和光晕衰减。
- `texture`：保持可靠区域的对数亮度梯度，并轻微强调高亮细节。
- `recovery`：用合成过曝前的当前图像监督恢复 RGB 与置信度。
- `smoothness`：在不跨越真实边缘的前提下约束色调场和光晕场。

损失的曝光阈值、最低压暗尺度、光晕限制和纹理可靠范围直接复用模型配置，不维护第二套同义参数。

## 训练

```bash
python train.py \
  --config config/slotDCE_v9_8_detector_lite.yaml \
  --data /path/to/train/images \
  --output /path/to/run
```

可用 `--weights` 从同一架构 checkpoint 初始化并重置优化器，或用 `--resume` 精确续训。旧架构 checkpoint 不再自动迁移。

训练采样由 YAML 中 `training.exposure_sampling` 控制，包括完整图缩放、普通随机裁剪、白光感知裁剪和混合曝光裁剪；找不到目标区域时自动回退到随机裁剪。

## 推理

```bash
python infer.py \
  --config config/slotDCE_v9_8_detector_lite.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --input /path/to/images \
  --output outputs/infer
```

`inference.max_side` 仅控制送入模型和保存结果的图像尺寸；模型内部数据流与训练一致。

## 检测器集成

```python
from models import SlotDCEDetectionAdapter

enhancer_layer = SlotDCEDetectionAdapter(slotdce_model)
enhanced_images = enhancer_layer(images)
```

当前模型包含 68,819 个可学习参数。所有学习型空间卷积都位于 H/4 网格，恢复头同样使用 `PixelUnshuffle(4)`，适合直接作为检测器的可微分前置模块。
