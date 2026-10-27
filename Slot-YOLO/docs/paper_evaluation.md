# 论文对比实验统一评估

在服务器已有的训练环境中直接运行普通 Python 脚本，无需 pytest、pycocotools 或新的测试框架：

```bash
cd /home/zhangyuhang/proj/ultralytics
python Slot-YOLO/run_paper_evaluation.py
```

当前默认依次评估 **OCM val、CMDPC val**，分别加载配置中的 best.pt，最后统一汇总。OCM 与旧 `test.py --config ./ex_config/config_RTDETRV2_hyE_yolo_geometry.yaml` 的实际划分保持一致。配置文件为 `Slot-YOLO/ex_config/paper_evaluation.yaml`；运行前确认 GPU 编号与数据根目录适合服务器。

```bash
# 指定 GPU，或只运行一个数据集
python Slot-YOLO/run_paper_evaluation.py --device 0
python Slot-YOLO/run_paper_evaluation.py --only OCM
# 真正评估 OCM 测试集；该结果不能与 val 的 0.803/0.418 混称为同一划分
python Slot-YOLO/run_paper_evaluation.py --only OCM --split test
# 明确复核旧 test.py 的验证集成绩
python Slot-YOLO/run_paper_evaluation.py --only OCM --split val
# 换配置即可加入其他方法
python Slot-YOLO/run_paper_evaluation.py --config /path/to/comparisons.yaml
```

每次运行创建独立目录 `Slot-YOLO/newoutput/paper_evaluation/evaluation_时间/`，避免覆盖历史结果。一个模型失败后仍尝试其余模型，最后返回非零退出码并列出失败原因。

## OCM 旧成绩与新成绩为什么不同

旧脚本虽然叫 `test.py`，配置中 `test.split` 实际为 `val`。用户提供的服务器日志明确读取 `labels/val.cache`，共 1371 张图、9785 个 GT，得到 mAP50=0.80333466、mAP50-95=0.41825700。此前论文脚本指定 `test`，共 1370 张图、10182 个 GT，并改为 batch=1、rect=false，因此不是相同的评估条件。

本地同一份 OCM 权重的三次完整原生验证复核如下；数值为 0～1，硬件/软件差异可能产生很小的数值变化：

| 数据划分 | batch | rect | 实际图像输入 H×W | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| val，1371 张 | 16 | true | 384×672 | 0.803269 | 0.418256 |
| test，1370 张 | 16 | true | 384×672 | 0.798424 | 0.385607 |
| test，1370 张 | 1 | false | 640×640 | 0.788354 | 0.381461 |

`imgsz=640` 在矩形填充模式下不是承诺每次输入均为 640×640；原生验证数据加载器按长宽比和 stride 补边。新配置的精度评估恢复 `batch=16, rect=true`，FPS 则另用 `batch=1, rect=false` 的完整遍历测量。两个阶段的实际尺寸、精度和图像集合哈希分别保存，测速阶段不会覆盖 AP。所有对比模型应统一各阶段配置，论文中的 val 和 test 必须按真实划分标注。

旧输出的 `AP=38.15` 就是 **mAP50-95=38.15%=0.3815**，此前名称不够明确；现在终端和汇总表直接显示 `mAP50-95(%)`，另有 0～1 的完整精度行，JSON 中也有 `overall_metrics_0_1` 别名。

## 指标与论文口径

| 输出 | 定义 |
| --- | --- |
| mAP50-95（JSON 内部键 AP） | IoU 阈值 0.50 到 0.95、步长 0.05；先对阈值平均，再对有 GT 的类别宏平均 |
| mAP50、mAP75 | 分别取 0.50、0.75 阈值下的类别宏平均 |
| 每类 AP50 | 每个类别单独在 0.50 阈值下的 AP；不是 `metrics.box.maps`，后者是每类 AP50-95 |
| APS、APM、APL | 分别对 S/M/L 组计算 AP50-95；不是 AP50，也不是先算每张图 AP 再平均 |
| Params(M) | checkpoint 模型在推理融合前的全部唯一参数量 / 1e6，包含增强、骨干、颈部和检测头 |
| GFLOPs | batch=1、完整 640×640 输入的一次 FP32 forward；分辨率取独立测速的 imgsz，1 MAC 按 2 FLOPs 计 |
| FPS(inference) | 独立测速阶段的 `1000 / native_validator.speed["inference"]`，默认论文表格使用这一列 |
| FPS(pipeline) | `1000 / (preprocess + inference + postprocess)`，三个时间均为原生验证器的 ms/image |
| Backbone | 配置声明并核对 checkpoint 的骨干类型；Slot-YOLO 为 YOLOv8s，C2f + SPPF |
| Epochs | 优先使用配置明确指定的实际训练轮数，否则读取权重运行目录的 results.csv；不把 best epoch 当作总训练轮数 |

表格和 CSV 中 AP 乘以 100，以百分数显示；JSON 保留 0～1 原始值。无 GT 的类别、尺寸组显示 `--` / JSON null，并从对应宏平均中排除；有 GT 但没有命中的 AP 为 0。

### CMDPC 的标注编号与检测头编号

CMDPC 训练时采用五类定义，编号 3 的 `Drill_pipec` 为空，编号 4 为 `Drill_tail`。用户之后将钻尾的标注编号改成了 3，导致旧五类配置把 2133 个钻尾误当作 `Drill_pipec`。现在按用户要求，将服务器上的标签文件恢复为训练时的五类编号。

上传 `restore_cmdpc_label_ids.py` 后，在服务器已有环境中运行（仅使用 Python 标准库，无需安装包）：

```bash
cd /home/zhangyuhang/proj/ultralytics/Slot-YOLO
python restore_cmdpc_label_ids.py
python run_paper_evaluation.py --only CMDPC --device 1
```

默认数据根目录是 `/home/zhangyuhang/dataset/Coal_mine/CMDPC_yolo_obb`，可用 `--root /实际目录` 覆盖；`--dry-run` 仅检查计数，不修改文件。运行转换期间不要同时训练、评估或修改这个数据集。

脚本处理 `labels/train`、`labels/val`，以及存在时的 `labels/test`：只将每行开头的类别 `3` 改成 `4`，其他类别和坐标原样保留。已有的 `4` 保持不变，因此可重复运行。执行前先完整验证所有标注，并将标签、原有 `data.yaml`、已有 `classes.txt` 和旧标签缓存打包到数据集旁边的 `CMDPC_yolo_obb_labels_backup_*.zip`，校验 ZIP 后才开始修改。更新数据集根目录的 `data.yaml` 为五类定义（如已有 `classes.txt` 也同步），清理 `.cache` / `.cache.npy` 标签缓存；写入使用逐文件原子替换，普通异常会尝试回滚，ZIP 始终保留。

恢复后的编号与验证集 GT 数量应为：

| 真实类别 | 恢复后的标注 ID | 模型 ID | 本次 val GT 数量 |
| --- | --- | --- | --- |
| Drill_body | 0 | 0 | 2159 |
| Drill_head | 1 | 1 | 2022 |
| Drill_pipe | 2 | 2 | 3910 |
| Drill_tail | 4 | 4 | 2133 |
| Drill_pipec（空输出类别） | 无 | 3 | 0 |

当前评估配置已切回 `datasets_config/CMDPC.yaml`，移除了额外的 `model_names`，标注与模型编号一致，不再执行先前的 `3 -> 4` 内存转换。`expected_empty_classes: [Drill_pipec]` 会在编号 3 仍有 GT 时停止评估，提示先恢复标签，防止再次生成错误指标。空类别显示 `--`，不参与 AP 宏平均。

通用评估器仍支持可选的 `model_names`，供其他确实需要按名称转换类别编号的实验使用；`class_mapping.json` 保存对应关系和计数。含非恒等映射信息的 `data.resolved.yaml` 是内部配置，不能绕过本脚本直接使用。恢复后的 CMDPC 不应再搭配四类 `CMDPC_2.yaml`；四类配置仅用于仍采用 `0,1,2,3` 编号的数据副本。

### 尺寸指标与速度

S/M/L 沿用 COCO 的面积阈值，单位为**原图像素平方**，与当前推理输入是否为 640 无关：

- S：0～32²，即 0～1024 px²。
- M：32²～96²，即 1024～9216 px²。
- L：96² 及以上，即 9216 px²及以上。

代码与 COCO 的 `areaRng` 一样使用闭区间，因此恰好等于 1024 或 9216 的目标可能计入相邻两组。OBB 使用原图旋转矩形 `width × height` 的面积，不使用包围旋转矩形的水平框面积，也不依据目标长边判组。若以后使用自定义尺寸阈值，应在论文中明确写出，所有模型共同重新评估。

**本脚本是 Ultralytics OBB 口径的尺寸扩展，不是原生 COCOeval。** 当前 OBB 使用 ProbIoU，原生 COCO bbox 使用水平框 IoU；COCO 的置信度优先匹配、AP 取样平均和默认 maxDets=100 也与这里的 YOLO 规则不同。这里将总 AP 与尺寸 AP 统一为：Ultralytics 默认 IoU 优先的一对一匹配、101 点插值后梯形积分、max_det=300。尺寸组先匹配组内 GT，再匹配组外 GT；组外 GT 的匹配预测被忽略，剩余未匹配且面积在组外的预测被忽略。每个 IoU 阈值单独处理忽略掩码。不能仅把其他尺寸 GT 删掉后，将检测到的这些目标全算作误检。

原生 YOLO 总 AP 也保存在 `metrics.json`。脚本会将其与固定核心的 AP 比较；差异超过 1e-4 时提示检查版本或适配层后再用于论文。若后续比较其他 OBB 项目，应同样输出 ProbIoU；水平框检测结果应另用 `task: detect` 和水平框数据单独比较。

速度来自单独一次 `model.val()` 原有的同步计时和预热流程，尺寸统计在它的计时区间之外。官方验证器直接给出的是各阶段延迟；这里明确给出两种 FPS 换算，避免只写“官方 FPS”造成歧义。这两种 FPS 均不包含磁盘读取、图像解码、指标累计和文件保存；pipeline 也不能称为完整应用端到端速度。所有方法的**测速阶段**固定同一 GPU、软件环境、640×640、batch=1、FP32、conf=0.001、NMS IoU=0.7、max_det=300、无 TTA。`speed_benchmark.rect: false` 保证测速输入为固定正方形；精度阶段使用独立的 `evaluation.batch: 16, evaluation.rect: true`。后处理阈值、数据划分、设备等由精度配置继承。

两次遍历同一划分，耗时比原先单次评估更长。若关闭 `speed_benchmark.enabled`，FPS 改为直接使用精度阶段的计时，日志会明确说明，这种 FPS 不能与 batch=1 固定方形测速的值混在同一论文列。旧日志的 8.7 ms 对应其 batch=16 矩形输入，不能用于声称 batch=1、640×640 的速度。外部配置没有 `speed_benchmark` 时也沿用精度阶段计时。

GFLOPs 默认使用 PyTorch `FlopCounterMode`，在完整分辨率下统计支持的算子，包括卷积和注意力矩阵乘法。不能将小分辨率的 FLOPs 简单按面积放大，注意力复杂度不一定线性。部分逐元素、采样、FFT 或自定义算子未被 profiler 覆盖，所以这是指定 profiler 口径的 FLOPs，而不是所有硬件指令数。所有对比模型使用同一种 profiler。失败会输出 `--` 并记录原因，不会用 0 冒充成功，也不会自动改用 THOP。`torch` 模式依赖现有 PyTorch 提供 `torch.utils.flop_counter`；服务器的 PyTorch 2.5.1 支持这一接口。若选择 `thop`，需使用环境中已有的 THOP，并留意其对 functional attention 的漏计。

## 输出文件

- `run.log`：与终端一致的简洁日志，运行中约每 20 秒提示一次进度。
- `summary.csv`、`summary.md`：两个数据集的汇总，包含数据划分、骨干及论文所需指标。
- `summary.tex`：每个数据集单独一张 booktabs 表格，与示例的列结构对应，增加 Backbone 列；主文档需加载 `booktabs`。
- `per_class_ap50.csv`：逐类 AP50 和 GT 数量。
- `protocol.json`、`config.snapshot.yaml`：共同评估口径、指标核心哈希和配置。
- 每个实验目录中的 `metrics.json`：原始数值、S/M/L 数量、逐类结果、两种 FPS、原生 YOLO 对照值、权重哈希、硬件及软件版本。`runtime` / `accuracy_settings` 记录精度阶段；`speed_runtime` / `speed_settings` / `speed_source` 记录测速阶段；`accuracy_speed_ms` 另存精度阶段的原生延迟用于核对。实际输入 batch、dtype 或固定方形尺寸偏离配置会报错，矩形模式按实记录 H/W。
- `data.resolved.yaml`、`evaluated_images.txt`：本次实际数据根目录、划分和图像清单。
- `class_mapping.json`：原始标注名称、模型名称、标签编号映射和映射前后的 GT 数量；映射在内存中应用。
- `raw.log`：原生库的详细输出；失败时额外保存 `error.txt`。

不同数据集分别展示，不跨数据集平均 AP。当前 OCM 默认配置和 CMDPC 的划分都是 val，表题与 Split 列按实输出；不能标成 test。需要 OCM test 时显式切换 `--only OCM --split test`。空类别显示 `--` 并排除宏平均，不会因类别为空而强制报错。

## 添加对比模型

同一个 Ultralytics 环境能加载的模型，只需复制 experiments 中一个条目，更改 id、method、weights、backbone；普通 YOLO 可将 `imports: []` 并删除 `expected_backbone`。数据集及 `evaluation` 保持一致。RT-DETR 可设置 `loader: ultralytics:RTDETR`；模型必须与所用数据的检测任务相符。

此文件同时支持较早版本的 tensor 预测和较新版本的 dict 预测，并根据环境选择 `half` 或 `quantize` 精度参数。若项目的验证器 API 继续变化，只调整标有 **ULTRALYTICS ADAPTER** 的部分。S/M/L 划分、匹配和 AP 积分集中在 **METRIC CORE**，不要因适配模型改变它。

完全不同的框架建议使用其独立 Python 环境，复制同一脚本与配置，并设置 `adapter: your_adapter:run`。函数接口：

```python
def run(spec, settings, run_dir, progress, raw_log):
    # 按 settings 评估精度；按 settings["speed_benchmark"] 的覆盖项另做测速。
    # records 只返回精度阶段预测，每张图一条，含无 GT/无预测图。
    # 必须按 spec["class_mapping"]["label_to_model_id"] 将原始 GT ID 转成模型 ID。
    return {
        "records": [{
            "image_id": "唯一图像路径",
            "gt_cls": ...,       # (G,) 类别 ID；顺序须对应 spec["names"]
            "pred_cls": ...,     # (D,)
            "confidence": ...,   # (D,) 置信度；应用公共 conf/NMS/max_det 后的预测
            "gt_area": ...,      # (G,) 原图像素面积
            "pred_area": ...,    # (D,) 原图像素面积
            "overlap": ...,      # (G,D)，obb 必须 ProbIoU；detect 必须水平框 IoU
        }],
        "overlap_kind": "ProbIoU", # 或 task=detect 时的 box_IoU
        "speed_ms": {"preprocess": ..., "inference": ..., "postprocess": ...}, # 每张图的毫秒数
        "speed_source": "separate_adapter_benchmark", # 开启独立测速时必须返回
        "speed_settings": {"imgsz": 640, "batch": 1, "rect": False, "half": False},
        "runtime": {"input_shapes_bchw": ..., "input_dtypes": ...}, # 精度阶段
        "speed_runtime": {"input_shapes_bchw": ..., "input_dtypes": ...}, # 测速阶段
        "complexity": {"parameters": ..., "gflops": ..., "flops_error": None},
        "metadata": {"backbone": "...", "epochs": ...},
        "environment": {"framework": "...", "device_name": "..."},
    }
```

`spec["resolved_data"]` 是本次实际使用的 YAML。相同框架通常无需写外部 adapter；不同框架的适配层需要验证预处理、后处理及计时范围相同。框架在 CPU 解码/缩放与 GPU preprocess 间的工作分配不同，pipeline FPS 的比较尤其需要注明这一点。`protocol.json` 中的 `kernel_sha256` 可检查指标核心是否一致；框架和版本仍需一并记录。不要把缺失的速度或 FLOPs 写成 0。

参考：

- [Ultralytics validation](https://docs.ultralytics.com/modes/val/)
- [Ultralytics validator timing](https://docs.ultralytics.com/reference/engine/validator/)
- [COCO area ranges and ignore rules](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

## OBB 预测宽高为负时的面积处理

部分 DFL-free 检测头（如 YOLO26）可能输出宽或高为负的预测框。原生 ProbIoU 用宽、高的平方计算重叠，因此本脚本保留原预测、置信度和匹配，仅在尺寸分组中按 `abs(w) * abs(h)` 计算预测框面积，再还原到原图尺度。不删除此类预测，不把面积截断为零。GT 宽高为负或数值非有限时仍报错。

日志中的 `OBB area audit` 显示这类预测的数量；每个 OBB 实验保存 `obb_geometry_audit.json`，包含涉及图像数、预测数及部分原始框示例，也写入 `metrics.json` 的 `runtime.obb_geometry`。`protocol.json` 记录 `obb_area_policy: absolute_predicted_sides_v1`。总 AP 的匹配与积分核心未改动；正宽高预测的面积结果不变。各对比项目已同步这一处理规则。
