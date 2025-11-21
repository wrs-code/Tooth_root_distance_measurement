# 牙齿实例分割功能集成说明

本文档说明如何使用新集成的Instance_seg_teeth牙齿实例分割功能。

## 简介

我们已经将[Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)仓库的OralBBNet方法集成到本项目中。该方法结合了YOLOv8目标检测和U-Net语义分割，在牙齿实例分割任务上取得了出色的性能。

### 性能指标（UFBA-425数据集）

| 牙齿类型 | Dice系数 |
|---------|---------|
| 门牙    | 89.34%  |
| 犬齿    | 88.40%  |
| 前磨牙  | 88.38%  |
| 磨牙    | 87.87%  |

## 项目结构

```
Tooth_root_distance_measurement/
├── Instance_seg_teeth/          # 克隆的原始仓库（用于参考）
├── test_instance_seg_teeth/     # 整合的实例分割模块
│   ├── __init__.py             # 模块初始化
│   ├── unet_model.py           # U-Net模型定义
│   ├── teeth_segmentation.py  # 主要分割类
│   ├── demo.py                 # 演示脚本
│   ├── download_models.py      # 模型下载工具
│   ├── requirements.txt        # 依赖列表
│   ├── README.md              # 模块文档
│   └── MODEL_DOWNLOAD.md      # 模型下载说明
├── test/
│   └── test_instance_seg_teeth.py  # 测试脚本
└── models/                     # 模型存放目录
    ├── yolov8_teeth.pt        # YOLOv8模型（需要下载/训练）
    └── unet_teeth.h5          # U-Net模型（需要训练）
```

## 快速开始

### 1. 安装依赖

```bash
# 安装项目依赖（已包含牙齿分割所需的所有依赖）
pip install -r requirements.txt
```

### 2. 准备模型

#### 选项A: 下载YOLOv8基础模型（仅用于测试）

```bash
python test_instance_seg_teeth/download_models.py
```

这将下载YOLOv8的基础模型。注意：基础模型并非专门为牙齿检测训练，效果有限。

#### 选项B: 训练专用模型（推荐）

1. **下载UFBA-425数据集**
   - 访问: https://figshare.com/articles/dataset/UFBA-425/29827475
   - 下载并解压数据集

2. **训练YOLOv8模型**
   ```bash
   # 参考 Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb
   ```

3. **训练U-Net模型**
   ```bash
   # 参考 Instance_seg_teeth/notebooks/yolov8+unet/yolov8+unet_training.ipynb
   ```

详细说明请参考: `test_instance_seg_teeth/MODEL_DOWNLOAD.md`

### 3. 运行测试

```bash
# 运行测试脚本
python test/test_instance_seg_teeth.py
```

测试脚本包含：
- 环境检查
- 单张图像分割
- 批量处理
- 可视化

## 使用方法

### 方法1: 使用测试脚本（推荐初学者）

```bash
python test/test_instance_seg_teeth.py
```

### 方法2: 在Python代码中使用

#### 基本用法

```python
from test_instance_seg_teeth import TeethSegmentation

# 初始化分割器
segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5",
    img_size=512
)

# 分割单张图像
pred_mask = segmentor.segment("input/image.jpg")

# 可视化结果
segmentor.visualize_segmentation(
    "input/image.jpg",
    pred_mask,
    save_path="output/result.jpg"
)
```

#### 批量处理

```python
from test_instance_seg_teeth import TeethSegmentation

segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5"
)

# 批量处理
image_paths = ["input/image1.jpg", "input/image2.jpg"]
results = segmentor.segment_batch(
    image_paths,
    output_dir="output/batch_results"
)
```

#### 调整参数

```python
# 调整YOLO检测参数
pred_mask = segmentor.segment(
    "input/image.jpg",
    yolo_iou=0.7,    # IOU阈值
    yolo_conf=0.5    # 置信度阈值
)
```

### 方法3: 运行演示脚本

```bash
python test_instance_seg_teeth/demo.py
```

## API文档

### TeethSegmentation类

#### 初始化参数

- `yolo_model_path` (str): YOLOv8模型权重路径
- `unet_model_path` (str): U-Net模型权重路径
- `img_size` (int): 处理图像大小，默认512

#### 主要方法

##### segment(image_path, yolo_iou=0.7, yolo_conf=0.5)

对单张图像进行分割。

**返回**: numpy.ndarray - 形状为(H, W, 32)的分割掩码

##### visualize_segmentation(image_path, pred_mask, save_path=None)

可视化分割结果。

**返回**: numpy.ndarray - 可视化图像

##### segment_batch(image_paths, output_dir=None)

批量处理多张图像。

**返回**: list - 预测掩码列表

## 技术细节

### OralBBNet架构

1. **YOLOv8检测**: 检测牙齿位置和边界框
2. **边界框掩码**: 将检测结果转换为32通道掩码
3. **U-Net分割**: 基于边界框引导进行精细分割

### 工作流程

```
输入图像
  ↓
YOLOv8检测 → 边界框 + 类别
  ↓
创建32通道掩码
  ↓
图像预处理（CLAHE + 归一化）
  ↓
合并：掩码(32通道) + 图像(3通道) = 35通道输入
  ↓
U-Net分割
  ↓
输出：32通道分割掩码（每颗牙齿一个通道）
```

## 对比：两种分割方法

本项目现在提供两种牙齿分割方法：

### 方法1: U-Net分割（原有方法）

- 位置: `teeth_analysis/`
- 技术: 纯U-Net语义分割
- 优点: 简单快速
- 适用: 快速原型、整体分割

```python
from teeth_analysis import TeethSegmentationPipeline
pipeline = TeethSegmentationPipeline()
results = pipeline.analyze_image('input/image.png')
```

### 方法2: OralBBNet实例分割（新方法）

- 位置: `test_instance_seg_teeth/`
- 技术: YOLOv8 + U-Net
- 优点: 高精度、实例级分割
- 适用: 研究、精确分析

```python
from test_instance_seg_teeth import TeethSegmentation
segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5"
)
pred_mask = segmentor.segment('input/image.png')
```

## 注意事项

1. **模型要求**: 需要两个模型配合使用（YOLOv8 + U-Net）
2. **计算资源**: 推理需要一定的计算资源，建议使用GPU
3. **输入格式**: 适用于全景X光片
4. **牙齿数量**: 设计用于最多32颗牙齿

## 故障排查

### 问题1: 模型文件不存在

```
错误: YOLO模型文件不存在
```

**解决方案**:
1. 运行 `python test_instance_seg_teeth/download_models.py` 下载基础模型
2. 或参考 `test_instance_seg_teeth/MODEL_DOWNLOAD.md` 训练专用模型

### 问题2: 依赖缺失

```
ModuleNotFoundError: No module named 'ultralytics'
```

**解决方案**:
```bash
pip install -r requirements.txt
# 或
pip install -r test_instance_seg_teeth/requirements.txt
```

### 问题3: 未检测到牙齿

**可能原因**:
- YOLOv8使用的是基础模型而非牙齿专用模型
- 图像质量问题

**解决方案**:
1. 在牙齿数据集上训练YOLOv8模型
2. 调整检测参数（降低置信度阈值）

### 问题4: GPU内存不足

**解决方案**:
1. 使用CPU推理（速度较慢但不占用GPU）
2. 减小batch size
3. 使用更小的图像尺寸

## 参考资源

- **原始仓库**: https://github.com/devichand579/Instance_seg_teeth
- **数据集**: https://figshare.com/articles/dataset/UFBA-425/29827475
- **论文**: https://arxiv.org/abs/2406.03747
- **YOLOv8文档**: https://docs.ultralytics.com/

## 更新日志

### 2024-11-21
- ✅ 集成Instance_seg_teeth仓库
- ✅ 创建test_instance_seg_teeth模块
- ✅ 实现OralBBNet推理功能
- ✅ 添加测试脚本和文档
- ✅ 更新项目依赖

## 贡献

欢迎贡献改进！如有问题请提交Issue。

## 许可

本功能集成遵循MIT许可证。原始Instance_seg_teeth仓库的许可请参考原仓库。
