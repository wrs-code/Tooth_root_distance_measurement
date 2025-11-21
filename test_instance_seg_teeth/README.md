# 牙齿实例分割模块

基于 [Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth) 仓库的 OralBBNet 方法进行牙齿实例分割。

## 简介

本模块基于 Instance_seg_teeth 仓库，使用 YOLOv8 + U-Net (OralBBNet) 架构实现高精度的牙齿实例分割：

- **YOLOv8**: 检测牙齿边界框和分类
- **U-Net**: 基于边界框引导进行精细分割
- **性能**: 在 UFBA-425 数据集上达到 89.34% Dice 系数

## 目录结构

```
test_instance_seg_teeth/
├── __init__.py         # 模块初始化
├── inference.py        # 推理核心代码（基于Instance_seg_teeth）
├── simple_demo.py      # 简单调用示例
├── batch_demo.py       # 批量处理示例
└── README.md          # 本文件
```

## 依赖安装

```bash
# 安装项目依赖（已包含所需库）
pip install -r requirements.txt
```

主要依赖：
- `tensorflow>=2.8.0`
- `ultralytics>=8.0.28` (YOLOv8)
- `scikit-image>=0.19.0`
- `opencv-python>=4.5.0`

## 模型准备

本模块需要两个训练好的模型：

### 1. YOLO v8 模型

用于检测牙齿边界框。

**训练方法**:
参考 `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb`

**放置位置**:
```
models/yolov8_teeth.pt
```

### 2. U-Net 模型

用于精细分割。

**训练方法**:
参考 `Instance_seg_teeth/notebooks/yolov8+unet/yolov8+unet_training.ipynb`

**放置位置**:
```
models/unet_teeth.h5
```

### 数据集

训练需要 UFBA-425 数据集：
- **下载**: https://figshare.com/articles/dataset/UFBA-425/29827475
- **说明**: 425张全景X光片，包含边界框和分割掩码标注

## 快速开始

### 方法1: 运行简单示例

```bash
python test_instance_seg_teeth/simple_demo.py
```

### 方法2: 在Python代码中使用

#### 基本用法

```python
from test_instance_seg_teeth import InstanceSegmentationPipeline

# 创建流水线
pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5',
    img_size=512
)

# 分割单张图像
results = pipeline.segment_image(
    image_path='input/image.jpg',
    output_dir='output/results'
)

# 查看结果
if results:
    print(f"预测掩码形状: {results['pred_mask'].shape}")
```

#### 批量处理

```python
from test_instance_seg_teeth import InstanceSegmentationPipeline

# 创建流水线
pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5'
)

# 批量分割
results = pipeline.batch_segment(
    input_dir='input',
    output_dir='output/batch_results'
)

# 统计结果
success_count = sum(1 for r in results if r is not None)
print(f"成功: {success_count}/{len(results)}")
```

## API 文档

### InstanceSegmentationPipeline 类

主要的牙齿实例分割类。

#### 初始化参数

```python
InstanceSegmentationPipeline(
    yolo_model_path=None,  # YOLO模型路径
    unet_model_path=None,  # U-Net模型路径
    img_size=512           # 处理图像大小
)
```

#### 主要方法

##### segment_image(image_path, output_dir=None)

分割单张图像。

**参数**:
- `image_path` (str): 图像路径
- `output_dir` (str, optional): 输出目录，如果指定则保存可视化结果

**返回**:
- `dict`: 包含以下键：
  - `pred_mask`: 预测掩码 (H, W, 32)
  - `binary_mask`: 二值掩码 (32, H, W)
  - `image_path`: 图像路径

##### batch_segment(input_dir, output_dir)

批量处理多张图像。

**参数**:
- `input_dir` (str): 输入目录
- `output_dir` (str): 输出目录

**返回**:
- `list`: 结果列表

## 工作流程

1. **YOLO检测**: 检测牙齿位置和边界框
2. **创建掩码**: 将检测结果转换为32通道二值掩码
3. **图像预处理**: CLAHE增强 + 归一化
4. **合并输入**: 掩码(32通道) + 图像(3通道) = 35通道
5. **U-Net分割**: 基于边界框引导进行精细分割
6. **输出**: 32通道分割掩码（每颗牙齿一个通道）

## 文件说明

### inference.py

包含从 Instance_seg_teeth 仓库提取的核心推理代码：

- `_build_unet_model()`: 构建 U-Net 模型架构
- `_dice_loss_with_l2_regularization()`: Dice 损失函数
- `apply_clahe()`: CLAHE 图像增强
- `detect_with_yolo()`: YOLO 检测
- `create_binary_masks()`: 创建二值掩码
- `segment_image()`: 主分割方法

### simple_demo.py

简单调用示例，包含：
- 基础使用
- 自定义输出路径
- 获取掩码数据

### batch_demo.py

批量处理示例，包含：
- 简单批量处理
- 带统计信息的批量处理
- 自定义批量处理流程

## 示例

### 示例1: 单张图像分割

```python
from test_instance_seg_teeth import InstanceSegmentationPipeline

pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5'
)

results = pipeline.segment_image(
    'input/107.png',
    output_dir='output/seg_result'
)
```

### 示例2: 批量处理

```python
from test_instance_seg_teeth import InstanceSegmentationPipeline

pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5'
)

results = pipeline.batch_segment(
    input_dir='input',
    output_dir='output/batch'
)
```

### 示例3: 获取掩码数据

```python
import cv2
import numpy as np
from test_instance_seg_teeth import InstanceSegmentationPipeline

pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5'
)

results = pipeline.segment_image('input/107.png')

if results:
    pred_mask = results['pred_mask']  # (H, W, 32)

    # 处理每颗牙齿
    for i in range(32):
        tooth_mask = (pred_mask[:, :, i] * 255).astype(np.uint8)
        if np.any(tooth_mask > 128):
            cv2.imwrite(f'output/tooth_{i+1}.png', tooth_mask)
```

## 性能指标

根据原论文，在 UFBA-425 数据集上的性能：

| 牙齿类型 | Dice系数 |
|---------|---------|
| 门牙    | 89.34%  |
| 犬齿    | 88.40%  |
| 前磨牙  | 88.38%  |
| 磨牙    | 87.87%  |

## 注意事项

1. **模型依赖**: 需要两个模型配合使用（YOLO + U-Net）
2. **输入格式**: 适用于全景X光片
3. **计算资源**: 推理需要一定的计算资源，建议使用GPU
4. **牙齿数量**: 设计用于最多32颗牙齿（成人恒牙）

## 故障排查

### 问题1: 模型文件不存在

**错误**:
```
⚠️ 警告: YOLO模型不存在: models/yolov8_teeth.pt
```

**解决方案**:
- 参考 `Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb` 训练 YOLO 模型
- 下载 UFBA-425 数据集
- 将训练好的模型放到 `models/` 目录

### 问题2: 依赖缺失

**错误**:
```
ModuleNotFoundError: No module named 'ultralytics'
```

**解决方案**:
```bash
pip install -r requirements.txt
```

### 问题3: 未检测到牙齿

**可能原因**:
- YOLO 置信度阈值过高
- 输入图像质量问题

**解决方案**:
调整检测参数（在 `inference.py` 中的 `detect_with_yolo` 方法）

## 参考资源

- **Instance_seg_teeth 仓库**: https://github.com/devichand579/Instance_seg_teeth
- **UFBA-425 数据集**: https://figshare.com/articles/dataset/UFBA-425/29827475
- **OralBBNet 论文**: https://arxiv.org/abs/2406.03747
- **YOLOv8 文档**: https://docs.ultralytics.com/

## 引用

如果使用本模块进行研究，请引用原始论文：

```bibtex
@misc{budagam2025oralbbnetspatiallyguideddental,
      title={OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors},
      author={Devichand Budagam and Azamat Zhanatuly Imanbayev and Iskander Rafailovich Akhmetov and Aleksandr Sinitca and Sergey Antonov and Dmitrii Kaplun},
      year={2025},
      eprint={2406.03747},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.03747},
}
```

## 许可

本模块遵循 MIT 许可证。Instance_seg_teeth 仓库的许可请参考原仓库。
