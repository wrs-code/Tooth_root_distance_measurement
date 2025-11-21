# 牙齿实例分割模块

基于[Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)仓库的OralBBNet方法实现的牙齿实例分割模块。

## 简介

本模块集成了YOLOv8和U-Net模型，实现了高精度的牙齿实例分割：

1. **YOLOv8**: 用于检测牙齿的边界框和分类
2. **U-Net**: 结合边界框信息进行精细的牙齿分割

这种方法被称为OralBBNet（Spatially Guided Dental Segmentation），在UFBA-425数据集上取得了优秀的性能。

## 目录结构

```
test_instance_seg_teeth/
├── __init__.py              # 模块初始化文件
├── unet_model.py            # U-Net模型定义
├── teeth_segmentation.py   # 牙齿分割主类
├── demo.py                  # 演示脚本
├── MODEL_DOWNLOAD.md        # 模型下载说明
└── README.md               # 本文件
```

## 安装依赖

```bash
pip install ultralytics tensorflow scikit-image opencv-python pillow numpy
```

或者使用项目根目录的requirements.txt：

```bash
pip install -r requirements.txt
```

## 模型准备

在使用本模块之前，你需要准备两个模型：

1. **YOLOv8模型权重** (`yolov8_teeth.pt`)
2. **U-Net模型权重** (`unet_teeth.h5`)

详细的模型获取方式请参考 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)

## 快速开始

### 单张图像分割

```python
from test_instance_seg_teeth import TeethSegmentation

# 初始化分割器
segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5",
    img_size=512
)

# 对图像进行分割
pred_mask = segmentor.segment("input/test_image.jpg")

# 可视化结果
segmentor.visualize_segmentation(
    "input/test_image.jpg",
    pred_mask,
    save_path="output/result.jpg"
)
```

### 批量处理

```python
from test_instance_seg_teeth import TeethSegmentation

# 初始化分割器
segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5"
)

# 批量处理
image_paths = [
    "input/image1.jpg",
    "input/image2.jpg",
    "input/image3.jpg"
]

results = segmentor.segment_batch(
    image_paths,
    output_dir="output/batch_results"
)
```

### 运行演示脚本

```bash
# 单张图像演示
python test_instance_seg_teeth/demo.py

# 或者使用Python模块方式运行
python -m test_instance_seg_teeth.demo
```

## API文档

### TeethSegmentation类

主要的牙齿分割类。

#### 初始化参数

- `yolo_model_path` (str): YOLOv8模型权重路径
- `unet_model_path` (str): U-Net模型权重路径
- `img_size` (int): 处理图像大小，默认512

#### 主要方法

##### segment(image_path, yolo_iou=0.7, yolo_conf=0.5)

对单张图像进行分割。

**参数:**
- `image_path` (str): 图像路径
- `yolo_iou` (float): YOLO的IOU阈值，默认0.7
- `yolo_conf` (float): YOLO的置信度阈值，默认0.5

**返回:**
- `numpy.ndarray`: 形状为(H, W, 32)的分割掩码，32对应32颗牙齿

##### visualize_segmentation(image_path, pred_mask, save_path=None)

可视化分割结果。

**参数:**
- `image_path` (str): 原始图像路径
- `pred_mask` (numpy.ndarray): 预测掩码
- `save_path` (str, optional): 保存路径

**返回:**
- `numpy.ndarray`: 可视化图像

##### segment_batch(image_paths, output_dir=None)

批量处理多张图像。

**参数:**
- `image_paths` (list): 图像路径列表
- `output_dir` (str, optional): 输出目录

**返回:**
- `list`: 预测掩码列表

## 工作流程

1. **牙齿检测**: 使用YOLOv8检测图像中的牙齿，获取每颗牙齿的边界框和分类
2. **创建掩码**: 根据检测结果创建32通道的二值掩码（每颗牙齿一个通道）
3. **图像预处理**: 对输入图像进行CLAHE增强、缩放和归一化
4. **输入准备**: 将边界框掩码和预处理后的图像合并为35通道的输入
5. **U-Net分割**: U-Net模型基于边界框引导进行精细分割
6. **输出**: 返回32通道的分割掩码，每个通道对应一颗牙齿的概率图

## 技术细节

### OralBBNet架构

- **编码器**: 5层卷积块，逐步下采样
- **边界框分支**: 并行处理边界框信息
- **解码器**: 5层上采样块，结合跳跃连接和边界框引导
- **损失函数**: Dice Loss + L2正则化

### 数据增强

- CLAHE对比度增强
- 图像归一化
- 随机水平翻转（训练时）

## 性能指标

根据原论文，在UFBA-425数据集上的性能：

| 牙齿类型 | Dice系数 |
|---------|---------|
| 门牙    | 89.34   |
| 犬齿    | 88.40   |
| 前磨牙  | 88.38   |
| 磨牙    | 87.87   |

## 限制和注意事项

1. **模型依赖**: 需要两个预训练模型，且两者需要配合使用
2. **输入要求**: 输入应为全景X光片，其他类型的牙科影像可能效果不佳
3. **计算资源**: U-Net推理需要一定的计算资源，建议使用GPU
4. **牙齿数量**: 模型设计用于处理最多32颗牙齿（成人恒牙）

## 参考

- 原始仓库: [Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)
- 数据集: [UFBA-425](https://figshare.com/articles/dataset/UFBA-425/29827475)
- 论文: [OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors](https://arxiv.org/abs/2406.03747)

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

本模块遵循MIT许可证。原始Instance_seg_teeth仓库的许可证请参考原仓库。

## 贡献

欢迎提交问题和改进建议！

## 联系

如有问题，请在项目仓库中提交Issue。
