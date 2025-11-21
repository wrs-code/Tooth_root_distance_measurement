# 模型下载说明

本模块需要两个预训练模型才能正常工作：

## 1. YOLOv8模型

YOLOv8用于检测牙齿的边界框。

### 下载方式

根据Instance_seg_teeth仓库，你需要训练或下载YOLOv8模型权重。

**选项A: 使用预训练的YOLOv8基础模型（用于测试）**
```bash
# 下载YOLOv8基础模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8_teeth.pt
```

**选项B: 训练自己的模型**

参考Instance_seg_teeth仓库中的`notebooks/yolov8/yolov8_train.ipynb`来训练自己的YOLOv8模型。

### 放置位置
```
Tooth_root_distance_measurement/
└── models/
    └── yolov8_teeth.pt
```

## 2. U-Net模型

U-Net模型用于精细的牙齿分割。

### 训练方式

U-Net模型需要在牙齿数据集上训练。参考Instance_seg_teeth仓库中的训练notebook：

- `notebooks/yolov8+unet/yolov8+unet_training.ipynb` - 训练OralBBNet模型
- `notebooks/yolov8+unet/yolov8+unet+cv.ipynb` - 使用交叉验证训练

训练完成后，模型会保存为`.h5`或`.keras`格式。

### 放置位置
```
Tooth_root_distance_measurement/
└── models/
    └── unet_teeth.h5
```

## 3. 数据集

### UFBA-425数据集

Instance_seg_teeth仓库使用UFBA-425数据集，包含425张带标注的全景X光片。

**下载链接:**
- [FigShare - UFBA-425](https://figshare.com/articles/dataset/UFBA-425/29827475)

### 数据集结构

下载后，数据集应该包含：
- 图像文件（.jpg格式）
- 边界框标注（YOLO格式）
- 分割掩码（.tiff格式）

## 4. 快速开始（使用现有模型）

如果你只是想测试功能而不想训练模型：

1. **下载YOLOv8基础模型**（临时解决方案）
```bash
mkdir -p models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8_teeth.pt
```

2. **对于U-Net模型**
   - 你需要在牙齿数据集上训练，或者联系原仓库作者获取预训练权重
   - 原仓库地址: https://github.com/devichand579/Instance_seg_teeth

## 5. 模型配置

在使用模型时，可以在代码中指定模型路径：

```python
from test_instance_seg_teeth import TeethSegmentation

# 初始化模型
segmentor = TeethSegmentation(
    yolo_model_path="models/yolov8_teeth.pt",
    unet_model_path="models/unet_teeth.h5",
    img_size=512
)
```

## 6. 依赖安装

确保安装了所需的依赖：

```bash
pip install ultralytics tensorflow scikit-image opencv-python pillow numpy
```

## 注意事项

1. YOLOv8基础模型可能无法直接用于牙齿检测，因为它不是在牙齿数据集上训练的
2. 为了获得最佳效果，建议在UFBA-425数据集上训练两个模型
3. U-Net模型的输入需要包含YOLO检测的边界框信息，因此两个模型是配合使用的
4. 模型训练可能需要GPU支持以获得合理的训练时间

## 参考资源

- Instance_seg_teeth仓库: https://github.com/devichand579/Instance_seg_teeth
- UFBA-425数据集: https://figshare.com/articles/dataset/UFBA-425/29827475
- OralBBNet论文: https://arxiv.org/abs/2406.03747
- YOLOv8文档: https://docs.ultralytics.com/
