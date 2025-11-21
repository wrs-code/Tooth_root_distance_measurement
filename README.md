# Tooth Root Distance Measurement

口腔正畸牙根间距自动测量与风险标注软件

## 项目简介

本项目旨在基于全景X光片自动测量牙根间距，并进行正畸风险标注。通过深度学习技术实现牙齿分割和根部距离的精确测量，为口腔正畸提供辅助诊断工具。

## 技术基础

本项目集成了两种先进的牙齿分割方案：

### 方案1: U-Net语义分割
- **基于项目**: [Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)
- **架构**: U-Net
- **特点**: 快速、简单、适合整体分割

### 方案2: OralBBNet实例分割 ⭐ NEW
- **基于项目**: [Instance_seg_teeth](https://github.com/devichand579/Instance_seg_teeth)
- **架构**: YOLOv8 + U-Net (OralBBNet)
- **性能**: 在UFBA-425数据集上达到89.34% Dice系数
- **特点**: 高精度、实例级分割、可识别单颗牙齿

**其他技术**:
- **测量算法**: 自动计算相邻牙根间的距离
- **风险评估**: 根据测量结果进行正畸风险标注

## 主要功能

1. **牙齿分割**: 从全景X光片中自动识别和分割单颗牙齿
2. **根部定位**: 精确定位每颗牙齿的根部位置
3. **距离测量**: 自动测量相邻牙根间的距离
4. **风险标注**: 基于测量数据进行正畸风险评估和可视化标注

## 项目结构

```
Tooth_root_distance_measurement/
├── Instance_seg_teeth/          # Instance_seg_teeth仓库（用于牙齿实例分割） ⭐ NEW
│   ├── notebooks/               # 训练和测试 Jupyter notebooks
│   │   ├── yolov8/             # YOLOv8 训练
│   │   ├── Unet/               # U-Net 训练
│   │   └── yolov8+unet/        # OralBBNet 训练
│   └── Dataset/                # 数据集处理代码
├── test_instance_seg_teeth/     # 牙齿实例分割推理模块 ⭐ NEW
│   ├── __init__.py             # 模块初始化
│   ├── inference.py            # 推理核心（调用Instance_seg_teeth）
│   ├── simple_demo.py          # 简单调用示例
│   ├── batch_demo.py           # 批量处理示例
│   └── README.md               # 模块文档
├── teeth_analysis/              # U-Net语义分割模块
│   ├── core/                    # 核心功能
│   ├── pipeline/                # 流水线
│   └── visualization/           # 可视化
├── test/                        # 测试和演示脚本
│   ├── simple_demo.py          # U-Net语义分割简单演示
│   ├── advanced_demo.py        # U-Net语义分割高级演示
│   └── batch_demo.py           # U-Net语义分割批量处理
├── models/                      # 模型文件目录
│   ├── dental_xray_seg.h5      # U-Net模型（语义分割）
│   ├── yolov8_teeth.pt         # YOLOv8模型（实例分割用）
│   └── unet_teeth.h5           # U-Net模型（实例分割用）
├── input/                       # 输入图像
├── output/                      # 输出结果
├── example_usage.py            # 使用示例
├── requirements.txt            # 项目依赖
└── README.md                   # 本文件
```

## 开发计划

- [ ] 搭建基础开发环境
- [ ] 集成牙齿分割模型
- [ ] 实现根部检测算法
- [ ] 开发距离测量功能
- [ ] 实现风险标注系统
- [ ] 构建用户界面
- [ ] 测试与优化

## 环境要求

- Python 3.x
- TensorFlow / PyTorch
- OpenCV
- NumPy
- 其他依赖见各模块的 requirements.txt

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:wrs-code/Tooth_root_distance_measurement.git
cd Tooth_root_distance_measurement
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 选择分割方案

#### 方案A: 使用U-Net语义分割（简单快速）

```bash
# 运行简单演示
python test/simple_demo.py

# 或使用Python代码
python -c "
from teeth_analysis import TeethSegmentationPipeline
pipeline = TeethSegmentationPipeline()
results = pipeline.analyze_image('input/107.png', output_dir='output')
"
```

#### 方案B: 使用OralBBNet实例分割（高精度）⭐ 推荐

```bash
# 1. 训练模型（参考Instance_seg_teeth仓库）
# - YOLOv8: Instance_seg_teeth/notebooks/yolov8/yolov8_train.ipynb
# - U-Net: Instance_seg_teeth/notebooks/yolov8+unet/yolov8+unet_training.ipynb

# 2. 运行简单示例
python test_instance_seg_teeth/simple_demo.py

# 或使用Python代码
python -c "
from test_instance_seg_teeth import InstanceSegmentationPipeline
pipeline = InstanceSegmentationPipeline(
    yolo_model_path='models/yolov8_teeth.pt',
    unet_model_path='models/unet_teeth.h5'
)
results = pipeline.segment_image('input/107.png', output_dir='output')
"
```

### 4. 查看结果

结果将保存在 `output/` 目录下。

### 5. 详细文档

- **实例分割模块**: [test_instance_seg_teeth/README.md](test_instance_seg_teeth/README.md)
- **U-Net语义分割**: [test/README.md](test/README.md)
- **Instance_seg_teeth仓库**: [Instance_seg_teeth/README.md](Instance_seg_teeth/README.md)

## 参考资料

- [技术需求文档](口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf)
- [牙齿分割开源项目](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)

## 许可证

待定

## 作者

wrs-code

---

*本项目正在开发中，更多功能即将推出*
