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
├── Instance_seg_teeth/          # 克隆的Instance_seg_teeth仓库（参考）
├── test_instance_seg_teeth/     # 牙齿实例分割模块 ⭐ NEW
│   ├── unet_model.py           # U-Net模型定义
│   ├── teeth_segmentation.py  # 分割主类
│   ├── demo.py                 # 演示脚本
│   ├── download_models.py      # 模型下载工具
│   ├── README.md              # 模块文档
│   └── MODEL_DOWNLOAD.md      # 模型下载说明
├── teeth_analysis/             # 原有牙齿分割模块
│   ├── core/                   # 核心功能
│   ├── pipeline/               # 流水线
│   └── visualization/          # 可视化
├── test/                       # 测试和演示脚本
│   ├── test_instance_seg_teeth.py  # 实例分割测试 ⭐ NEW
│   ├── simple_demo.py         # 简单演示
│   ├── advanced_demo.py       # 高级演示
│   └── batch_demo.py          # 批量处理演示
├── models/                     # 模型文件
├── input/                      # 输入图像
├── output/                     # 输出结果
├── example_usage.py           # 使用示例
├── requirements.txt           # 项目依赖
├── INSTANCE_SEG_INTEGRATION.md  # 实例分割集成说明 ⭐ NEW
└── README.md                  # 本文件
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
# 1. 下载基础模型（用于测试）
python test_instance_seg_teeth/download_models.py

# 2. 运行实例分割测试
python test/test_instance_seg_teeth.py

# 或使用Python代码
python -c "
from test_instance_seg_teeth import TeethSegmentation
segmentor = TeethSegmentation(
    yolo_model_path='models/yolov8_base.pt',
    unet_model_path='models/unet_teeth.h5'
)
pred_mask = segmentor.segment('input/107.png')
"
```

### 4. 查看结果

结果将保存在 `output/` 目录下。

### 5. 详细文档

- **实例分割集成说明**: [INSTANCE_SEG_INTEGRATION.md](INSTANCE_SEG_INTEGRATION.md)
- **测试脚本说明**: [test/README.md](test/README.md)
- **模块文档**: [test_instance_seg_teeth/README.md](test_instance_seg_teeth/README.md)

## 参考资料

- [技术需求文档](口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf)
- [牙齿分割开源项目](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)

## 许可证

待定

## 作者

wrs-code

---

*本项目正在开发中，更多功能即将推出*
