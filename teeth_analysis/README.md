# 牙齿分割分析工具包

模块化的牙齿分割和分析工具，基于U-Net深度学习模型，与开源仓库 [Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net) 完全一致的实现。

## 🏗️ 模块架构

```
teeth_analysis/
├── core/                          # 核心功能模块
│   ├── image_preprocessor.py      # 图像预处理器
│   ├── mask_postprocessor.py      # 掩码后处理器
│   ├── teeth_contour_detector.py  # 牙齿轮廓检测器
│   └── unet_inference_engine.py   # U-Net推理引擎
├── visualization/                 # 可视化模块
│   └── teeth_visualizer.py        # 牙齿可视化器
└── pipeline/                      # 流水线模块
    └── teeth_segmentation_pipeline.py  # 牙齿分割流水线
```

## 📦 模块说明

### 1. ImagePreprocessor（图像预处理器）
**职责**：图像预处理操作

**主要功能**：
- `convert_to_grayscale()` - 转换为灰度图
- `apply_clahe()` - CLAHE对比度增强
- `apply_bilateral_filter()` - 双边滤波降噪
- `resize_image()` - 调整图像大小
- `normalize_image()` - 归一化
- `prepare_for_unet()` - 为U-Net准备图像

**使用示例**：
```python
from teeth_analysis.core import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(512, 512))
preprocessed, original_size = preprocessor.prepare_for_unet(image)
```

---

### 2. UNetInferenceEngine（U-Net推理引擎）
**职责**：加载和运行U-Net模型

**主要功能**：
- `predict()` - 执行模型推理
- `get_model_info()` - 获取模型信息

**使用示例**：
```python
from teeth_analysis.core import UNetInferenceEngine

engine = UNetInferenceEngine(model_path='models/dental_xray_seg.h5')
prediction = engine.predict(preprocessed_image)
```

---

### 3. MaskPostprocessor（掩码后处理器）⚙️
**职责**：掩码后处理（与开源仓库一致）

**主要功能**：
- `apply_opening()` - 形态学开运算（去噪）
- `apply_sharpening()` - 锐化滤波（增强边缘）
- `apply_erosion()` - 腐蚀操作（分离牙齿）⭐
- `refine_mask()` - 完整的细化流程
- `postprocess_prediction()` - 从预测到细化掩码

**⚙️ 关键参数调整位置**：

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `kernel_size` | 5 | 形态学核大小 | 通常不需要改变 |
| `open_iteration` | 2 | 开运算迭代次数 | 2-3次（去噪） |
| `erode_iteration` | 1 | **腐蚀迭代次数** ⭐ | **调整牙齿分离程度** |

**腐蚀参数调整指南**：
```python
# erode_iteration=0: 无腐蚀，保持原始边界
# erode_iteration=1: 轻度腐蚀（默认，与开源仓库一致）
# erode_iteration=2: 中度腐蚀，适用于牙齿紧密相连
# erode_iteration=3+: 强腐蚀，适用于严重粘连
```

**使用示例**：
```python
from teeth_analysis.core import MaskPostprocessor

# 创建后处理器
postprocessor = MaskPostprocessor(
    kernel_size=5,
    open_iteration=2,
    erode_iteration=1  # ⚙️ 腐蚀参数
)

# 后处理掩码
binary_mask, refined_mask = postprocessor.postprocess_prediction(
    prediction, original_size, threshold=0.5
)

# 动态调整参数
postprocessor.update_parameters(erode_iteration=2)
```

---

### 4. TeethContourDetector（牙齿轮廓检测器）
**职责**：从掩码中提取单个牙齿轮廓

**主要功能**：
- `extract_connected_components()` - 连通组件分析（CCA）
- `filter_by_area()` - 面积过滤
- `extract_contour()` - 提取轮廓
- `compute_bounding_box()` - 计算边界框
- `extract_teeth_from_mask()` - 完整的提取流程

**与开源仓库一致的参数**：
- `connectivity=8` - 8邻域连通性
- `min_area=2000` - 最小面积阈值（c_area > 2000）

**使用示例**：
```python
from teeth_analysis.core import TeethContourDetector

detector = TeethContourDetector(min_area=2000, connectivity=8)
teeth_data = detector.extract_teeth_from_mask(refined_mask)
```

---

### 5. TeethVisualizer（牙齿可视化器）
**职责**：绘制和可视化结果

**主要功能**：
- `draw_mask_overlay()` - 掩码叠加
- `draw_teeth_contours()` - 绘制牙齿轮廓
- `create_comparison_figure()` - 创建对比图
- `save_visualization()` - 保存结果
- `visualize_segmentation_result()` - 完整的可视化流程

**使用示例**：
```python
from teeth_analysis.visualization import TeethVisualizer

visualizer = TeethVisualizer()
visualizer.visualize_segmentation_result(
    original_image, mask, teeth_data, 'output/result.png'
)
```

---

### 6. TeethSegmentationPipeline（牙齿分割流水线）
**职责**：整合所有模块，提供完整流程

**主要功能**：
- `segment_teeth()` - 完整的分割流程
- `analyze_image()` - 分析单张图像
- `batch_analyze()` - 批量分析
- `update_erosion_parameters()` - 更新腐蚀参数 ⚙️
- `update_area_threshold()` - 更新面积阈值

**使用示例**：
```python
from teeth_analysis import TeethSegmentationPipeline

# 创建流水线
pipeline = TeethSegmentationPipeline(
    model_path='models/dental_xray_seg.h5',
    open_iteration=2,
    erode_iteration=1,  # ⚙️ 腐蚀参数
    min_area=2000
)

# 分析单张图像
results = pipeline.analyze_image('input/107.png', output_dir='output')

# 批量分析
all_results = pipeline.batch_analyze(input_dir='input', output_dir='output')

# 动态调整参数
pipeline.update_erosion_parameters(erode_iteration=2)
```

---

## 🔄 完整处理流程

```
输入图像
    ↓
[ImagePreprocessor] 图像预处理
    ├─ 转灰度图
    ├─ resize到512×512
    └─ 归一化到[0,1]
    ↓
[UNetInferenceEngine] U-Net推理
    └─ 模型预测
    ↓
[MaskPostprocessor] 掩码后处理 ⚙️
    ├─ 调整回原始尺寸
    ├─ 二值化（threshold=0.5）
    ├─ 开运算去噪（5×5核，2次迭代）
    ├─ 锐化增强边缘
    └─ 腐蚀分离牙齿（5×5核，1次迭代）⭐
    ↓
[TeethContourDetector] 轮廓检测
    ├─ 连通组件分析（8邻域）
    ├─ 面积过滤（>2000像素）
    ├─ 轮廓提取
    └─ 从左到右排序
    ↓
[TeethVisualizer] 可视化
    ├─ 绘制轮廓
    ├─ 标注编号
    └─ 保存结果
    ↓
输出结果
```

---

## 📝 快速开始

### 方式1：使用流水线（推荐）

```python
from teeth_analysis import TeethSegmentationPipeline

# 创建流水线
pipeline = TeethSegmentationPipeline()

# 分析图像
results = pipeline.analyze_image('input/107.png')

print(f"检测到 {len(results['teeth_data'])} 颗牙齿")
```

### 方式2：单独使用各个模块

```python
from teeth_analysis.core import ImagePreprocessor, UNetInferenceEngine
from teeth_analysis.core import MaskPostprocessor, TeethContourDetector
from teeth_analysis.visualization import TeethVisualizer
import cv2

# 1. 读取图像
image = cv2.imread('input/107.png')

# 2. 创建模块
preprocessor = ImagePreprocessor()
inference_engine = UNetInferenceEngine()
postprocessor = MaskPostprocessor()
detector = TeethContourDetector()
visualizer = TeethVisualizer()

# 3. 执行流程
preprocessed, original_size = preprocessor.prepare_for_unet(image)
prediction = inference_engine.predict(preprocessed)
binary_mask, refined_mask = postprocessor.postprocess_prediction(prediction, original_size)
teeth_data = detector.extract_teeth_from_mask(refined_mask)
visualizer.visualize_segmentation_result(image, refined_mask, teeth_data, 'output/result.png')
```

---

## ⚙️ 参数调整指南

### 调整腐蚀程度（牙齿分离）

**位置**：`MaskPostprocessor.erode_iteration`

```python
# 方法1：创建时指定
pipeline = TeethSegmentationPipeline(erode_iteration=2)

# 方法2：动态更新
pipeline.update_erosion_parameters(erode_iteration=2)

# 方法3：直接使用模块
postprocessor = MaskPostprocessor(erode_iteration=2)
```

### 调整面积阈值（过滤小区域）

**位置**：`TeethContourDetector.min_area`

```python
# 方法1：创建时指定
pipeline = TeethSegmentationPipeline(min_area=3000)

# 方法2：动态更新
pipeline.update_area_threshold(min_area=3000)

# 方法3：直接使用模块
detector = TeethContourDetector(min_area=3000)
```

---

## ✅ 与开源仓库一致性验证

| 项目 | 开源仓库 | 当前实现 | 一致性 |
|------|----------|----------|--------|
| 后处理顺序 | 开运算→锐化→腐蚀 | 开运算→锐化→腐蚀 | ✅ |
| 核大小 | 5×5 | 5×5 | ✅ |
| 锐化核 | [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]] | [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]] | ✅ |
| 连通性 | connectivity=8 | connectivity=8 | ✅ |
| 面积阈值 | c_area > 2000 | area > 2000 | ✅ |
| 默认参数 | open_iteration, erode_iteration | open_iteration=2, erode_iteration=1 | ✅ |

---

## 📄 更多示例

查看 `example_usage.py` 获取更多使用示例。

---

## 🔗 参考

- 开源仓库: [SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)
- 关键文件: `CCA_Analysis.py`
