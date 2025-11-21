# 牙齿分割工具 - Demo 使用说明

本文件夹包含了如何调用本开源代码仓库的完整示例。

## 📁 文件说明

### 1. simple_demo.py - 简单调用示例
**适用对象**: 初学者、快速上手

**包含示例**:
- 基础使用：分析单张图像
- 自定义输出路径
- 获取掩码数据进行自定义处理

**运行方式**:
```bash
cd /path/to/Tooth_root_distance_measurement
python test/simple_demo.py
```

### 2. advanced_demo.py - 高级调用示例
**适用对象**: 高级用户、需要精细控制处理流程

**包含示例**:
- 逐步调用各个独立模块
- 使用自定义参数
- 对比不同参数的效果
- 访问各个组件的高级功能

**运行方式**:
```bash
cd /path/to/Tooth_root_distance_measurement
python test/advanced_demo.py
```

### 3. batch_demo.py - 批量处理示例
**适用对象**: 需要处理大量图像的用户

**包含示例**:
- 简单批量处理
- 自定义批量处理流程
- 带时间统计的批量处理
- 带错误处理的批量处理
- 带结果过滤的批量处理

**运行方式**:
```bash
cd /path/to/Tooth_root_distance_measurement
python test/batch_demo.py
```

## 🚀 快速开始

### 前置要求

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保有以下文件：
- `models/dental_xray_seg.h5` - U-Net模型文件
- `input/` 文件夹中有测试图像

### 最简单的调用方式

```python
from teeth_analysis import TeethSegmentationPipeline

# 创建流水线
pipeline = TeethSegmentationPipeline()

# 分析图像
results = pipeline.analyze_image('input/image.png', output_dir='output')

# 查看结果
if results:
    print(f"检测到 {len(results['teeth_data'])} 颗牙齿")
```

## 📖 代码结构说明

### 主要模块

本项目提供了以下核心模块（位于 `teeth_analysis/` 文件夹）：

#### 1. 流水线模块（推荐使用）
```python
from teeth_analysis import TeethSegmentationPipeline

pipeline = TeethSegmentationPipeline()
results = pipeline.analyze_image('image.png', output_dir='output')
```

#### 2. 核心模块（高级使用）
```python
from teeth_analysis import (
    ImagePreprocessor,        # 图像预处理
    UNetInferenceEngine,      # U-Net推理引擎
    MaskPostprocessor,        # 掩码后处理
    TeethContourDetector,     # 牙齿轮廓检测
    TeethVisualizer          # 可视化
)
```

### 模块功能说明

| 模块 | 功能 | 位置 |
|------|------|------|
| `ImagePreprocessor` | 图像预处理、归一化、调整尺寸 | `teeth_analysis/core/image_preprocessor.py` |
| `UNetInferenceEngine` | U-Net模型推理 | `teeth_analysis/core/unet_inference_engine.py` |
| `MaskPostprocessor` | 掩码后处理、开运算、腐蚀 | `teeth_analysis/core/mask_postprocessor.py` |
| `TeethContourDetector` | 提取牙齿轮廓、计算特征 | `teeth_analysis/core/teeth_contour_detector.py` |
| `TeethVisualizer` | 可视化结果 | `teeth_analysis/visualization/teeth_visualizer.py` |
| `TeethSegmentationPipeline` | 整合所有功能的流水线 | `teeth_analysis/pipeline/teeth_segmentation_pipeline.py` |

## 🎯 常见使用场景

### 场景1：分析单张图像（最常用）
```python
from teeth_analysis import TeethSegmentationPipeline

pipeline = TeethSegmentationPipeline()
results = pipeline.analyze_image('input/image.png', output_dir='output')
```

### 场景2：批量处理多张图像
```python
from teeth_analysis import TeethSegmentationPipeline

pipeline = TeethSegmentationPipeline()
results = pipeline.batch_analyze(input_dir='input', output_dir='output')
```

### 场景3：调整参数以获得更好的分割效果
```python
from teeth_analysis import TeethSegmentationPipeline

# 创建流水线时指定参数
pipeline = TeethSegmentationPipeline(
    model_path='models/dental_xray_seg.h5',
    open_iteration=3,      # 开运算次数（去噪）
    erode_iteration=2,     # 腐蚀次数（分离牙齿）
    min_area=3000          # 最小面积阈值
)

results = pipeline.analyze_image('input/image.png', output_dir='output')
```

### 场景4：获取原始数据进行自定义处理
```python
import cv2
from teeth_analysis import TeethSegmentationPipeline

pipeline = TeethSegmentationPipeline()
image = cv2.imread('input/image.png')

# 只获取分割结果，不保存
results = pipeline.segment_teeth(image)

# 访问各种数据
binary_mask = results['binary_mask']      # 二值掩码
refined_mask = results['refined_mask']    # 细化掩码
teeth_data = results['teeth_data']        # 牙齿信息列表

# 自定义处理
for tooth in teeth_data:
    print(f"面积: {tooth['area']}, 中心: {tooth['centroid']}")
```

### 场景5：使用独立模块（完全自定义流程）
```python
import cv2
from teeth_analysis import (
    ImagePreprocessor,
    UNetInferenceEngine,
    MaskPostprocessor,
    TeethContourDetector,
    TeethVisualizer
)

# 创建各个模块
preprocessor = ImagePreprocessor()
inference_engine = UNetInferenceEngine()
postprocessor = MaskPostprocessor(erode_iteration=2)
detector = TeethContourDetector(min_area=3000)
visualizer = TeethVisualizer()

# 执行各个步骤
image = cv2.imread('input/image.png')
preprocessed, original_size = preprocessor.prepare_for_unet(image)
prediction = inference_engine.predict(preprocessed)
binary_mask, refined_mask = postprocessor.postprocess_prediction(prediction, original_size)
teeth_data = detector.extract_teeth_from_mask(refined_mask)
visualizer.visualize_segmentation_result(image, refined_mask, teeth_data, 'output.png')
```

## ⚙️ 参数说明

### TeethSegmentationPipeline 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `'models/dental_xray_seg.h5'` | U-Net模型文件路径 |
| `open_iteration` | `2` | 开运算迭代次数（去噪） |
| `erode_iteration` | `1` | 腐蚀迭代次数（分离牙齿） |
| `min_area` | `2000` | 最小牙齿面积阈值（像素） |

### 参数调整建议

- **牙齿分离不够**：增加 `erode_iteration`（如 2 或 3）
- **检测到太多噪声**：增加 `open_iteration` 或 `min_area`
- **丢失小牙齿**：减少 `min_area`

## 📊 输出说明

### analyze_image 返回值

```python
results = {
    'image_path': str,           # 原图像路径
    'binary_mask': np.ndarray,   # 二值掩码 (H, W)
    'refined_mask': np.ndarray,  # 细化掩码 (H, W)
    'teeth_data': [              # 牙齿数据列表
        {
            'contour': np.ndarray,      # 轮廓点坐标
            'area': float,              # 面积
            'centroid': tuple,          # 中心点 (x, y)
            'bbox': tuple,              # 边界框 (x, y, w, h)
            'perimeter': float          # 周长
        },
        ...
    ]
}
```

### 生成的文件

- `{image_name}_comparison.png` - 包含原图、掩码、轮廓的对比图
- `summary_report.txt` - 批量处理的汇总报告

## 🔧 故障排查

### 问题1：找不到模块
```
ModuleNotFoundError: No module named 'teeth_analysis'
```
**解决方案**：确保从项目根目录运行，或在代码中添加：
```python
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
```

### 问题2：找不到模型文件
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/dental_xray_seg.h5'
```
**解决方案**：确保 `models/dental_xray_seg.h5` 文件存在

### 问题3：未检测到牙齿
**解决方案**：尝试调整参数：
```python
pipeline = TeethSegmentationPipeline(
    erode_iteration=0,    # 减少腐蚀
    min_area=1000         # 降低面积阈值
)
```

## 📚 更多资源

- **项目主页**: [GitHub](https://github.com/wrs-code/Tooth_root_distance_measurement)
- **原始开源项目**: [Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)
- **环境安装**: 参见项目根目录的 `install_env.md`

## 💡 提示

1. **首次使用**建议从 `simple_demo.py` 开始
2. **需要自定义**可以参考 `advanced_demo.py`
3. **批量处理**可以使用 `batch_demo.py`
4. 所有demo都可以直接运行，会自动处理 `input/` 文件夹中的图像
5. 结果会保存在 `test/output_*` 文件夹中

## 📝 许可证

本项目代码遵循原开源项目的许可证。
