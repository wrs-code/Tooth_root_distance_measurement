# 牙根安全距离测量Demo说明
# Dental Root Safety Distance Measurement Demo

## 项目概述

本Demo实现了基于CEJ线（釉牙骨质界）到牙根尖的**梯形安全区域测量**，这是一个创新的二维面积测量方法，能够根据垂直深度动态计算安全距离要求。

## 核心创新：基于深度的梯形安全区域

### 传统方法 vs 本方法

**传统方法**：
- 简单测量相邻牙根的最短水平距离
- 使用固定阈值（如3.2mm）判断风险
- 没有考虑深度因素

**本方法（梯形安全区域）**：
- 将安全距离定义为从CEJ到根尖的一个**类似梯形的区域**
- 安全宽度随深度动态变化：**越深，要求的宽度越大**
- 在多个深度点检查是否满足安全距离要求
- 更符合临床实际的生物力学需求

## 安全区域标准（基于文献研究）

根据互联网搜索的最新文献和临床标准，本Demo采用以下深度-宽度映射：

| 距CEJ的垂直深度 | 最小安全宽度（单侧） | 临床意义 |
|----------------|---------------------|----------|
| 0mm (CEJ处) | 1.0mm | 牙颈部，骨量较少 |
| 3mm | 1.5mm | 早期根部区域 |
| 6mm | 2.0mm | **推荐正畸种植体位置**（文献支持） |
| 9mm | 2.5mm | 中根部 |
| 12mm | 3.0mm | 接近根尖区域 |

### 科学依据

1. **文献支持**：
   - 在CEJ下方5.8mm处是安全的种植体位置（第一、第二磨牙间）
   - 在CEJ下方6mm或更远推荐用于下颌种植
   - 从CEJ向根尖方向2mm处，牙间骨宽度开始增加

2. **生物力学原理**：
   - 越接近CEJ，牙槽骨越薄，需要更小的操作空间
   - 越深入根部，骨量增加，可以承受更大的移动范围
   - 根尖区域需要足够的骨支持（2-3mm）

## 风险等级标准

根据PDF文档要求：

| 风险等级 | 距离阈值 | 颜色标注 | 临床建议 |
|---------|---------|---------|----------|
| 危险 | < 3.2mm | 红色 | 禁止或需特别小心 |
| 相对安全 | 3.2-4.0mm | 黄色 | 可以进行，需密切监测 |
| 安全 | ≥ 4.0mm | 绿色 | 可以安全进行正畸治疗 |

## Demo功能特点

### 1. 梯形安全区域计算
```python
# 核心算法
def calculate_trapezoid_safety_zone(root):
    # 在从CEJ到根尖的路径上采样多个点
    # 在每个深度点，根据深度计算所需的安全宽度
    # 使用线性插值获得平滑的梯形边界
```

### 2. 多深度合规性检查
不仅测量最短距离，还在多个关键深度（3, 5, 7, 9mm）检查是否满足安全要求：

```python
violations = []
for depth in [3, 5, 7, 9]:
    required_width = get_required_width_at_depth(depth)
    actual_distance = measure_at_depth(depth)
    if actual_distance < required_width:
        violations.append({depth, required, actual})
```

### 3. 可视化展示
- 绿色点：CEJ位置
- 红色点：根尖位置
- 淡蓝色区域：梯形安全区域
- 连接线：相邻牙根的最短距离（颜色表示风险等级）

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行Demo
```bash
python dental_safety_demo.py
```

### 输出结果
1. **控制台输出**：
   - 相邻牙根距离测量结果
   - 安全区域合规性检查结果
   - 违规详情（如果有）

2. **可视化图像**：
   - `demo_output.png`：包含所有标注的完整可视化

## 代码结构

```
dental_safety_demo.py
├── SafetyZoneConfig          # 配置类：深度-宽度映射、阈值
├── ToothRoot                  # 数据类：牙根信息
├── SafetyZoneCalculator       # 核心计算类
│   ├── get_required_width_at_depth()      # 深度→宽度映射
│   ├── calculate_trapezoid_safety_zone()  # 梯形区域计算
│   ├── measure_root_proximity()           # 距离测量
│   └── check_safety_zone_compliance()     # 合规性检查
├── DentalSafetyVisualizer     # 可视化类
└── create_synthetic_tooth_root()  # 合成数据生成
```

## 关键算法

### 1. 深度插值算法
```python
def get_required_width_at_depth(depth_mm):
    # 使用线性插值在标准点之间平滑过渡
    return np.interp(depth_mm, standard_depths, standard_widths)
```

### 2. 梯形生成算法
```python
for each depth_point from CEJ to apex:
    current_position = CEJ + direction * depth
    required_width = get_required_width(depth)
    left_boundary = current_position - perpendicular * width/2
    right_boundary = current_position + perpendicular * width/2
```

### 3. 最短距离计算
```python
min_distance = min(
    norm(pt1 - pt2)
    for pt1 in root1.contour
    for pt2 in root2.contour
)
```

## 实际应用场景

1. **术前规划**：
   - 评估正畸治疗的可行性
   - 确定安全的牙齿移动范围
   - 预测潜在的根吸收风险

2. **术中监测**：
   - 实时检查牙齿移动是否进入危险区域
   - 在不同深度验证安全距离

3. **术后评估**：
   - 评估治疗后的根间距离
   - 生成治疗报告

## 与PDF需求的对应关系

| PDF需求 | Demo实现 |
|--------|---------|
| 牙根识别与分割 | ✓ ToothRoot数据结构 |
| 牙根间距计算 | ✓ measure_root_proximity() |
| 风险等级标注 | ✓ 三级颜色标注系统 |
| 结果可视化 | ✓ DentalSafetyVisualizer |
| 测量精度要求 | ✓ 基于轮廓的精确计算 |

## 扩展方向

### 短期扩展
1. **真实图像处理**：
   - 集成U-Net分割模型
   - 自动检测CEJ线和根尖点
   - 从全景片提取牙根轮廓

2. **解剖结构识别**：
   - 下颌神经管识别
   - 上颌窦识别
   - 计算牙根与解剖结构的距离

3. **报告生成**：
   - 生成Excel/PDF报告
   - 包含所有测量数据
   - 风险等级汇总

### 长期扩展
1. **3D CBCT支持**：
   - 三维梯形安全区域（锥体）
   - 体积测量
   - 多平面分析

2. **AI辅助**：
   - 训练深度学习模型自动识别CEJ
   - 预测正畸治疗后的根位置
   - 风险预测模型

3. **临床验证**：
   - 与实际临床数据对比
   - 优化深度-宽度映射参数
   - 建立个性化安全标准

## 科学创新点

1. **首次提出梯形安全区域概念**：
   - 传统方法只看单点距离
   - 本方法考虑整个根部的安全空间

2. **基于深度的动态标准**：
   - 符合牙槽骨的生物学特性
   - 更精确的风险评估

3. **多深度验证机制**：
   - 不仅测量最小距离
   - 在关键深度点全面检查

## 参考文献依据

根据文献研究，本Demo的标准设定基于：
- CEJ下方6mm是推荐的种植体位置（成功率高）
- 根距离<0.6mm有根吸收风险
- 推荐2-3mm的正畸治疗间隙
- 从CEJ向根尖方向，骨宽度逐渐增加

## 联系与反馈

如有问题或建议，请通过GitHub Issues提交。

---

**开发者**: wrs-code
**最后更新**: 2025-11-19
**版本**: 1.0.0 - Initial Demo Release
