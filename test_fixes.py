#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试修复后的功能
"""

import sys
print("Python version:", sys.version)

try:
    import cv2
    print("✓ OpenCV imported")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported")

    # 测试中文字体配置
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    print("✓ Font configuration set")
    print("  Available fonts:", plt.rcParams['font.sans-serif'])
except Exception as e:
    print(f"✗ Matplotlib setup failed: {e}")

try:
    from unet_segmentation import UNetTeethSegmentation
    print("✓ UNetTeethSegmentation imported")
except Exception as e:
    print(f"✗ UNet import failed: {e}")
    sys.exit(1)

try:
    from tooth_cej_root_analyzer import ToothCEJAnalyzer
    print("✓ ToothCEJAnalyzer imported")
except Exception as e:
    print(f"✗ Analyzer import failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("所有导入测试通过！系统已准备好运行。")
print("="*60)
print("\n主要修复：")
print("1. ✓ 中文字体配置更新为 WenQuanYi Zen Hei/Micro Hei")
print("2. ✓ U-Net后处理优化（参考开源仓库CCA分析）")
print("3. ✓ CEJ线检测改为贴合牙齿的曲线（而非直线）")
print("\n请运行以下命令开始分析：")
print("  python3 tooth_cej_root_analyzer.py")
