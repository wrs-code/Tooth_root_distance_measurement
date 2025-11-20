#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的传统牙齿分割方法
当U-Net模型不可用时使用
"""

import cv2
import numpy as np


def improved_traditional_segmentation(image):
    """
    改进的传统牙齿分割方法（不依赖深度学习）

    参数:
        image: 输入的全景X光图像（BGR或灰度）

    返回:
        mask: 分割掩码
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. CLAHE增强对比度（更激进的参数）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. 双边滤波（保留边缘的同时降噪）
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. 多尺度形态学操作
    # 使用多个核尺寸来捕获不同大小的结构
    masks = []

    for kernel_size in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 顶帽变换（提取亮区域 - 牙齿）
        tophat = cv2.morphologyEx(bilateral, cv2.MORPH_TOPHAT, kernel)

        # 黑帽变换（提取暗区域）
        blackhat = cv2.morphologyEx(bilateral, cv2.MORPH_BLACKHAT, kernel)

        # 组合
        combined = cv2.add(bilateral, tophat)
        combined = cv2.subtract(combined, blackhat)

        # Otsu二值化
        _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        masks.append(thresh)

    # 4. 合并多尺度结果（取并集）
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 5. 自适应阈值（局部二值化）作为补充
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )

    # 6. 结合全局和局部二值化
    final_mask = cv2.bitwise_and(combined_mask, adaptive)

    # 7. 形态学后处理
    # 开运算去噪
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 距离变换 + watershed分割（尝试分离粘连的牙齿）
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 膨胀得到背景区域
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(opened, kernel_dilate, iterations=3)

    # 未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    image_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    markers = cv2.watershed(image_3ch, markers)

    # 提取分割结果
    watershed_mask = np.zeros_like(gray)
    watershed_mask[markers > 1] = 255

    return watershed_mask


def enhanced_cca_parameters():
    """
    返回针对传统方法优化的CCA参数

    返回:
        dict: 参数字典
    """
    return {
        'erode_iteration': 1,      # 减少腐蚀（避免牙齿粘连）
        'open_iteration': 3,       # 增加开运算（更强的去噪）
        'min_area': 1000,          # 降低最小面积阈值
        'kernel_size': (3, 3)      # 使用更小的核（减少过度处理）
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python improved_traditional_segmentation.py <图像路径>")
        sys.exit(1)

    # 读取图像
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        sys.exit(1)

    print(f"处理图像: {image_path}")
    print(f"图像尺寸: {image.shape}")

    # 分割
    mask = improved_traditional_segmentation(image)

    # 保存结果
    output_path = "output/improved_traditional_mask.png"
    import os
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(output_path, mask)

    print(f"分割掩码已保存: {output_path}")

    # 显示统计
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    coverage = (white_pixels / total_pixels) * 100

    print(f"牙齿区域覆盖率: {coverage:.2f}%")

    # CCA统计
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    print(f"连通组件数量: {num_labels - 1}")  # 减去背景

    # 推荐参数
    params = enhanced_cca_parameters()
    print(f"\n推荐的CCA参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
