#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试U-Net分割 - 查看中间结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from unet_segmentation import UNetTeethSegmentation

def debug_segmentation(image_path='input/image.png'):
    """
    调试U-Net分割过程，输出中间结果

    参数:
        image_path: 测试图像路径
    """
    print("=" * 60)
    print("U-Net分割调试")
    print("=" * 60)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    print(f"✓ 读取图像: {image_path}")
    print(f"  图像尺寸: {image.shape}")

    # 创建分割器
    try:
        segmenter = UNetTeethSegmentation()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 预处理
    print("\n正在预处理图像...")
    processed, original_size = segmenter.preprocess_image(image)
    print(f"  预处理后尺寸: {processed.shape}")
    print(f"  原始尺寸: {original_size}")

    # U-Net推理
    print("\n正在进行U-Net推理...")
    prediction = segmenter.model.predict(processed, verbose=0)
    print(f"  预测输出尺寸: {prediction.shape}")
    print(f"  预测值范围: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"  预测值均值: {prediction.mean():.4f}")
    print(f"  预测值标准差: {prediction.std():.4f}")

    # 分析预测值分布
    pred_flat = prediction.flatten()
    print(f"\n预测值分布统计:")
    print(f"  > 0.1 的像素比例: {(pred_flat > 0.1).sum() / len(pred_flat) * 100:.2f}%")
    print(f"  > 0.3 的像素比例: {(pred_flat > 0.3).sum() / len(pred_flat) * 100:.2f}%")
    print(f"  > 0.5 的像素比例: {(pred_flat > 0.5).sum() / len(pred_flat) * 100:.2f}%")
    print(f"  > 0.7 的像素比例: {(pred_flat > 0.7).sum() / len(pred_flat) * 100:.2f}%")

    # 尝试不同的阈值
    print("\n测试不同的二值化阈值:")
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        mask = segmenter.postprocess_mask(prediction, original_size, threshold=threshold)
        white_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        print(f"  阈值 {threshold}: {white_pixels} 白色像素 ({white_pixels/total_pixels*100:.2f}%)")

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        print(f"    连通组件数量: {num_labels - 1}")  # 减去背景

        # 显示各组件面积
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            print(f"    组件面积范围: [{areas.min()}, {areas.max()}]")
            print(f"    组件面积均值: {areas.mean():.1f}")
            # 计算有多少组件在500-50000范围内
            valid_count = np.sum((areas >= 500) & (areas <= 50000))
            print(f"    面积在[500, 50000]范围内的组件: {valid_count}")

    # 使用默认阈值0.5进行完整分割
    print("\n使用默认阈值0.5进行分割...")
    mask, refined_mask = segmenter.segment_teeth(image)

    # 保存中间结果图像
    print("\n保存调试图像...")
    import os
    os.makedirs('debug_output', exist_ok=True)

    # 保存原始预测掩码（归一化到0-255）
    pred_vis = np.squeeze(prediction)
    pred_vis_resized = cv2.resize(pred_vis, (original_size[1], original_size[0]))
    pred_vis_255 = (pred_vis_resized * 255).astype(np.uint8)
    cv2.imwrite('debug_output/1_prediction_raw.png', pred_vis_255)
    print("  ✓ 保存原始预测: debug_output/1_prediction_raw.png")

    # 保存二值化掩码
    cv2.imwrite('debug_output/2_mask_binary.png', mask)
    print("  ✓ 保存二值化掩码: debug_output/2_mask_binary.png")

    # 保存细化后的掩码
    cv2.imwrite('debug_output/3_mask_refined.png', refined_mask)
    print("  ✓ 保存细化掩码: debug_output/3_mask_refined.png")

    # 提取单个牙齿
    print("\n提取单个牙齿（默认参数 min_area=500, max_area=50000）...")
    teeth_data = segmenter.extract_individual_teeth(refined_mask, min_area=500, max_area=50000)
    print(f"  ✓ 检测到 {len(teeth_data)} 颗牙齿")

    # 尝试调整面积参数
    print("\n尝试不同的面积阈值:")
    for min_area, max_area in [(100, 100000), (200, 80000), (1000, 50000)]:
        teeth = segmenter.extract_individual_teeth(refined_mask, min_area=min_area, max_area=max_area)
        print(f"  min_area={min_area}, max_area={max_area}: 检测到 {len(teeth)} 颗牙齿")

    # 可视化结果
    if len(teeth_data) > 0:
        vis_image = segmenter.visualize_segmentation(image, refined_mask, teeth_data)
        cv2.imwrite('debug_output/4_visualization.png', vis_image)
        print("\n  ✓ 保存可视化结果: debug_output/4_visualization.png")
    else:
        # 即使没有检测到牙齿，也显示掩码覆盖
        vis_image = segmenter.visualize_segmentation(image, refined_mask, None)
        cv2.imwrite('debug_output/4_visualization_no_teeth.png', vis_image)
        print("\n  ✓ 保存可视化结果（无牙齿检测）: debug_output/4_visualization_no_teeth.png")

    print("\n" + "=" * 60)
    print("调试完成！请查看 debug_output/ 文件夹中的图像")
    print("=" * 60)


if __name__ == "__main__":
    debug_segmentation()
