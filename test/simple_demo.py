#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单调用示例
演示如何快速使用本项目进行牙齿分割分析
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from teeth_analysis import TeethSegmentationPipeline


def demo_basic_usage():
    """
    基础使用示例：分析单张图像
    这是最简单、最常用的调用方式
    """
    print("\n" + "="*60)
    print("示例1：基础使用 - 分析单张图像")
    print("="*60)

    # 步骤1：创建流水线对象（使用默认参数）
    # 模型会自动从 models/dental_xray_seg.h5 加载
    pipeline = TeethSegmentationPipeline()

    # 步骤2：分析图像
    # 输入：图像路径
    # 输出：分析结果（包含掩码、牙齿数据等）
    results = pipeline.analyze_image(
        image_path='input/image.png',  # 输入图像路径
        output_dir='output'            # 输出目录
    )

    # 步骤3：查看结果
    if results:
        print(f"\n✓ 分析完成！")
        print(f"  结果已保存到: output/image_comparison.png")

        # 打印牙齿轮廓信息
        print(f"\n牙齿轮廓信息：")
        for i, tooth in enumerate(results['teeth_data'], 1):
            print(f"  轮廓 {i}: 面积={tooth['area']:.0f}, "
                  f"中心=({tooth['centroid'][0]:.1f}, {tooth['centroid'][1]:.1f})")
    else:
        print("\n❌ 分析失败")


def demo_custom_output():
    """
    自定义输出示例：指定输出路径
    """
    print("\n" + "="*60)
    print("示例2：自定义输出路径")
    print("="*60)

    pipeline = TeethSegmentationPipeline()

    # 可以指定不同的输出目录
    results = pipeline.analyze_image(
        image_path='input/image.png',
        output_dir='test/output_demo1'  # 自定义输出目录
    )

    if results:
        print(f"\n✓ 结果已保存到: test/output_demo1/")


def demo_get_mask_data():
    """
    获取掩码数据示例：用于进一步处理
    """
    print("\n" + "="*60)
    print("示例3：获取掩码数据进行自定义处理")
    print("="*60)

    import cv2
    import numpy as np

    # 创建流水线
    pipeline = TeethSegmentationPipeline()

    # 读取图像
    image = cv2.imread('input/image.png')

    # 执行分割（不保存，只获取数据）
    results = pipeline.segment_teeth(image)

    # 获取各种数据
    binary_mask = results['binary_mask']      # 二值化掩码
    refined_mask = results['refined_mask']    # 细化后的掩码
    teeth_data = results['teeth_data']        # 牙齿轮廓数据列表

    print(f"\n获取到的数据：")
    print(f"  二值掩码尺寸: {binary_mask.shape}")
    print(f"  细化掩码尺寸: {refined_mask.shape}")
    print(f"  轮廓数据数量: {len(teeth_data)}")

    # 可以对掩码进行自定义处理
    # 例如：保存掩码图像
    cv2.imwrite('test/output_demo1/mask.png', binary_mask)
    cv2.imwrite('test/output_demo1/refined_mask.png', refined_mask)
    print(f"\n✓ 掩码已保存到: test/output_demo1/")


def main():
    """主函数"""
    print("="*60)
    print("牙齿分割工具 - 简单调用示例")
    print("="*60)

    # 运行示例1：基础使用
    demo_basic_usage()

    # 运行示例2：自定义输出
    demo_custom_output()

    # 运行示例3：获取掩码数据
    demo_get_mask_data()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
