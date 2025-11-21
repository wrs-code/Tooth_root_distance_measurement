#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单调用示例 - 牙齿实例分割
演示如何使用Instance_seg_teeth仓库进行牙齿实例分割
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from test_instance_seg_teeth import InstanceSegmentationPipeline


def demo_basic_usage():
    """
    基础使用示例：分析单张图像
    这是最简单、最常用的调用方式
    """
    print("\n" + "="*60)
    print("示例1：基础使用 - 牙齿实例分割")
    print("="*60)

    # 步骤1：创建流水线对象
    # 需要指定YOLO和U-Net模型路径
    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',  # YOLO模型
        unet_model_path='models/unet_teeth.h5',    # U-Net模型
        img_size=512
    )

    # 步骤2：分割图像
    # 输入：图像路径
    # 输出：分割结果（包含pred_mask等）
    results = pipeline.segment_image(
        image_path='input/107.png',
        output_dir='output/instance_seg'
    )

    # 步骤3：查看结果
    if results:
        print(f"\n✓ 分割完成！")
        print(f"  预测掩码形状: {results['pred_mask'].shape}")
        print(f"  结果已保存到: output/instance_seg/")
    else:
        print("\n❌ 分割失败")


def demo_custom_output():
    """
    自定义输出示例：指定输出路径
    """
    print("\n" + "="*60)
    print("示例2：自定义输出路径")
    print("="*60)

    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',
        unet_model_path='models/unet_teeth.h5'
    )

    # 可以指定不同的输出目录
    results = pipeline.segment_image(
        image_path='input/107.png',
        output_dir='test_instance_seg_teeth/output_demo1'  # 自定义输出目录
    )

    if results:
        print(f"\n✓ 结果已保存到: test_instance_seg_teeth/output_demo1/")


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
    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',
        unet_model_path='models/unet_teeth.h5'
    )

    # 执行分割（不保存，只获取数据）
    results = pipeline.segment_image(
        image_path='input/107.png',
        output_dir=None  # 不保存
    )

    if results:
        # 获取各种数据
        pred_mask = results['pred_mask']        # 预测掩码 (H, W, 32)
        binary_mask = results['binary_mask']    # 二值掩码 (32, H, W)

        print(f"\n获取到的数据：")
        print(f"  预测掩码形状: {pred_mask.shape}")
        print(f"  二值掩码形状: {binary_mask.shape}")

        # 可以对掩码进行自定义处理
        # 例如：保存每颗牙齿的掩码
        os.makedirs('test_instance_seg_teeth/output_demo1', exist_ok=True)

        for i in range(32):
            tooth_mask = (pred_mask[:, :, i] * 255).astype(np.uint8)
            if np.any(tooth_mask > 128):
                save_path = f'test_instance_seg_teeth/output_demo1/tooth_{i+1}.png'
                cv2.imwrite(save_path, tooth_mask)

        print(f"\n✓ 牙齿掩码已保存到: test_instance_seg_teeth/output_demo1/")


def main():
    """主函数"""
    print("="*60)
    print("牙齿实例分割工具 - 简单调用示例")
    print("基于Instance_seg_teeth (OralBBNet)")
    print("="*60)

    # 检查模型文件
    yolo_model = 'models/yolov8_teeth.pt'
    unet_model = 'models/unet_teeth.h5'

    if not os.path.exists(yolo_model):
        print(f"\n⚠️ 警告: YOLO模型不存在: {yolo_model}")
        print("请先下载或训练模型，参考: test_instance_seg_teeth/README.md")
        return

    if not os.path.exists(unet_model):
        print(f"\n⚠️ 警告: U-Net模型不存在: {unet_model}")
        print("请先训练模型，参考: test_instance_seg_teeth/README.md")
        return

    # 运行示例1：基础使用
    demo_basic_usage()

    # 运行示例2：自定义输出
    # demo_custom_output()

    # 运行示例3：获取掩码数据
    # demo_get_mask_data()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
