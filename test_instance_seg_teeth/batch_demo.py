#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理示例 - 牙齿实例分割
演示如何批量处理多张牙齿X光片
"""

import sys
import os
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from test_instance_seg_teeth import InstanceSegmentationPipeline


def demo_simple_batch():
    """
    示例1：简单批量处理
    """
    print("\n" + "="*60)
    print("示例1：简单批量处理")
    print("="*60)

    # 创建流水线
    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',
        unet_model_path='models/unet_teeth.h5'
    )

    # 批量分割input文件夹中的所有图像
    results = pipeline.batch_segment(
        input_dir='input',
        output_dir='output/instance_seg_batch'
    )

    # 统计结果
    success_count = sum(1 for r in results if r is not None)

    print(f"\n批量处理完成！")
    print(f"  总数: {len(results)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {len(results) - success_count}")


def demo_batch_with_stats():
    """
    示例2：带统计信息的批量处理
    """
    print("\n" + "="*60)
    print("示例2：带统计信息的批量处理")
    print("="*60)

    # 创建流水线
    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',
        unet_model_path='models/unet_teeth.h5'
    )

    # 记录开始时间
    start_time = time.time()

    # 批量处理
    results = pipeline.batch_segment(
        input_dir='input',
        output_dir='output/instance_seg_batch_stats'
    )

    # 计算耗时
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r is not None)

    print(f"\n{'='*60}")
    print("批量处理统计")
    print(f"{'='*60}")
    print(f"总图像数: {len(results)}")
    print(f"成功数量: {success_count}")
    print(f"失败数量: {len(results) - success_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    if len(results) > 0:
        print(f"平均耗时: {total_time/len(results):.2f} 秒/张")


def demo_custom_batch():
    """
    示例3：自定义批量处理流程
    """
    print("\n" + "="*60)
    print("示例3：自定义批量处理流程")
    print("="*60)

    import glob

    # 创建流水线
    pipeline = InstanceSegmentationPipeline(
        yolo_model_path='models/yolov8_teeth.pt',
        unet_model_path='models/unet_teeth.h5'
    )

    # 自定义：只处理PNG文件
    png_files = glob.glob('input/*.png')

    print(f"\n找到 {len(png_files)} 个PNG文件")

    results = []
    for i, image_path in enumerate(png_files, 1):
        print(f"\n处理 [{i}/{len(png_files)}]: {os.path.basename(image_path)}")

        try:
            result = pipeline.segment_image(
                image_path,
                output_dir='output/instance_seg_custom'
            )
            results.append(result)
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append(None)

    # 统计
    success_count = sum(1 for r in results if r is not None)
    print(f"\n处理完成: {success_count}/{len(png_files)} 成功")


def main():
    """主函数"""
    print("="*60)
    print("牙齿实例分割工具 - 批量处理示例")
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

    # 运行示例1：简单批量处理
    demo_simple_batch()

    # 运行示例2：带统计信息的批量处理
    # demo_batch_with_stats()

    # 运行示例3：自定义批量处理
    # demo_custom_batch()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
