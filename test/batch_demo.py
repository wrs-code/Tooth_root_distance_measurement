#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理示例
演示如何批量处理多张图像
适合需要处理大量X光片的场景
"""

import sys
import os
import glob
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from teeth_analysis import TeethSegmentationPipeline


def demo_batch_process_simple():
    """
    示例1：简单批量处理
    使用内置的批量处理功能
    """
    print("\n" + "="*60)
    print("示例1：简单批量处理")
    print("="*60)

    # 创建流水线
    pipeline = TeethSegmentationPipeline()

    # 批量处理input文件夹中的所有图像
    # 会自动查找所有支持的图像格式（png, jpg, jpeg, bmp, tiff）
    results = pipeline.batch_analyze(
        input_dir='input',
        output_dir='test/output_batch'
    )

    # 打印汇总信息
    print(f"\n处理完成！")
    print(f"  成功处理: {len(results)} 张图像")

    # 统计总的牙齿数
    total_teeth = sum(len(r['teeth_data']) for r in results)
    print(f"  检测牙齿总数: {total_teeth}")
    print(f"  平均每张: {total_teeth/len(results):.1f} 颗")


def demo_batch_process_custom():
    """
    示例2：自定义批量处理
    手动控制批量处理流程
    """
    print("\n" + "="*60)
    print("示例2：自定义批量处理")
    print("="*60)

    # 创建流水线
    pipeline = TeethSegmentationPipeline()

    # 手动查找图像文件
    input_dir = 'input'
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(input_dir, '*.jpg')))

    print(f"找到 {len(image_files)} 张图像")

    # 创建输出目录
    output_dir = 'test/output_batch_custom'
    os.makedirs(output_dir, exist_ok=True)

    # 逐个处理
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理 [{i}/{len(image_files)}]: {os.path.basename(image_path)}")

        result = pipeline.analyze_image(image_path, output_dir)

        if result:
            results.append(result)
            print(f"  ✓ 检测到 {len(result['teeth_data'])} 颗牙齿")
        else:
            print(f"  ✗ 处理失败")

    print(f"\n批量处理完成！共处理 {len(results)}/{len(image_files)} 张图像")


def demo_batch_with_timing():
    """
    示例3：带时间统计的批量处理
    测量处理速度和性能
    """
    print("\n" + "="*60)
    print("示例3：带时间统计的批量处理")
    print("="*60)

    # 创建流水线
    start_time = time.time()
    pipeline = TeethSegmentationPipeline()
    init_time = time.time() - start_time
    print(f"流水线初始化耗时: {init_time:.2f} 秒")

    # 查找图像
    image_files = glob.glob('input/*.png')
    print(f"\n找到 {len(image_files)} 张图像")

    # 批量处理（带时间统计）
    output_dir = 'test/output_batch_timing'
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    processing_times = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")

        # 计时单张图像处理
        img_start = time.time()
        result = pipeline.analyze_image(image_path, output_dir)
        img_time = time.time() - img_start

        processing_times.append(img_time)

        if result:
            print(f"  检测到 {len(result['teeth_data'])} 颗牙齿")
            print(f"  处理耗时: {img_time:.2f} 秒")

    total_time = time.time() - total_start

    # 打印统计信息
    print("\n" + "="*60)
    print("性能统计")
    print("="*60)
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每张: {total_time/len(image_files):.2f} 秒")
    print(f"最快: {min(processing_times):.2f} 秒")
    print(f"最慢: {max(processing_times):.2f} 秒")
    print(f"处理速度: {len(image_files)/total_time:.2f} 张/秒")


def demo_batch_with_error_handling():
    """
    示例4：带错误处理的批量处理
    处理可能出现的各种错误
    """
    print("\n" + "="*60)
    print("示例4：带错误处理的批量处理")
    print("="*60)

    pipeline = TeethSegmentationPipeline()

    # 查找图像
    image_files = glob.glob('input/*.png')
    output_dir = 'test/output_batch_error_handling'
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    fail_count = 0
    failed_files = []

    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")

            result = pipeline.analyze_image(image_path, output_dir)

            if result and len(result['teeth_data']) > 0:
                success_count += 1
                print(f"  ✓ 成功 - 检测到 {len(result['teeth_data'])} 颗牙齿")
            else:
                fail_count += 1
                failed_files.append(image_path)
                print(f"  ✗ 失败 - 未检测到牙齿")

        except Exception as e:
            fail_count += 1
            failed_files.append(image_path)
            print(f"  ✗ 错误: {str(e)}")

    # 打印汇总
    print("\n" + "="*60)
    print("处理汇总")
    print("="*60)
    print(f"总计: {len(image_files)} 张图像")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")

    if failed_files:
        print(f"\n失败的文件：")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")


def demo_batch_with_filtering():
    """
    示例5：带结果过滤的批量处理
    只保存满足特定条件的结果
    """
    print("\n" + "="*60)
    print("示例5：带结果过滤的批量处理")
    print("="*60)

    pipeline = TeethSegmentationPipeline()

    # 查找图像
    image_files = glob.glob('input/*.png')
    output_dir = 'test/output_batch_filtered'
    os.makedirs(output_dir, exist_ok=True)

    # 设置过滤条件
    MIN_TEETH_COUNT = 10  # 最少牙齿数
    MAX_TEETH_COUNT = 40  # 最多牙齿数

    print(f"过滤条件: 牙齿数在 {MIN_TEETH_COUNT}-{MAX_TEETH_COUNT} 之间")

    valid_results = []
    invalid_results = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")

        result = pipeline.analyze_image(image_path, output_dir)

        if result:
            teeth_count = len(result['teeth_data'])
            print(f"  检测到 {teeth_count} 颗牙齿", end="")

            # 检查是否满足条件
            if MIN_TEETH_COUNT <= teeth_count <= MAX_TEETH_COUNT:
                valid_results.append({
                    'path': image_path,
                    'count': teeth_count,
                    'result': result
                })
                print(" - ✓ 有效")
            else:
                invalid_results.append({
                    'path': image_path,
                    'count': teeth_count,
                    'reason': '牙齿数超出范围'
                })
                print(" - ✗ 无效（超出范围）")

    # 打印汇总
    print("\n" + "="*60)
    print("过滤汇总")
    print("="*60)
    print(f"有效结果: {len(valid_results)} 张")
    print(f"无效结果: {len(invalid_results)} 张")

    if valid_results:
        print(f"\n有效图像：")
        for r in valid_results:
            print(f"  {os.path.basename(r['path'])}: {r['count']} 颗牙齿")

    if invalid_results:
        print(f"\n无效图像：")
        for r in invalid_results:
            print(f"  {os.path.basename(r['path'])}: {r['count']} 颗牙齿 ({r['reason']})")


def main():
    """主函数"""
    print("="*60)
    print("牙齿分割工具 - 批量处理示例")
    print("="*60)

    # 运行示例1：简单批量处理
    demo_batch_process_simple()

    # 运行示例2：自定义批量处理
    # demo_batch_process_custom()

    # 运行示例3：带时间统计
    # demo_batch_with_timing()

    # 运行示例4：带错误处理
    # demo_batch_with_error_handling()

    # 运行示例5：带结果过滤
    # demo_batch_with_filtering()

    print("\n" + "="*60)
    print("批量处理示例运行完成！")
    print("提示：可以取消注释其他示例来运行更多功能")
    print("="*60)


if __name__ == "__main__":
    main()
