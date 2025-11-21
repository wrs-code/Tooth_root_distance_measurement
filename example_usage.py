#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化牙齿分割工具使用示例
演示如何使用重构后的模块化代码
"""

import cv2
from teeth_analysis import TeethSegmentationPipeline


def example1_simple_usage():
    """示例1：简单使用 - 分析单张图像"""
    print("\n" + "="*60)
    print("示例1：简单使用")
    print("="*60)

    # 创建流水线（使用默认参数）
    pipeline = TeethSegmentationPipeline()

    # 分析单张图像
    results = pipeline.analyze_image('input/107.png', output_dir='output')

    if results:
        print(f"\n分析完成！")


def example2_batch_processing():
    """示例2：批量处理"""
    print("\n" + "="*60)
    print("示例2：批量处理")
    print("="*60)

    # 创建流水线
    pipeline = TeethSegmentationPipeline()

    # 批量分析input文件夹中的所有图像
    all_results = pipeline.batch_analyze(
        input_dir='input',
        output_dir='output'
    )

    print(f"\n批量处理完成！共处理 {len(all_results)} 张图像")


def example3_adjust_erosion():
    """示例3：调整腐蚀参数"""
    print("\n" + "="*60)
    print("示例3：调整腐蚀参数")
    print("="*60)

    # 创建流水线
    pipeline = TeethSegmentationPipeline()

    # 调整腐蚀参数（增强牙齿分离）
    print("\n调整腐蚀参数：erode_iteration=2（中度腐蚀）")
    pipeline.update_erosion_parameters(erode_iteration=2)

    # 分析图像
    results = pipeline.analyze_image('input/107.png', output_dir='output_erosion2')

    if results:
        print(f"\n分析完成！")


def example4_custom_parameters():
    """示例4：自定义参数"""
    print("\n" + "="*60)
    print("示例4：自定义参数")
    print("="*60)

    # 创建流水线时指定自定义参数
    pipeline = TeethSegmentationPipeline(
        model_path='models/dental_xray_seg.h5',
        open_iteration=3,      # 增强开运算
        erode_iteration=2,     # 增强腐蚀
        min_area=3000          # 提高面积阈值
    )

    # 分析图像
    results = pipeline.analyze_image('input/107.png', output_dir='output_custom')

    if results:
        print(f"\n分析完成！")


def example5_advanced_usage():
    """示例5：高级使用 - 单独使用各个模块"""
    print("\n" + "="*60)
    print("示例5：高级使用 - 单独使用各个模块")
    print("="*60)

    from teeth_analysis.core import ImagePreprocessor, MaskPostprocessor
    from teeth_analysis.core import TeethContourDetector, UNetInferenceEngine
    from teeth_analysis.visualization import TeethVisualizer

    # 1. 读取图像
    image = cv2.imread('input/107.png')

    # 2. 创建各个模块
    preprocessor = ImagePreprocessor()
    inference_engine = UNetInferenceEngine()
    postprocessor = MaskPostprocessor(erode_iteration=1)
    detector = TeethContourDetector(min_area=2000)
    visualizer = TeethVisualizer()

    # 3. 执行各个步骤
    print("\n步骤1：预处理图像...")
    preprocessed, original_size = preprocessor.prepare_for_unet(image)

    print("步骤2：U-Net推理...")
    prediction = inference_engine.predict(preprocessed)

    print("步骤3：后处理掩码...")
    binary_mask, refined_mask = postprocessor.postprocess_prediction(
        prediction, original_size
    )

    print("步骤4：提取牙齿轮廓...")
    teeth_data = detector.extract_teeth_from_mask(refined_mask)

    print("步骤5：可视化结果...")
    visualizer.visualize_segmentation_result(
        image, refined_mask, teeth_data,
        'output/advanced_result.png'
    )

    print(f"\n完成！")


def main():
    """主函数"""
    print("="*60)
    print("模块化牙齿分割工具 - 使用示例")
    print("="*60)

    # 运行各个示例
    example1_simple_usage()
    # example2_batch_processing()
    # example3_adjust_erosion()
    # example4_custom_parameters()
    # example5_advanced_usage()

    print("\n" + "="*60)
    print("所有示例完成！")
    print("="*60)


if __name__ == "__main__":
    main()
