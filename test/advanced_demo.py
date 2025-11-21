#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级调用示例
演示如何使用各个独立模块进行自定义处理
适合需要精细控制处理流程的高级用户
"""

import sys
import os
import cv2
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from teeth_analysis import (
    ImagePreprocessor,
    UNetInferenceEngine,
    MaskPostprocessor,
    TeethContourDetector,
    TeethVisualizer
)


def demo_step_by_step():
    """
    示例1：逐步调用各个模块
    完全控制每个处理步骤
    """
    print("\n" + "="*60)
    print("示例1：逐步调用各个模块")
    print("="*60)

    # ========== 步骤1：读取图像 ==========
    print("\n步骤1：读取图像...")
    image_path = 'input/image.png'
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    print(f"✓ 图像尺寸: {image.shape[1]}x{image.shape[0]}")

    # ========== 步骤2：图像预处理 ==========
    print("\n步骤2：图像预处理...")
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    preprocessed, original_size = preprocessor.prepare_for_unet(image)
    print(f"✓ 预处理后尺寸: {preprocessed.shape}")
    print(f"✓ 原始尺寸: {original_size}")

    # ========== 步骤3：U-Net推理 ==========
    print("\n步骤3：U-Net推理...")
    inference_engine = UNetInferenceEngine(model_path='models/dental_xray_seg.h5')
    prediction = inference_engine.predict(preprocessed, verbose=1)
    print(f"✓ 预测结果尺寸: {prediction.shape}")
    print(f"✓ 预测值范围: [{prediction.min():.3f}, {prediction.max():.3f}]")

    # ========== 步骤4：掩码后处理 ==========
    print("\n步骤4：掩码后处理...")
    postprocessor = MaskPostprocessor(
        kernel_size=5,
        open_iteration=2,  # 开运算迭代次数
        erode_iteration=1  # 腐蚀迭代次数
    )
    binary_mask, refined_mask = postprocessor.postprocess_prediction(
        prediction, original_size, threshold=0.5
    )
    print(f"✓ 二值掩码尺寸: {binary_mask.shape}")
    print(f"✓ 细化掩码尺寸: {refined_mask.shape}")

    # ========== 步骤5：提取牙齿轮廓 ==========
    print("\n步骤5：提取牙齿轮廓...")
    detector = TeethContourDetector(min_area=2000, connectivity=8)
    teeth_data = detector.extract_teeth_from_mask(refined_mask)
    print(f"✓ 检测到 {len(teeth_data)} 颗牙齿")

    # ========== 步骤6：可视化结果 ==========
    print("\n步骤6：可视化结果...")
    visualizer = TeethVisualizer()
    os.makedirs('test/output_advanced', exist_ok=True)
    visualizer.visualize_segmentation_result(
        image, refined_mask, teeth_data,
        output_path='test/output_advanced/step_by_step_result.png'
    )
    print(f"✓ 结果已保存")

    # ========== 打印详细信息 ==========
    print("\n牙齿轮廓信息：")
    for i, tooth in enumerate(teeth_data, 1):
        print(f"  轮廓 {i}:")
        print(f"    - 面积: {tooth['area']:.0f} 像素")
        print(f"    - 中心: ({tooth['centroid'][0]:.1f}, {tooth['centroid'][1]:.1f})")
        print(f"    - 边界框: {tooth['bbox']}")


def demo_custom_parameters():
    """
    示例2：使用自定义参数
    调整各个模块的参数以适应不同的图像
    """
    print("\n" + "="*60)
    print("示例2：使用自定义参数")
    print("="*60)

    image = cv2.imread('input/image.png')

    # 使用自定义参数
    print("\n使用自定义参数：")
    print("  - 开运算迭代次数: 3（更强的去噪）")
    print("  - 腐蚀迭代次数: 2（更好的牙齿分离）")
    print("  - 最小面积阈值: 3000（过滤更小的噪声）")

    # 创建各模块（使用自定义参数）
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    inference_engine = UNetInferenceEngine(model_path='models/dental_xray_seg.h5')
    postprocessor = MaskPostprocessor(
        kernel_size=5,
        open_iteration=3,   # 增强去噪
        erode_iteration=2   # 增强分离
    )
    detector = TeethContourDetector(
        min_area=3000,      # 提高阈值
        connectivity=8
    )
    visualizer = TeethVisualizer()

    # 执行处理
    preprocessed, original_size = preprocessor.prepare_for_unet(image)
    prediction = inference_engine.predict(preprocessed, verbose=0)
    binary_mask, refined_mask = postprocessor.postprocess_prediction(
        prediction, original_size
    )
    teeth_data = detector.extract_teeth_from_mask(refined_mask)

    print(f"\n✓ 牙齿分割完成")

    # 保存结果
    visualizer.visualize_segmentation_result(
        image, refined_mask, teeth_data,
        output_path='test/output_advanced/custom_parameters_result.png'
    )


def demo_compare_parameters():
    """
    示例3：对比不同参数的效果
    """
    print("\n" + "="*60)
    print("示例3：对比不同参数的效果")
    print("="*60)

    image = cv2.imread('input/image.png')
    preprocessor = ImagePreprocessor()
    inference_engine = UNetInferenceEngine(model_path='models/dental_xray_seg.h5')

    # 预处理和推理（只做一次）
    preprocessed, original_size = preprocessor.prepare_for_unet(image)
    prediction = inference_engine.predict(preprocessed, verbose=0)

    # 测试不同的腐蚀参数
    erode_values = [0, 1, 2, 3]

    print("\n对比不同腐蚀参数的效果：")
    for erode in erode_values:
        postprocessor = MaskPostprocessor(erode_iteration=erode)
        detector = TeethContourDetector(min_area=2000)
        visualizer = TeethVisualizer()

        binary_mask, refined_mask = postprocessor.postprocess_prediction(
            prediction, original_size
        )
        teeth_data = detector.extract_teeth_from_mask(refined_mask)

        print(f"  腐蚀次数={erode}: 轮廓数量 {len(teeth_data)}")

        # 保存结果
        visualizer.visualize_segmentation_result(
            image, refined_mask, teeth_data,
            output_path=f'test/output_advanced/erode_{erode}_result.png'
        )

    print(f"\n✓ 所有对比结果已保存到: test/output_advanced/")


def demo_access_individual_components():
    """
    示例4：访问和使用各个组件的高级功能
    """
    print("\n" + "="*60)
    print("示例4：访问各个组件的高级功能")
    print("="*60)

    image = cv2.imread('input/image.png')

    # ========== 预处理器高级功能 ==========
    print("\n预处理器高级功能：")
    preprocessor = ImagePreprocessor()

    # 归一化图像
    normalized = preprocessor.normalize_image(image)
    print(f"  归一化后值范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # 调整尺寸
    resized = preprocessor.resize_image(image, (512, 512))
    print(f"  调整后尺寸: {resized.shape}")

    # ========== 掩码后处理器高级功能 ==========
    print("\n掩码后处理器高级功能：")
    postprocessor = MaskPostprocessor()

    # 创建一个测试掩码
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[30:70, 30:70] = 255

    # 开运算
    opened = postprocessor.apply_morphological_opening(test_mask)
    print(f"  开运算后非零像素数: {np.count_nonzero(opened)}")

    # 腐蚀
    eroded = postprocessor.apply_erosion(test_mask)
    print(f"  腐蚀后非零像素数: {np.count_nonzero(eroded)}")

    # ========== 轮廓检测器高级功能 ==========
    print("\n轮廓检测器高级功能：")
    detector = TeethContourDetector()

    # 执行完整的分割流程获取掩码
    preprocessed, original_size = preprocessor.prepare_for_unet(image)
    inference_engine = UNetInferenceEngine()
    prediction = inference_engine.predict(preprocessed, verbose=0)
    _, refined_mask = postprocessor.postprocess_prediction(prediction, original_size)

    # 查找所有轮廓（包括小的）
    all_contours = detector.find_all_contours(refined_mask)
    print(f"  找到所有轮廓数: {len(all_contours)}")

    # 过滤轮廓（只保留大的）
    filtered = detector.filter_contours_by_area(all_contours, min_area=2000)
    print(f"  过滤后轮廓数: {len(filtered)}")


def main():
    """主函数"""
    print("="*60)
    print("牙齿分割工具 - 高级调用示例")
    print("="*60)

    # 运行示例1：逐步调用
    demo_step_by_step()

    # 运行示例2：自定义参数
    demo_custom_parameters()

    # 运行示例3：对比参数
    demo_compare_parameters()

    # 运行示例4：访问高级功能
    demo_access_individual_components()

    print("\n" + "="*60)
    print("所有高级示例运行完成！")
    print("结果保存在: test/output_advanced/")
    print("="*60)


if __name__ == "__main__":
    main()
