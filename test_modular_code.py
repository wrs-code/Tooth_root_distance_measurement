#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模块化代码的导入和基本功能
"""

import sys


def test_imports():
    """测试所有模块是否可以正确导入"""
    print("="*60)
    print("测试模块导入")
    print("="*60)

    try:
        print("\n1. 测试导入核心模块...")
        from teeth_analysis.core import ImagePreprocessor
        from teeth_analysis.core import MaskPostprocessor
        from teeth_analysis.core import TeethContourDetector
        from teeth_analysis.core import UNetInferenceEngine
        print("   ✓ 核心模块导入成功")

        print("\n2. 测试导入可视化模块...")
        from teeth_analysis.visualization import TeethVisualizer
        print("   ✓ 可视化模块导入成功")

        print("\n3. 测试导入流水线模块...")
        from teeth_analysis.pipeline import TeethSegmentationPipeline
        print("   ✓ 流水线模块导入成功")

        print("\n4. 测试从顶层导入...")
        from teeth_analysis import TeethSegmentationPipeline as Pipeline
        print("   ✓ 顶层导入成功")

        return True

    except Exception as e:
        print(f"\n   ❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_instantiation():
    """测试模块实例化"""
    print("\n" + "="*60)
    print("测试模块实例化")
    print("="*60)

    try:
        from teeth_analysis.core import ImagePreprocessor
        from teeth_analysis.core import MaskPostprocessor
        from teeth_analysis.core import TeethContourDetector
        from teeth_analysis.visualization import TeethVisualizer

        print("\n1. 实例化ImagePreprocessor...")
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        print(f"   ✓ 成功 - target_size: {preprocessor.target_size}")

        print("\n2. 实例化MaskPostprocessor...")
        postprocessor = MaskPostprocessor(
            kernel_size=5,
            open_iteration=2,
            erode_iteration=1
        )
        print(f"   ✓ 成功 - open_iteration: {postprocessor.open_iteration}, "
              f"erode_iteration: {postprocessor.erode_iteration}")

        print("\n3. 实例化TeethContourDetector...")
        detector = TeethContourDetector(min_area=2000, connectivity=8)
        print(f"   ✓ 成功 - min_area: {detector.min_area}, "
              f"connectivity: {detector.connectivity}")

        print("\n4. 实例化TeethVisualizer...")
        visualizer = TeethVisualizer()
        print("   ✓ 成功")

        return True

    except Exception as e:
        print(f"\n   ❌ 实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_updates():
    """测试参数更新功能"""
    print("\n" + "="*60)
    print("测试参数更新功能")
    print("="*60)

    try:
        from teeth_analysis.core import MaskPostprocessor
        from teeth_analysis.core import TeethContourDetector

        print("\n1. 测试MaskPostprocessor参数更新...")
        postprocessor = MaskPostprocessor(erode_iteration=1)
        print(f"   初始值: erode_iteration={postprocessor.erode_iteration}")

        postprocessor.update_parameters(erode_iteration=2)
        print(f"   更新后: erode_iteration={postprocessor.erode_iteration}")
        print("   ✓ 参数更新成功")

        print("\n2. 测试TeethContourDetector参数更新...")
        detector = TeethContourDetector(min_area=2000)
        print(f"   初始值: min_area={detector.min_area}")

        detector.update_min_area(3000)
        print(f"   更新后: min_area={detector.min_area}")
        print("   ✓ 参数更新成功")

        return True

    except Exception as e:
        print(f"\n   ❌ 参数更新失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_processing():
    """测试图像处理功能"""
    print("\n" + "="*60)
    print("测试图像处理功能")
    print("="*60)

    try:
        import cv2
        import numpy as np
        from teeth_analysis.core import ImagePreprocessor
        from teeth_analysis.core import MaskPostprocessor

        print("\n1. 创建测试图像...")
        test_image = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
        print(f"   测试图像尺寸: {test_image.shape}")

        print("\n2. 测试图像预处理...")
        preprocessor = ImagePreprocessor()
        gray = preprocessor.convert_to_grayscale(test_image)
        print(f"   灰度图尺寸: {gray.shape}")

        enhanced = preprocessor.apply_clahe(gray)
        print(f"   CLAHE增强后尺寸: {enhanced.shape}")

        resized, original_size = preprocessor.resize_image(gray, size=(512, 512))
        print(f"   调整大小后: {resized.shape}, 原始尺寸: {original_size}")

        print("\n3. 测试掩码后处理...")
        test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
        postprocessor = MaskPostprocessor()

        binary = postprocessor.binarize_mask(test_mask / 255.0, threshold=0.5)
        print(f"   二值化后尺寸: {binary.shape}")

        opened = postprocessor.apply_opening(binary)
        print(f"   开运算后尺寸: {opened.shape}")

        sharpened = postprocessor.apply_sharpening(opened)
        print(f"   锐化后尺寸: {sharpened.shape}")

        eroded = postprocessor.apply_erosion(sharpened)
        print(f"   腐蚀后尺寸: {eroded.shape}")

        print("\n   ✓ 图像处理功能正常")
        return True

    except Exception as e:
        print(f"\n   ❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("模块化代码测试")
    print("="*60)

    results = []

    # 测试1: 导入
    results.append(("模块导入", test_imports()))

    # 测试2: 实例化
    results.append(("模块实例化", test_module_instantiation()))

    # 测试3: 参数更新
    results.append(("参数更新", test_parameter_updates()))

    # 测试4: 图像处理
    results.append(("图像处理", test_image_processing()))

    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"{test_name:20s} : {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
