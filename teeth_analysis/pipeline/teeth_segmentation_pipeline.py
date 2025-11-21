#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿分割流水线模块
整合所有功能模块，提供完整的牙齿分割和分析流程
"""

import cv2
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_preprocessor import ImagePreprocessor
from core.mask_postprocessor import MaskPostprocessor
from core.teeth_contour_detector import TeethContourDetector
from core.unet_inference_engine import UNetInferenceEngine
from visualization.teeth_visualizer import TeethVisualizer


class TeethSegmentationPipeline:
    """
    牙齿分割流水线
    整合图像预处理、U-Net推理、掩码后处理、轮廓检测和可视化
    """

    def __init__(self, model_path='models/dental_xray_seg.h5',
                 open_iteration=2, erode_iteration=1, min_area=2000):
        """
        初始化分割流水线

        参数:
            model_path: U-Net模型路径
            open_iteration: 开运算迭代次数（默认2，与开源仓库一致）
            erode_iteration: 腐蚀迭代次数（默认1，与开源仓库一致）⚙️
            min_area: 最小牙齿面积阈值（默认2000，与开源仓库一致）
        """
        # 初始化各个功能模块
        print("正在初始化牙齿分割流水线...")

        # 1. 图像预处理器
        self.preprocessor = ImagePreprocessor(target_size=(512, 512))

        # 2. U-Net推理引擎
        try:
            self.inference_engine = UNetInferenceEngine(model_path=model_path)
            self.use_unet = True
        except Exception as e:
            print(f"⚠ U-Net推理引擎初始化失败: {e}")
            print("  流水线将不可用")
            self.use_unet = False
            raise

        # 3. 掩码后处理器
        self.postprocessor = MaskPostprocessor(
            kernel_size=5,
            open_iteration=open_iteration,
            erode_iteration=erode_iteration
        )

        # 4. 牙齿轮廓检测器
        self.contour_detector = TeethContourDetector(
            min_area=min_area,
            connectivity=8
        )

        # 5. 可视化器
        self.visualizer = TeethVisualizer()

        print("✓ 流水线初始化完成")

    def segment_teeth(self, image, threshold=0.5):
        """
        完整的牙齿分割流程

        参数:
            image: 输入图像（BGR或灰度）
            threshold: 二值化阈值（默认0.5）

        返回:
            results: 分割结果字典
                - binary_mask: 二值化掩码
                - refined_mask: 细化后的掩码
                - teeth_data: 牙齿数据列表
        """
        if not self.use_unet:
            raise RuntimeError("U-Net未初始化，无法进行分割")

        # Step 1: 图像预处理（为U-Net准备）
        preprocessed, original_size = self.preprocessor.prepare_for_unet(image)

        # Step 2: U-Net推理
        prediction = self.inference_engine.predict(preprocessed, verbose=0)

        # Step 3: 掩码后处理
        binary_mask, refined_mask = self.postprocessor.postprocess_prediction(
            prediction, original_size, threshold=threshold
        )

        # Step 4: 提取牙齿轮廓
        teeth_data = self.contour_detector.extract_teeth_from_mask(refined_mask)

        results = {
            'binary_mask': binary_mask,
            'refined_mask': refined_mask,
            'teeth_data': teeth_data
        }

        return results

    def analyze_image(self, image_path, output_dir='output', save_comparison=True):
        """
        分析单张图像的完整流程

        参数:
            image_path: 图像路径
            output_dir: 输出目录
            save_comparison: 是否保存对比图

        返回:
            results: 分析结果
        """
        print(f"\n{'='*60}")
        print(f"分析图像: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 1. 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        print(f"✓ 图像尺寸: {image.shape[1]}x{image.shape[0]}")

        # 2. 分割牙齿
        print("正在进行牙齿分割...")
        results = self.segment_teeth(image)

        if len(results['teeth_data']) == 0:
            print("❌ 未检测到牙齿轮廓")
            return None

        print(f"✓ 牙齿分割完成")

        # 3. 可视化并保存结果
        print("正在生成可视化...")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if save_comparison:
            # 保存对比图
            output_path = os.path.join(output_dir, f'{base_name}_comparison.png')
            self.visualizer.visualize_segmentation_result(
                image,
                results['refined_mask'],
                results['teeth_data'],
                output_path
            )
        else:
            # 仅保存轮廓结果
            output_path = os.path.join(output_dir, f'{base_name}_contours.png')
            self.visualizer.create_simple_result(
                image,
                results['teeth_data'],
                output_path
            )

        # 4. 添加图像路径到结果
        results['image_path'] = image_path

        return results

    def batch_analyze(self, input_dir='input', output_dir='output'):
        """
        批量分析图像

        参数:
            input_dir: 输入目录
            output_dir: 输出目录

        返回:
            all_results: 所有图像的分析结果列表
        """
        print(f"\n{'='*60}")
        print(f"批量分析模式")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}")

        # 查找所有图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []

        import glob
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if len(image_files) == 0:
            print(f"❌ 在 {input_dir} 中未找到图像文件")
            return []

        print(f"\n找到 {len(image_files)} 张图像")

        all_results = []

        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]")
            result = self.analyze_image(image_path, output_dir)
            if result is not None:
                all_results.append(result)

        # 生成汇总报告
        self._generate_summary_report(all_results, output_dir)

        return all_results

    def _generate_summary_report(self, all_results, output_dir):
        """
        生成汇总报告

        参数:
            all_results: 所有分析结果
            output_dir: 输出目录
        """
        if len(all_results) == 0:
            return

        print(f"\n{'='*60}")
        print(f"汇总报告")
        print(f"{'='*60}")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("牙齿分割分析 - 汇总报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        for result in all_results:
            image_name = os.path.basename(result['image_path'])
            report_lines.append(f"图像: {image_name}")
            report_lines.append(f"  状态: 分割完成")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("总体统计")
        report_lines.append("=" * 80)
        report_lines.append(f"处理图像数: {len(all_results)}")
        report_lines.append(f"成功分割: {len(all_results)}")
        report_lines.append("=" * 80)

        # 打印到控制台
        for line in report_lines:
            print(line)

        # 保存到文件
        report_path = os.path.join(output_dir, 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n✓ 汇总报告已保存: {report_path}")

    def update_erosion_parameters(self, erode_iteration):
        """
        动态更新腐蚀参数
        ⚙️ 用于调整牙齿分离程度

        参数:
            erode_iteration: 新的腐蚀迭代次数

        使用示例:
            >>> pipeline = TeethSegmentationPipeline()
            >>> pipeline.update_erosion_parameters(erode_iteration=2)
        """
        self.postprocessor.update_parameters(erode_iteration=erode_iteration)
        print(f"✓ 腐蚀参数已更新: erode_iteration={erode_iteration}")

    def update_area_threshold(self, min_area):
        """
        动态更新最小面积阈值

        参数:
            min_area: 新的最小面积阈值

        使用示例:
            >>> pipeline = TeethSegmentationPipeline()
            >>> pipeline.update_area_threshold(min_area=3000)
        """
        self.contour_detector.update_min_area(min_area)
        print(f"✓ 面积阈值已更新: min_area={min_area}")


def main():
    """主函数：演示流水线的使用"""
    print("=" * 60)
    print("牙齿分割流水线")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. 使用U-Net深度学习模型进行牙齿分割")
    print("2. 提取牙齿轮廓和形态学特征")
    print("3. 与开源仓库完全一致的后处理流程")
    print("4. 生成可视化结果和汇总报告")
    print()
    print("-" * 60)

    # 创建流水线
    try:
        pipeline = TeethSegmentationPipeline()
    except Exception as e:
        print(f"❌ 流水线初始化失败: {e}")
        return

    # 批量处理图像
    pipeline.batch_analyze(input_dir='input', output_folder='output')

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
