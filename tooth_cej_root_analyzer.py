#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿轮廓检测系统
基于全景X光片的牙齿分析，使用U-Net深度学习模型进行牙齿边缘检测和分割
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 导入U-Net分割模块
from unet_segmentation import UNetTeethSegmentation

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ToothCEJAnalyzer:
    """牙齿轮廓分析器"""

    def __init__(self):

        # 初始化U-Net分割器
        try:
            self.unet_segmenter = UNetTeethSegmentation()
            self.use_unet = True
            print("✓ U-Net分割器已初始化")
        except Exception as e:
            print(f"⚠ U-Net分割器初始化失败: {e}")
            print("  将使用传统方法（不推荐）")
            self.unet_segmenter = None
            self.use_unet = False

    def order_points(self, pts):
        """
        对矩形的4个顶点排序：左上、右上、右下、左下

        参数:
            pts: 4个点的坐标数组

        返回:
            ordered: 排序后的点
        """
        # 初始化坐标点数组
        rect = np.zeros((4, 2), dtype=np.float32)

        # 计算左上和右下
        # 左上的点具有最小的和，右下的点具有最大的和
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 计算右上和左下
        # 右上的点具有最小的差，左下的点具有最大的差
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def preprocess_image(self, image):
        """
        预处理X光图像

        参数:
            image: 输入图像（BGR或灰度）

        返回:
            processed: 预处理后的二值图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 应用CLAHE（对比度限制自适应直方图均衡化）增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 使用双边滤波保留边缘的同时降噪
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised, gray

    def detect_teeth_contours(self, image):
        """
        检测牙齿轮廓
        使用U-Net深度学习模型进行牙齿分割

        参数:
            image: 预处理后的图像或原始图像

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        if self.use_unet and self.unet_segmenter is not None:
            # 使用U-Net深度学习方法（推荐）
            return self._detect_teeth_with_unet(image)
        else:
            # 使用传统方法（已弃用，不推荐）
            print("⚠ 警告：使用传统分割方法，效果可能不佳")
            return self._detect_teeth_traditional(image)

    def _detect_teeth_with_unet(self, image):
        """
        使用U-Net深度学习模型检测牙齿
        与开源仓库逻辑一致

        参数:
            image: 输入图像

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        # 使用U-Net进行分割
        mask, refined_mask = self.unet_segmenter.segment_teeth(image)

        # 提取单个牙齿（使用开源仓库的默认参数：min_area=2000）
        teeth_data = self.unet_segmenter.extract_individual_teeth(refined_mask)

        # 为每个牙齿添加排序后的box（确保兼容性）
        for tooth in teeth_data:
            if 'box' in tooth:
                tooth['box'] = self.order_points(tooth['box'])

        return teeth_data

    def _detect_teeth_traditional(self, image):
        """
        使用传统方法检测牙齿轮廓（已弃用）
        仅作为后备方案，不推荐使用

        参数:
            image: 预处理后的图像

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        # 二值化 - 使用Otsu自动阈值
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 反转二值图（牙齿区域为白色）
        binary = cv2.bitwise_not(binary)

        # 形态学操作 - 去除小噪声
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        teeth_data = []

        # 遍历每个连通组件
        for label in range(1, num_labels):  # 跳过背景（label=0）
            # 获取组件统计信息
            area = stats[label, cv2.CC_STAT_AREA]

            # 过滤太小的区域（噪声）和太大的区域
            if area < 500 or area > 50000:
                continue

            # 创建单个牙齿的掩码
            tooth_mask = np.zeros(binary.shape, dtype=np.uint8)
            tooth_mask[labels == label] = 255

            # 查找轮廓
            contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            contour = contours[0]

            # 计算轮廓的边界框和中心
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤不合理的形状（宽高比）
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 2.0 or aspect_ratio < 0.2:
                continue

            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            # 对矩形顶点排序（左上、右上、右下、左下）
            box = self.order_points(box)

            teeth_data.append({
                'label': label,
                'contour': contour,
                'mask': tooth_mask,
                'bbox': (x, y, w, h),
                'rect': rect,
                'box': box,
                'centroid': centroids[label],
                'area': area
            })

        # 按X坐标排序（从左到右）
        teeth_data.sort(key=lambda t: t['centroid'][0])

        return teeth_data

    def analyze_single_image(self, image_path, output_dir='output'):
        """
        分析单张全景X光图像

        参数:
            image_path: 图像路径
            output_dir: 输出目录

        返回:
            results: 分析结果
        """
        print(f"\n{'='*60}")
        print(f"分析图像: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        print(f"✓ 图像尺寸: {original_image.shape[1]}x{original_image.shape[0]}")

        # 预处理
        print("正在预处理图像...")
        processed, gray = self.preprocess_image(original_image)

        # 检测牙齿轮廓
        print("正在检测牙齿轮廓...")
        # 如果使用U-Net，传入原始图像；否则使用预处理后的图像
        if self.use_unet:
            teeth_data = self.detect_teeth_contours(original_image)
        else:
            teeth_data = self.detect_teeth_contours(processed)
        print(f"✓ 检测到 {len(teeth_data)} 颗牙齿")

        if len(teeth_data) == 0:
            print("❌ 未检测到牙齿")
            return None

        # 可视化结果
        print("正在生成可视化...")
        self.visualize_results(original_image, teeth_data, image_path, output_dir)

        results = {
            'image_path': image_path,
            'teeth_data': teeth_data
        }

        return results

    def visualize_results(self, original_image, teeth_data, image_path, output_dir):
        """
        可视化分析结果

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            image_path: 原始图像路径
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建图像副本用于绘制
        result_image = original_image.copy()

        # 绘制每颗牙齿的轮廓
        for i, tooth in enumerate(teeth_data):
            contour = tooth['contour']
            # 为每颗牙齿使用不同的颜色
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.drawContours(result_image, [contour], -1, color, 2)

            # 获取牙齿中心位置用于标注编号
            centroid = tooth['centroid']
            # 标注牙齿编号
            cv2.putText(result_image, f'T{i+1}',
                       (int(centroid[0]) - 10, int(centroid[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax.set_title('牙齿轮廓检测', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 保存图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_contours.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化结果已保存: {output_path}")

    def process_input_folder(self, input_folder='input', output_folder='output'):
        """
        处理input文件夹中的所有图像

        参数:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径

        返回:
            all_results: 所有图像的分析结果
        """
        print(f"\n{'='*60}")
        print(f"批量处理模式")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        print(f"{'='*60}")

        # 查找所有图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

        if len(image_files) == 0:
            print(f"❌ 在 {input_folder} 中未找到图像文件")
            return []

        print(f"\n找到 {len(image_files)} 张图像")

        all_results = []

        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]")
            result = self.analyze_single_image(image_path, output_folder)
            if result is not None:
                all_results.append(result)

        # 生成汇总报告
        self.generate_summary_report(all_results, output_folder)

        return all_results

    def generate_summary_report(self, all_results, output_folder):
        """
        生成汇总报告

        参数:
            all_results: 所有分析结果
            output_folder: 输出文件夹
        """
        if len(all_results) == 0:
            return

        print(f"\n{'='*60}")
        print(f"汇总报告")
        print(f"{'='*60}")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("牙齿轮廓检测 - 汇总报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        total_teeth = 0

        for result in all_results:
            image_name = os.path.basename(result['image_path'])
            teeth_count = len(result['teeth_data'])

            total_teeth += teeth_count

            report_lines.append(f"图像: {image_name}")
            report_lines.append(f"  检测到牙齿数量: {teeth_count}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("总体统计")
        report_lines.append("=" * 80)
        report_lines.append(f"处理图像数: {len(all_results)}")
        report_lines.append(f"检测牙齿总数: {total_teeth}")
        report_lines.append("=" * 80)

        # 打印到控制台
        for line in report_lines:
            print(line)

        # 保存到文件
        report_path = os.path.join(output_folder, 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n✓ 汇总报告已保存: {report_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("牙齿轮廓检测系统")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. 自动检测全景X光片中的每颗牙齿")
    print("2. 使用U-Net深度学习模型进行牙齿分割")
    print("3. 绘制并保存牙齿轮廓图像")
    print()
    print("-" * 60)

    # 创建分析器
    analyzer = ToothCEJAnalyzer()

    # 处理input文件夹中的所有图像
    analyzer.process_input_folder(input_folder='input', output_folder='output')

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
