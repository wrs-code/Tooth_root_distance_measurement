#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿可视化器模块
负责绘制和可视化牙齿分割结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class TeethVisualizer:
    """牙齿分割结果可视化器"""

    def __init__(self):
        """初始化可视化器"""
        # 配置matplotlib支持中文显示
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def draw_mask_overlay(self, image, mask, color=(0, 0, 255), alpha=0.3):
        """
        在图像上叠加掩码

        参数:
            image: 原始图像
            mask: 二值掩码
            color: 掩码颜色 (B, G, R)
            alpha: 透明度

        返回:
            overlay: 叠加后的图像
        """
        # 确保图像是BGR格式
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # 创建掩码覆盖层
        mask_color = np.zeros_like(vis_image)
        mask_color[:, :, 0] = mask * (color[0] / 255)  # B
        mask_color[:, :, 1] = mask * (color[1] / 255)  # G
        mask_color[:, :, 2] = mask * (color[2] / 255)  # R

        # 半透明叠加
        overlay = cv2.addWeighted(vis_image, 1 - alpha, mask_color, alpha, 0)

        return overlay

    def draw_teeth_contours(self, image, teeth_data, draw_bbox=True, draw_centroid=True, draw_label=True):
        """
        在图像上绘制牙齿轮廓

        参数:
            image: 原始图像
            teeth_data: 牙齿数据列表
            draw_bbox: 是否绘制边界框
            draw_centroid: 是否绘制质心
            draw_label: 是否标注编号

        返回:
            result_image: 绘制后的图像
        """
        # 确保图像是BGR格式
        if len(image.shape) == 2:
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result_image = image.copy()

        # 绘制每颗牙齿的轮廓
        for i, tooth in enumerate(teeth_data):
            contour = tooth['contour']

            # 为每颗牙齿使用不同的颜色
            color = tuple(np.random.randint(100, 255, 3).tolist())

            # 绘制轮廓（蓝色）
            cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 2)

            # 获取牙齿中心位置
            centroid = tooth['centroid']
            centroid_int = (int(centroid[0]), int(centroid[1]))

            # 绘制中心点
            if draw_centroid:
                cv2.circle(result_image, centroid_int, 5, (0, 255, 0), -1)

            # 绘制边界框
            if draw_bbox:
                x, y, w, h = tooth['bbox']
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 1)

            # 标注牙齿编号
            if draw_label:
                label_pos = (centroid_int[0] - 10, centroid_int[1])
                cv2.putText(result_image, f'T{i+1}',
                           label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return result_image

    def create_comparison_figure(self, original, mask, result, teeth_count):
        """
        创建对比图：原图 | 掩码 | 结果

        参数:
            original: 原始图像
            mask: 分割掩码
            result: 结果图像
            teeth_count: 检测到的牙齿数量

        返回:
            fig: matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 分割掩码
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('牙齿分割掩码', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 检测结果
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'牙齿轮廓检测 (检测到 {teeth_count} 颗)', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        return fig

    def save_visualization(self, image, output_path, dpi=300):
        """
        保存可视化结果

        参数:
            image: 图像（OpenCV格式或matplotlib figure）
            output_path: 输出路径
            dpi: 分辨率
        """
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 判断是OpenCV图像还是matplotlib figure
        if isinstance(image, plt.Figure):
            image.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(image)
        else:
            cv2.imwrite(output_path, image)

    def visualize_segmentation_result(self, original_image, mask, teeth_data, output_path):
        """
        完整的可视化流程：绘制并保存结果

        参数:
            original_image: 原始图像
            mask: 分割掩码
            teeth_data: 牙齿数据列表
            output_path: 输出路径
        """
        # 绘制轮廓
        result_image = self.draw_teeth_contours(original_image, teeth_data)

        # 创建对比图
        fig = self.create_comparison_figure(
            original_image,
            mask,
            result_image,
            len(teeth_data)
        )

        # 保存结果
        self.save_visualization(fig, output_path)

        print(f"✓ 可视化结果已保存: {output_path}")

    def create_simple_result(self, original_image, teeth_data, output_path):
        """
        创建简单的结果图（仅轮廓）

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            output_path: 输出路径
        """
        # 绘制轮廓
        result_image = self.draw_teeth_contours(original_image, teeth_data)

        # 创建单图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax.set_title('牙齿轮廓检测', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 保存结果
        self.save_visualization(fig, output_path)

        print(f"✓ 可视化结果已保存: {output_path}")

    def draw_individual_teeth(self, original_image, teeth_data, output_dir):
        """
        单独保存每颗牙齿的图像

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, tooth in enumerate(teeth_data):
            # 获取边界框
            x, y, w, h = tooth['bbox']

            # 添加边距
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(original_image.shape[1], x + w + margin)
            y2 = min(original_image.shape[0], y + h + margin)

            # 裁剪牙齿区域
            tooth_roi = original_image[y1:y2, x1:x2]

            # 保存
            tooth_path = os.path.join(output_dir, f'tooth_{i+1}.png')
            cv2.imwrite(tooth_path, tooth_roi)

        print(f"✓ 单个牙齿图像已保存到: {output_dir}")
