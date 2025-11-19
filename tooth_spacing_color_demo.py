#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿间距深度测量与颜色编码Demo
在CEJ线下方不同深度处测量牙齿间距，并根据间距值进行颜色编码
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ToothSpacingAnalyzer:
    """牙齿间距分析器"""

    def __init__(self):
        # 间距阈值 (mm)
        self.DANGER_THRESHOLD = 3.2  # 红色：< 3.2mm
        self.WARNING_THRESHOLD = 4.0  # 黄色：3.2-4.0mm，绿色：>= 4.0mm

    def create_tooth_contour(self, center_x, width=8, crown_height=10, root_height=12):
        """
        创建牙齿轮廓（简化的牙齿形状）

        参数:
            center_x: 牙齿中心X坐标
            width: 牙齿宽度
            crown_height: 牙冠高度
            root_height: 牙根高度

        返回:
            x, y坐标数组
        """
        # 牙冠部分（梯形）
        crown_top_width = width * 0.8
        crown_x = [
            center_x - crown_top_width/2,
            center_x + crown_top_width/2,
            center_x + width/2,
            center_x - width/2
        ]
        crown_y = [crown_height, crown_height, 0, 0]

        # 牙根部分（锥形）
        root_bottom_width = width * 0.3
        root_x = [
            center_x + width/2,
            center_x + root_bottom_width/2,
            center_x - root_bottom_width/2,
            center_x - width/2
        ]
        root_y = [0, -root_height, -root_height, 0]

        # 合并轮廓
        x = crown_x + root_x
        y = crown_y + root_y

        return np.array(x), np.array(y)

    def get_tooth_width_at_depth(self, tooth_contour_x, tooth_contour_y, depth):
        """
        获取牙齿在指定深度处的左右边界

        参数:
            tooth_contour_x: 牙齿轮廓X坐标
            tooth_contour_y: 牙齿轮廓Y坐标
            depth: 深度（负值，从CEJ线向下）

        返回:
            left_x, right_x: 左右边界X坐标，如果该深度没有牙齿则返回None
        """
        # 找到该深度的交点
        intersections = []

        for i in range(len(tooth_contour_x)):
            j = (i + 1) % len(tooth_contour_x)
            y1, y2 = tooth_contour_y[i], tooth_contour_y[j]
            x1, x2 = tooth_contour_x[i], tooth_contour_x[j]

            # 检查线段是否跨越该深度
            if (y1 <= depth <= y2) or (y2 <= depth <= y1):
                if abs(y2 - y1) < 1e-6:  # 水平线
                    intersections.extend([x1, x2])
                else:
                    # 线性插值计算交点
                    t = (depth - y1) / (y2 - y1)
                    x = x1 + t * (x2 - x1)
                    intersections.append(x)

        if len(intersections) < 2:
            return None, None

        intersections = sorted(intersections)
        return intersections[0], intersections[-1]

    def measure_spacing_at_depth(self, tooth1_x, tooth1_y, tooth2_x, tooth2_y, depth):
        """
        测量两颗牙齿在指定深度处的间距

        参数:
            tooth1_x, tooth1_y: 第一颗牙齿的轮廓坐标
            tooth2_x, tooth2_y: 第二颗牙齿的轮廓坐标
            depth: 深度（负值）

        返回:
            spacing: 间距值（mm），如果无法测量则返回None
        """
        left1, right1 = self.get_tooth_width_at_depth(tooth1_x, tooth1_y, depth)
        left2, right2 = self.get_tooth_width_at_depth(tooth2_x, tooth2_y, depth)

        if left1 is None or left2 is None:
            return None

        # 假设tooth1在左侧，tooth2在右侧
        if right1 < left2:
            spacing = left2 - right1
        elif right2 < left1:
            spacing = left1 - right2
        else:
            # 重叠
            spacing = 0

        return spacing

    def get_color_for_spacing(self, spacing):
        """
        根据间距值返回对应的颜色

        参数:
            spacing: 间距值（mm）

        返回:
            color: 颜色字符串
            label: 风险级别标签
        """
        if spacing < self.DANGER_THRESHOLD:
            return '#FF4444', '危险'  # 红色
        elif spacing < self.WARNING_THRESHOLD:
            return '#FFDD44', '相对安全'  # 黄色
        else:
            return '#44FF44', '安全'  # 绿色

    def analyze_and_visualize(self, tooth1_center=20, tooth2_center=32,
                             cej_depth=0, max_depth=15, depth_step=0.5):
        """
        分析并可视化牙齿间距

        参数:
            tooth1_center: 第一颗牙齿中心X坐标
            tooth2_center: 第二颗牙齿中心X坐标
            cej_depth: CEJ线深度（通常为0）
            max_depth: 最大测量深度
            depth_step: 深度步长
        """
        # 创建牙齿轮廓
        tooth1_x, tooth1_y = self.create_tooth_contour(tooth1_center)
        tooth2_x, tooth2_y = self.create_tooth_contour(tooth2_center)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # ===== 左图：牙齿轮廓和颜色编码区域 =====
        ax1.set_xlim(0, 60)
        ax1.set_ylim(-max_depth - 2, 15)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('水平位置 (mm)', fontsize=12)
        ax1.set_ylabel('深度 (mm)', fontsize=12)
        ax1.set_title('牙齿间距深度分析与颜色编码', fontsize=14, fontweight='bold')

        # 绘制CEJ线
        ax1.axhline(y=cej_depth, color='blue', linewidth=2,
                   linestyle='--', label='CEJ线（釉牙骨质界）', alpha=0.7)

        # 绘制牙齿轮廓
        ax1.plot(np.append(tooth1_x, tooth1_x[0]),
                np.append(tooth1_y, tooth1_y[0]),
                'k-', linewidth=2, label='牙齿轮廓')
        ax1.plot(np.append(tooth2_x, tooth2_x[0]),
                np.append(tooth2_y, tooth2_y[0]),
                'k-', linewidth=2)

        # 在不同深度处测量间距并绘制颜色区域
        depths = np.arange(cej_depth - depth_step, -max_depth, -depth_step)
        color_patches = []

        for i in range(len(depths) - 1):
            depth_top = depths[i]
            depth_bottom = depths[i + 1]
            depth_mid = (depth_top + depth_bottom) / 2

            # 测量中间深度的间距
            spacing = self.measure_spacing_at_depth(
                tooth1_x, tooth1_y, tooth2_x, tooth2_y, depth_mid
            )

            if spacing is not None:
                # 获取该深度两颗牙齿的边界
                left1, right1 = self.get_tooth_width_at_depth(
                    tooth1_x, tooth1_y, depth_mid
                )
                left2, right2 = self.get_tooth_width_at_depth(
                    tooth2_x, tooth2_y, depth_mid
                )

                # 确定间隙的左右边界
                gap_left = min(right1, right2)
                gap_right = max(left1, left2)
                if right1 < left2:
                    gap_left = right1
                    gap_right = left2
                elif right2 < left1:
                    gap_left = right2
                    gap_right = left1

                # 根据间距获取颜色
                color, _ = self.get_color_for_spacing(spacing)

                # 创建填充区域（矩形）
                rect_x = [gap_left, gap_right, gap_right, gap_left]
                rect_y = [depth_top, depth_top, depth_bottom, depth_bottom]
                polygon = Polygon(list(zip(rect_x, rect_y)),
                                facecolor=color, edgecolor='none', alpha=0.6)
                color_patches.append(polygon)
                ax1.add_patch(polygon)

        # 添加图例
        danger_patch = mpatches.Patch(color='#FF4444', label='危险 (< 3.2mm)', alpha=0.6)
        warning_patch = mpatches.Patch(color='#FFDD44', label='相对安全 (3.2-4.0mm)', alpha=0.6)
        safe_patch = mpatches.Patch(color='#44FF44', label='安全 (≥ 4.0mm)', alpha=0.6)

        handles, labels = ax1.get_legend_handles_labels()
        handles.extend([danger_patch, warning_patch, safe_patch])
        ax1.legend(handles=handles, loc='upper right', fontsize=10)

        # ===== 右图：间距-深度曲线 =====
        ax2.set_xlabel('间距 (mm)', fontsize=12)
        ax2.set_ylabel('CEJ线下深度 (mm)', fontsize=12)
        ax2.set_title('间距随深度变化曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 计算并绘制间距曲线
        measurement_depths = np.arange(cej_depth - 0.2, -max_depth, -0.2)
        spacings = []
        valid_depths = []
        colors = []

        for depth in measurement_depths:
            spacing = self.measure_spacing_at_depth(
                tooth1_x, tooth1_y, tooth2_x, tooth2_y, depth
            )
            if spacing is not None:
                spacings.append(spacing)
                valid_depths.append(-depth)  # 转换为正值显示
                color, _ = self.get_color_for_spacing(spacing)
                colors.append(color)

        # 绘制散点，颜色编码
        for i in range(len(spacings)):
            ax2.scatter(spacings[i], valid_depths[i],
                       c=colors[i], s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

        # 绘制曲线
        ax2.plot(spacings, valid_depths, 'b-', alpha=0.3, linewidth=1)

        # 添加阈值线
        ax2.axvline(x=self.DANGER_THRESHOLD, color='red',
                   linestyle='--', linewidth=2, alpha=0.5, label=f'危险阈值 ({self.DANGER_THRESHOLD}mm)')
        ax2.axvline(x=self.WARNING_THRESHOLD, color='orange',
                   linestyle='--', linewidth=2, alpha=0.5, label=f'警告阈值 ({self.WARNING_THRESHOLD}mm)')

        # 填充颜色区域
        ax2.axvspan(0, self.DANGER_THRESHOLD, alpha=0.2, color='red', label='危险区')
        ax2.axvspan(self.DANGER_THRESHOLD, self.WARNING_THRESHOLD,
                   alpha=0.2, color='yellow', label='警告区')
        ax2.axvspan(self.WARNING_THRESHOLD, max(spacings) * 1.1,
                   alpha=0.2, color='green', label='安全区')

        ax2.legend(loc='best', fontsize=10)
        ax2.invert_yaxis()  # 深度向下增加

        # 添加统计信息
        min_spacing = min(spacings)
        max_spacing = max(spacings)
        avg_spacing = np.mean(spacings)

        min_color, min_label = self.get_color_for_spacing(min_spacing)

        stats_text = f'统计信息:\n'
        stats_text += f'最小间距: {min_spacing:.2f}mm ({min_label})\n'
        stats_text += f'最大间距: {max_spacing:.2f}mm\n'
        stats_text += f'平均间距: {avg_spacing:.2f}mm\n'
        stats_text += f'测量深度范围: 0-{max_depth}mm'

        ax2.text(0.98, 0.02, stats_text,
                transform=ax2.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        plt.tight_layout()
        plt.savefig('tooth_spacing_depth_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ 可视化图像已保存为: tooth_spacing_depth_analysis.png")
        plt.show()

        return spacings, valid_depths


def main():
    """主函数"""
    print("=" * 60)
    print("牙齿间距深度测量与颜色编码Demo")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. 在CEJ线（釉牙骨质界）下方不同深度处测量牙齿间距")
    print("2. 根据间距值进行颜色编码：")
    print("   • 间距 < 3.2mm     → 红色（危险）")
    print("   • 3.2mm ≤ 间距 < 4.0mm → 黄色（相对安全）")
    print("   • 间距 ≥ 4.0mm     → 绿色（安全）")
    print("3. 可视化展示深度-间距的颜色热力图")
    print()
    print("-" * 60)

    # 创建分析器
    analyzer = ToothSpacingAnalyzer()

    # 分析并可视化（模拟两颗牙齿，间距逐渐变化）
    print("正在分析牙齿间距...")
    spacings, depths = analyzer.analyze_and_visualize(
        tooth1_center=20,    # 第一颗牙齿中心
        tooth2_center=32,    # 第二颗牙齿中心（间距约12mm，但根部会更近）
        cej_depth=0,         # CEJ线在Y=0处
        max_depth=15,        # 测量深度15mm
        depth_step=0.5       # 每0.5mm测量一次
    )

    print()
    print("=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
