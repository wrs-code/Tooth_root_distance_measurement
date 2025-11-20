#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿CEJ线检测与根部距离测量系统
基于全景X光片的牙齿分析，包括：
1. 牙齿边缘检测和分割（使用U-Net深度学习模型）
2. CEJ线（釉牙骨质界）识别
3. 基于CEJ线法线方向的深度测量
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import matplotlib.patches as mpatches
from scipy.spatial import distance as dist
from scipy import ndimage
import os
import glob
from pathlib import Path

# 导入U-Net分割模块
from unet_segmentation import UNetTeethSegmentation

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ToothCEJAnalyzer:
    """牙齿CEJ线分析器"""

    def __init__(self):
        # 间距阈值 (mm)
        self.DANGER_THRESHOLD = 3.2  # 红色：< 3.2mm
        self.WARNING_THRESHOLD = 4.0  # 黄色：3.2-4.0mm，绿色：>= 4.0mm

        # 像素到毫米的转换比例（需要根据实际X光片校准）
        self.pixels_per_mm = 10  # 默认值，可调整

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

        参数:
            image: 输入图像

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        # 使用U-Net进行分割
        mask, refined_mask = self.unet_segmenter.segment_teeth(image)

        # 提取单个牙齿，先尝试默认参数
        teeth_data = self.unet_segmenter.extract_individual_teeth(
            refined_mask, min_area=500, max_area=50000
        )

        # 如果没有检测到牙齿，尝试更宽松的参数
        if len(teeth_data) == 0:
            print("  未检测到牙齿，尝试调整参数...")
            # 尝试更小的最小面积和更大的最大面积
            teeth_data = self.unet_segmenter.extract_individual_teeth(
                refined_mask, min_area=100, max_area=100000
            )
            if len(teeth_data) > 0:
                print(f"  使用调整后的参数检测到 {len(teeth_data)} 颗牙齿")

        # 如果还是没有检测到，尝试直接使用未细化的掩码
        if len(teeth_data) == 0:
            print("  尝试使用原始掩码...")
            teeth_data = self.unet_segmenter.extract_individual_teeth(
                mask, min_area=100, max_area=100000
            )
            if len(teeth_data) > 0:
                print(f"  使用原始掩码检测到 {len(teeth_data)} 颗牙齿")

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

    def detect_cej_line(self, tooth_data, original_image):
        """
        检测单颗牙齿的CEJ线（釉牙骨质界）
        CEJ线是沿着牙齿轮廓的曲线，位于牙冠和牙根的交界处

        参数:
            tooth_data: 牙齿数据字典
            original_image: 原始图像

        返回:
            cej_curve: CEJ线的曲线点列表 [(x1, y1), (x2, y2), ...]
            cej_center: CEJ线的中心点坐标
            cej_normal: CEJ线中心处的法线方向向量
        """
        contour = tooth_data['contour']
        x, y, w, h = tooth_data['bbox']

        # 分析牙齿轮廓在垂直方向上的宽度变化
        # CEJ线通常位于牙齿高度的30%-50%处（从顶部开始）

        width_profile = []
        y_positions = []
        left_edges = []
        right_edges = []

        # 从上到下扫描牙齿
        for scan_y in range(y, y + h, 2):
            # 找到该Y坐标处与轮廓的交点
            intersections = []

            for point in contour:
                px, py = point[0]
                if abs(py - scan_y) <= 2:
                    intersections.append(px)

            if len(intersections) >= 2:
                intersections.sort()
                width = max(intersections) - min(intersections)
                width_profile.append(width)
                y_positions.append(scan_y)
                left_edges.append(min(intersections))
                right_edges.append(max(intersections))

        if len(width_profile) < 3:
            # 如果无法检测，使用牙齿高度的40%作为估计
            cej_y = y + int(h * 0.4)
            cej_center = (int(x + w / 2), cej_y)
            cej_curve = [(x, cej_y), (x + w, cej_y)]
            cej_normal = np.array([0, 1])  # 默认垂直向下
            return cej_curve, cej_center, cej_normal

        # 计算宽度变化率
        width_profile = np.array(width_profile)

        # 使用高斯平滑减少噪声
        from scipy.ndimage import gaussian_filter1d
        smoothed_width = gaussian_filter1d(width_profile, sigma=2)

        # 寻找宽度开始显著减小的位置（牙冠到牙根的过渡）
        # 计算一阶导数
        gradient = np.gradient(smoothed_width)

        # 寻找最大负梯度的位置（宽度减小最快）
        # 限制在牙齿的中上部（20%-60%）
        search_start = int(len(gradient) * 0.2)
        search_end = int(len(gradient) * 0.6)

        if search_end > search_start:
            search_region = gradient[search_start:search_end]
            if len(search_region) > 0:
                min_gradient_idx = np.argmin(search_region)
                cej_idx = search_start + min_gradient_idx
            else:
                cej_idx = len(y_positions) // 3
        else:
            cej_idx = len(y_positions) // 3

        # 获取CEJ线的Y坐标
        cej_y = y_positions[min(cej_idx, len(y_positions) - 1)]

        # 提取CEJ线附近的轮廓点（±5个像素范围）
        cej_curve_points = []
        for point in contour:
            px, py = point[0]
            if abs(py - cej_y) <= 5:
                cej_curve_points.append((int(px), int(py)))

        # 如果找到CEJ曲线点，按X坐标排序
        if len(cej_curve_points) >= 2:
            cej_curve_points.sort(key=lambda p: p[0])

            # 计算中心点
            left_point = cej_curve_points[0]
            right_point = cej_curve_points[-1]
            cej_center = ((left_point[0] + right_point[0]) // 2, cej_y)

            # 计算CEJ线的切线方向（使用平均方向）
            if len(cej_curve_points) >= 3:
                # 使用中间部分的点计算切线
                mid_idx = len(cej_curve_points) // 2
                p1 = cej_curve_points[max(0, mid_idx - 1)]
                p2 = cej_curve_points[min(len(cej_curve_points) - 1, mid_idx + 1)]
                tangent = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
            else:
                tangent = np.array([right_point[0] - left_point[0], right_point[1] - left_point[1]], dtype=float)

            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)

            # 法线方向（垂直于切线，指向牙根）
            cej_normal = np.array([-tangent[1], tangent[0]])

            # 确保法线指向下方（牙根方向）
            if cej_normal[1] < 0:
                cej_normal = -cej_normal

            cej_curve = cej_curve_points
        else:
            # 使用简单的水平线作为后备
            cej_center = (int(x + w / 2), cej_y)
            cej_curve = [(x, cej_y), (x + w, cej_y)]
            cej_normal = np.array([0, 1])

        return cej_curve, cej_center, cej_normal

    def measure_root_depth_along_normal(self, tooth_data, cej_point, cej_normal, max_depth_mm=15):
        """
        沿着CEJ线的法线方向测量牙根深度

        参数:
            tooth_data: 牙齿数据
            cej_point: CEJ线中心点
            cej_normal: CEJ线法线方向（单位向量）
            max_depth_mm: 最大测量深度（毫米）

        返回:
            depth_profile: 深度轮廓数据
        """
        contour = tooth_data['contour']
        max_depth_pixels = int(max_depth_mm * self.pixels_per_mm)

        depth_profile = {
            'depths': [],  # 深度值（mm）
            'widths': [],  # 该深度处的牙齿宽度（mm）
            'positions': []  # 该深度处的采样点坐标
        }

        # 沿法线方向采样
        for depth_pixel in range(0, max_depth_pixels, int(self.pixels_per_mm * 0.5)):
            depth_mm = depth_pixel / self.pixels_per_mm

            # 计算采样点位置
            sample_point = cej_point + cej_normal * depth_pixel
            sample_x, sample_y = int(sample_point[0]), int(sample_point[1])

            # 在该深度处，垂直于法线方向寻找牙齿边界
            # 切线方向（垂直于法线）
            tangent = np.array([-cej_normal[1], cej_normal[0]])

            # 沿切线方向寻找左右边界
            intersections = []

            for offset in range(-100, 101, 2):
                check_point = sample_point + tangent * offset
                check_x, check_y = int(check_point[0]), int(check_point[1])

                # 检查该点是否在牙齿轮廓内
                if cv2.pointPolygonTest(contour, (float(check_x), float(check_y)), False) >= 0:
                    intersections.append(offset)

            if len(intersections) >= 2:
                width_pixels = max(intersections) - min(intersections)
                width_mm = width_pixels / self.pixels_per_mm

                depth_profile['depths'].append(depth_mm)
                depth_profile['widths'].append(width_mm)
                depth_profile['positions'].append((sample_x, sample_y))
            else:
                # 已经超出牙齿范围（到达根尖）
                break

        return depth_profile

    def measure_spacing_between_teeth(self, tooth1_data, tooth2_data, cej1_point, cej2_point,
                                     cej1_normal, cej2_normal, max_depth_mm=15):
        """
        测量两颗相邻牙齿在不同深度的间距

        参数:
            tooth1_data, tooth2_data: 两颗牙齿的数据
            cej1_point, cej2_point: 两颗牙齿的CEJ点
            cej1_normal, cej2_normal: 两颗牙齿的CEJ法线方向
            max_depth_mm: 最大测量深度

        返回:
            spacing_profile: 间距轮廓数据
        """
        max_depth_pixels = int(max_depth_mm * self.pixels_per_mm)

        spacing_profile = {
            'depths': [],
            'spacings': [],
            'colors': [],
            'tooth1_edges': [],
            'tooth2_edges': []
        }

        # 在不同深度采样
        for depth_pixel in range(0, max_depth_pixels, int(self.pixels_per_mm * 0.5)):
            depth_mm = depth_pixel / self.pixels_per_mm

            # 计算两颗牙齿在该深度的位置
            point1 = cej1_point + cej1_normal * depth_pixel
            point2 = cej2_point + cej2_normal * depth_pixel

            # 获取两颗牙齿在该深度的边界
            # 对于tooth1，找右边界
            contour1 = tooth1_data['contour']
            rightmost1 = None
            for point in contour1:
                px, py = point[0]
                if abs(py - point1[1]) <= 5:  # 在相近的Y坐标
                    if rightmost1 is None or px > rightmost1[0]:
                        rightmost1 = (px, py)

            # 对于tooth2，找左边界
            contour2 = tooth2_data['contour']
            leftmost2 = None
            for point in contour2:
                px, py = point[0]
                if abs(py - point2[1]) <= 5:
                    if leftmost2 is None or px < leftmost2[0]:
                        leftmost2 = (px, py)

            if rightmost1 is not None and leftmost2 is not None:
                # 计算间距
                spacing_pixels = leftmost2[0] - rightmost1[0]
                spacing_mm = spacing_pixels / self.pixels_per_mm

                # 获取颜色编码
                color, _ = self.get_color_for_spacing(spacing_mm)

                spacing_profile['depths'].append(depth_mm)
                spacing_profile['spacings'].append(spacing_mm)
                spacing_profile['colors'].append(color)
                spacing_profile['tooth1_edges'].append(rightmost1)
                spacing_profile['tooth2_edges'].append(leftmost2)
            else:
                # 已经超出有效测量范围
                break

        return spacing_profile

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

        # 为每颗牙齿检测CEJ线
        print("正在检测CEJ线...")
        for i, tooth in enumerate(teeth_data):
            cej_curve, cej_center, cej_normal = self.detect_cej_line(tooth, processed)
            tooth['cej_curve'] = cej_curve  # CEJ曲线点列表
            tooth['cej_point'] = cej_center  # CEJ中心点（用于深度测量）
            tooth['cej_normal'] = cej_normal
            print(f"  牙齿 {i+1}: CEJ线包含 {len(cej_curve)} 个点，中心位于 Y={cej_center[1]}")

        # 测量每颗牙齿的根部深度
        print("正在测量牙根深度...")
        for i, tooth in enumerate(teeth_data):
            depth_profile = self.measure_root_depth_along_normal(
                tooth, tooth['cej_point'], tooth['cej_normal']
            )
            tooth['depth_profile'] = depth_profile
            if len(depth_profile['depths']) > 0:
                max_depth = max(depth_profile['depths'])
                print(f"  牙齿 {i+1}: 根部深度 {max_depth:.1f}mm")

        # 测量相邻牙齿间距
        print("正在测量牙齿间距...")
        spacing_results = []
        for i in range(len(teeth_data) - 1):
            tooth1 = teeth_data[i]
            tooth2 = teeth_data[i + 1]

            spacing_profile = self.measure_spacing_between_teeth(
                tooth1, tooth2,
                tooth1['cej_point'], tooth2['cej_point'],
                tooth1['cej_normal'], tooth2['cej_normal']
            )

            if len(spacing_profile['spacings']) > 0:
                min_spacing = min(spacing_profile['spacings'])
                avg_spacing = np.mean(spacing_profile['spacings'])
                _, risk_label = self.get_color_for_spacing(min_spacing)

                spacing_results.append({
                    'tooth_pair': (i, i + 1),
                    'profile': spacing_profile,
                    'min_spacing': min_spacing,
                    'avg_spacing': avg_spacing,
                    'risk_label': risk_label
                })

                print(f"  牙齿 {i+1}-{i+2}: 最小间距 {min_spacing:.2f}mm ({risk_label})")

        # 可视化结果
        print("正在生成可视化...")
        self.visualize_results(original_image, teeth_data, spacing_results, image_path, output_dir)

        results = {
            'image_path': image_path,
            'teeth_data': teeth_data,
            'spacing_results': spacing_results
        }

        return results

    def visualize_results(self, original_image, teeth_data, spacing_results, image_path, output_dir):
        """
        可视化分析结果

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            spacing_results: 间距测量结果
            image_path: 原始图像路径
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建图形
        fig = plt.figure(figsize=(20, 12))

        # 1. 原始图像 + 牙齿轮廓 + CEJ线
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('牙齿检测与CEJ线标注', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 绘制每颗牙齿的轮廓和CEJ线
        for i, tooth in enumerate(teeth_data):
            # 绘制轮廓
            contour = tooth['contour']
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.drawContours(original_image, [contour], -1, color, 2)

            # 绘制CEJ线（作为曲线）
            cej_curve = tooth['cej_curve']
            cej_point = tooth['cej_point']

            if len(cej_curve) >= 2:
                # 将CEJ曲线点转换为数组用于绘图
                cej_x = [p[0] for p in cej_curve]
                cej_y = [p[1] for p in cej_curve]
                ax1.plot(cej_x, cej_y, 'b-', linewidth=3, alpha=0.8, label=f'CEJ {i+1}' if i == 0 else '')

            # 绘制CEJ中心点
            ax1.plot(cej_point[0], cej_point[1], 'ro', markersize=8)

            # 标注牙齿编号
            ax1.text(cej_point[0], cej_point[1] - 20, f'T{i+1}',
                    fontsize=10, color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # 2. CEJ线深度测量示意图
        ax2 = plt.subplot(2, 2, 2)
        ax2.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('CEJ线法线方向深度测量', fontsize=14, fontweight='bold')
        ax2.axis('off')

        for i, tooth in enumerate(teeth_data):
            cej_point = np.array(tooth['cej_point'])
            cej_normal = tooth['cej_normal']
            depth_profile = tooth['depth_profile']

            # 绘制法线方向
            normal_end = cej_point + cej_normal * 100
            ax2.arrow(cej_point[0], cej_point[1],
                     normal_end[0] - cej_point[0], normal_end[1] - cej_point[1],
                     head_width=10, head_length=15, fc='cyan', ec='cyan', linewidth=2)

            # 绘制深度采样点
            for pos in depth_profile['positions'][::2]:  # 每隔一个点绘制
                ax2.plot(pos[0], pos[1], 'g.', markersize=4)

        # 3. 牙齿间距热力图
        ax3 = plt.subplot(2, 2, 3)

        # 创建一个副本用于绘制间距区域
        spacing_vis = original_image.copy()

        for result in spacing_results:
            profile = result['profile']

            # 绘制间距区域的颜色编码
            for i in range(len(profile['depths']) - 1):
                edge1 = profile['tooth1_edges'][i]
                edge2 = profile['tooth2_edges'][i]
                edge1_next = profile['tooth1_edges'][i + 1]
                edge2_next = profile['tooth2_edges'][i + 1]

                # 创建四边形填充
                pts = np.array([edge1, edge2, edge2_next, edge1_next], dtype=np.int32)

                # 获取颜色
                color_hex = profile['colors'][i]
                color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                # 绘制半透明填充
                overlay = spacing_vis.copy()
                cv2.fillPoly(overlay, [pts], color_bgr)
                cv2.addWeighted(overlay, 0.5, spacing_vis, 0.5, 0, spacing_vis)

        ax3.imshow(cv2.cvtColor(spacing_vis, cv2.COLOR_BGR2RGB))
        ax3.set_title('牙齿间距颜色编码', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 添加图例
        danger_patch = mpatches.Patch(color='#FF4444', label=f'危险 (< {self.DANGER_THRESHOLD}mm)', alpha=0.7)
        warning_patch = mpatches.Patch(color='#FFDD44', label=f'相对安全 ({self.DANGER_THRESHOLD}-{self.WARNING_THRESHOLD}mm)', alpha=0.7)
        safe_patch = mpatches.Patch(color='#44FF44', label=f'安全 (≥ {self.WARNING_THRESHOLD}mm)', alpha=0.7)
        ax3.legend(handles=[danger_patch, warning_patch, safe_patch], loc='upper right', fontsize=10)

        # 4. 间距-深度曲线图
        ax4 = plt.subplot(2, 2, 4)

        for result in spacing_results:
            i, j = result['tooth_pair']
            profile = result['profile']

            if len(profile['depths']) > 0:
                # 绘制曲线
                ax4.plot(profile['spacings'], profile['depths'],
                        marker='o', markersize=4, linewidth=2,
                        label=f'牙齿 {i+1}-{j+1}', alpha=0.7)

        ax4.axvline(x=self.DANGER_THRESHOLD, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'危险阈值 ({self.DANGER_THRESHOLD}mm)')
        ax4.axvline(x=self.WARNING_THRESHOLD, color='orange', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'警告阈值 ({self.WARNING_THRESHOLD}mm)')

        ax4.set_xlabel('间距 (mm)', fontsize=12)
        ax4.set_ylabel('CEJ线下深度 (mm)', fontsize=12)
        ax4.set_title('间距随深度变化曲线', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=9)
        ax4.invert_yaxis()

        # 保存图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_analysis.png')
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
        report_lines.append("牙齿CEJ线检测与根部距离测量 - 汇总报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        total_teeth = 0
        total_spacings = 0
        danger_count = 0
        warning_count = 0
        safe_count = 0

        for result in all_results:
            image_name = os.path.basename(result['image_path'])
            teeth_count = len(result['teeth_data'])
            spacing_results = result['spacing_results']

            total_teeth += teeth_count

            report_lines.append(f"图像: {image_name}")
            report_lines.append(f"  检测到牙齿数量: {teeth_count}")
            report_lines.append(f"  相邻牙齿对数: {len(spacing_results)}")
            report_lines.append("")

            for spacing_result in spacing_results:
                i, j = spacing_result['tooth_pair']
                min_spacing = spacing_result['min_spacing']
                avg_spacing = spacing_result['avg_spacing']
                risk_label = spacing_result['risk_label']

                total_spacings += 1

                if risk_label == '危险':
                    danger_count += 1
                elif risk_label == '相对安全':
                    warning_count += 1
                else:
                    safe_count += 1

                report_lines.append(f"    牙齿 {i+1} - {j+1}:")
                report_lines.append(f"      最小间距: {min_spacing:.2f}mm")
                report_lines.append(f"      平均间距: {avg_spacing:.2f}mm")
                report_lines.append(f"      风险等级: {risk_label}")
                report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("总体统计")
        report_lines.append("=" * 80)
        report_lines.append(f"处理图像数: {len(all_results)}")
        report_lines.append(f"检测牙齿总数: {total_teeth}")
        report_lines.append(f"测量间距总数: {total_spacings}")
        report_lines.append(f"")
        report_lines.append(f"风险分布:")
        report_lines.append(f"  危险 (< {self.DANGER_THRESHOLD}mm): {danger_count} ({danger_count/max(total_spacings,1)*100:.1f}%)")
        report_lines.append(f"  相对安全 ({self.DANGER_THRESHOLD}-{self.WARNING_THRESHOLD}mm): {warning_count} ({warning_count/max(total_spacings,1)*100:.1f}%)")
        report_lines.append(f"  安全 (≥ {self.WARNING_THRESHOLD}mm): {safe_count} ({safe_count/max(total_spacings,1)*100:.1f}%)")
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
    print("牙齿CEJ线检测与根部距离测量系统")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. 自动检测全景X光片中的每颗牙齿")
    print("2. 识别每颗牙齿的CEJ线（釉牙骨质界）")
    print("3. 沿CEJ线法线方向测量牙根深度")
    print("4. 测量相邻牙齿在不同深度的间距")
    print("5. 根据间距进行风险等级颜色编码")
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
