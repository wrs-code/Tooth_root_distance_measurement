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

    def separate_upper_lower_jaws(self, teeth_data, image_height):
        """
        将牙齿分离为上颌和下颌两组

        参数:
            teeth_data: 所有牙齿的数据列表
            image_height: 图像高度

        返回:
            upper_teeth: 上颌牙齿列表
            lower_teeth: 下颌牙齿列表
        """
        if len(teeth_data) == 0:
            return [], []

        # 计算所有牙齿中心的Y坐标
        y_coords = [tooth['centroid'][1] for tooth in teeth_data]

        # 使用K-means或简单的中位数分割
        # 这里使用简单方法：如果牙齿明显分为上下两组，用间隙分割
        y_sorted = sorted(y_coords)

        # 寻找最大的Y坐标间隙
        if len(y_sorted) >= 2:
            max_gap = 0
            split_y = image_height // 2

            for i in range(len(y_sorted) - 1):
                gap = y_sorted[i + 1] - y_sorted[i]
                if gap > max_gap:
                    max_gap = gap
                    split_y = (y_sorted[i] + y_sorted[i + 1]) / 2

            # 如果最大间隙足够大（说明有明显的上下颌分离）
            if max_gap > image_height * 0.05:  # 间隙大于图像高度的5%
                threshold_y = split_y
            else:
                # 否则使用图像中心
                threshold_y = image_height // 2
        else:
            threshold_y = image_height // 2

        upper_teeth = [tooth for tooth in teeth_data if tooth['centroid'][1] < threshold_y]
        lower_teeth = [tooth for tooth in teeth_data if tooth['centroid'][1] >= threshold_y]

        print(f"  上颌牙齿: {len(upper_teeth)} 颗，下颌牙齿: {len(lower_teeth)} 颗")

        return upper_teeth, lower_teeth

    def create_convex_hull_roi(self, teeth_group, image_shape):
        """
        为一组牙齿创建基于凸包的ROI区域（只有凸面，不扩展）

        参数:
            teeth_group: 牙齿组（上颌或下颌）
            image_shape: 图像形状 (height, width)

        返回:
            convex_hull: 凸包轮廓点 (N, 1, 2)
            roi_mask: ROI区域的二值掩码
            teeth_mask: 牙齿区域的掩码
        """
        if len(teeth_group) == 0:
            return None, None, None

        height, width = image_shape[:2]

        # 收集所有牙齿的轮廓点
        all_points = []
        teeth_mask = np.zeros((height, width), dtype=np.uint8)

        for tooth in teeth_group:
            # 收集该牙齿的所有轮廓点
            contour = tooth['contour']
            points = contour.reshape(-1, 2)  # (N, 2)
            all_points.extend(points)

            # 合并牙齿掩码
            teeth_mask = cv2.bitwise_or(teeth_mask, tooth['mask'])

        if len(all_points) == 0:
            return None, None, None

        # 转换为numpy数组
        all_points = np.array(all_points, dtype=np.int32)

        # 计算凸包（不扩展，直接使用原始凸包）
        convex_hull = cv2.convexHull(all_points)

        # 创建ROI掩码
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [convex_hull], 255)

        return convex_hull, roi_mask, teeth_mask

    def create_alveolar_bone_image(self, image, roi_mask, teeth_mask, buffer_pixels=5):
        """
        创建只包含牙槽骨区域的图像（删除牙齿区域及其边缘缓冲区）

        参数:
            image: 原始图像
            roi_mask: ROI区域掩码
            teeth_mask: 牙齿区域掩码
            buffer_pixels: 牙齿边缘缓冲区像素数（默认5像素）

        返回:
            alveolar_image: 只包含牙槽骨的图像
        """
        # 对牙齿mask进行膨胀操作，创建缓冲区
        # 这样可以排除牙齿轮廓周围几个像素的区域，避免边缘检测到牙齿边缘
        kernel = np.ones((buffer_pixels*2+1, buffer_pixels*2+1), np.uint8)
        dilated_teeth_mask = cv2.dilate(teeth_mask, kernel, iterations=1)

        # 创建牙槽骨掩码：在ROI内但不在扩展后的牙齿区域
        alveolar_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(dilated_teeth_mask))

        # 应用掩码到图像
        alveolar_image = cv2.bitwise_and(image, image, mask=alveolar_mask)

        return alveolar_image

    def check_point_distribution(self, points, min_points=13):
        """
        检查点的分布是否均匀

        参数:
            points: 点列表 [(x, y), ...]
            min_points: 最小点数

        返回:
            is_valid: 是否满足要求（点数>=min_points且分布均匀）
            uniformity_score: 均匀性评分（0-1，越高越均匀）
        """
        if len(points) < min_points:
            return False, 0.0

        # 按X坐标排序
        sorted_points = sorted(points, key=lambda p: p[0])

        # 计算相邻点之间的X方向间距
        x_gaps = []
        for i in range(len(sorted_points) - 1):
            gap = sorted_points[i+1][0] - sorted_points[i][0]
            x_gaps.append(gap)

        if len(x_gaps) == 0:
            return False, 0.0

        # 计算间距的标准差和平均值
        mean_gap = np.mean(x_gaps)
        std_gap = np.std(x_gaps)

        # 计算变异系数（CV = std/mean）
        # CV越小说明分布越均匀
        if mean_gap > 0:
            cv = std_gap / mean_gap
            # 将CV转换为均匀性评分（0-1）
            # CV=0时最均匀，CV>1时很不均匀
            uniformity_score = max(0, 1 - cv)
        else:
            uniformity_score = 0.0

        # 判断是否均匀：CV < 0.5 认为是均匀的
        is_uniform = cv < 0.5 if mean_gap > 0 else False

        return is_uniform and len(points) >= min_points, uniformity_score

    def extract_uniform_edge_points(self, alveolar_image, teeth_group, convex_hull,
                                    min_points=13, canny_low=50, canny_high=150):
        """
        从牙槽骨图像中提取均匀分布的CEJ边缘点

        参数:
            alveolar_image: 牙槽骨图像（已删除牙齿）
            teeth_group: 牙齿组
            convex_hull: 凸包轮廓
            min_points: 最小边缘点数
            canny_low: Canny低阈值
            canny_high: Canny高阈值

        返回:
            edge_points: 边缘点列表 [(x, y), ...]
            is_valid: 是否满足要求
        """
        # 1. 使用Canny边缘检测
        edges = cv2.Canny(alveolar_image, canny_low, canny_high)

        # 2. 形态学操作连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 3. 确定CEJ搜索区域（在牙齿颈部附近）
        # 计算所有牙齿的Y坐标范围
        if len(teeth_group) == 0:
            return [], False

        all_y_coords = []
        for tooth in teeth_group:
            x, y, w, h = tooth['bbox']
            # CEJ通常在牙齿的25%-60%高度范围内
            cej_y_start = y + int(h * 0.25)
            cej_y_end = y + int(h * 0.6)
            all_y_coords.extend([cej_y_start, cej_y_end])

        if len(all_y_coords) == 0:
            return [], False

        # 搜索区域的Y范围
        search_y_min = min(all_y_coords)
        search_y_max = max(all_y_coords)

        # 4. 提取边缘点
        edge_coords = np.column_stack(np.where(edges > 0))  # (y, x)

        # 5. 筛选在搜索区域内的点
        cej_candidate_points = []
        for y, x in edge_coords:
            if search_y_min <= y <= search_y_max:
                cej_candidate_points.append((x, y))

        if len(cej_candidate_points) < min_points:
            print(f"    警告: 边缘点数不足 ({len(cej_candidate_points)} < {min_points})")
            return cej_candidate_points, False

        # 6. 检查点的分布均匀性
        is_valid, uniformity_score = self.check_point_distribution(cej_candidate_points, min_points)

        print(f"    提取边缘点: {len(cej_candidate_points)} 个, 均匀性评分: {uniformity_score:.2f}")

        # 7. 如果分布不均匀，尝试采样使其更均匀
        if not is_valid and len(cej_candidate_points) >= min_points:
            print("    分布不够均匀，尝试重新采样...")
            # 按X坐标排序
            sorted_points = sorted(cej_candidate_points, key=lambda p: p[0])

            # 计算X范围
            x_min = sorted_points[0][0]
            x_max = sorted_points[-1][0]
            x_range = x_max - x_min

            if x_range > 0:
                # 将X范围分成bins，每个bin取一个代表点
                num_bins = max(min_points, len(cej_candidate_points) // 3)
                bin_width = x_range / num_bins

                resampled_points = []
                for i in range(num_bins):
                    bin_x_min = x_min + i * bin_width
                    bin_x_max = x_min + (i + 1) * bin_width

                    # 找到该bin内的所有点
                    bin_points = [p for p in sorted_points if bin_x_min <= p[0] < bin_x_max]

                    if bin_points:
                        # 取该bin的中位数点
                        mid_idx = len(bin_points) // 2
                        resampled_points.append(bin_points[mid_idx])

                # 再次检查分布
                is_valid_resampled, uniformity_score_resampled = self.check_point_distribution(
                    resampled_points, min_points)

                if is_valid_resampled:
                    print(f"    重新采样成功: {len(resampled_points)} 个点, 均匀性: {uniformity_score_resampled:.2f}")
                    return resampled_points, True
                else:
                    # 即使不够均匀，如果点数足够，也可以使用
                    if len(resampled_points) >= min_points:
                        print(f"    使用重采样点 (均匀性欠佳)")
                        return resampled_points, True

        return cej_candidate_points, is_valid

    def detect_cej_points_in_roi(self, image, teeth_group, roi_bbox, teeth_mask):
        """
        在ROI区域内检测CEJ候选点

        参数:
            image: 预处理后的图像
            teeth_group: 牙齿组
            roi_bbox: ROI边界框
            teeth_mask: 牙齿掩码

        返回:
            cej_points: CEJ候选点列表 [(x, y), ...]
        """
        if roi_bbox is None or len(teeth_group) == 0:
            return []

        x, y, w, h = roi_bbox

        # 提取ROI区域
        roi_image = image[y:y+h, x:x+w]
        roi_mask = teeth_mask[y:y+h, x:x+w]

        # 边缘检测
        edges = cv2.Canny(roi_image, 50, 150)

        # 使用形态学操作连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        cej_points = []

        # 对每颗牙齿，找到CEJ候选点
        for tooth in teeth_group:
            tooth_x, tooth_y, tooth_w, tooth_h = tooth['bbox']
            contour = tooth['contour']

            # 计算牙齿在ROI中的相对位置
            rel_x = tooth_x - x
            rel_y = tooth_y - y

            # 在牙齿的30%-50%高度范围内搜索CEJ点
            search_y_start = max(0, rel_y + int(tooth_h * 0.25))
            search_y_end = min(h, rel_y + int(tooth_h * 0.6))
            search_x_start = max(0, rel_x - 10)
            search_x_end = min(w, rel_x + tooth_w + 10)

            # 分析牙齿轮廓的宽度变化，找到收缩点
            width_profile = []
            y_positions = []

            for scan_y in range(tooth_y, tooth_y + tooth_h, 2):
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

            if len(width_profile) >= 5:
                # 平滑宽度曲线
                from scipy.ndimage import gaussian_filter1d
                smoothed_width = gaussian_filter1d(width_profile, sigma=2)

                # 计算梯度（宽度变化率）
                gradient = np.gradient(smoothed_width)

                # 在搜索范围内找到最大负梯度（宽度收缩最快的位置）
                search_start_idx = max(0, int(len(gradient) * 0.25))
                search_end_idx = min(len(gradient), int(len(gradient) * 0.6))

                if search_end_idx > search_start_idx:
                    search_region = gradient[search_start_idx:search_end_idx]
                    min_grad_idx = np.argmin(search_region)
                    cej_idx = search_start_idx + min_grad_idx
                    cej_y = y_positions[cej_idx]

                    # 在该Y坐标处找到牙齿轮廓的边缘点
                    for point in contour:
                        px, py = point[0]
                        if abs(py - cej_y) <= 3:
                            cej_points.append((px, py))

        # 去重并排序
        if len(cej_points) > 0:
            cej_points = list(set(cej_points))
            cej_points.sort(key=lambda p: p[0])

        return cej_points

    def fit_cej_curve(self, cej_points, image_width, degree=3):
        """
        使用多项式拟合CEJ曲线

        参数:
            cej_points: CEJ候选点
            image_width: 图像宽度
            degree: 多项式阶数

        返回:
            fitted_curve: 拟合后的曲线点 [(x, y), ...]
            poly_coeffs: 多项式系数（用于计算法线）
        """
        if len(cej_points) < 3:
            return None, None

        # 提取x和y坐标
        x_coords = np.array([p[0] for p in cej_points])
        y_coords = np.array([p[1] for p in cej_points])

        # 使用RANSAC去除离群点
        from sklearn.linear_model import RANSACRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        try:
            # 创建多项式回归模型
            X = x_coords.reshape(-1, 1)
            y = y_coords

            # 使用RANSAC拟合
            ransac = RANSACRegressor(
                estimator=make_pipeline(PolynomialFeatures(degree),
                                       type('LinearRegression', (), {'fit': lambda self, X, y: self,
                                                                     'predict': lambda self, X: np.polyval(np.polyfit(X.flatten(), y, degree), X.flatten())}
                                       )()),
                min_samples=max(3, len(cej_points) // 3),
                residual_threshold=10,
                max_trials=100
            )

            # 简化：直接使用numpy的polyfit
            # 去除明显的离群点（y坐标标准差的2倍之外）
            y_mean = np.mean(y_coords)
            y_std = np.std(y_coords)

            # 过滤离群点
            valid_mask = np.abs(y_coords - y_mean) < 2 * y_std
            x_filtered = x_coords[valid_mask]
            y_filtered = y_coords[valid_mask]

            if len(x_filtered) < 3:
                x_filtered = x_coords
                y_filtered = y_coords

            # 多项式拟合
            poly_coeffs = np.polyfit(x_filtered, y_filtered, degree)

            # 生成拟合曲线
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            x_fit = np.linspace(x_min, x_max, max(100, x_max - x_min))
            y_fit = np.polyval(poly_coeffs, x_fit)

            fitted_curve = [(int(x), int(y)) for x, y in zip(x_fit, y_fit)]

            return fitted_curve, poly_coeffs

        except Exception as e:
            print(f"    拟合失败: {e}，使用简单平均")
            # 后备方案：使用简单的线性拟合
            poly_coeffs = np.polyfit(x_coords, y_coords, 1)
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            x_fit = np.linspace(x_min, x_max, x_max - x_min)
            y_fit = np.polyval(poly_coeffs, x_fit)
            fitted_curve = [(int(x), int(y)) for x, y in zip(x_fit, y_fit)]
            return fitted_curve, poly_coeffs

    def detect_global_cej_lines(self, teeth_data, image, image_shape):
        """
        检测全局的CEJ线（上颌和下颌各一条）- 新算法

        新算法基于：
        1. 凸包ROI提取（只有凸面）
        2. 删除牙齿区域，只保留牙槽骨
        3. Canny边缘检测提取牙槽骨边缘
        4. 确保边缘点数>=13且分布均匀
        5. 多项式曲线拟合

        参数:
            teeth_data: 所有牙齿数据
            image: 预处理后的图像
            image_shape: 图像形状

        返回:
            upper_cej: 上颌CEJ线数据 {'curve': [...], 'coeffs': [...]}
            lower_cej: 下颌CEJ线数据
        """
        print("正在检测全局CEJ线（使用凸包ROI和牙槽骨边缘检测）...")

        height, width = image_shape[:2]

        # 1. 分离上下颌牙齿
        upper_teeth, lower_teeth = self.separate_upper_lower_jaws(teeth_data, height)

        upper_cej = None
        lower_cej = None

        # 2. 处理上颌
        if len(upper_teeth) > 0:
            print("  处理上颌CEJ线...")

            # 2.1 创建凸包ROI（不扩展）
            convex_hull, roi_mask, teeth_mask = self.create_convex_hull_roi(
                upper_teeth, image_shape)

            if convex_hull is not None:
                # 2.2 创建只包含牙槽骨的图像（删除牙齿及其边缘缓冲区）
                alveolar_image = self.create_alveolar_bone_image(image, roi_mask, teeth_mask, buffer_pixels=5)

                # 2.3 提取均匀分布的边缘点
                edge_points, is_valid = self.extract_uniform_edge_points(
                    alveolar_image, upper_teeth, convex_hull, min_points=13)

                if is_valid and len(edge_points) >= 3:
                    # 2.4 拟合CEJ曲线
                    fitted_curve, poly_coeffs = self.fit_cej_curve(edge_points, width)

                    if fitted_curve is not None:
                        upper_cej = {
                            'curve': fitted_curve,
                            'coeffs': poly_coeffs,
                            'points': edge_points,
                            'teeth': upper_teeth,
                            'convex_hull': convex_hull
                        }
                        print(f"    ✓ 上颌CEJ线已检测，包含 {len(edge_points)} 个边缘点，{len(fitted_curve)} 个拟合点")
                else:
                    print(f"    ✗ 上颌CEJ线检测失败：边缘点不足或分布不均")

        # 3. 处理下颌
        if len(lower_teeth) > 0:
            print("  处理下颌CEJ线...")

            # 3.1 创建凸包ROI（不扩展）
            convex_hull, roi_mask, teeth_mask = self.create_convex_hull_roi(
                lower_teeth, image_shape)

            if convex_hull is not None:
                # 3.2 创建只包含牙槽骨的图像（删除牙齿及其边缘缓冲区）
                alveolar_image = self.create_alveolar_bone_image(image, roi_mask, teeth_mask, buffer_pixels=5)

                # 3.3 提取均匀分布的边缘点
                edge_points, is_valid = self.extract_uniform_edge_points(
                    alveolar_image, lower_teeth, convex_hull, min_points=13)

                if is_valid and len(edge_points) >= 3:
                    # 3.4 拟合CEJ曲线
                    fitted_curve, poly_coeffs = self.fit_cej_curve(edge_points, width)

                    if fitted_curve is not None:
                        lower_cej = {
                            'curve': fitted_curve,
                            'coeffs': poly_coeffs,
                            'points': edge_points,
                            'teeth': lower_teeth,
                            'convex_hull': convex_hull
                        }
                        print(f"    ✓ 下颌CEJ线已检测，包含 {len(edge_points)} 个边缘点，{len(fitted_curve)} 个拟合点")
                else:
                    print(f"    ✗ 下颌CEJ线检测失败：边缘点不足或分布不均")

        return upper_cej, lower_cej

    def detect_cej_line(self, tooth_data, original_image):
        """
        检测单颗牙齿的CEJ线（釉牙骨质界）- 旧方法，保留用于向后兼容

        注意：此方法已被 detect_global_cej_lines 替代
        建议使用全局CEJ检测方法

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

        # 使用新的全局CEJ线检测方法
        print("正在检测全局CEJ线...")
        upper_cej, lower_cej = self.detect_global_cej_lines(teeth_data, processed, original_image.shape)

        # 为了向后兼容，为每颗牙齿分配CEJ线信息
        # 根据牙齿所属的上颌或下颌，使用对应的全局CEJ线
        for i, tooth in enumerate(teeth_data):
            # 确定牙齿属于上颌还是下颌
            centroid_y = tooth['centroid'][1]

            # 如果有上下颌CEJ线，根据牙齿位置选择
            if upper_cej is not None and lower_cej is not None:
                # 判断牙齿属于哪个颌
                is_upper = any(t['label'] == tooth['label'] for t in upper_cej['teeth'])
                cej_data = upper_cej if is_upper else lower_cej
            elif upper_cej is not None:
                cej_data = upper_cej
            elif lower_cej is not None:
                cej_data = lower_cej
            else:
                # 回退到旧方法
                print(f"  警告：无法使用全局CEJ线，使用单个牙齿检测方法")
                cej_curve, cej_center, cej_normal = self.detect_cej_line(tooth, processed)
                tooth['cej_curve'] = cej_curve
                tooth['cej_point'] = cej_center
                tooth['cej_normal'] = cej_normal
                continue

            # 从全局CEJ曲线中找到该牙齿对应的CEJ点
            tooth_x = tooth['centroid'][0]
            cej_curve = cej_data['curve']

            # 找到最接近牙齿X坐标的CEJ点
            closest_points = []
            for point in cej_curve:
                if abs(point[0] - tooth_x) < tooth['bbox'][2] / 2:  # 在牙齿宽度的一半范围内
                    closest_points.append(point)

            if len(closest_points) > 0:
                # 使用这些点作为该牙齿的CEJ点
                tooth['cej_curve'] = closest_points
                # 计算CEJ中心点（用于深度测量）
                avg_x = int(np.mean([p[0] for p in closest_points]))
                avg_y = int(np.mean([p[1] for p in closest_points]))
                tooth['cej_point'] = (avg_x, avg_y)

                # 计算法线（从曲线的导数）
                # 使用多项式系数计算切线方向
                poly_coeffs = cej_data['coeffs']
                # dy/dx = poly'(x)
                poly_derivative = np.polyder(poly_coeffs)
                slope = np.polyval(poly_derivative, avg_x)
                tangent = np.array([1.0, slope])
                tangent = tangent / np.linalg.norm(tangent)
                # 法线垂直于切线
                cej_normal = np.array([-tangent[1], tangent[0]])
                # 确保法线指向下方
                if cej_normal[1] < 0:
                    cej_normal = -cej_normal
                tooth['cej_normal'] = cej_normal
            else:
                # 如果没有找到对应的点，使用曲线上最近的点
                min_dist = float('inf')
                closest_point = cej_curve[0]
                for point in cej_curve:
                    dist = abs(point[0] - tooth_x)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = point

                tooth['cej_curve'] = [closest_point]
                tooth['cej_point'] = closest_point
                tooth['cej_normal'] = np.array([0, 1])  # 默认垂直向下

        # 保存全局CEJ线数据到结果中
        global_cej_data = {
            'upper_cej': upper_cej,
            'lower_cej': lower_cej
        }

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
        self.visualize_results(original_image, teeth_data, spacing_results, image_path, output_dir, global_cej_data)

        results = {
            'image_path': image_path,
            'teeth_data': teeth_data,
            'spacing_results': spacing_results,
            'global_cej': global_cej_data
        }

        return results

    def visualize_results(self, original_image, teeth_data, spacing_results, image_path, output_dir, global_cej_data=None):
        """
        可视化分析结果

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            spacing_results: 间距测量结果
            image_path: 原始图像路径
            output_dir: 输出目录
            global_cej_data: 全局CEJ线数据（可选）
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建图形
        fig = plt.figure(figsize=(20, 12))

        # 1. 原始图像 + 牙齿轮廓 + 全局CEJ线
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('牙齿检测与全局CEJ线标注', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 绘制每颗牙齿的轮廓
        for i, tooth in enumerate(teeth_data):
            # 绘制轮廓
            contour = tooth['contour']
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.drawContours(original_image, [contour], -1, color, 2)

            # 标注牙齿编号
            centroid = tuple(map(int, tooth['centroid']))
            ax1.text(centroid[0], centroid[1] - 20, f'T{i+1}',
                    fontsize=10, color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # 绘制全局CEJ线（上颌和下颌各一条）
        if global_cej_data is not None:
            upper_cej = global_cej_data.get('upper_cej')
            lower_cej = global_cej_data.get('lower_cej')

            # 绘制上颌CEJ线
            if upper_cej is not None and 'curve' in upper_cej:
                cej_curve = upper_cej['curve']
                cej_x = [p[0] for p in cej_curve]
                cej_y = [p[1] for p in cej_curve]
                ax1.plot(cej_x, cej_y, 'b-', linewidth=4, alpha=0.9, label='上颌CEJ线')

                # 绘制CEJ候选点
                if 'points' in upper_cej:
                    points = upper_cej['points']
                    points_x = [p[0] for p in points]
                    points_y = [p[1] for p in points]
                    ax1.scatter(points_x, points_y, c='cyan', s=20, alpha=0.6, marker='o')

            # 绘制下颌CEJ线
            if lower_cej is not None and 'curve' in lower_cej:
                cej_curve = lower_cej['curve']
                cej_x = [p[0] for p in cej_curve]
                cej_y = [p[1] for p in cej_curve]
                ax1.plot(cej_x, cej_y, 'r-', linewidth=4, alpha=0.9, label='下颌CEJ线')

                # 绘制CEJ候选点
                if 'points' in lower_cej:
                    points = lower_cej['points']
                    points_x = [p[0] for p in points]
                    points_y = [p[1] for p in points]
                    ax1.scatter(points_x, points_y, c='orange', s=20, alpha=0.6, marker='o')

            ax1.legend(loc='upper right', fontsize=10)
        else:
            # 如果没有全局CEJ数据，绘制单个牙齿的CEJ线（兼容旧方法）
            for i, tooth in enumerate(teeth_data):
                cej_curve = tooth.get('cej_curve', [])
                cej_point = tooth.get('cej_point')

                if len(cej_curve) >= 2:
                    # 将CEJ曲线点转换为数组用于绘图
                    cej_x = [p[0] for p in cej_curve]
                    cej_y = [p[1] for p in cej_curve]
                    ax1.plot(cej_x, cej_y, 'b-', linewidth=3, alpha=0.8, label=f'CEJ {i+1}' if i == 0 else '')

                if cej_point is not None:
                    # 绘制CEJ中心点
                    ax1.plot(cej_point[0], cej_point[1], 'ro', markersize=8)

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
