#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿根部距离测量系统（形态学方法）
基于全景X光片的牙齿分析，包括：
1. 牙齿边缘检测和分割（使用U-Net深度学习模型）
2. 牙齿长轴计算（参考SerdarHelli/CCA_Analysis.py）
3. 牙齿间隙骨嵴边界检测（形态学方法）
4. 沿牙齿长轴的根部间距测量
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
    """牙齿根部距离分析器（形态学方法）"""

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

    def midpoint(self, ptA, ptB):
        """
        计算两点的中点（参考SerdarHelli/CCA_Analysis.py）

        参数:
            ptA: 第一个点 (x, y)
            ptB: 第二个点 (x, y)

        返回:
            中点坐标 (x, y)
        """
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def compute_tooth_long_axis(self, tooth_data, debug_dir=None, tooth_id=None):
        """
        计算牙齿的长轴方向和关键点（参考SerdarHelli/CCA_Analysis.py）

        参数:
            tooth_data: 牙齿数据
            debug_dir: debug目录（可选）
            tooth_id: 牙齿ID（用于debug文件命名）

        返回:
            long_axis_data: 包含长轴信息的字典
        """
        contour = tooth_data['contour']

        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        # 排序顶点：左上、右上、右下、左下
        box = self.order_points(box)
        (tl, tr, br, bl) = box

        # 计算四条边的中点
        top_mid = self.midpoint(tl, tr)      # 上边中点
        bottom_mid = self.midpoint(bl, br)   # 下边中点
        left_mid = self.midpoint(tl, bl)     # 左边中点
        right_mid = self.midpoint(tr, br)    # 右边中点

        # 计算两个轴的长度
        dA = dist.euclidean(top_mid, bottom_mid)   # 主轴长度
        dB = dist.euclidean(left_mid, right_mid)   # 副轴长度

        # 长轴是较长的那个
        if dA >= dB:
            # 主轴为长轴（通常是牙齿的竖直方向）
            axis_start = top_mid
            axis_end = bottom_mid
            axis_length = dA
            is_vertical = True
        else:
            # 副轴为长轴（水平方向）
            axis_start = left_mid
            axis_end = right_mid
            axis_length = dB
            is_vertical = False

        # 计算长轴方向向量（归一化）
        axis_vector = np.array([
            axis_end[0] - axis_start[0],
            axis_end[1] - axis_start[1]
        ])
        axis_vector_norm = axis_vector / (np.linalg.norm(axis_vector) + 1e-6)

        # 确保长轴指向下方（牙根方向）
        if axis_vector_norm[1] < 0:  # y方向向上
            axis_vector_norm = -axis_vector_norm
            axis_start, axis_end = axis_end, axis_start

        long_axis_data = {
            'box': box,
            'top_mid': top_mid,
            'bottom_mid': bottom_mid,
            'left_mid': left_mid,
            'right_mid': right_mid,
            'axis_start': axis_start,
            'axis_end': axis_end,
            'axis_vector': axis_vector_norm,
            'axis_length': axis_length,
            'is_vertical': is_vertical
        }

        # 保存debug可视化
        if debug_dir is not None and tooth_id is not None:
            self._save_long_axis_debug(tooth_data, long_axis_data, debug_dir, tooth_id)

        return long_axis_data

    def _save_long_axis_debug(self, tooth_data, long_axis_data, debug_dir, tooth_id):
        """保存长轴debug可视化"""
        # 创建可视化图像
        h, w = tooth_data['mask'].shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # 绘制牙齿轮廓
        cv2.drawContours(vis, [tooth_data['contour']], -1, (255, 255, 255), 2)

        # 绘制最小外接矩形
        box_int = np.array(long_axis_data['box'], dtype=np.int32)
        cv2.drawContours(vis, [box_int], 0, (0, 255, 0), 2)

        # 绘制四个中点
        for point_name, point in [
            ('top_mid', long_axis_data['top_mid']),
            ('bottom_mid', long_axis_data['bottom_mid']),
            ('left_mid', long_axis_data['left_mid']),
            ('right_mid', long_axis_data['right_mid'])
        ]:
            cv2.circle(vis, (int(point[0]), int(point[1])), 5, (255, 255, 0), -1)

        # 绘制长轴（蓝色粗线）
        start = long_axis_data['axis_start']
        end = long_axis_data['axis_end']
        cv2.line(vis, (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])), (255, 0, 0), 3)

        # 绘制箭头指向牙根
        cv2.arrowedLine(vis, (int(start[0]), int(start[1])),
                       (int(end[0]), int(end[1])), (0, 0, 255), 2, tipLength=0.3)

        # 保存
        output_path = os.path.join(debug_dir, f'tooth_{tooth_id}_long_axis.png')
        cv2.imwrite(output_path, vis)

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

    def create_gap_mask(self, tooth1_mask, tooth2_mask, expansion=10):
        """
        创建两颗牙齿之间的间隙mask

        参数:
            tooth1_mask: 第一颗牙齿的mask
            tooth2_mask: 第二颗牙齿的mask
            expansion: 向外扩展的像素数

        返回:
            gap_mask: 间隙区域的mask
        """
        # 合并两颗牙齿的mask
        combined_mask = cv2.bitwise_or(tooth1_mask, tooth2_mask)

        # 膨胀合并后的mask，创建搜索区域
        kernel = np.ones((expansion*2+1, expansion*2+1), np.uint8)
        expanded_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # 间隙区域 = 扩展区域 - 原始牙齿区域
        gap_mask = cv2.subtract(expanded_mask, combined_mask)

        return gap_mask

    def detect_bone_crest_in_gap(self, image, gap_mask, tooth1_axis_data, tooth2_axis_data,
                                 debug_dir=None, gap_id=None):
        """
        在牙齿间隙检测骨嵴点（牙槽骨边界）

        算法流程：
        1. 在间隙区域应用边缘检测
        2. 找到最靠近牙根末端的边缘点（骨嵴位置）
        3. 返回骨嵴点及其相关信息

        参数:
            image: 原始图像
            gap_mask: 间隙区域mask
            tooth1_axis_data: 第一颗牙齿的长轴数据
            tooth2_axis_data: 第二颗牙齿的长轴数据
            debug_dir: debug目录
            gap_id: 间隙ID

        返回:
            bone_crest_data: 骨嵴数据字典
        """
        # 1. 在间隙区域应用边缘检测
        gap_image = cv2.bitwise_and(image, image, mask=gap_mask)
        gap_edges = cv2.Canny(gap_image, 50, 150)

        # 2. 提取边缘点
        edge_coords = np.column_stack(np.where(gap_edges > 0))  # (y, x)
        edge_points = [(x, y) for y, x in edge_coords]

        if len(edge_points) == 0:
            return None

        # 3. 确定搜索区域（基于两颗牙齿的根部位置）
        # 取两颗牙齿长轴末端的中间高度作为参考
        root1_y = tooth1_axis_data['axis_end'][1]
        root2_y = tooth2_axis_data['axis_end'][1]
        ref_y = (root1_y + root2_y) / 2

        # 4. 找到最接近参考高度的边缘点（骨嵴点）
        # 骨嵴通常在牙根附近，但略高于牙根末端
        search_range = 50  # 搜索范围（像素）

        # 筛选在搜索范围内的点
        candidate_points = [
            p for p in edge_points
            if abs(p[1] - ref_y) < search_range
        ]

        if len(candidate_points) == 0:
            # 如果没有找到，使用所有点中y坐标最小的（最高的）
            candidate_points = edge_points

        # 找到最高的点（y坐标最小）作为骨嵴点
        bone_crest_point = min(candidate_points, key=lambda p: p[1])

        # 5. 计算X方向的中心点
        # 在骨嵴高度附近，找到所有边缘点的X坐标范围
        y_threshold = 10
        points_at_crest = [
            p for p in edge_points
            if abs(p[1] - bone_crest_point[1]) < y_threshold
        ]

        if len(points_at_crest) > 0:
            # 使用中位数x坐标
            x_coords = [p[0] for p in points_at_crest]
            median_x = int(np.median(x_coords))
            bone_crest_point = (median_x, bone_crest_point[1])

        bone_crest_data = {
            'point': bone_crest_point,
            'ref_y': ref_y,
            'edge_points': edge_points,
            'points_at_crest': points_at_crest
        }

        # Debug可视化
        if debug_dir is not None and gap_id is not None:
            self._save_gap_detection_debug(
                image, gap_mask, gap_edges, bone_crest_data,
                tooth1_axis_data, tooth2_axis_data,
                debug_dir, gap_id
            )

        return bone_crest_data

    def _save_gap_detection_debug(self, image, gap_mask, gap_edges, bone_crest_data,
                                  tooth1_axis_data, tooth2_axis_data, debug_dir, gap_id):
        """保存间隙检测debug可视化"""
        # 创建可视化图像
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        # 应用gap mask（半透明蓝色）
        gap_overlay = vis.copy()
        gap_overlay[gap_mask > 0] = [255, 100, 0]  # 蓝色
        vis = cv2.addWeighted(vis, 0.7, gap_overlay, 0.3, 0)

        # 绘制两颗牙齿的根部末端
        root1 = tooth1_axis_data['axis_end']
        root2 = tooth2_axis_data['axis_end']
        cv2.circle(vis, (int(root1[0]), int(root1[1])), 6, (0, 255, 0), -1)
        cv2.circle(vis, (int(root2[0]), int(root2[1])), 6, (0, 255, 0), -1)

        # 绘制参考线
        ref_y = int(bone_crest_data['ref_y'])
        cv2.line(vis, (0, ref_y), (vis.shape[1], ref_y), (255, 255, 0), 1)

        # 绘制所有边缘点（黄色小点）
        for x, y in bone_crest_data['edge_points']:
            cv2.circle(vis, (x, y), 1, (0, 255, 255), -1)

        # 绘制骨嵴点（红色大圆圈）
        crest_point = bone_crest_data['point']
        cv2.circle(vis, crest_point, 8, (0, 0, 255), 3)
        cv2.circle(vis, crest_point, 2, (255, 255, 255), -1)

        # 保存
        output_path = os.path.join(debug_dir, f'gap_{gap_id}_bone_crest.png')
        cv2.imwrite(output_path, vis)

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

    def detect_roi_edges_simple(self, image, roi_mask, teeth_mask, convex_hull,
                                canny_low=50, canny_high=150, debug_dir=None, jaw_name=''):
        """
        直接在凸包ROI中检测边缘，并移除牙齿内部的边缘点

        新算法流程：
        1. 在ROI区域内进行Canny边缘检测
        2. 提取所有边缘点
        3. 清除牙齿内部的边缘点（使用膨胀后的牙齿mask）
        4. 返回剩余的边缘点

        参数:
            image: 原始图像（灰度或预处理后）
            roi_mask: ROI区域掩码
            teeth_mask: 牙齿区域掩码
            convex_hull: 凸包轮廓
            canny_low: Canny低阈值
            canny_high: Canny高阈值
            debug_dir: debug图像保存目录
            jaw_name: 颌名称（用于debug文件命名）

        返回:
            edge_points: 边缘点列表 [(x, y), ...]
            debug_images: debug图像字典
        """
        debug_images = {}

        # 1. 在ROI区域应用Canny边缘检测
        print(f"    步骤1: 在ROI区域进行边缘检测...")
        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)
        edges = cv2.Canny(roi_image, canny_low, canny_high)
        debug_images['edges_raw'] = edges.copy()

        # 2. 形态学操作连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        debug_images['edges_morphology'] = edges.copy()

        # 3. 膨胀牙齿mask，创建排除区域
        print(f"    步骤2: 创建牙齿排除区域...")
        buffer_pixels = 5
        kernel_dilate = np.ones((buffer_pixels*2+1, buffer_pixels*2+1), np.uint8)
        dilated_teeth_mask = cv2.dilate(teeth_mask, kernel_dilate, iterations=1)
        debug_images['teeth_mask_dilated'] = dilated_teeth_mask.copy()

        # 4. 从边缘中移除牙齿内部的点
        print(f"    步骤3: 移除牙齿内部的边缘点...")
        edges_outside_teeth = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_teeth_mask))
        debug_images['edges_filtered'] = edges_outside_teeth.copy()

        # 5. 提取所有边缘点
        edge_coords = np.column_stack(np.where(edges_outside_teeth > 0))  # (y, x)
        edge_points = [(x, y) for y, x in edge_coords]

        print(f"    提取到 {len(edge_points)} 个边缘点")

        # 6. 保存debug图像（如果指定了目录）
        if debug_dir is not None and jaw_name:
            os.makedirs(debug_dir, exist_ok=True)

            # 保存ROI区域
            roi_vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(roi_vis, [convex_hull], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_01_roi_region.png'), roi_vis)

            # 保存原始边缘
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_02_edges_raw.png'),
                       debug_images['edges_raw'])

            # 保存形态学处理后的边缘
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_03_edges_morphology.png'),
                       debug_images['edges_morphology'])

            # 保存膨胀后的牙齿mask
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_04_teeth_mask_dilated.png'),
                       dilated_teeth_mask)

            # 保存过滤后的边缘
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_05_edges_filtered.png'),
                       edges_outside_teeth)

            # 保存边缘点可视化
            edge_vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            for x, y in edge_points:
                cv2.circle(edge_vis, (x, y), 2, (0, 255, 255), -1)
            cv2.imwrite(os.path.join(debug_dir, f'{jaw_name}_06_edge_points.png'), edge_vis)

            print(f"    Debug图像已保存到: {debug_dir}")

        return edge_points, debug_images

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

    def detect_root_boundary_lines(self, teeth_data, image, image_shape, debug_dir=None):
        """
        检测牙根边界线（上颌和下颌各一条）- 简化算法

        简化算法流程：
        1. 创建凸包ROI（只有凸面）
        2. 在ROI区域内进行Canny边缘检测
        3. 清除牙齿内部的边缘点
        4. 用剩余边缘点拟合曲线

        参数:
            teeth_data: 所有牙齿数据
            image: 预处理后的图像
            image_shape: 图像形状
            debug_dir: debug图像保存目录

        返回:
            upper_boundary: 上颌边界线数据 {'curve': [...], 'coeffs': [...]}
            lower_boundary: 下颌边界线数据
        """
        print("正在检测牙根边界线（使用凸包ROI边缘检测）...")

        height, width = image_shape[:2]

        # 1. 分离上下颌牙齿
        upper_teeth, lower_teeth = self.separate_upper_lower_jaws(teeth_data, height)

        upper_boundary = None
        lower_boundary = None

        # 2. 处理上颌
        if len(upper_teeth) > 0:
            print("  处理上颌边界线...")

            # 2.1 创建凸包ROI
            convex_hull, roi_mask, teeth_mask = self.create_convex_hull_roi(
                upper_teeth, image_shape)

            if convex_hull is not None:
                # 2.2 检测ROI中的边缘，并移除牙齿内部的点
                edge_points, debug_images = self.detect_roi_edges_simple(
                    image, roi_mask, teeth_mask, convex_hull,
                    canny_low=50, canny_high=150,
                    debug_dir=debug_dir, jaw_name='upper'
                )

                if len(edge_points) >= 3:
                    # 2.3 拟合曲线
                    fitted_curve, poly_coeffs = self.fit_cej_curve(edge_points, width, degree=3)

                    if fitted_curve is not None:
                        upper_boundary = {
                            'curve': fitted_curve,
                            'coeffs': poly_coeffs,
                            'points': edge_points,
                            'teeth': upper_teeth,
                            'convex_hull': convex_hull,
                            'debug_images': debug_images
                        }
                        print(f"    ✓ 上颌边界线已检测，包含 {len(edge_points)} 个边缘点，{len(fitted_curve)} 个拟合点")

                        # 保存拟合曲线可视化
                        if debug_dir is not None:
                            self._save_fitted_curve_debug(
                                image, edge_points, fitted_curve, convex_hull,
                                debug_dir, 'upper'
                            )
                else:
                    print(f"    ✗ 上颌边界线检测失败：边缘点不足 ({len(edge_points)} < 3)")

        # 3. 处理下颌
        if len(lower_teeth) > 0:
            print("  处理下颌边界线...")

            # 3.1 创建凸包ROI
            convex_hull, roi_mask, teeth_mask = self.create_convex_hull_roi(
                lower_teeth, image_shape)

            if convex_hull is not None:
                # 3.2 检测ROI中的边缘，并移除牙齿内部的点
                edge_points, debug_images = self.detect_roi_edges_simple(
                    image, roi_mask, teeth_mask, convex_hull,
                    canny_low=50, canny_high=150,
                    debug_dir=debug_dir, jaw_name='lower'
                )

                if len(edge_points) >= 3:
                    # 3.3 拟合曲线
                    fitted_curve, poly_coeffs = self.fit_cej_curve(edge_points, width, degree=3)

                    if fitted_curve is not None:
                        lower_boundary = {
                            'curve': fitted_curve,
                            'coeffs': poly_coeffs,
                            'points': edge_points,
                            'teeth': lower_teeth,
                            'convex_hull': convex_hull,
                            'debug_images': debug_images
                        }
                        print(f"    ✓ 下颌边界线已检测，包含 {len(edge_points)} 个边缘点，{len(fitted_curve)} 个拟合点")

                        # 保存拟合曲线可视化
                        if debug_dir is not None:
                            self._save_fitted_curve_debug(
                                image, edge_points, fitted_curve, convex_hull,
                                debug_dir, 'lower'
                            )
                else:
                    print(f"    ✗ 下颌边界线检测失败：边缘点不足 ({len(edge_points)} < 3)")

        return upper_boundary, lower_boundary

    def _save_fitted_curve_debug(self, image, edge_points, fitted_curve, convex_hull,
                                 debug_dir, jaw_name):
        """
        保存拟合曲线的debug可视化

        参数:
            image: 原始图像
            edge_points: 边缘点列表
            fitted_curve: 拟合曲线
            convex_hull: 凸包轮廓
            debug_dir: debug目录
            jaw_name: 颌名称
        """
        # 创建可视化图像
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        # 绘制凸包
        cv2.drawContours(vis_image, [convex_hull], -1, (0, 255, 0), 2)

        # 绘制边缘点（黄色小圆点）
        for x, y in edge_points:
            cv2.circle(vis_image, (x, y), 2, (0, 255, 255), -1)

        # 绘制拟合曲线（蓝色粗线）
        if len(fitted_curve) > 1:
            pts = np.array(fitted_curve, dtype=np.int32)
            cv2.polylines(vis_image, [pts], False, (255, 0, 0), 3)

        # 保存
        output_path = os.path.join(debug_dir, f'{jaw_name}_07_fitted_curve.png')
        cv2.imwrite(output_path, vis_image)
        print(f"    拟合曲线可视化已保存: {output_path}")

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

    def measure_root_spacing_along_axis(self, tooth1_data, tooth2_data, bone_crest_point,
                                        tooth1_axis, tooth2_axis, debug_dir=None, pair_id=None):
        """
        沿牙齿长轴测量牙根间距（新方法 - 基于形态学）

        算法流程：
        1. 从骨嵴点开始，沿两颗牙齿的长轴向下采样
        2. 在每个深度，找到牙齿边缘的最近点
        3. 计算两颗牙齿之间的距离

        参数:
            tooth1_data: 第一颗牙齿数据
            tooth2_data: 第二颗牙齿数据
            bone_crest_point: 骨嵴点（起始测量点）
            tooth1_axis: 第一颗牙齿的长轴数据
            tooth2_axis: 第二颗牙齿的长轴数据
            debug_dir: debug目录
            pair_id: 牙齿对ID

        返回:
            spacing_profile: 间距轮廓数据
        """
        max_depth_mm = 15
        max_depth_pixels = int(max_depth_mm * self.pixels_per_mm)

        spacing_profile = {
            'depths': [],
            'spacings': [],
            'colors': [],
            'tooth1_edges': [],
            'tooth2_edges': [],
            'measurement_lines': []
        }

        # 采样步长（每0.5mm）
        step_pixels = int(self.pixels_per_mm * 0.5)

        # 从骨嵴点开始向下采样
        for depth_pixel in range(0, max_depth_pixels, step_pixels):
            depth_mm = depth_pixel / self.pixels_per_mm

            # 沿第一颗牙齿的长轴向下移动
            tooth1_point = np.array(bone_crest_point) + tooth1_axis['axis_vector'] * depth_pixel

            # 沿第二颗牙齿的长轴向下移动
            tooth2_point = np.array(bone_crest_point) + tooth2_axis['axis_vector'] * depth_pixel

            # 在tooth1的轮廓上找最近的边缘点
            contour1 = tooth1_data['contour'].reshape(-1, 2)
            dists1 = np.linalg.norm(contour1 - tooth1_point, axis=1)
            nearest_idx1 = np.argmin(dists1)
            edge1 = tuple(contour1[nearest_idx1].astype(int))

            # 在tooth2的轮廓上找最近的边缘点
            contour2 = tooth2_data['contour'].reshape(-1, 2)
            dists2 = np.linalg.norm(contour2 - tooth2_point, axis=1)
            nearest_idx2 = np.argmin(dists2)
            edge2 = tuple(contour2[nearest_idx2].astype(int))

            # 计算两个边缘点之间的距离
            spacing_pixels = dist.euclidean(edge1, edge2)
            spacing_mm = spacing_pixels / self.pixels_per_mm

            # 只保留合理的间距（避免牙齿重叠区域）
            if spacing_mm >= 0 and spacing_mm < 30:
                # 获取颜色编码
                color, _ = self.get_color_for_spacing(spacing_mm)

                spacing_profile['depths'].append(depth_mm)
                spacing_profile['spacings'].append(spacing_mm)
                spacing_profile['colors'].append(color)
                spacing_profile['tooth1_edges'].append(edge1)
                spacing_profile['tooth2_edges'].append(edge2)
                spacing_profile['measurement_lines'].append((edge1, edge2))

        # Debug可视化
        if debug_dir is not None and pair_id is not None:
            self._save_spacing_measurement_debug(
                tooth1_data, tooth2_data, bone_crest_point,
                spacing_profile, debug_dir, pair_id
            )

        return spacing_profile

    def _save_spacing_measurement_debug(self, tooth1_data, tooth2_data, bone_crest_point,
                                       spacing_profile, debug_dir, pair_id):
        """保存间距测量debug可视化"""
        # 创建空白图像
        h = max(tooth1_data['mask'].shape[0], tooth2_data['mask'].shape[0])
        w = max(tooth1_data['mask'].shape[1], tooth2_data['mask'].shape[1])
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # 绘制两颗牙齿轮廓
        cv2.drawContours(vis, [tooth1_data['contour']], -1, (255, 255, 255), 2)
        cv2.drawContours(vis, [tooth2_data['contour']], -1, (255, 255, 255), 2)

        # 绘制骨嵴点
        cv2.circle(vis, bone_crest_point, 8, (0, 255, 255), -1)

        # 绘制所有测量线（颜色编码）
        for i, (edge1, edge2) in enumerate(spacing_profile['measurement_lines']):
            # 根据间距获取颜色
            color_hex = spacing_profile['colors'][i]
            # 将hex颜色转换为BGR
            color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # 绘制测量线
            cv2.line(vis, edge1, edge2, color_bgr, 2)

            # 绘制端点
            cv2.circle(vis, edge1, 3, (255, 255, 255), -1)
            cv2.circle(vis, edge2, 3, (255, 255, 255), -1)

        # 保存
        output_path = os.path.join(debug_dir, f'pair_{pair_id}_spacing_measurement.png')
        cv2.imwrite(output_path, vis)

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
        分析单张全景X光图像（新方法 - 形态学+间隙检测）

        流程：
        1. U-Net检测牙齿
        2. 计算每颗牙齿的长轴（参考SerdarHelli/CCA_Analysis.py）
        3. 在牙齿间隙检测骨嵴边界
        4. 沿长轴测量牙根间距

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
        if self.use_unet:
            teeth_data = self.detect_teeth_contours(original_image)
        else:
            teeth_data = self.detect_teeth_contours(processed)
        print(f"✓ 检测到 {len(teeth_data)} 颗牙齿")

        if len(teeth_data) == 0:
            print("❌ 未检测到牙齿")
            return None

        # 创建debug输出目录
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = os.path.join(output_dir, f'{base_name}_debug')
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug图像将保存到: {debug_dir}")

        # 计算每颗牙齿的长轴
        print("正在计算牙齿长轴...")
        for i, tooth in enumerate(teeth_data):
            long_axis_data = self.compute_tooth_long_axis(
                tooth, debug_dir=debug_dir, tooth_id=i+1
            )
            tooth['long_axis'] = long_axis_data
            print(f"  牙齿 {i+1}: 长轴长度 {long_axis_data['axis_length']/self.pixels_per_mm:.1f}mm")

        # 测量相邻牙齿间距（使用新方法）
        print("正在测量牙根间距（基于间隙骨嵴检测）...")
        spacing_results = []

        for i in range(len(teeth_data) - 1):
            tooth1 = teeth_data[i]
            tooth2 = teeth_data[i + 1]

            print(f"  处理牙齿对 {i+1}-{i+2}...")

            # 创建间隙mask
            gap_mask = self.create_gap_mask(tooth1['mask'], tooth2['mask'], expansion=10)

            # 检测间隙中的骨嵴点
            bone_crest_data = self.detect_bone_crest_in_gap(
                processed, gap_mask,
                tooth1['long_axis'], tooth2['long_axis'],
                debug_dir=debug_dir, gap_id=f"{i+1}_{i+2}"
            )

            if bone_crest_data is None:
                print(f"    ⚠ 无法检测骨嵴点，跳过")
                continue

            # 沿长轴测量间距
            spacing_profile = self.measure_root_spacing_along_axis(
                tooth1, tooth2, bone_crest_data['point'],
                tooth1['long_axis'], tooth2['long_axis'],
                debug_dir=debug_dir, pair_id=f"{i+1}_{i+2}"
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
                    'risk_label': risk_label,
                    'bone_crest_point': bone_crest_data['point']
                })

                print(f"    ✓ 最小间距: {min_spacing:.2f}mm ({risk_label})")
            else:
                print(f"    ⚠ 无法测量间距")

        # 可视化结果
        print("正在生成可视化...")
        self.visualize_results_v2(original_image, teeth_data, spacing_results, image_path, output_dir)

        results = {
            'image_path': image_path,
            'teeth_data': teeth_data,
            'spacing_results': spacing_results
        }

        return results

    def visualize_results_v2(self, original_image, teeth_data, spacing_results, image_path, output_dir):
        """
        可视化分析结果（新版本 - 基于形态学方法）

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

        # 1. 原始图像 + 牙齿轮廓 + 长轴
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('牙齿检测与长轴标注', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 绘制每颗牙齿的轮廓和长轴
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

            # 绘制长轴
            if 'long_axis' in tooth:
                axis_data = tooth['long_axis']
                start = axis_data['axis_start']
                end = axis_data['axis_end']
                ax1.plot([start[0], end[0]], [start[1], end[1]],
                        'b-', linewidth=3, alpha=0.8)
                # 箭头指向牙根
                ax1.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # 绘制骨嵴点
        for result in spacing_results:
            if 'bone_crest_point' in result:
                point = result['bone_crest_point']
                ax1.scatter(point[0], point[1], c='red', s=100, marker='o',
                          edgecolors='white', linewidths=2, zorder=5)

        # 2. 牙齿间距测量示意图
        ax2 = plt.subplot(2, 2, 2)
        ax2.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('牙根间距测量（沿长轴）', fontsize=14, fontweight='bold')
        ax2.axis('off')

        for result in spacing_results:
            profile = result['profile']

            # 绘制测量线（颜色编码）
            for i, (edge1, edge2) in enumerate(profile['measurement_lines']):
                color_hex = profile['colors'][i]
                color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
                color_rgb = tuple(c/255.0 for c in color_rgb)

                ax2.plot([edge1[0], edge2[0]], [edge1[1], edge2[1]],
                        color=color_rgb, linewidth=2, alpha=0.7)

        # 3. 牙齿间距热力图
        ax3 = plt.subplot(2, 2, 3)
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
        ax4.set_ylabel('从骨嵴向下深度 (mm)', fontsize=12)
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

    def visualize_results(self, original_image, teeth_data, spacing_results, image_path, output_dir, boundary_data=None):
        """
        可视化分析结果

        参数:
            original_image: 原始图像
            teeth_data: 牙齿数据列表
            spacing_results: 间距测量结果
            image_path: 原始图像路径
            output_dir: 输出目录
            boundary_data: 边界线数据（可选）
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

        # 绘制边界线（上颌和下颌各一条）
        if boundary_data is not None:
            upper_boundary = boundary_data.get('upper_boundary')
            lower_boundary = boundary_data.get('lower_boundary')

            # 绘制上颌边界线
            if upper_boundary is not None and 'curve' in upper_boundary:
                boundary_curve = upper_boundary['curve']
                curve_x = [p[0] for p in boundary_curve]
                curve_y = [p[1] for p in boundary_curve]
                ax1.plot(curve_x, curve_y, 'b-', linewidth=4, alpha=0.9, label='上颌边界线')

                # 绘制边缘点
                if 'points' in upper_boundary:
                    points = upper_boundary['points']
                    points_x = [p[0] for p in points]
                    points_y = [p[1] for p in points]
                    ax1.scatter(points_x, points_y, c='cyan', s=20, alpha=0.6, marker='o')

            # 绘制下颌边界线
            if lower_boundary is not None and 'curve' in lower_boundary:
                boundary_curve = lower_boundary['curve']
                curve_x = [p[0] for p in boundary_curve]
                curve_y = [p[1] for p in boundary_curve]
                ax1.plot(curve_x, curve_y, 'r-', linewidth=4, alpha=0.9, label='下颌边界线')

                # 绘制边缘点
                if 'points' in lower_boundary:
                    points = lower_boundary['points']
                    points_x = [p[0] for p in points]
                    points_y = [p[1] for p in points]
                    ax1.scatter(points_x, points_y, c='orange', s=20, alpha=0.6, marker='o')

            ax1.legend(loc='upper right', fontsize=10)
        else:
            # 如果没有边界线数据，绘制单个牙齿的边界点（兼容旧方法）
            for i, tooth in enumerate(teeth_data):
                cej_curve = tooth.get('cej_curve', [])
                cej_point = tooth.get('cej_point')

                if len(cej_curve) >= 2:
                    cej_x = [p[0] for p in cej_curve]
                    cej_y = [p[1] for p in cej_curve]
                    ax1.plot(cej_x, cej_y, 'b-', linewidth=3, alpha=0.8, label=f'边界 {i+1}' if i == 0 else '')

                if cej_point is not None:
                    ax1.plot(cej_point[0], cej_point[1], 'ro', markersize=8)

        # 2. 边界线深度测量示意图
        ax2 = plt.subplot(2, 2, 2)
        ax2.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('边界线法线方向深度测量', fontsize=14, fontweight='bold')
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
    print("牙齿根部距离测量系统（形态学方法）")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. 自动检测全景X光片中的每颗牙齿（U-Net）")
    print("2. 计算每颗牙齿的长轴和方向")
    print("3. 在牙齿间隙检测骨嵴边界（形态学方法）")
    print("4. 沿牙齿长轴测量根部间距")
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
