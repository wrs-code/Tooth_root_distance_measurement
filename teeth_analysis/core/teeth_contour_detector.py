#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿轮廓检测器模块
使用连通组件分析（CCA）从分割掩码中提取单个牙齿轮廓
与开源仓库CCA_Analysis.py完全一致
"""

import cv2
import numpy as np


class TeethContourDetector:
    """
    牙齿轮廓检测器
    使用连通组件分析提取单个牙齿
    """

    def __init__(self, min_area=2000, connectivity=8):
        """
        初始化轮廓检测器

        参数:
            min_area: 最小牙齿面积阈值（默认2000，与开源仓库一致）
            connectivity: 连通性（4或8，默认8）
        """
        self.min_area = min_area
        self.connectivity = connectivity

    def extract_connected_components(self, mask):
        """
        使用连通组件分析提取各个区域

        参数:
            mask: 二值掩码

        返回:
            num_labels: 标签数量
            labels: 标签图
            stats: 统计信息
            centroids: 质心坐标
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=self.connectivity
        )
        return num_labels, labels, stats, centroids

    def filter_by_area(self, area):
        """
        根据面积判断是否为有效牙齿
        与开源仓库一致：c_area > 2000

        参数:
            area: 区域面积

        返回:
            is_valid: 是否为有效牙齿
        """
        return area > self.min_area

    def create_single_tooth_mask(self, labels, label_id, mask_shape):
        """
        创建单个牙齿的掩码

        参数:
            labels: 标签图
            label_id: 牙齿标签ID
            mask_shape: 掩码形状

        返回:
            tooth_mask: 单个牙齿的二值掩码
        """
        tooth_mask = np.zeros(mask_shape, dtype=np.uint8)
        tooth_mask[labels == label_id] = 255
        return tooth_mask

    def extract_contour(self, tooth_mask):
        """
        从牙齿掩码中提取轮廓

        参数:
            tooth_mask: 单个牙齿的二值掩码

        返回:
            contour: 轮廓点（如果找到）
            None: 未找到轮廓
        """
        contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # 返回最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        return contour

    def compute_bounding_box(self, contour):
        """
        计算轮廓的边界框

        参数:
            contour: 轮廓点

        返回:
            bbox: 边界框 (x, y, w, h)
        """
        return cv2.boundingRect(contour)

    def compute_rotated_rect(self, contour):
        """
        计算轮廓的最小外接矩形

        参数:
            contour: 轮廓点

        返回:
            rect: 旋转矩形 ((center_x, center_y), (width, height), angle)
            box: 矩形的4个顶点坐标
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        return rect, box

    def order_box_points(self, pts):
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

    def extract_teeth_from_mask(self, mask):
        """
        从分割掩码中提取单个牙齿
        与开源仓库CCA_Analysis.py一致：
        - connectivity=8
        - 面积阈值 > 2000

        参数:
            mask: 牙齿分割掩码（二值图像）

        返回:
            teeth_data: 包含每颗牙齿信息的列表
                每个元素包含：
                - label: 标签ID
                - contour: 轮廓点
                - mask: 单个牙齿掩码
                - bbox: 边界框 (x, y, w, h)
                - rect: 旋转矩形
                - box: 矩形顶点（已排序）
                - centroid: 质心坐标 (x, y)
                - area: 面积
        """
        # 1. 连通组件分析（connectivity=8，与开源仓库一致）
        num_labels, labels, stats, centroids = self.extract_connected_components(mask)

        teeth_data = []

        # 2. 遍历每个连通组件（跳过背景label=0）
        for label in range(1, num_labels):
            # 获取组件统计信息
            area = stats[label, cv2.CC_STAT_AREA]

            # 3. 面积过滤：与开源仓库一致（c_area > 2000）
            if not self.filter_by_area(area):
                continue

            # 4. 创建单个牙齿的掩码
            tooth_mask = self.create_single_tooth_mask(labels, label, mask.shape)

            # 5. 查找轮廓
            contour = self.extract_contour(tooth_mask)
            if contour is None:
                continue

            # 6. 计算边界框
            bbox = self.compute_bounding_box(contour)

            # 7. 计算最小外接矩形
            rect, box = self.compute_rotated_rect(contour)

            # 8. 对矩形顶点排序（左上、右上、右下、左下）
            box = self.order_box_points(box)

            teeth_data.append({
                'label': label,
                'contour': contour,
                'mask': tooth_mask,
                'bbox': bbox,
                'rect': rect,
                'box': box,
                'centroid': centroids[label],
                'area': area
            })

        # 9. 按X坐标排序（从左到右）
        teeth_data.sort(key=lambda t: t['centroid'][0])

        return teeth_data

    def filter_by_aspect_ratio(self, teeth_data, min_ratio=0.2, max_ratio=2.0):
        """
        根据宽高比过滤牙齿（可选的额外过滤步骤）

        参数:
            teeth_data: 牙齿数据列表
            min_ratio: 最小宽高比
            max_ratio: 最大宽高比

        返回:
            filtered_data: 过滤后的牙齿数据
        """
        filtered_data = []

        for tooth in teeth_data:
            x, y, w, h = tooth['bbox']
            aspect_ratio = float(w) / h if h > 0 else 0

            if min_ratio <= aspect_ratio <= max_ratio:
                filtered_data.append(tooth)

        return filtered_data

    def update_min_area(self, min_area):
        """
        动态更新最小面积阈值

        参数:
            min_area: 新的最小面积阈值

        使用示例:
            >>> detector = TeethContourDetector()
            >>> detector.update_min_area(3000)  # 提高阈值，过滤更多小区域
        """
        self.min_area = min_area
