#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理器模块
负责X光图像的预处理操作，包括增强、降噪和归一化
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """X光图像预处理器"""

    def __init__(self, target_size=(512, 512)):
        """
        初始化预处理器

        参数:
            target_size: 目标图像尺寸（用于深度学习模型输入）
        """
        self.target_size = target_size

    def convert_to_grayscale(self, image):
        """
        转换为灰度图

        参数:
            image: 输入图像（BGR或灰度）

        返回:
            gray: 灰度图像
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def apply_clahe(self, gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        应用CLAHE（对比度限制自适应直方图均衡化）

        参数:
            gray_image: 灰度图像
            clip_limit: 对比度限制阈值
            tile_grid_size: 分块网格大小

        返回:
            enhanced: 增强后的图像
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_image)

    def apply_bilateral_filter(self, image, diameter=9, sigma_color=75, sigma_space=75):
        """
        应用双边滤波（保留边缘的降噪）

        参数:
            image: 输入图像
            diameter: 滤波器直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差

        返回:
            filtered: 滤波后的图像
        """
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    def resize_image(self, image, size=None, interpolation=cv2.INTER_AREA):
        """
        调整图像大小

        参数:
            image: 输入图像
            size: 目标尺寸，默认使用self.target_size
            interpolation: 插值方法

        返回:
            resized: 调整后的图像
            original_size: 原始尺寸
        """
        if size is None:
            size = self.target_size

        original_size = image.shape[:2]  # (height, width)
        resized = cv2.resize(image, size, interpolation=interpolation)
        return resized, original_size

    def normalize_image(self, image, range_min=0.0, range_max=1.0):
        """
        归一化图像到指定范围

        参数:
            image: 输入图像
            range_min: 最小值
            range_max: 最大值

        返回:
            normalized: 归一化后的图像
        """
        normalized = image.astype(np.float32)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        normalized = normalized * (range_max - range_min) + range_min
        return normalized

    def prepare_for_unet(self, image):
        """
        为U-Net模型准备图像

        参数:
            image: 输入图像（BGR或灰度）

        返回:
            processed: 预处理后的图像 (1, H, W, 1)
            original_size: 原始图像尺寸
        """
        # 1. 转换为灰度图
        gray = self.convert_to_grayscale(image)

        # 2. 调整大小到目标尺寸
        resized, original_size = self.resize_image(gray)

        # 3. 归一化到[0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # 4. 添加批次维度和通道维度: (1, H, W, 1)
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)

        return processed, original_size

    def enhance_xray(self, image):
        """
        增强X光图像（用于传统方法）

        参数:
            image: 输入图像（BGR或灰度）

        返回:
            enhanced: 增强后的图像
            gray: 原始灰度图
        """
        # 1. 转换为灰度图
        gray = self.convert_to_grayscale(image)

        # 2. 应用CLAHE增强对比度
        enhanced = self.apply_clahe(gray)

        # 3. 应用双边滤波保留边缘的同时降噪
        enhanced = self.apply_bilateral_filter(enhanced)

        return enhanced, gray
