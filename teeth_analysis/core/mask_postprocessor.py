#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码后处理器模块
负责对U-Net预测掩码进行后处理，包括形态学操作和锐化
与开源仓库CCA_Analysis.py完全一致
"""

import cv2
import numpy as np


class MaskPostprocessor:
    """
    掩码后处理器
    实现与开源仓库一致的后处理流程
    """

    def __init__(self, kernel_size=5, open_iteration=2, erode_iteration=1):
        """
        初始化后处理器

        参数:
            kernel_size: 形态学核大小（默认5，与开源仓库一致）
            open_iteration: 开运算迭代次数（默认2）
            erode_iteration: 腐蚀迭代次数（默认1）⚙️ 主要参数
        """
        self.kernel_size = kernel_size
        self.open_iteration = open_iteration
        self.erode_iteration = erode_iteration

        # 创建形态学核
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 创建锐化核（与开源仓库一致）
        self.sharpening_kernel = np.array([[-1, -1, -1],
                                           [-1,  9, -1],
                                           [-1, -1, -1]])

    def resize_to_original(self, mask, original_size):
        """
        将掩码调整回原始尺寸

        参数:
            mask: U-Net预测的掩码
            original_size: 原始图像尺寸 (height, width)

        返回:
            resized_mask: 调整后的掩码
        """
        # 移除批次维度和通道维度（如果存在）
        mask = np.squeeze(mask)

        # 调整大小回原始尺寸
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]),
                                  interpolation=cv2.INTER_LINEAR)

        return mask_resized

    def binarize_mask(self, mask, threshold=0.5):
        """
        二值化掩码

        参数:
            mask: 连续值掩码
            threshold: 二值化阈值

        返回:
            binary_mask: 二值掩码 (0或255)
        """
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        return binary_mask

    def apply_opening(self, mask, iterations=None):
        """
        应用形态学开运算去除小噪声

        参数:
            mask: 二值掩码
            iterations: 迭代次数，默认使用self.open_iteration

        返回:
            opened: 开运算后的掩码
        """
        if iterations is None:
            iterations = self.open_iteration

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=iterations)
        return opened

    def apply_sharpening(self, mask):
        """
        应用锐化滤波器增强边缘
        使用与开源仓库一致的锐化核

        参数:
            mask: 输入掩码

        返回:
            sharpened: 锐化后的掩码
        """
        sharpened = cv2.filter2D(mask, -1, self.sharpening_kernel)
        return sharpened

    def apply_erosion(self, mask, iterations=None):
        """
        应用腐蚀操作以分离相邻牙齿
        ⚙️ 这是调整牙齿分离程度的关键参数

        参数:
            mask: 二值掩码
            iterations: 迭代次数，默认使用self.erode_iteration

        返回:
            eroded: 腐蚀后的掩码

        调整建议:
            - iterations=0: 无腐蚀，保持原始边界
            - iterations=1: 轻度腐蚀（默认，与开源仓库一致）
            - iterations=2: 中度腐蚀，适用于牙齿紧密相连
            - iterations=3+: 强腐蚀，适用于严重粘连
        """
        if iterations is None:
            iterations = self.erode_iteration

        eroded = cv2.erode(mask, self.kernel, iterations=iterations)
        return eroded

    def refine_mask(self, mask):
        """
        完整的掩码细化流程
        与开源仓库CCA_Analysis.py完全一致的处理顺序：
        1. 形态学开运算去除小噪声
        2. 锐化滤波器增强边缘
        3. 腐蚀以分离相邻牙齿

        参数:
            mask: 二值掩码

        返回:
            refined: 细化后的掩码
        """
        # 确保mask是uint8类型
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # 1. 形态学开运算去除小噪声（5x5核）
        opened = self.apply_opening(mask)

        # 2. 应用锐化滤波器增强边缘
        sharpened = self.apply_sharpening(opened)

        # 3. 腐蚀以分离相邻牙齿
        eroded = self.apply_erosion(sharpened)

        return eroded

    def postprocess_prediction(self, prediction, original_size, threshold=0.5):
        """
        完整的后处理流程：从U-Net预测到细化掩码

        参数:
            prediction: U-Net模型预测输出
            original_size: 原始图像尺寸 (height, width)
            threshold: 二值化阈值

        返回:
            binary_mask: 二值化掩码
            refined_mask: 细化后的掩码
        """
        # 1. 调整大小回原始尺寸
        mask_resized = self.resize_to_original(prediction, original_size)

        # 2. 二值化
        binary_mask = self.binarize_mask(mask_resized, threshold=threshold)

        # 3. 细化掩码（开运算 -> 锐化 -> 腐蚀）
        refined_mask = self.refine_mask(binary_mask)

        return binary_mask, refined_mask

    def update_parameters(self, open_iteration=None, erode_iteration=None, kernel_size=None):
        """
        动态更新后处理参数

        参数:
            open_iteration: 开运算迭代次数
            erode_iteration: 腐蚀迭代次数
            kernel_size: 核大小

        使用示例:
            >>> processor = MaskPostprocessor()
            >>> processor.update_parameters(erode_iteration=2)  # 增强腐蚀
        """
        if open_iteration is not None:
            self.open_iteration = open_iteration

        if erode_iteration is not None:
            self.erode_iteration = erode_iteration

        if kernel_size is not None:
            self.kernel_size = kernel_size
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
