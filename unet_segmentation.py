#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net牙齿分割模块
基于预训练的U-Net深度学习模型进行牙齿语义分割
"""

import cv2
import numpy as np
import os
import sys

# 尝试导入TensorFlow，提供友好的错误信息
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("错误：未安装TensorFlow。请运行: pip install tensorflow")
    sys.exit(1)


class UNetTeethSegmentation:
    """U-Net牙齿分割器"""

    def __init__(self, model_path='models/dental_xray_seg.h5'):
        """
        初始化U-Net分割器

        参数:
            model_path: 预训练模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.input_size = (512, 512)  # U-Net输入尺寸

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载预训练的U-Net模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            print(f"正在加载U-Net模型: {self.model_path}")
            # 尝试使用Keras加载模型
            self.model = keras.models.load_model(self.model_path, compile=False)
            print("✓ U-Net模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def preprocess_image(self, image):
        """
        预处理图像以供U-Net使用

        参数:
            image: 输入图像（BGR或灰度）

        返回:
            processed: 预处理后的图像
            original_size: 原始图像尺寸
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        original_size = gray.shape

        # 调整大小到512x512
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)

        # 归一化到[0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # 添加批次维度和通道维度: (1, 512, 512, 1)
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)

        return processed, original_size

    def postprocess_mask(self, mask, original_size, threshold=0.5):
        """
        后处理预测掩码

        参数:
            mask: U-Net预测的掩码
            original_size: 原始图像尺寸
            threshold: 二值化阈值

        返回:
            binary_mask: 二值化后的掩码
        """
        # 移除批次维度和通道维度
        mask = np.squeeze(mask)

        # 调整大小回原始尺寸
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]),
                                  interpolation=cv2.INTER_LINEAR)

        # 二值化
        binary_mask = (mask_resized > threshold).astype(np.uint8) * 255

        return binary_mask

    def refine_mask(self, mask, open_iteration=2, erode_iteration=3):
        """
        使用形态学操作细化掩码
        与开源仓库CCA_Analysis.py完全一致的后处理流程
        参考：https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net/blob/master/CCA_Analysis.py

        参数:
            mask: 二值掩码
            open_iteration: 开运算迭代次数（默认2，与开源一致）
            erode_iteration: 腐蚀迭代次数（默认3，与开源一致）

        返回:
            refined: 细化后的掩码
        """
        # 与开源仓库完全相同的参数
        # kernel1 = np.ones((5,5), dtype=np.float32)
        kernel1 = np.ones((5, 5), dtype=np.float32)
        kernel_sharpening = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])

        image = mask

        # 1. 形态学开运算去除小噪声
        # image=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1,iterations=open_iteration )
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1, iterations=open_iteration)

        # 2. 应用锐化滤波器增强边缘
        # image = cv2.filter2D(image, -1, kernel_sharpening)
        image = cv2.filter2D(image, -1, kernel_sharpening)

        # 3. 腐蚀以分离相邻牙齿
        # image=cv2.erode(image,kernel1,iterations =erode_iteration)
        image = cv2.erode(image, kernel1, iterations=erode_iteration)

        return image

    def segment_teeth(self, image, threshold=0.5):
        """
        使用U-Net分割牙齿

        参数:
            image: 输入图像（BGR或灰度）
            threshold: 二值化阈值（默认0.5）

        返回:
            mask: 牙齿分割掩码（二值图像）
            refined_mask: 细化后的掩码
        """
        # 预处理
        processed, original_size = self.preprocess_image(image)

        # U-Net推理
        prediction = self.model.predict(processed, verbose=0)

        # 后处理
        mask = self.postprocess_mask(prediction, original_size, threshold=threshold)

        # 细化掩码
        refined_mask = self.refine_mask(mask)

        return mask, refined_mask

    def _find_optimal_threshold(self, prediction):
        """
        自动寻找最佳二值化阈值

        参数:
            prediction: U-Net预测输出

        返回:
            optimal_threshold: 最佳阈值
        """
        pred_flat = prediction.flatten()

        # 分析预测值分布
        pred_mean = pred_flat.mean()
        pred_std = pred_flat.std()
        pred_max = pred_flat.max()

        # 如果最大值很小，说明模型预测置信度很低
        if pred_max < 0.3:
            print(f"  警告：模型预测置信度很低 (max={pred_max:.3f})")
            return 0.1  # 使用更低的阈值

        # 如果预测值普遍较低
        if pred_mean < 0.2:
            return max(0.1, pred_mean + pred_std)

        # 使用Otsu方法寻找最佳阈值
        # 将预测值转换为0-255范围
        pred_255 = (pred_flat * 255).astype(np.uint8)

        # 计算直方图
        hist, bin_edges = np.histogram(pred_255, bins=256, range=(0, 256))

        # Otsu算法
        total = len(pred_255)
        sum_total = np.dot(np.arange(256), hist)

        sum_background = 0
        weight_background = 0
        max_variance = 0
        optimal_threshold_255 = 0

        for t in range(256):
            weight_background += hist[t]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * hist[t]

            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            if variance > max_variance:
                max_variance = variance
                optimal_threshold_255 = t

        optimal_threshold = optimal_threshold_255 / 255.0

        # 限制阈值范围
        optimal_threshold = max(0.1, min(0.7, optimal_threshold))

        return optimal_threshold

    def extract_individual_teeth(self, mask, min_area=2000):
        """
        从分割掩码中提取单个牙齿
        使用连通组件分析（CCA）
        与开源仓库CCA_Analysis.py完全一致
        参考：https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net/blob/master/CCA_Analysis.py

        参数:
            mask: 牙齿分割掩码
            min_area: 最小牙齿面积（默认2000，与开源仓库一致：c_area>2000）

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        # 与开源仓库完全一致的连通组件分析
        # labels=cv2.connectedComponents(thresh,connectivity=8)[1]
        labels = cv2.connectedComponents(mask, connectivity=8)[1]
        a = np.unique(labels)

        teeth_data = []

        # 遍历每个label（与开源仓库完全一致的循环）
        # for label in a:
        for label in a:
            if label == 0:
                continue

            # Create a mask
            # mask = np.zeros(thresh.shape, dtype="uint8")
            # mask[labels == label] = 255
            tooth_mask = np.zeros(mask.shape, dtype="uint8")
            tooth_mask[labels == label] = 255

            # Find contours and determine contour area
            # cnts,hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = cnts[0]
            cnts, hieararch = cv2.findContours(tooth_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            cnts = cnts[0]

            # c_area = cv2.contourArea(cnts)
            c_area = cv2.contourArea(cnts)

            # threshhold for tooth count
            # if c_area>2000:
            if c_area > min_area:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(cnts)

                # 计算最小外接矩形（与开源仓库一致）
                # rect = cv2.minAreaRect(cnts)
                # box = cv2.boxPoints(rect)
                # box = np.array(box, dtype="int")
                rect = cv2.minAreaRect(cnts)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype="int")

                # 计算质心（近似）
                M = cv2.moments(cnts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                teeth_data.append({
                    'label': label,
                    'contour': cnts,
                    'mask': tooth_mask,
                    'bbox': (x, y, w, h),
                    'rect': rect,
                    'box': box,
                    'centroid': (cx, cy),
                    'area': c_area
                })

        # 按X坐标排序（从左到右）
        teeth_data.sort(key=lambda t: t['centroid'][0])

        return teeth_data

    def visualize_segmentation(self, image, mask, teeth_data=None):
        """
        可视化分割结果

        参数:
            image: 原始图像
            mask: 分割掩码
            teeth_data: 牙齿数据列表（可选）

        返回:
            vis_image: 可视化图像
        """
        # 确保图像是BGR格式
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # 创建掩码覆盖层（红色）
        mask_color = np.zeros_like(vis_image)
        mask_color[:, :, 2] = mask  # 红色通道

        # 半透明叠加
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_color, 0.3, 0)

        # 如果提供了牙齿数据，绘制轮廓和编号
        if teeth_data is not None:
            for i, tooth in enumerate(teeth_data):
                # 绘制轮廓（蓝色）
                cv2.drawContours(vis_image, [tooth['contour']], -1, (255, 0, 0), 2)

                # 绘制中心点
                centroid = tuple(map(int, tooth['centroid']))
                cv2.circle(vis_image, centroid, 5, (0, 255, 0), -1)

                # 标注编号
                cv2.putText(vis_image, f'T{i+1}',
                           (centroid[0] - 10, centroid[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return vis_image


def test_unet_segmentation(image_path='input/107.png'):
    """
    测试U-Net分割功能

    参数:
        image_path: 测试图像路径
    """
    print("=" * 60)
    print("U-Net牙齿分割测试")
    print("=" * 60)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    print(f"✓ 读取图像: {image_path}")
    print(f"  图像尺寸: {image.shape}")

    # 创建分割器
    try:
        segmenter = UNetTeethSegmentation()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 分割牙齿
    print("\n正在进行U-Net分割...")
    mask, refined_mask = segmenter.segment_teeth(image)
    print("✓ 分割完成")

    # 提取单个牙齿
    print("\n正在提取单个牙齿...")
    teeth_data = segmenter.extract_individual_teeth(refined_mask)
    print(f"✓ 检测到 {len(teeth_data)} 颗牙齿")

    # 可视化
    print("\n正在生成可视化...")
    vis_image = segmenter.visualize_segmentation(image, refined_mask, teeth_data)

    # 保存结果
    os.makedirs('output', exist_ok=True)
    output_path = 'output/unet_segmentation_test.png'
    cv2.imwrite(output_path, vis_image)
    print(f"✓ 可视化结果已保存: {output_path}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 如果直接运行此脚本，执行测试
    test_unet_segmentation()
