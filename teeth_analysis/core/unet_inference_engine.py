#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net推理引擎模块
负责加载和运行U-Net深度学习模型
"""

import os
import sys

# 尝试导入TensorFlow，提供友好的错误信息
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("错误：未安装TensorFlow。请运行: pip install tensorflow")
    sys.exit(1)


class UNetInferenceEngine:
    """U-Net推理引擎"""

    def __init__(self, model_path='models/dental_xray_seg.h5'):
        """
        初始化U-Net推理引擎

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
            # 使用Keras加载模型（不编译以加快加载速度）
            self.model = keras.models.load_model(self.model_path, compile=False)
            print("✓ U-Net模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def predict(self, preprocessed_image, verbose=0):
        """
        使用U-Net模型进行推理

        参数:
            preprocessed_image: 预处理后的图像 (1, H, W, 1)
            verbose: 是否显示推理进度

        返回:
            prediction: 模型预测输出
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        prediction = self.model.predict(preprocessed_image, verbose=verbose)
        return prediction

    def get_model_info(self):
        """
        获取模型信息

        返回:
            info: 模型信息字典
        """
        if self.model is None:
            return None

        info = {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'layers': len(self.model.layers)
        }

        return info

    def print_model_summary(self):
        """打印模型摘要"""
        if self.model is not None:
            self.model.summary()
        else:
            print("模型未加载")
