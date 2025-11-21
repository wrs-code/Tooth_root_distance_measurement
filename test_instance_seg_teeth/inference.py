#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿实例分割推理模块
基于Instance_seg_teeth仓库的OralBBNet方法 (YOLOv8 + U-Net)
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from skimage import exposure

# 添加Instance_seg_teeth到路径（如果需要的话）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class InstanceSegmentationPipeline:
    """
    牙齿实例分割流水线
    结合YOLO检测和U-Net分割
    """

    def __init__(self, yolo_model_path=None, unet_model_path=None, img_size=512):
        """
        初始化实例分割流水线

        Args:
            yolo_model_path: YOLO模型路径
            unet_model_path: U-Net模型路径
            img_size: 图像大小
        """
        self.img_size = img_size
        self.yolo_model = None
        self.unet_model = None

        # 加载YOLO模型
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                from ultralytics import YOLO
                print(f"正在加载YOLO模型: {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
                print("✓ YOLO模型加载成功")
            except Exception as e:
                print(f"✗ YOLO模型加载失败: {e}")

        # 加载U-Net模型
        if unet_model_path and os.path.exists(unet_model_path):
            try:
                print(f"正在加载U-Net模型: {unet_model_path}")
                self.unet_model = self._load_unet_model(unet_model_path)
                print("✓ U-Net模型加载成功")
            except Exception as e:
                print(f"✗ U-Net模型加载失败: {e}")

    def _load_unet_model(self, model_path):
        """加载U-Net模型"""
        from tensorflow.keras.metrics import Precision, Recall

        # 定义模型架构（从notebook提取）
        model = self._build_unet_model()

        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(0.0003),
            loss=self._dice_loss_with_l2_regularization,
            metrics=[Precision(), Recall(), self._dice_coef]
        )

        # 加载权重
        model.load_weights(model_path)
        return model

    def _build_unet_model(self, img_size=(512, 512), num_classes=32, drop_rate=0.12):
        """构建U-Net模型（从Instance_seg_teeth notebook提取）"""
        inputs = keras.Input(shape=img_size + (35,))

        inputs0 = inputs[:, :, :, 32:]  # 图像
        inputs1 = inputs[:, :, :, :32]  # 边界框
        skip_connections = []
        bb_out = []

        # Entry block
        x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(inputs0)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        skip_connections.append(x)

        # Encoder
        for filters in [128, 256, 512, 1024]:
            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
            x = keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(drop_rate)(x)

            x = keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(drop_rate)(x)
            skip_connections.append(x)

        # Bounding box branch
        bb = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=1)(inputs1)
        bb = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bb)
        bb = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                    activation=tf.nn.sigmoid)(bb)
        bb_out.append(bb)

        for idx, filters in enumerate([128, 256, 512]):
            bb = tf.keras.layers.MaxPool2D(pool_size=(pow(2, idx+1), pow(2, idx+1)),
                                           strides=pow(2, idx+1))(inputs1)
            bb = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(bb)
            bb = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
                                        activation=tf.nn.sigmoid)(bb)
            bb_out.append(bb)

        skip_connections.pop()

        # Decoder
        for filters in [512, 256, 128]:
            x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(drop_rate)(x)

            skip_connection = skip_connections.pop()
            bb_layer = bb_out.pop()
            out = tf.multiply(skip_connection, bb_layer)
            x = keras.layers.concatenate([x, out])

            x = keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(drop_rate)(x)

            x = keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.SpatialDropout2D(drop_rate)(x)

        filters = 64
        x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        skip_connection = skip_connections.pop()
        bb_layer = bb_out.pop()
        out = tf.multiply(skip_connection, bb_layer)
        x = keras.layers.concatenate([x, out])

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)

        outputs = keras.layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

        model = keras.Model(inputs, outputs)
        return model

    def _dice_loss_with_l2_regularization(self, target, predicted, epsilon=1e-7, l2_weight=0.1):
        """Dice损失函数（从notebook提取）"""
        intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
        predicted_square = tf.square(predicted)
        target_square = tf.square(target)
        union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
        dice = (2 * intersection + epsilon) / (union + epsilon)
        mean_dice_loss = tf.reduce_mean(dice)

        l2_norm = tf.reduce_sum(tf.square(predicted - target), axis=[1, 2])
        l2_regularization = l2_weight * tf.reduce_mean(l2_norm)

        total_loss = mean_dice_loss + l2_regularization
        return total_loss

    def _dice_coef(self, target, predicted, epsilon=1e-7):
        """Dice系数（从notebook提取）"""
        predicted = tf.where(predicted < 0.51, 0.00, 1.00)
        intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
        predicted_square = tf.square(predicted)
        target_square = tf.square(target)
        union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
        dice = (2 * intersection + epsilon) / (union + epsilon)
        mean_dice_loss = -tf.reduce_sum(dice)
        return -mean_dice_loss

    def apply_clahe(self, image, clip_limit=0.02):
        """应用CLAHE（从notebook提取）"""
        image_float = image.astype(float) / 255.0
        image_clahe = exposure.equalize_adapthist(image_float, clip_limit=clip_limit)
        image_clahe = (image_clahe * 255).astype(image.dtype)
        return image_clahe

    def normalize(self, image_data):
        """归一化图像"""
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val > min_val:
            image_data = (image_data - min_val) / (max_val - min_val)
        return image_data

    def detect_with_yolo(self, image_path, iou=0.7, conf=0.5):
        """使用YOLO检测牙齿（从notebook提取）"""
        if self.yolo_model is None:
            raise ValueError("YOLO模型未加载")

        results = self.yolo_model.predict(source=image_path, iou=iou, conf=conf, save=False)

        boxes_list = []
        classes_list = []

        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            boxes_list.append(boxes)
            classes_list.append(classes)

        return boxes_list, classes_list

    def create_binary_masks(self, boxes_list, classes_list, img_width=640, img_height=640):
        """创建二值掩码（从notebook提取）"""
        bin_masks = []

        for boxes, classes in zip(boxes_list, classes_list):
            binary_map = np.zeros((32, img_height, img_width), dtype=np.uint8)

            for box, class_id in zip(boxes.tolist(), classes.tolist()):
                x, y, w, h = box
                x1 = max(0, min(int(round(x - w/2)), img_width))
                y1 = max(0, min(int(round(y - h/2)), img_height))
                x2 = max(0, min(int(round(x + w/2)), img_width))
                y2 = max(0, min(int(round(y + h/2)), img_height))

                binary_map[int(class_id), y1:y2, x1:x2] = 1

            binary_map = np.nan_to_num(binary_map, nan=0)
            bin_masks.append(binary_map)

        return bin_masks

    def segment_image(self, image_path, output_dir=None):
        """
        分割单张图像

        Args:
            image_path: 图像路径
            output_dir: 输出目录

        Returns:
            dict: 包含pred_mask, binary_mask等的结果字典
        """
        if self.yolo_model is None or self.unet_model is None:
            raise ValueError("模型未完全加载")

        print(f"\n处理图像: {os.path.basename(image_path)}")

        # 1. YOLO检测
        print("  步骤1: YOLO检测...")
        boxes_list, classes_list = self.detect_with_yolo(image_path)

        if len(boxes_list) == 0 or len(boxes_list[0]) == 0:
            print("  ✗ 未检测到牙齿")
            return None

        print(f"  ✓ 检测到 {len(boxes_list[0])} 个边界框")

        # 2. 创建二值掩码
        print("  步骤2: 创建二值掩码...")
        bin_masks = self.create_binary_masks(boxes_list, classes_list)

        # 3. 预处理图像
        print("  步骤3: 预处理图像...")
        img = Image.open(image_path)
        img_data = np.array(img, dtype=np.float32)
        img_data = self.apply_clahe(img_data)
        img_data = cv2.resize(img_data, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)
        img_data = self.normalize(img_data)

        # 4. 准备U-Net输入
        print("  步骤4: 准备U-Net输入...")
        bin_mask = bin_masks[0]
        bin_mask = np.transpose(bin_mask, axes=[1, 2, 0])
        bin_mask = cv2.resize(bin_mask, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.repeat(img_data, 3, axis=-1)
        elif img_data.shape[-1] == 1:
            img_data = np.repeat(img_data, 3, axis=-1)

        combined_input = np.concatenate((bin_mask, img_data), axis=2)
        combined_input = np.expand_dims(combined_input, axis=0)

        # 5. U-Net预测
        print("  步骤5: U-Net分割...")
        predictions = self.unet_model.predict(combined_input, verbose=0)
        pred_mask = predictions[0]

        print("  ✓ 分割完成")

        # 6. 可视化（如果指定了输出目录）
        if output_dir:
            self.visualize_result(image_path, pred_mask, output_dir)

        return {
            'pred_mask': pred_mask,
            'binary_mask': bin_masks[0],
            'image_path': image_path
        }

    def visualize_result(self, image_path, pred_mask, output_dir):
        """可视化结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 读取原图
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 创建彩色掩码
        colored_mask = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        for i in range(32):
            mask = pred_mask[:, :, i] > 0.5
            if np.any(mask):
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask[mask] = color

        # 叠加
        overlay = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)

        # 保存
        basename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"seg_{basename}")
        cv2.imwrite(save_path, overlay)
        print(f"  ✓ 结果已保存: {save_path}")

    def batch_segment(self, input_dir, output_dir):
        """
        批量分割

        Args:
            input_dir: 输入目录
            output_dir: 输出目录

        Returns:
            list: 结果列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]")
            try:
                result = self.segment_image(image_path, output_dir=output_dir)
                results.append(result)
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results.append(None)

        return results
