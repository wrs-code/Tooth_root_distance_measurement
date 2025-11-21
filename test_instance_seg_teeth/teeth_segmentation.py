"""
牙齿实例分割推理类
结合YOLOv8检测和U-Net分割
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
from skimage import exposure

from .unet_model import get_model, dice_coef, dice_loss_with_l2_regularization


class TeethSegmentation:
    """牙齿实例分割类"""

    def __init__(self, yolo_model_path=None, unet_model_path=None, img_size=512):
        """
        初始化牙齿分割模型

        Args:
            yolo_model_path: YOLOv8模型权重路径
            unet_model_path: U-Net模型权重路径
            img_size: 处理图像大小，默认512
        """
        self.img_size = img_size
        self.yolo_model = None
        self.unet_model = None

        # 加载YOLO模型
        if yolo_model_path and os.path.exists(yolo_model_path):
            print(f"加载YOLO模型: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
        else:
            print("警告: YOLO模型路径未提供或不存在")

        # 加载U-Net模型
        if unet_model_path and os.path.exists(unet_model_path):
            print(f"加载U-Net模型: {unet_model_path}")
            self.unet_model = self._load_unet_model(unet_model_path)
        else:
            print("警告: U-Net模型路径未提供或不存在")

    def _load_unet_model(self, model_path):
        """加载U-Net模型"""
        from tensorflow.keras.metrics import Precision, Recall

        model = get_model(img_size=(self.img_size, self.img_size), num_classes=32)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss=dice_loss_with_l2_regularization,
            metrics=[Precision(), Recall(), dice_coef]
        )
        model.load_weights(model_path)
        return model

    def apply_clahe(self, image, clip_limit=0.02):
        """
        应用CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Args:
            image: 输入图像
            clip_limit: 对比度限制

        Returns:
            处理后的图像
        """
        image_float = image.astype(float) / 255.0
        image_clahe = exposure.equalize_adapthist(image_float, clip_limit=clip_limit)
        image_clahe = (image_clahe * 255).astype(image.dtype)
        return image_clahe

    def normalize(self, image_data):
        """归一化图像数据"""
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        if max_val > min_val:
            image_data = (image_data - min_val) / (max_val - min_val)
        return image_data

    def detect_teeth_yolo(self, image_path, iou=0.7, conf=0.5):
        """
        使用YOLO检测牙齿

        Args:
            image_path: 图像路径
            iou: IOU阈值
            conf: 置信度阈值

        Returns:
            检测结果 (boxes, classes)
        """
        if self.yolo_model is None:
            raise ValueError("YOLO模型未加载")

        results = self.yolo_model.predict(source=image_path, iou=iou, conf=conf, save=False)

        boxes_list = []
        classes_list = []

        for result in results:
            boxes = result.boxes
            classes = boxes.cls
            boxes_xywh = boxes.xywh

            boxes_list.append(boxes_xywh.cpu().numpy())
            classes_list.append(classes.cpu().numpy())

        return boxes_list, classes_list

    def create_binary_masks(self, boxes_list, classes_list, img_width=640, img_height=640):
        """
        从YOLO检测结果创建二值掩码

        Args:
            boxes_list: 边界框列表
            classes_list: 类别列表
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            二值掩码数组
        """
        bin_masks = []

        for boxes, classes in zip(boxes_list, classes_list):
            binary_map = np.zeros((32, img_height, img_width), dtype=np.uint8)

            for box, class_id in zip(boxes.tolist(), classes.tolist()):
                x, y, w, h = box
                x1, y1 = int(round(x - w/2)), int(round(y - h/2))
                x2, y2 = int(round(x + w/2)), int(round(y + h/2))

                # 确保坐标在有效范围内
                x1 = max(0, min(x1, img_width))
                x2 = max(0, min(x2, img_width))
                y1 = max(0, min(y1, img_height))
                y2 = max(0, min(y2, img_height))

                binary_map[int(class_id), y1:y2, x1:x2] = 1

            # 移除NaN值
            binary_map = np.nan_to_num(binary_map, nan=0)
            bin_masks.append(binary_map)

        return bin_masks

    def preprocess_image(self, image_path):
        """
        预处理图像

        Args:
            image_path: 图像路径

        Returns:
            预处理后的图像数组
        """
        img = Image.open(image_path)
        img_data = np.array(img, dtype=np.float32)

        # 应用CLAHE
        img_data = self.apply_clahe(img_data)

        # 调整大小
        img_data = cv2.resize(img_data, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        # 归一化
        img_data = self.normalize(img_data)

        return img_data

    def segment(self, image_path, yolo_iou=0.7, yolo_conf=0.5):
        """
        对图像进行牙齿分割

        Args:
            image_path: 图像路径
            yolo_iou: YOLO的IOU阈值
            yolo_conf: YOLO的置信度阈值

        Returns:
            分割掩码
        """
        if self.yolo_model is None or self.unet_model is None:
            raise ValueError("模型未完全加载，请检查模型路径")

        # 1. 使用YOLO检测牙齿
        print("步骤1: 使用YOLO检测牙齿...")
        boxes_list, classes_list = self.detect_teeth_yolo(image_path, iou=yolo_iou, conf=yolo_conf)

        if len(boxes_list) == 0 or len(boxes_list[0]) == 0:
            print("警告: 未检测到牙齿")
            return None

        # 2. 创建二值掩码
        print("步骤2: 创建边界框掩码...")
        bin_masks = self.create_binary_masks(boxes_list, classes_list)

        # 3. 预处理图像
        print("步骤3: 预处理图像...")
        img_data = self.preprocess_image(image_path)

        # 4. 准备U-Net输入
        print("步骤4: 准备U-Net输入...")
        bin_mask = bin_masks[0]  # 取第一张图像的掩码
        bin_mask = np.transpose(bin_mask, axes=[1, 2, 0])  # (H, W, 32)
        bin_mask = cv2.resize(bin_mask, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        # 扩展图像维度以匹配输入
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.repeat(img_data, 3, axis=-1)
        elif img_data.shape[-1] == 1:
            img_data = np.repeat(img_data, 3, axis=-1)

        # 合并图像和边界框掩码
        combined_input = np.concatenate((bin_mask, img_data), axis=2)  # (H, W, 35)
        combined_input = np.expand_dims(combined_input, axis=0)  # (1, H, W, 35)

        # 5. U-Net预测
        print("步骤5: 使用U-Net进行分割...")
        predictions = self.unet_model.predict(combined_input)

        # 6. 后处理
        print("步骤6: 后处理结果...")
        pred_mask = predictions[0]  # (H, W, 32)

        return pred_mask

    def visualize_segmentation(self, image_path, pred_mask, save_path=None):
        """
        可视化分割结果

        Args:
            image_path: 原始图像路径
            pred_mask: 预测掩码 (H, W, 32)
            save_path: 保存路径

        Returns:
            可视化图像
        """
        # 读取原始图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 创建彩色掩码
        colored_mask = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 为每个牙齿类别分配不同的颜色
        for i in range(32):
            mask = pred_mask[:, :, i] > 0.5
            if np.any(mask):
                # 生成随机颜色
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask[mask] = color

        # 叠加到原始图像
        overlay = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)

        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"可视化结果已保存到: {save_path}")

        return overlay

    def segment_batch(self, image_paths, output_dir=None):
        """
        批量处理多张图像

        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录

        Returns:
            预测掩码列表
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"\n处理图像 {i+1}/{len(image_paths)}: {image_path}")

            try:
                pred_mask = self.segment(image_path)

                if pred_mask is not None:
                    results.append(pred_mask)

                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        save_path = os.path.join(output_dir,
                                                f"seg_{os.path.basename(image_path)}")
                        self.visualize_segmentation(image_path, pred_mask, save_path)
                else:
                    results.append(None)

            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
                results.append(None)

        return results
