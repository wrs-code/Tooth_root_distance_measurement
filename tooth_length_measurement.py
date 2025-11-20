#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿长度测量系统
完全复刻 SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net
基于 CCA_Analysis.py 的实现

功能：
1. 牙齿分割（U-Net）
2. 连通组件分析（CCA）
3. 测量牙齿长轴和短轴
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

# 导入U-Net分割模块
from unet_segmentation import UNetTeethSegmentation

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def midpoint(ptA, ptB):
    """
    计算两点的中点

    参数:
        ptA: 第一个点 (x, y)
        ptB: 第二个点 (x, y)

    返回:
        中点坐标 (x, y)
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def order_points(pts):
    """
    对矩形的4个顶点排序：左上、右上、右下、左下

    参数:
        pts: 4个点的坐标数组

    返回:
        ordered: 排序后的点 [tl, tr, br, bl]
    """
    # 初始化坐标点数组
    rect = np.zeros((4, 2), dtype=np.float32)

    # 计算左上和右下
    # 左上的点具有最小的和，右下的点具有最大的和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # 计算右上和左下
    # 右上的点具有最小的差，左下的点具有最大的差
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


class ToothLengthMeasurement:
    """牙齿长度测量器（复刻SerdarHelli实现）"""

    def __init__(self, pixels_per_mm=10):
        """
        初始化

        参数:
            pixels_per_mm: 像素到毫米的转换比例
        """
        self.pixels_per_mm = pixels_per_mm

        # 初始化U-Net分割器
        try:
            self.unet_segmenter = UNetTeethSegmentation()
            self.use_unet = True
            print("✓ U-Net分割器已初始化")
        except Exception as e:
            print(f"⚠ U-Net分割器初始化失败: {e}")
            print("  将使用传统方法")
            self.unet_segmenter = None
            self.use_unet = False

    def CCA_Analysis(self, orig_image, predict_image, erode_iteration=2, open_iteration=2):
        """
        连通组件分析 (CCA) - 提取牙齿轮廓并计算长短轴位置

        参数:
            orig_image: 原始图像
            predict_image: 预测/分割后的图像
            erode_iteration: 腐蚀迭代次数
            open_iteration: 开运算迭代次数

        返回:
            result_image: 标注后的图像
            teeth_data: 每颗牙齿的数据列表（包含轮廓和轴信息）
        """
        # 复制图像
        image = predict_image.copy()
        image2 = orig_image.copy()

        # 1. 形态学处理
        # 创建5x5卷积核
        kernel1 = np.ones((5, 5), np.uint8)

        # 开运算（先腐蚀后膨胀，用于去除噪声）
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1, iterations=open_iteration)

        # 锐化滤波核
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel_sharpening)

        # 腐蚀操作
        image = cv2.erode(image, kernel1, iterations=erode_iteration)

        # 2. 转换为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Otsu二值化
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 4. 连通组件分析（8-连通）
        output = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]

        # 5. 遍历每个连通组件并保存牙齿数据
        teeth_data = []

        for i in range(1, num_labels):  # 跳过背景（标签0）
            # 获取组件面积
            c_area = stats[i, cv2.CC_STAT_AREA]

            # 面积阈值：大于2000像素才认为是牙齿
            if c_area > 2000:
                # 创建单个组件的mask
                componentMask = (labels == i).astype("uint8") * 255

                # 查找轮廓
                cnts = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                c = max(cnts, key=cv2.contourArea)

                # 6. 计算最小外接矩形
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.float32)

                # 排序顶点
                box = order_points(box)
                (tl, tr, br, bl) = box

                # 7. 计算四条边的中点
                (tltrX, tltrY) = midpoint(tl, tr)  # 上边中点
                (blbrX, blbrY) = midpoint(bl, br)  # 下边中点
                (tlblX, tlblY) = midpoint(tl, bl)  # 左边中点
                (trbrX, trbrY) = midpoint(tr, br)  # 右边中点

                # 8. 计算轴长（欧氏距离）
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # 主轴
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # 副轴

                # 判断长轴和短轴
                if dA >= dB:
                    # dA是长轴，dB是短轴
                    long_axis_start = (tltrX, tltrY)
                    long_axis_end = (blbrX, blbrY)
                    short_axis_start = (tlblX, tlblY)
                    short_axis_end = (trbrX, trbrY)
                    long_axis_length = dA
                    short_axis_length = dB
                else:
                    # dB是长轴，dA是短轴
                    long_axis_start = (tlblX, tlblY)
                    long_axis_end = (trbrX, trbrY)
                    short_axis_start = (tltrX, tltrY)
                    short_axis_end = (blbrX, blbrY)
                    long_axis_length = dB
                    short_axis_length = dA

                # 计算长轴方向向量（归一化）
                long_axis_vector = np.array([
                    long_axis_end[0] - long_axis_start[0],
                    long_axis_end[1] - long_axis_start[1]
                ])
                long_axis_vector_norm = long_axis_vector / (np.linalg.norm(long_axis_vector) + 1e-6)

                # 计算短轴方向向量（归一化）
                short_axis_vector = np.array([
                    short_axis_end[0] - short_axis_start[0],
                    short_axis_end[1] - short_axis_start[1]
                ])
                short_axis_vector_norm = short_axis_vector / (np.linalg.norm(short_axis_vector) + 1e-6)

                # 转换为毫米
                long_axis_mm = long_axis_length / self.pixels_per_mm
                short_axis_mm = short_axis_length / self.pixels_per_mm

                # 计算轮廓中心
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = int((tl[0] + br[0]) / 2), int((tl[1] + br[1]) / 2)

                # 保存牙齿数据
                tooth_info = {
                    'id': len(teeth_data) + 1,
                    'contour': c,
                    'mask': componentMask,
                    'area': c_area,
                    'center': (cX, cY),
                    'box': box,
                    'box_vertices': {
                        'tl': tl, 'tr': tr, 'br': br, 'bl': bl
                    },
                    'long_axis': {
                        'start': long_axis_start,
                        'end': long_axis_end,
                        'vector': long_axis_vector_norm,
                        'length_px': long_axis_length,
                        'length_mm': long_axis_mm
                    },
                    'short_axis': {
                        'start': short_axis_start,
                        'end': short_axis_end,
                        'vector': short_axis_vector_norm,
                        'length_px': short_axis_length,
                        'length_mm': short_axis_mm
                    }
                }
                teeth_data.append(tooth_info)

                # 9. 可视化绘制
                # 绘制完整的牙齿轮廓（绿色）
                cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)

                # 绘制最小外接矩形（浅蓝色虚线，参考用）
                # cv2.drawContours(image2, [box.astype("int")], -1, (255, 200, 0), 1)

                # 绘制长轴（红色粗线）
                cv2.line(image2,
                        (int(long_axis_start[0]), int(long_axis_start[1])),
                        (int(long_axis_end[0]), int(long_axis_end[1])),
                        (0, 0, 255), 3)

                # 绘制长轴端点（红色圆点）
                cv2.circle(image2, (int(long_axis_start[0]), int(long_axis_start[1])), 5, (0, 0, 255), -1)
                cv2.circle(image2, (int(long_axis_end[0]), int(long_axis_end[1])), 5, (0, 0, 255), -1)

                # 绘制短轴（蓝色粗线）
                cv2.line(image2,
                        (int(short_axis_start[0]), int(short_axis_start[1])),
                        (int(short_axis_end[0]), int(short_axis_end[1])),
                        (255, 0, 0), 3)

                # 绘制短轴端点（蓝色圆点）
                cv2.circle(image2, (int(short_axis_start[0]), int(short_axis_start[1])), 5, (255, 0, 0), -1)
                cv2.circle(image2, (int(short_axis_end[0]), int(short_axis_end[1])), 5, (255, 0, 0), -1)

                # 绘制中心点（黄色圆点）
                cv2.circle(image2, (cX, cY), 6, (0, 255, 255), -1)

                # 10. 标注信息
                # 长轴标注（红色文字）
                cv2.putText(image2, "L:{:.1f}mm".format(long_axis_mm),
                           (int(long_axis_start[0] - 15), int(long_axis_start[1] - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 短轴标注（蓝色文字）
                cv2.putText(image2, "S:{:.1f}mm".format(short_axis_mm),
                           (int(short_axis_start[0] + 10), int(short_axis_start[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 牙齿编号（黄色文字）
                cv2.putText(image2, "#{}".format(len(teeth_data)),
                           (cX - 15, cY - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)

        return image2, teeth_data

    def analyze_single_image(self, image_path, output_dir='output'):
        """
        分析单张全景X光图像

        参数:
            image_path: 图像路径
            output_dir: 输出目录

        返回:
            result_image: 标注后的图像
            teeth_count: 检测到的牙齿数量
        """
        print(f"\n{'='*60}")
        print(f"分析图像: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None, 0

        print(f"✓ 图像尺寸: {original_image.shape[1]}x{original_image.shape[0]}")

        # 1. 使用U-Net进行牙齿分割
        if self.use_unet:
            print("正在使用U-Net进行牙齿分割...")
            mask, refined_mask = self.unet_segmenter.segment_teeth(original_image)
            predict_image = refined_mask
        else:
            print("使用传统方法...")
            # 简单的预处理
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            _, predict_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            predict_image = cv2.bitwise_not(predict_image)

        # 2. 执行CCA分析
        print("正在进行连通组件分析（CCA）...")
        result_image, teeth_data = self.CCA_Analysis(
            original_image,
            predict_image,
            erode_iteration=2,
            open_iteration=2
        )

        teeth_count = len(teeth_data)
        print(f"✓ 检测到 {teeth_count} 颗牙齿")

        # 打印每颗牙齿的轴信息
        for tooth in teeth_data:
            print(f"  牙齿 #{tooth['id']}:")
            print(f"    长轴: {tooth['long_axis']['length_mm']:.1f}mm, 位置: {tooth['long_axis']['start']} -> {tooth['long_axis']['end']}")
            print(f"    短轴: {tooth['short_axis']['length_mm']:.1f}mm, 位置: {tooth['short_axis']['start']} -> {tooth['short_axis']['end']}")

        # 3. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_tooth_axis.png')

        cv2.imwrite(output_path, result_image)
        print(f"✓ 可视化结果已保存: {output_path}")

        # 4. 保存轴数据到JSON
        import json
        axes_data = {
            'image': os.path.basename(image_path),
            'teeth_count': teeth_count,
            'teeth': []
        }

        for tooth in teeth_data:
            axes_data['teeth'].append({
                'id': tooth['id'],
                'center': tooth['center'],
                'long_axis': {
                    'start': [float(tooth['long_axis']['start'][0]), float(tooth['long_axis']['start'][1])],
                    'end': [float(tooth['long_axis']['end'][0]), float(tooth['long_axis']['end'][1])],
                    'vector': [float(tooth['long_axis']['vector'][0]), float(tooth['long_axis']['vector'][1])],
                    'length_mm': float(tooth['long_axis']['length_mm'])
                },
                'short_axis': {
                    'start': [float(tooth['short_axis']['start'][0]), float(tooth['short_axis']['start'][1])],
                    'end': [float(tooth['short_axis']['end'][0]), float(tooth['short_axis']['end'][1])],
                    'vector': [float(tooth['short_axis']['vector'][0]), float(tooth['short_axis']['vector'][1])],
                    'length_mm': float(tooth['short_axis']['length_mm'])
                }
            })

        json_path = os.path.join(output_dir, f'{base_name}_tooth_axes.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(axes_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 轴位置数据已保存: {json_path}")

        return result_image, teeth_data

    def process_input_folder(self, input_folder='input', output_folder='output'):
        """
        处理input文件夹中的所有图像

        参数:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径

        返回:
            results: 所有图像的分析结果
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

        results = []
        total_teeth = 0

        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]")
            result_image, teeth_data = self.analyze_single_image(image_path, output_folder)

            if result_image is not None:
                teeth_count = len(teeth_data)
                results.append({
                    'image_path': image_path,
                    'teeth_count': teeth_count,
                    'teeth_data': teeth_data
                })
                total_teeth += teeth_count

        # 打印汇总
        print(f"\n{'='*60}")
        print(f"处理完成！")
        print(f"{'='*60}")
        print(f"处理图像数: {len(results)}")
        print(f"检测牙齿总数: {total_teeth}")

        return results


def main():
    """主函数"""
    print("=" * 60)
    print("牙齿长度测量系统")
    print("复刻 SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray")
    print("=" * 60)
    print()
    print("功能说明：")
    print("1. U-Net牙齿分割")
    print("2. 连通组件分析（CCA）")
    print("3. 测量每颗牙齿的长轴和短轴")
    print("4. 可视化标注（复刻exampleofcca.png效果）")
    print()
    print("-" * 60)

    # 创建分析器
    analyzer = ToothLengthMeasurement(pixels_per_mm=10)

    # 处理input文件夹中的所有图像
    analyzer.process_input_folder(input_folder='input', output_folder='output')

    print()
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
