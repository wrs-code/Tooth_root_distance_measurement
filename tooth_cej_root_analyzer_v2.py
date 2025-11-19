#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿CEJ线检测与根部距离测量系统 V2
改进的边缘检测算法，不依赖深度学习模型
使用分水岭算法和多重阈值方法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from scipy.spatial import distance as dist
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import os
import glob

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ToothCEJAnalyzer:
    """牙齿CEJ线分析器 - 改进版"""

    def __init__(self):
        # 间距阈值 (mm)
        self.DANGER_THRESHOLD = 3.2
        self.WARNING_THRESHOLD = 4.0

        # 像素到毫米的转换比例
        self.pixels_per_mm = 10

    def order_points(self, pts):
        """对矩形的4个顶点排序"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def preprocess_and_segment_teeth(self, image):
        """
        预处理并分割牙齿区域
        使用改进的阈值方法和分水岭算法

        参数:
            image: 输入的全景X光图像

        返回:
            teeth_mask: 牙齿区域的二值掩码
            processed: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. 高斯模糊去噪
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 3. 使用Otsu阈值 - 牙齿通常是高亮区域
        # 先计算阈值
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 使用更高的阈值来只选择牙齿（通常比Otsu阈值高20-40）
        high_thresh = otsu_thresh + 30
        _, teeth_binary = cv2.threshold(blurred, high_thresh, 255, cv2.THRESH_BINARY)

        # 4. 形态学操作去除小噪声
        kernel = np.ones((3, 3), np.uint8)

        # 开运算去除小亮点
        teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 闭运算填充牙齿内部的小孔
        teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        return teeth_binary, enhanced

    def separate_teeth_watershed(self, teeth_binary):
        """
        使用分水岭算法分离粘连的牙齿

        参数:
            teeth_binary: 牙齿二值掩码

        返回:
            labels: 分离后的标签图
        """
        # 1. 腐蚀操作，确保分离
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(teeth_binary, kernel, iterations=3)

        # 2. 距离变换
        distance_map = ndimage.distance_transform_edt(eroded)

        # 3. 应用最大滤波器增强局部最大值
        distance_map = ndimage.maximum_filter(distance_map, size=15, mode='constant')

        # 4. 找到局部最大值作为种子点
        local_max = peak_local_max(distance_map, indices=False, min_distance=30, labels=teeth_binary)

        # 5. 标记种子点
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]

        # 6. 分水岭算法
        labels = watershed(-distance_map, markers, mask=teeth_binary)

        return labels

    def detect_teeth_contours(self, image):
        """
        检测牙齿轮廓 - 改进版

        参数:
            image: 原始X光图像

        返回:
            teeth_data: 包含每颗牙齿信息的列表
        """
        print("  步骤1: 预处理和分割牙齿区域...")
        teeth_binary, processed = self.preprocess_and_segment_teeth(image)

        print("  步骤2: 使用分水岭算法分离粘连的牙齿...")
        labels = self.separate_teeth_watershed(teeth_binary)

        print("  步骤3: 提取单颗牙齿轮廓...")
        teeth_data = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == 0:  # 跳过背景
                continue

            # 创建单个牙齿的掩码
            tooth_mask = np.zeros(teeth_binary.shape, dtype=np.uint8)
            tooth_mask[labels == label] = 255

            # 计算面积
            area = np.sum(tooth_mask == 255)

            # 过滤太小的区域（面积阈值根据图像大小调整）
            img_area = image.shape[0] * image.shape[1]
            min_area = img_area * 0.001  # 至少占图像的0.1%
            max_area = img_area * 0.05   # 最多占图像的5%

            if area < min_area or area > max_area:
                continue

            # 查找轮廓
            contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            contour = contours[0]

            # 计算轮廓周长和紧凑度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # 紧凑度 = 4π * 面积 / 周长²
            # 圆形的紧凑度为1，越不规则越小
            compactness = 4 * np.pi * area / (perimeter * perimeter)

            # 过滤太不规则的形状（如细长的伪影）
            if compactness < 0.1:
                continue

            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤不合理的宽高比
            aspect_ratio = float(w) / h if h > 0 else 0
            # 牙齿的宽高比通常在 0.3 到 1.5 之间
            if aspect_ratio < 0.3 or aspect_ratio > 1.5:
                continue

            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            box = self.order_points(box)

            # 计算质心
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
            else:
                centroid = (x + w//2, y + h//2)

            teeth_data.append({
                'label': label,
                'contour': contour,
                'mask': tooth_mask,
                'bbox': (x, y, w, h),
                'rect': rect,
                'box': box,
                'centroid': centroid,
                'area': area,
                'compactness': compactness
            })

        # 按X坐标排序（从左到右）
        teeth_data.sort(key=lambda t: t['centroid'][0])

        # 重新分配标签为连续的编号
        for i, tooth in enumerate(teeth_data):
            tooth['label'] = i + 1

        return teeth_data

    def detect_cej_line(self, tooth_data, original_image):
        """
        检测单颗牙齿的CEJ线

        使用轮廓宽度变化梯度检测CEJ线位置
        """
        contour = tooth_data['contour']
        x, y, w, h = tooth_data['bbox']

        # 分析牙齿轮廓在垂直方向上的宽度变化
        width_profile = []
        y_positions = []

        # 从上到下扫描牙齿
        for scan_y in range(y, y + h, 2):
            intersections = []

            for point in contour:
                px, py = point[0]
                if abs(py - scan_y) <= 2:
                    intersections.append(px)

            if len(intersections) >= 2:
                width = max(intersections) - min(intersections)
                width_profile.append(width)
                y_positions.append(scan_y)

        if len(width_profile) < 3:
            # 使用默认位置
            cej_y = y + int(h * 0.4)
            cej_point = (int(x + w / 2), cej_y)
            cej_normal = np.array([0, 1])
            return cej_point, cej_y, cej_normal

        # 计算宽度变化率
        width_profile = np.array(width_profile)

        # 高斯平滑
        from scipy.ndimage import gaussian_filter1d
        smoothed_width = gaussian_filter1d(width_profile, sigma=2)

        # 计算一阶导数
        gradient = np.gradient(smoothed_width)

        # 在牙齿中上部寻找最大负梯度
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

        # 获取CEJ线处的左右边界点
        cej_intersections = []
        for point in contour:
            px, py = point[0]
            if abs(py - cej_y) <= 3:
                cej_intersections.append((px, py))

        if len(cej_intersections) >= 2:
            cej_intersections.sort(key=lambda p: p[0])
            left_point = cej_intersections[0]
            right_point = cej_intersections[-1]
            cej_point = ((left_point[0] + right_point[0]) // 2, cej_y)

            # 计算CEJ线的切线和法线
            tangent = np.array([right_point[0] - left_point[0], right_point[1] - left_point[1]], dtype=float)
            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)

            # 法线方向（垂直于切线，指向牙根）
            cej_normal = np.array([tangent[1], -tangent[0]])

            # 确保法线指向下方
            if cej_normal[1] < 0:
                cej_normal = -cej_normal
        else:
            cej_point = (int(x + w / 2), cej_y)
            cej_normal = np.array([0, 1])

        return cej_point, cej_y, cej_normal

    def measure_root_depth_along_normal(self, tooth_data, cej_point, cej_normal, max_depth_mm=15):
        """沿CEJ线法线方向测量牙根深度"""
        contour = tooth_data['contour']
        max_depth_pixels = int(max_depth_mm * self.pixels_per_mm)

        depth_profile = {
            'depths': [],
            'widths': [],
            'positions': []
        }

        for depth_pixel in range(0, max_depth_pixels, int(self.pixels_per_mm * 0.5)):
            depth_mm = depth_pixel / self.pixels_per_mm

            sample_point = np.array(cej_point) + cej_normal * depth_pixel
            sample_x, sample_y = int(sample_point[0]), int(sample_point[1])

            # 垂直于法线方向
            tangent = np.array([-cej_normal[1], cej_normal[0]])

            intersections = []

            for offset in range(-100, 101, 2):
                check_point = sample_point + tangent * offset
                check_x, check_y = int(check_point[0]), int(check_point[1])

                if cv2.pointPolygonTest(contour, (float(check_x), float(check_y)), False) >= 0:
                    intersections.append(offset)

            if len(intersections) >= 2:
                width_pixels = max(intersections) - min(intersections)
                width_mm = width_pixels / self.pixels_per_mm

                depth_profile['depths'].append(depth_mm)
                depth_profile['widths'].append(width_mm)
                depth_profile['positions'].append((sample_x, sample_y))
            else:
                break

        return depth_profile

    def measure_spacing_between_teeth(self, tooth1_data, tooth2_data, cej1_point, cej2_point,
                                     cej1_normal, cej2_normal, max_depth_mm=15):
        """测量两颗相邻牙齿在不同深度的间距"""
        max_depth_pixels = int(max_depth_mm * self.pixels_per_mm)

        spacing_profile = {
            'depths': [],
            'spacings': [],
            'colors': [],
            'tooth1_edges': [],
            'tooth2_edges': []
        }

        for depth_pixel in range(0, max_depth_pixels, int(self.pixels_per_mm * 0.5)):
            depth_mm = depth_pixel / self.pixels_per_mm

            point1 = np.array(cej1_point) + cej1_normal * depth_pixel
            point2 = np.array(cej2_point) + cej2_normal * depth_pixel

            # 获取两颗牙齿在该深度的边界
            contour1 = tooth1_data['contour']
            contour2 = tooth2_data['contour']

            rightmost1 = None
            for point in contour1:
                px, py = point[0]
                if abs(py - point1[1]) <= 5:
                    if rightmost1 is None or px > rightmost1[0]:
                        rightmost1 = (px, py)

            leftmost2 = None
            for point in contour2:
                px, py = point[0]
                if abs(py - point2[1]) <= 5:
                    if leftmost2 is None or px < leftmost2[0]:
                        leftmost2 = (px, py)

            if rightmost1 is not None and leftmost2 is not None:
                spacing_pixels = leftmost2[0] - rightmost1[0]
                spacing_mm = spacing_pixels / self.pixels_per_mm

                color, _ = self.get_color_for_spacing(spacing_mm)

                spacing_profile['depths'].append(depth_mm)
                spacing_profile['spacings'].append(spacing_mm)
                spacing_profile['colors'].append(color)
                spacing_profile['tooth1_edges'].append(rightmost1)
                spacing_profile['tooth2_edges'].append(leftmost2)
            else:
                break

        return spacing_profile

    def get_color_for_spacing(self, spacing):
        """根据间距值返回对应的颜色"""
        if spacing < self.DANGER_THRESHOLD:
            return '#FF4444', '危险'
        elif spacing < self.WARNING_THRESHOLD:
            return '#FFDD44', '相对安全'
        else:
            return '#44FF44', '安全'

    def analyze_single_image(self, image_path, output_dir='output'):
        """分析单张全景X光图像"""
        print(f"\n{'='*60}")
        print(f"分析图像: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        print(f"✓ 图像尺寸: {original_image.shape[1]}x{original_image.shape[0]}")

        # 检测牙齿轮廓
        print("正在检测牙齿轮廓...")
        teeth_data = self.detect_teeth_contours(original_image)
        print(f"✓ 检测到 {len(teeth_data)} 颗牙齿")

        if len(teeth_data) == 0:
            print("❌ 未检测到牙齿")
            return None

        # 为每颗牙齿检测CEJ线
        print("正在检测CEJ线...")
        for i, tooth in enumerate(teeth_data):
            cej_point, cej_y, cej_normal = self.detect_cej_line(tooth, original_image)
            tooth['cej_point'] = cej_point
            tooth['cej_y'] = cej_y
            tooth['cej_normal'] = cej_normal
            print(f"  牙齿 {i+1}: CEJ线位于 Y={cej_y}")

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
        self.visualize_results(original_image, teeth_data, spacing_results, image_path, output_dir)

        return {
            'image_path': image_path,
            'teeth_data': teeth_data,
            'spacing_results': spacing_results
        }

    def visualize_results(self, original_image, teeth_data, spacing_results, image_path, output_dir):
        """可视化分析结果"""
        os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(20, 12))

        # 1. 原始图像 + 牙齿轮廓 + CEJ线
        ax1 = plt.subplot(2, 2, 1)
        display_img1 = original_image.copy()

        for i, tooth in enumerate(teeth_data):
            contour = tooth['contour']
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.drawContours(display_img1, [contour], -1, color, 2)

            cej_point = tooth['cej_point']
            cej_y = tooth['cej_y']
            x, y, w, h = tooth['bbox']

            cv2.line(display_img1, (x, cej_y), (x + w, cej_y), (0, 255, 255), 2)
            cv2.circle(display_img1, cej_point, 5, (0, 0, 255), -1)

        ax1.imshow(cv2.cvtColor(display_img1, cv2.COLOR_BGR2RGB))
        ax1.set_title('Teeth Detection & CEJ Lines', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. CEJ线法线方向
        ax2 = plt.subplot(2, 2, 2)
        display_img2 = original_image.copy()

        for tooth in enumerate(teeth_data):
            cej_point = np.array(tooth[1]['cej_point'])
            cej_normal = tooth[1]['cej_normal']

            normal_end = cej_point + cej_normal * 80
            cv2.arrowedLine(display_img2, tuple(cej_point.astype(int)), tuple(normal_end.astype(int)),
                          (0, 255, 255), 2, tipLength=0.3)

        ax2.imshow(cv2.cvtColor(display_img2, cv2.COLOR_BGR2RGB))
        ax2.set_title('CEJ Normal Directions', fontsize=14, fontweight='bold')
        ax2.axis('off')

        # 3. 间距热力图
        ax3 = plt.subplot(2, 2, 3)
        spacing_vis = original_image.copy()

        for result in spacing_results:
            profile = result['profile']

            for i in range(len(profile['depths']) - 1):
                edge1 = profile['tooth1_edges'][i]
                edge2 = profile['tooth2_edges'][i]
                edge1_next = profile['tooth1_edges'][i + 1]
                edge2_next = profile['tooth2_edges'][i + 1]

                pts = np.array([edge1, edge2, edge2_next, edge1_next], dtype=np.int32)

                color_hex = profile['colors'][i]
                color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                overlay = spacing_vis.copy()
                cv2.fillPoly(overlay, [pts], color_bgr)
                cv2.addWeighted(overlay, 0.5, spacing_vis, 0.5, 0, spacing_vis)

        ax3.imshow(cv2.cvtColor(spacing_vis, cv2.COLOR_BGR2RGB))
        ax3.set_title('Tooth Spacing Color Map', fontsize=14, fontweight='bold')
        ax3.axis('off')

        danger_patch = mpatches.Patch(color='#FF4444', label=f'Danger (< {self.DANGER_THRESHOLD}mm)', alpha=0.7)
        warning_patch = mpatches.Patch(color='#FFDD44', label=f'Warning ({self.DANGER_THRESHOLD}-{self.WARNING_THRESHOLD}mm)', alpha=0.7)
        safe_patch = mpatches.Patch(color='#44FF44', label=f'Safe (>= {self.WARNING_THRESHOLD}mm)', alpha=0.7)
        ax3.legend(handles=[danger_patch, warning_patch, safe_patch], loc='upper right', fontsize=10)

        # 4. 间距-深度曲线
        ax4 = plt.subplot(2, 2, 4)

        for result in spacing_results:
            i, j = result['tooth_pair']
            profile = result['profile']

            if len(profile['depths']) > 0:
                ax4.plot(profile['spacings'], profile['depths'],
                        marker='o', markersize=4, linewidth=2,
                        label=f'Teeth {i+1}-{j+1}', alpha=0.7)

        ax4.axvline(x=self.DANGER_THRESHOLD, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'Danger ({self.DANGER_THRESHOLD}mm)')
        ax4.axvline(x=self.WARNING_THRESHOLD, color='orange', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'Warning ({self.WARNING_THRESHOLD}mm)')

        ax4.set_xlabel('Spacing (mm)', fontsize=12)
        ax4.set_ylabel('Depth from CEJ (mm)', fontsize=12)
        ax4.set_title('Spacing vs Depth Curves', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=9)
        ax4.invert_yaxis()

        # 保存图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_analysis_v2.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化结果已保存: {output_path}")

    def process_input_folder(self, input_folder='input', output_folder='output'):
        """处理input文件夹中的所有图像"""
        print(f"\n{'='*60}")
        print(f"Batch Processing Mode")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*60}")

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

        if len(image_files) == 0:
            print(f"❌ No images found in {input_folder}")
            return []

        print(f"\nFound {len(image_files)} images")

        all_results = []

        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}]")
            result = self.analyze_single_image(image_path, output_folder)
            if result is not None:
                all_results.append(result)

        return all_results


def main():
    """主函数"""
    print("=" * 60)
    print("Tooth CEJ Detection & Root Distance Measurement System V2")
    print("Improved Edge Detection without Deep Learning")
    print("=" * 60)
    print()
    print("Features:")
    print("1. Advanced threshold-based tooth segmentation")
    print("2. Watershed algorithm for separating touching teeth")
    print("3. CEJ line detection using contour width gradient")
    print("4. Normal direction depth measurement")
    print("5. Risk assessment with color coding")
    print()
    print("-" * 60)

    analyzer = ToothCEJAnalyzer()
    analyzer.process_input_folder(input_folder='input', output_folder='output')

    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
