"""
牙根安全距离测量Demo
Dental Root Safety Distance Measurement Demo

基于CEJ线和牙根尖的梯形安全区域测量
Trapezoid safety zone measurement based on CEJ line and root apex
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SafetyZoneConfig:
    """安全区域配置 - Safety Zone Configuration"""
    # 基于深度的宽度要求 (depth_mm: width_mm)
    # Depth-based width requirements
    depth_width_map = {
        0.0: 1.0,   # CEJ处 - At CEJ
        3.0: 1.5,   # CEJ下3mm - 3mm below CEJ
        6.0: 2.0,   # CEJ下6mm - 6mm below CEJ (推荐种植体位置)
        9.0: 2.5,   # CEJ下9mm - 9mm below CEJ
        12.0: 3.0,  # CEJ下12mm - 12mm below CEJ (接近根尖)
    }

    # 风险等级阈值 (根据PDF文档)
    # Risk level thresholds (from PDF document)
    DANGER_THRESHOLD = 3.2      # <3.2mm = 危险 (红色)
    RELATIVE_SAFE_THRESHOLD = 4.0  # 3.2-4.0mm = 相对安全 (黄色)
    # >=4.0mm = 安全 (绿色)

    # 解剖结构安全距离阈值
    # Anatomical structure safety distance threshold
    ANATOMICAL_RISK_THRESHOLD = 2.0  # <2mm警告


@dataclass
class ToothRoot:
    """牙根数据结构"""
    tooth_id: int
    cej_point: Tuple[float, float]  # CEJ点坐标 (x, y)
    apex_point: Tuple[float, float]  # 根尖点坐标 (x, y)
    root_contour: np.ndarray  # 牙根轮廓点集

    def get_depth(self) -> float:
        """获取牙根深度（CEJ到根尖的距离）"""
        return np.sqrt(
            (self.apex_point[0] - self.cej_point[0])**2 +
            (self.apex_point[1] - self.cej_point[1])**2
        )

    def get_root_direction(self) -> np.ndarray:
        """获取牙根方向向量（单位向量）"""
        direction = np.array([
            self.apex_point[0] - self.cej_point[0],
            self.apex_point[1] - self.cej_point[1]
        ])
        return direction / np.linalg.norm(direction)


class SafetyZoneCalculator:
    """安全区域计算器"""

    def __init__(self, config: SafetyZoneConfig = None):
        self.config = config or SafetyZoneConfig()

    def get_required_width_at_depth(self, depth_mm: float) -> float:
        """
        根据深度获取所需的安全宽度
        Get required safety width at given depth

        使用线性插值计算任意深度处的安全宽度要求
        Use linear interpolation to calculate safety width at any depth
        """
        depths = sorted(self.config.depth_width_map.keys())
        widths = [self.config.depth_width_map[d] for d in depths]

        # 如果深度超出范围，使用边界值
        if depth_mm <= depths[0]:
            return widths[0]
        if depth_mm >= depths[-1]:
            return widths[-1]

        # 线性插值
        return np.interp(depth_mm, depths, widths)

    def calculate_trapezoid_safety_zone(self, root: ToothRoot) -> np.ndarray:
        """
        计算单个牙根的梯形安全区域
        Calculate trapezoid safety zone for a single tooth root

        返回：梯形的四个顶点坐标
        Returns: Four vertices of the trapezoid
        """
        # 获取根方向和垂直方向
        root_dir = root.get_root_direction()
        perpendicular = np.array([-root_dir[1], root_dir[0]])  # 垂直于根方向

        # 在不同深度采样点计算安全宽度
        num_samples = 20
        total_depth = root.get_depth()

        left_boundary = []
        right_boundary = []

        for i in range(num_samples + 1):
            # 当前深度比例
            depth_ratio = i / num_samples
            current_depth = total_depth * depth_ratio

            # 当前位置
            current_point = np.array(root.cej_point) + root_dir * current_depth

            # 所需宽度（单侧）
            required_width = self.get_required_width_at_depth(current_depth)
            half_width = required_width / 2

            # 左右边界点
            left_point = current_point - perpendicular * half_width
            right_point = current_point + perpendicular * half_width

            left_boundary.append(left_point)
            right_boundary.append(right_point)

        # 组合成闭合多边形：左边界 + 右边界反向
        polygon = np.array(left_boundary + right_boundary[::-1])
        return polygon

    def measure_root_proximity(self, root1: ToothRoot, root2: ToothRoot) -> Dict:
        """
        测量两个相邻牙根之间的距离
        Measure distance between two adjacent tooth roots

        返回：包含最小距离、位置、风险等级等信息
        Returns: Dictionary with minimum distance, location, risk level, etc.
        """
        # 计算两个根轮廓之间的最小距离
        min_distance = float('inf')
        closest_points = (None, None)

        for pt1 in root1.root_contour:
            for pt2 in root2.root_contour:
                dist = np.linalg.norm(pt1 - pt2)
                if dist < min_distance:
                    min_distance = dist
                    closest_points = (pt1, pt2)

        # 确定风险等级
        if min_distance < self.config.DANGER_THRESHOLD:
            risk_level = "危险"
            risk_color = (255, 0, 0)  # 红色
        elif min_distance < self.config.RELATIVE_SAFE_THRESHOLD:
            risk_level = "相对安全"
            risk_color = (255, 255, 0)  # 黄色
        else:
            risk_level = "安全"
            risk_color = (0, 255, 0)  # 绿色

        return {
            'distance': min_distance,
            'closest_points': closest_points,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'root1_id': root1.tooth_id,
            'root2_id': root2.tooth_id
        }

    def check_safety_zone_compliance(self, root: ToothRoot,
                                     adjacent_roots: List[ToothRoot]) -> Dict:
        """
        检查牙根是否满足梯形安全区域要求
        Check if tooth root meets trapezoid safety zone requirements

        在不同深度检查实际宽度是否满足要求
        Check if actual width meets requirements at different depths
        """
        root_dir = root.get_root_direction()
        perpendicular = np.array([-root_dir[1], root_dir[0]])
        total_depth = root.get_depth()

        violations = []

        # 在多个深度采样点检查
        for depth_mm in [3, 5, 7, 9]:
            if depth_mm > total_depth:
                break

            # 当前深度的位置
            current_point = np.array(root.cej_point) + root_dir * depth_mm

            # 所需宽度
            required_width = self.get_required_width_at_depth(depth_mm)

            # 计算到相邻根的实际距离
            for adj_root in adjacent_roots:
                # 找到相邻根轮廓上距离当前点最近的点
                distances = [np.linalg.norm(pt - current_point)
                           for pt in adj_root.root_contour]
                actual_distance = min(distances)

                # 检查是否违反安全距离
                if actual_distance < required_width:
                    violations.append({
                        'depth': depth_mm,
                        'required_width': required_width,
                        'actual_distance': actual_distance,
                        'deficit': required_width - actual_distance,
                        'adjacent_tooth': adj_root.tooth_id
                    })

        return {
            'tooth_id': root.tooth_id,
            'compliant': len(violations) == 0,
            'violations': violations
        }


class DentalSafetyVisualizer:
    """牙科安全区域可视化"""

    def __init__(self, image_size=(800, 1200)):
        self.image_size = image_size
        self.config = SafetyZoneConfig()

    def create_visualization(self, roots: List[ToothRoot],
                           measurements: List[Dict],
                           safety_zones: List[np.ndarray]) -> np.ndarray:
        """
        创建完整的可视化图像
        Create complete visualization image
        """
        # 创建白色背景
        img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 255

        # 1. 绘制安全区域（半透明）
        overlay = img.copy()
        for zone in safety_zones:
            zone_int = zone.astype(np.int32)
            cv2.fillPoly(overlay, [zone_int], (200, 200, 255))  # 淡蓝色
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # 2. 绘制牙根轮廓
        for root in roots:
            contour_int = root.root_contour.astype(np.int32)
            cv2.polylines(img, [contour_int], True, (100, 100, 100), 2)

            # 标记CEJ点和根尖点
            cv2.circle(img, tuple(map(int, root.cej_point)), 5, (0, 255, 0), -1)
            cv2.circle(img, tuple(map(int, root.apex_point)), 5, (0, 0, 255), -1)

            # 标注牙齿编号
            cv2.putText(img, f"#{root.tooth_id}",
                       tuple(map(int, root.cej_point - np.array([0, 20]))),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 3. 绘制距离测量线和标注
        for measurement in measurements:
            pt1, pt2 = measurement['closest_points']
            color = measurement['risk_color']
            distance = measurement['distance']

            # 绘制连接线
            cv2.line(img, tuple(map(int, pt1)), tuple(map(int, pt2)), color, 2)

            # 标注距离和风险等级
            mid_point = ((pt1 + pt2) / 2).astype(int)
            text = f"{distance:.2f}mm - {measurement['risk_level']}"
            cv2.putText(img, text, tuple(mid_point),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 4. 添加图例
        self._add_legend(img)

        return img

    def _add_legend(self, img: np.ndarray):
        """添加图例"""
        legend_x, legend_y = 20, 20
        line_height = 30

        legends = [
            ("CEJ点", (0, 255, 0)),
            ("根尖点", (0, 0, 255)),
            (f"危险 (<{self.config.DANGER_THRESHOLD}mm)", (255, 0, 0)),
            (f"相对安全 ({self.config.DANGER_THRESHOLD}-{self.config.RELATIVE_SAFE_THRESHOLD}mm)",
             (255, 255, 0)),
            (f"安全 (>={self.config.RELATIVE_SAFE_THRESHOLD}mm)", (0, 255, 0)),
        ]

        # 绘制白色背景
        cv2.rectangle(img, (legend_x-5, legend_y-5),
                     (legend_x+300, legend_y+len(legends)*line_height+5),
                     (255, 255, 255), -1)
        cv2.rectangle(img, (legend_x-5, legend_y-5),
                     (legend_x+300, legend_y+len(legends)*line_height+5),
                     (0, 0, 0), 2)

        for i, (text, color) in enumerate(legends):
            y = legend_y + i * line_height
            cv2.circle(img, (legend_x+10, y+10), 5, color, -1)
            cv2.putText(img, text, (legend_x+25, y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def create_synthetic_tooth_root(tooth_id: int, cej_point: Tuple[float, float],
                               apex_point: Tuple[float, float],
                               width: float = 8.0) -> ToothRoot:
    """
    创建合成的牙根数据（用于演示）
    Create synthetic tooth root data (for demonstration)
    """
    # 计算根方向
    root_dir = np.array([apex_point[0] - cej_point[0],
                        apex_point[1] - cej_point[1]])
    root_length = np.linalg.norm(root_dir)
    root_dir = root_dir / root_length

    # 垂直方向
    perpendicular = np.array([-root_dir[1], root_dir[0]])

    # 生成椭圆形牙根轮廓
    num_points = 50
    contour = []

    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        # 沿根方向的位置
        along_root = root_length * (1 + np.cos(t)) / 2
        # 垂直于根的宽度（中间较宽）
        cross_width = width * np.sin(t) * (0.5 + 0.5 * np.sin(along_root / root_length * np.pi))

        point = (np.array(cej_point) +
                root_dir * along_root +
                perpendicular * cross_width)
        contour.append(point)

    return ToothRoot(
        tooth_id=tooth_id,
        cej_point=cej_point,
        apex_point=apex_point,
        root_contour=np.array(contour)
    )


def main():
    """主演示函数"""
    print("=== 牙根安全距离测量Demo ===")
    print("=== Dental Root Safety Distance Measurement Demo ===\n")

    # 1. 创建合成牙根数据
    print("1. 创建合成牙根数据...")
    roots = [
        # 三颗相邻的牙齿
        create_synthetic_tooth_root(1, (200, 100), (200, 250), width=7),
        create_synthetic_tooth_root(2, (215, 100), (220, 250), width=7),  # 危险距离
        create_synthetic_tooth_root(3, (240, 100), (245, 250), width=7),  # 相对安全
        create_synthetic_tooth_root(4, (275, 100), (280, 250), width=7),  # 安全
    ]

    # 2. 计算安全区域
    print("2. 计算梯形安全区域...")
    calculator = SafetyZoneCalculator()
    safety_zones = [calculator.calculate_trapezoid_safety_zone(root) for root in roots]

    # 3. 测量相邻牙根距离
    print("3. 测量相邻牙根距离...")
    measurements = []
    for i in range(len(roots) - 1):
        measurement = calculator.measure_root_proximity(roots[i], roots[i+1])
        measurements.append(measurement)
        print(f"   牙齿 #{measurement['root1_id']} - #{measurement['root2_id']}: "
              f"{measurement['distance']:.2f}mm ({measurement['risk_level']})")

    # 4. 检查安全区域合规性
    print("\n4. 检查安全区域合规性...")
    for i, root in enumerate(roots):
        adjacent = [r for j, r in enumerate(roots) if abs(i-j) == 1]
        compliance = calculator.check_safety_zone_compliance(root, adjacent)

        status = "✓ 符合" if compliance['compliant'] else "✗ 违规"
        print(f"   牙齿 #{root.tooth_id}: {status}")

        if not compliance['compliant']:
            for v in compliance['violations']:
                print(f"      - 深度{v['depth']}mm: 需要{v['required_width']:.2f}mm, "
                      f"实际{v['actual_distance']:.2f}mm (不足{v['deficit']:.2f}mm)")

    # 5. 创建可视化
    print("\n5. 创建可视化...")
    visualizer = DentalSafetyVisualizer(image_size=(400, 600))
    result_img = visualizer.create_visualization(roots, measurements, safety_zones)

    # 6. 保存结果
    output_path = "/home/user/Tooth_root_distance_measurement/demo_output.png"
    cv2.imwrite(output_path, result_img)
    print(f"   结果已保存到: {output_path}")

    # 7. 显示深度-宽度映射表
    print("\n=== 安全区域标准（基于CEJ的垂直深度）===")
    print("深度(mm) | 最小宽度(mm) | 说明")
    print("-" * 50)
    config = SafetyZoneConfig()
    for depth, width in sorted(config.depth_width_map.items()):
        print(f"{depth:6.1f}   | {width:11.1f}   | ", end="")
        if depth == 0:
            print("CEJ处")
        elif depth == 6:
            print("推荐种植体位置")
        elif depth >= 12:
            print("接近根尖")
        else:
            print("")

    print("\n=== 风险等级标准 ===")
    print(f"危险（红色）: < {config.DANGER_THRESHOLD} mm")
    print(f"相对安全（黄色）: {config.DANGER_THRESHOLD} - {config.RELATIVE_SAFE_THRESHOLD} mm")
    print(f"安全（绿色）: >= {config.RELATIVE_SAFE_THRESHOLD} mm")

    print("\n演示完成！")


if __name__ == "__main__":
    main()
