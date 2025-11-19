"""
牙根安全距离测量Demo v2 - 改进的可视化
Dental Root Safety Distance Measurement Demo v2 - Improved Visualization
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SafetyZoneConfig:
    """安全区域配置"""
    depth_width_map = {
        0.0: 1.0,
        3.0: 1.5,
        6.0: 2.0,
        9.0: 2.5,
        12.0: 3.0,
    }

    DANGER_THRESHOLD = 3.2
    RELATIVE_SAFE_THRESHOLD = 4.0
    ANATOMICAL_RISK_THRESHOLD = 2.0


@dataclass
class ToothRoot:
    """牙根数据结构"""
    tooth_id: int
    cej_point: Tuple[float, float]
    apex_point: Tuple[float, float]
    root_contour: np.ndarray

    def get_depth(self) -> float:
        return np.sqrt(
            (self.apex_point[0] - self.cej_point[0])**2 +
            (self.apex_point[1] - self.cej_point[1])**2
        )

    def get_root_direction(self) -> np.ndarray:
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
        depths = sorted(self.config.depth_width_map.keys())
        widths = [self.config.depth_width_map[d] for d in depths]

        if depth_mm <= depths[0]:
            return widths[0]
        if depth_mm >= depths[-1]:
            return widths[-1]

        return np.interp(depth_mm, depths, widths)

    def calculate_trapezoid_safety_zone(self, root: ToothRoot) -> np.ndarray:
        root_dir = root.get_root_direction()
        perpendicular = np.array([-root_dir[1], root_dir[0]])

        num_samples = 30
        total_depth = root.get_depth()

        left_boundary = []
        right_boundary = []

        for i in range(num_samples + 1):
            depth_ratio = i / num_samples
            current_depth = total_depth * depth_ratio
            current_point = np.array(root.cej_point) + root_dir * current_depth
            required_width = self.get_required_width_at_depth(current_depth)
            half_width = required_width

            left_point = current_point - perpendicular * half_width
            right_point = current_point + perpendicular * half_width

            left_boundary.append(left_point)
            right_boundary.append(right_point)

        polygon = np.array(left_boundary + right_boundary[::-1])
        return polygon

    def measure_root_proximity(self, root1: ToothRoot, root2: ToothRoot) -> Dict:
        min_distance = float('inf')
        closest_points = (None, None)

        for pt1 in root1.root_contour:
            for pt2 in root2.root_contour:
                dist = np.linalg.norm(pt1 - pt2)
                if dist < min_distance:
                    min_distance = dist
                    closest_points = (pt1, pt2)

        if min_distance < self.config.DANGER_THRESHOLD:
            risk_level = "危险"
            risk_color = 'red'
        elif min_distance < self.config.RELATIVE_SAFE_THRESHOLD:
            risk_level = "相对安全"
            risk_color = 'yellow'
        else:
            risk_level = "安全"
            risk_color = 'green'

        return {
            'distance': min_distance,
            'closest_points': closest_points,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'root1_id': root1.tooth_id,
            'root2_id': root2.tooth_id
        }


def create_synthetic_tooth_root(tooth_id: int, cej_point: Tuple[float, float],
                               apex_point: Tuple[float, float],
                               width: float = 4.0) -> ToothRoot:
    """创建合成牙根"""
    root_dir = np.array([apex_point[0] - cej_point[0],
                        apex_point[1] - cej_point[1]])
    root_length = np.linalg.norm(root_dir)
    root_dir = root_dir / root_length

    perpendicular = np.array([-root_dir[1], root_dir[0]])

    num_points = 50
    contour = []

    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        along_root = root_length * (1 + np.cos(t)) / 2
        cross_width = width * np.sin(t) * (0.3 + 0.7 * np.sin(along_root / root_length * np.pi))

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


def visualize_with_matplotlib(roots: List[ToothRoot],
                             measurements: List[Dict],
                             safety_zones: List[np.ndarray],
                             config: SafetyZoneConfig):
    """使用matplotlib创建专业可视化"""

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # === 左图：梯形安全区域可视化 ===
    ax1.set_title('梯形安全区域可视化\nTrapezoid Safety Zone Visualization',
                  fontsize=14, fontweight='bold')

    # 绘制安全区域（半透明）
    for i, zone in enumerate(safety_zones):
        poly = Polygon(zone, alpha=0.3, facecolor='lightblue',
                      edgecolor='blue', linewidth=1.5, label='安全区域' if i == 0 else '')
        ax1.add_patch(poly)

    # 绘制牙根轮廓
    for root in roots:
        ax1.plot(root.root_contour[:, 0], root.root_contour[:, 1],
                'k-', linewidth=2, label='牙根轮廓' if root.tooth_id == 1 else '')

        # CEJ点
        ax1.plot(root.cej_point[0], root.cej_point[1],
                'go', markersize=10, label='CEJ点' if root.tooth_id == 1 else '')

        # 根尖点
        ax1.plot(root.apex_point[0], root.apex_point[1],
                'ro', markersize=10, label='根尖点' if root.tooth_id == 1 else '')

        # 标注牙齿编号
        ax1.text(root.cej_point[0], root.cej_point[1] - 5,
                f'#{root.tooth_id}', fontsize=12, ha='center', fontweight='bold')

    # 绘制距离测量线
    for measurement in measurements:
        pt1, pt2 = measurement['closest_points']
        color = measurement['risk_color']
        distance = measurement['distance']

        ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                color=color, linewidth=3, linestyle='--')

        mid_point = (pt1 + pt2) / 2
        ax1.text(mid_point[0], mid_point[1] - 3,
                f"{distance:.2f}mm\n{measurement['risk_level']}",
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    ax1.set_xlabel('水平位置 (像素)', fontsize=11)
    ax1.set_ylabel('垂直位置 (像素)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()

    # === 右图：深度-宽度关系图 ===
    ax2.set_title('安全区域标准：深度-宽度映射\nSafety Zone Standards: Depth-Width Mapping',
                  fontsize=14, fontweight='bold')

    # 绘制深度-宽度曲线
    depths = np.linspace(0, 15, 100)
    calculator = SafetyZoneCalculator(config)
    widths = [calculator.get_required_width_at_depth(d) for d in depths]

    ax2.plot(widths, depths, 'b-', linewidth=3, label='所需最小宽度')

    # 标准点
    for depth, width in config.depth_width_map.items():
        ax2.plot(width, depth, 'ro', markersize=10)
        label = ""
        if depth == 0:
            label = "CEJ处"
        elif depth == 6:
            label = "推荐种植体位置"
        elif depth == 12:
            label = "接近根尖"

        ax2.text(width + 0.15, depth, f'{depth:.0f}mm: {width:.1f}mm\n{label}',
                fontsize=10, va='center')

    # 填充区域
    ax2.fill_betweenx(depths, 0, widths, alpha=0.3, color='lightblue',
                      label='安全区域')
    ax2.fill_betweenx(depths, widths, 5, alpha=0.2, color='lightcoral',
                      label='危险区域')

    ax2.set_xlabel('最小安全宽度 (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('距离CEJ的垂直深度 (mm)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # 添加说明文本
    info_text = f"""
风险等级标准:
• 危险: < {config.DANGER_THRESHOLD} mm
• 相对安全: {config.DANGER_THRESHOLD}-{config.RELATIVE_SAFE_THRESHOLD} mm
• 安全: ≥ {config.RELATIVE_SAFE_THRESHOLD} mm

关键深度说明:
• 0mm: CEJ处，牙颈部
• 6mm: 推荐正畸种植体位置
• 12mm: 接近根尖，需最大空间
    """

    ax2.text(0.98, 0.98, info_text.strip(),
            transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    output_path = '/home/user/Tooth_root_distance_measurement/demo_output_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")

    return fig


def create_concept_diagram():
    """创建梯形概念示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    ax.set_title('梯形安全区域概念图\nTrapezoid Safety Zone Concept',
                fontsize=16, fontweight='bold')

    # 绘制单个牙根及其安全区域
    cej = np.array([5, 2])
    apex = np.array([5, 14])

    # 牙根轮廓（简化为椭圆）
    root_width = 1.5
    root_points = []
    for i in range(50):
        t = 2 * np.pi * i / 50
        y = 2 + 12 * (1 + np.cos(t)) / 2
        x = 5 + root_width * np.sin(t) * (0.5 + 0.5 * np.sin((y-2)/12 * np.pi))
        root_points.append([x, y])
    root_points = np.array(root_points)

    ax.fill(root_points[:, 0], root_points[:, 1], color='lightgray',
           edgecolor='black', linewidth=2, label='牙根')

    # 绘制梯形安全区域
    config = SafetyZoneConfig()
    calculator = SafetyZoneCalculator(config)

    depths_to_show = [0, 3, 6, 9, 12]
    colors_gradient = ['red', 'orange', 'yellow', 'lightgreen', 'green']

    for i, depth in enumerate(depths_to_show):
        if depth > 12:
            continue

        # 位置
        y = cej[1] + depth
        width = calculator.get_required_width_at_depth(depth)

        # 绘制宽度线
        ax.plot([5-width, 5+width], [y, y],
               color=colors_gradient[i], linewidth=3, alpha=0.7)

        # 标注
        ax.text(5 + width + 0.5, y,
               f'{depth}mm深度\n需要{width:.1f}mm宽',
               fontsize=10, va='center',
               bbox=dict(boxstyle='round', facecolor=colors_gradient[i], alpha=0.5))

    # 梯形边界
    left_trapezoid = []
    right_trapezoid = []
    for depth in np.linspace(0, 12, 50):
        y = cej[1] + depth
        width = calculator.get_required_width_at_depth(depth)
        left_trapezoid.append([5 - width, y])
        right_trapezoid.append([5 + width, y])

    trapezoid = np.array(left_trapezoid + right_trapezoid[::-1])
    poly = Polygon(trapezoid, alpha=0.2, facecolor='blue',
                  edgecolor='blue', linewidth=2, linestyle='--',
                  label='梯形安全区域')
    ax.add_patch(poly)

    # CEJ和根尖标记
    ax.plot(cej[0], cej[1], 'go', markersize=15, label='CEJ点', zorder=10)
    ax.plot(apex[0], apex[1], 'ro', markersize=15, label='根尖点', zorder=10)

    # 添加深度箭头
    ax.annotate('', xy=(7.5, cej[1]), xytext=(7.5, apex[1]),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(8, (cej[1] + apex[1])/2, '牙根深度', fontsize=11,
           rotation=-90, va='center')

    # 说明文字
    explanation = """
核心概念：

1. 安全距离不是固定值，而是随深度变化的区域

2. 梯形特性：
   - 顶部（CEJ处）最窄：1.0mm
   - 底部（根尖处）最宽：3.0mm
   - 中间平滑过渡

3. 生物学依据：
   - CEJ处牙槽骨薄，空间受限
   - 根部牙槽骨厚，可承受更大移动
   - 根尖需足够骨支持

4. 测量方法：
   在多个深度点检查是否满足宽度要求
    """

    ax.text(0.02, 0.98, explanation.strip(),
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel('水平距离 (mm)', fontsize=12)
    ax.set_ylabel('垂直深度 (mm)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 12)
    ax.set_ylim(0, 16)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # 保存
    output_path = '/home/user/Tooth_root_distance_measurement/trapezoid_concept.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"概念图已保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("牙根安全距离测量Demo v2")
    print("Dental Root Safety Distance Measurement Demo v2")
    print("=" * 60)
    print()

    # 创建测试数据（4颗牙齿，展示不同风险等级）
    print("1. 创建测试数据...")
    roots = [
        create_synthetic_tooth_root(1, (50, 20), (50, 120), width=3.5),
        create_synthetic_tooth_root(2, (56, 20), (58, 120), width=3.5),  # 危险距离
        create_synthetic_tooth_root(3, (72, 20), (75, 120), width=3.5),  # 相对安全
        create_synthetic_tooth_root(4, (95, 20), (98, 120), width=3.5),  # 安全
    ]

    # 计算
    print("2. 计算安全区域...")
    calculator = SafetyZoneCalculator()
    safety_zones = [calculator.calculate_trapezoid_safety_zone(root) for root in roots]

    print("3. 测量相邻牙根距离...")
    measurements = []
    for i in range(len(roots) - 1):
        measurement = calculator.measure_root_proximity(roots[i], roots[i+1])
        measurements.append(measurement)
        print(f"   牙齿 #{measurement['root1_id']} ↔ #{measurement['root2_id']}: "
              f"{measurement['distance']:.2f}mm ({measurement['risk_level']})")

    # 可视化
    print("\n4. 生成可视化...")
    config = SafetyZoneConfig()
    visualize_with_matplotlib(roots, measurements, safety_zones, config)

    print("\n5. 生成概念示意图...")
    create_concept_diagram()

    # 输出标准
    print("\n" + "=" * 60)
    print("安全区域标准（基于CEJ的垂直深度）")
    print("=" * 60)
    print(f"{'深度(mm)':>10} | {'最小宽度(mm)':>12} | {'说明':>20}")
    print("-" * 60)
    for depth, width in sorted(config.depth_width_map.items()):
        desc = ""
        if depth == 0:
            desc = "CEJ处"
        elif depth == 6:
            desc = "推荐种植体位置"
        elif depth == 12:
            desc = "接近根尖"
        print(f"{depth:>10.1f} | {width:>12.1f} | {desc:>20}")

    print("\n" + "=" * 60)
    print("风险等级标准")
    print("=" * 60)
    print(f"🔴 危险（红色）: < {config.DANGER_THRESHOLD} mm")
    print(f"🟡 相对安全（黄色）: {config.DANGER_THRESHOLD} - {config.RELATIVE_SAFE_THRESHOLD} mm")
    print(f"🟢 安全（绿色）: >= {config.RELATIVE_SAFE_THRESHOLD} mm")

    print("\n" + "=" * 60)
    print("演示完成！请查看生成的图像文件：")
    print("  - demo_output_v2.png: 完整测量结果")
    print("  - trapezoid_concept.png: 梯形概念示意图")
    print("=" * 60)


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    main()
