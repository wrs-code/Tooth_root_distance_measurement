#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙齿实例分割测试脚本
使用Instance_seg_teeth仓库的OralBBNet方法进行牙齿分割
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from test_instance_seg_teeth import TeethSegmentation


def test_single_image():
    """测试单张图像分割"""
    print("=" * 80)
    print("测试1：单张图像牙齿实例分割")
    print("=" * 80)

    # 模型路径配置
    yolo_model_path = os.path.join(project_root, "models", "yolov8_teeth.pt")
    unet_model_path = os.path.join(project_root, "models", "unet_teeth.h5")

    # 检查模型是否存在
    if not os.path.exists(yolo_model_path):
        print(f"\n警告: YOLO模型不存在: {yolo_model_path}")
        print("尝试使用基础模型...")
        yolo_model_path = os.path.join(project_root, "models", "yolov8_base.pt")

        if not os.path.exists(yolo_model_path):
            print("\n错误: 未找到任何YOLO模型")
            print("请运行以下命令下载基础模型:")
            print("  python test_instance_seg_teeth/download_models.py")
            print("\n或参考 test_instance_seg_teeth/MODEL_DOWNLOAD.md 训练专用模型")
            return

    if not os.path.exists(unet_model_path):
        print(f"\n错误: U-Net模型不存在: {unet_model_path}")
        print("U-Net模型需要在牙齿数据集上训练")
        print("参考: test_instance_seg_teeth/MODEL_DOWNLOAD.md")
        return

    # 初始化分割器
    print("\n初始化牙齿分割模型...")
    try:
        segmentor = TeethSegmentation(
            yolo_model_path=yolo_model_path,
            unet_model_path=unet_model_path,
            img_size=512
        )
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return

    # 测试图像路径
    test_image = os.path.join(project_root, "input", "107.png")

    if not os.path.exists(test_image):
        # 尝试其他图像
        input_dir = os.path.join(project_root, "input")
        if os.path.exists(input_dir):
            images = [f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(input_dir, images[0])
            else:
                print(f"\n错误: input目录下没有图像文件")
                return
        else:
            print(f"\n错误: input目录不存在")
            return

    print(f"\n处理图像: {test_image}")

    # 执行分割
    try:
        pred_mask = segmentor.segment(test_image, yolo_iou=0.7, yolo_conf=0.5)

        if pred_mask is not None:
            print("\n✓ 分割成功!")
            print(f"  预测掩码形状: {pred_mask.shape}")

            # 保存结果
            output_dir = os.path.join(project_root, "output", "instance_seg_teeth")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "single_result.jpg")
            segmentor.visualize_segmentation(test_image, pred_mask, output_path)

            print(f"\n✓ 结果已保存到: {output_path}")

        else:
            print("\n✗ 分割失败: 未检测到牙齿")

    except Exception as e:
        print(f"\n✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 80)
    print("测试2：批量牙齿实例分割")
    print("=" * 80)

    # 模型路径配置
    yolo_model_path = os.path.join(project_root, "models", "yolov8_teeth.pt")
    unet_model_path = os.path.join(project_root, "models", "unet_teeth.h5")

    # 检查模型
    if not os.path.exists(yolo_model_path):
        yolo_model_path = os.path.join(project_root, "models", "yolov8_base.pt")
        if not os.path.exists(yolo_model_path):
            print("\n错误: 未找到YOLO模型")
            return

    if not os.path.exists(unet_model_path):
        print("\n错误: 未找到U-Net模型")
        return

    # 初始化分割器
    print("\n初始化牙齿分割模型...")
    try:
        segmentor = TeethSegmentation(
            yolo_model_path=yolo_model_path,
            unet_model_path=unet_model_path,
            img_size=512
        )
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return

    # 获取输入图像列表
    input_dir = os.path.join(project_root, "input")
    if not os.path.exists(input_dir):
        print(f"\n错误: 输入目录不存在: {input_dir}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_paths:
        print(f"\n错误: 在 {input_dir} 目录下未找到图像")
        return

    print(f"\n找到 {len(image_paths)} 张图像")

    # 批量处理
    output_dir = os.path.join(project_root, "output", "instance_seg_teeth_batch")

    try:
        results = segmentor.segment_batch(image_paths, output_dir=output_dir)

        # 统计结果
        success_count = sum(1 for r in results if r is not None)

        print("\n" + "=" * 80)
        print("批量处理完成")
        print("=" * 80)
        print(f"总数: {len(image_paths)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(image_paths) - success_count}")
        print(f"结果保存在: {output_dir}")

    except Exception as e:
        print(f"\n✗ 批量处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_visualization():
    """测试可视化功能"""
    print("\n" + "=" * 80)
    print("测试3：牙齿分割可视化")
    print("=" * 80)

    # 这个测试需要先运行test_single_image生成预测结果
    print("\n此测试需要先运行test_single_image()")
    print("可视化结果将保存在 output/instance_seg_teeth/ 目录")


def check_environment():
    """检查环境和依赖"""
    print("=" * 80)
    print("环境检查")
    print("=" * 80)

    # 检查依赖
    dependencies = {
        'tensorflow': 'TensorFlow (用于U-Net)',
        'ultralytics': 'Ultralytics (用于YOLOv8)',
        'cv2': 'OpenCV (图像处理)',
        'numpy': 'NumPy (数值计算)',
        'PIL': 'Pillow (图像读取)',
        'skimage': 'scikit-image (图像增强)'
    }

    print("\n检查Python依赖:")
    missing_deps = []

    for module, description in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            elif module == 'skimage':
                from skimage import exposure
            else:
                __import__(module)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} - 未安装")
            missing_deps.append(module)

    if missing_deps:
        print("\n缺少依赖，请运行:")
        print("  pip install -r test_instance_seg_teeth/requirements.txt")
        return False

    # 检查模型文件
    print("\n检查模型文件:")
    models_to_check = [
        ("models/yolov8_teeth.pt", "YOLOv8牙齿检测模型"),
        ("models/yolov8_base.pt", "YOLOv8基础模型"),
        ("models/unet_teeth.h5", "U-Net牙齿分割模型")
    ]

    for model_path, description in models_to_check:
        full_path = os.path.join(project_root, model_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✓ {description} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {description} - 不存在")

    # 检查输入目录
    print("\n检查输入目录:")
    input_dir = os.path.join(project_root, "input")
    if os.path.exists(input_dir):
        images = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  ✓ input目录存在，包含 {len(images)} 张图像")
    else:
        print(f"  ✗ input目录不存在")

    print("\n" + "=" * 80)
    return len(missing_deps) == 0


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print(" " * 20 + "牙齿实例分割测试套件")
    print(" " * 15 + "基于Instance_seg_teeth (OralBBNet)")
    print("*" * 80)
    print("\n")

    # 检查环境
    if not check_environment():
        print("\n环境检查未通过，请先解决上述问题")
        return

    print("\n选择要运行的测试:")
    print("  1. 单张图像分割")
    print("  2. 批量图像分割")
    print("  3. 可视化测试")
    print("  4. 运行所有测试")

    # 默认运行单张图像测试
    choice = "1"

    if choice == "1":
        test_single_image()
    elif choice == "2":
        test_batch_processing()
    elif choice == "3":
        test_visualization()
    elif choice == "4":
        test_single_image()
        test_batch_processing()
        test_visualization()

    print("\n" + "*" * 80)
    print(" " * 30 + "测试完成")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
