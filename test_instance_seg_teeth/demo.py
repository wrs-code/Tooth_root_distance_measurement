"""
牙齿分割演示脚本
展示如何使用TeethSegmentation类进行牙齿实例分割
"""
import os
import sys

# 添加项目路径到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_instance_seg_teeth import TeethSegmentation


def main():
    """主函数"""
    print("=" * 60)
    print("牙齿实例分割演示")
    print("=" * 60)

    # 模型路径配置
    # 注意: 你需要先下载模型权重文件
    yolo_model_path = "models/yolov8_teeth.pt"  # YOLOv8模型路径
    unet_model_path = "models/unet_teeth.h5"    # U-Net模型权重路径

    # 检查模型文件是否存在
    if not os.path.exists(yolo_model_path):
        print(f"\n错误: YOLO模型文件不存在: {yolo_model_path}")
        print("请先下载YOLOv8模型权重并放置到正确位置")
        print("参考: test_instance_seg_teeth/MODEL_DOWNLOAD.md")
        return

    if not os.path.exists(unet_model_path):
        print(f"\n错误: U-Net模型文件不存在: {unet_model_path}")
        print("请先下载U-Net模型权重并放置到正确位置")
        print("参考: test_instance_seg_teeth/MODEL_DOWNLOAD.md")
        return

    # 初始化分割模型
    print("\n初始化模型...")
    segmentor = TeethSegmentation(
        yolo_model_path=yolo_model_path,
        unet_model_path=unet_model_path,
        img_size=512
    )

    # 测试图像路径
    test_image_path = "input/test_image.jpg"

    # 检查测试图像是否存在
    if not os.path.exists(test_image_path):
        print(f"\n错误: 测试图像不存在: {test_image_path}")
        print("请将测试图像放置到 input/ 目录下")
        return

    print(f"\n处理图像: {test_image_path}")

    # 进行分割
    try:
        pred_mask = segmentor.segment(test_image_path)

        if pred_mask is not None:
            print("\n分割成功!")
            print(f"预测掩码形状: {pred_mask.shape}")

            # 保存可视化结果
            output_dir = "output/teeth_segmentation"
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "segmentation_result.jpg")
            segmentor.visualize_segmentation(test_image_path, pred_mask, output_path)

            print(f"\n结果已保存到: {output_path}")
        else:
            print("\n分割失败: 未检测到牙齿")

    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


def batch_demo():
    """批量处理演示"""
    print("=" * 60)
    print("批量牙齿分割演示")
    print("=" * 60)

    # 模型路径配置
    yolo_model_path = "models/yolov8_teeth.pt"
    unet_model_path = "models/unet_teeth.h5"

    # 初始化分割模型
    print("\n初始化模型...")
    segmentor = TeethSegmentation(
        yolo_model_path=yolo_model_path,
        unet_model_path=unet_model_path,
        img_size=512
    )

    # 批量处理图像
    input_dir = "input"
    output_dir = "output/teeth_segmentation_batch"

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_paths:
        print(f"\n在 {input_dir} 目录下未找到图像文件")
        return

    print(f"\n找到 {len(image_paths)} 张图像，开始批量处理...")

    # 批量分割
    results = segmentor.segment_batch(image_paths, output_dir=output_dir)

    # 统计结果
    success_count = sum(1 for r in results if r is not None)
    print(f"\n处理完成!")
    print(f"成功: {success_count}/{len(image_paths)}")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    # 单张图像演示
    main()

    # 如果需要批量处理，取消下面的注释
    # batch_demo()
