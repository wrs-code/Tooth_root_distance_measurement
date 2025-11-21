"""
模型下载辅助脚本
用于下载YOLOv8基础模型
"""
import os
import urllib.request


def download_file(url, save_path):
    """
    下载文件

    Args:
        url: 下载链接
        save_path: 保存路径
    """
    print(f"下载: {url}")
    print(f"保存到: {save_path}")

    try:
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 下载文件
        urllib.request.urlretrieve(url, save_path)
        print("下载完成!")
        return True

    except Exception as e:
        print(f"下载失败: {str(e)}")
        return False


def download_yolov8_base():
    """下载YOLOv8基础模型"""
    print("=" * 60)
    print("下载YOLOv8基础模型")
    print("=" * 60)
    print("\n注意: 这是YOLOv8的基础模型，并非专门用于牙齿检测")
    print("为了获得最佳效果，建议在牙齿数据集上训练专用模型\n")

    # YOLOv8n (nano) 模型 - 最小的模型，适合测试
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    yolo_save_path = "../models/yolov8_base.pt"

    return download_file(yolo_url, yolo_save_path)


def main():
    """主函数"""
    print("\n牙齿分割模型下载工具\n")

    # 下载YOLOv8基础模型
    print("\n1. 下载YOLOv8基础模型...")
    yolo_success = download_yolov8_base()

    print("\n" + "=" * 60)
    print("下载总结")
    print("=" * 60)

    if yolo_success:
        print("✓ YOLOv8基础模型下载成功")
        print("\n模型位置:")
        print("  - YOLOv8: models/yolov8_base.pt")
    else:
        print("✗ YOLOv8基础模型下载失败")

    print("\n" + "=" * 60)
    print("重要提示")
    print("=" * 60)
    print("""
1. YOLOv8基础模型仅用于测试，实际使用需要在牙齿数据集上训练

2. U-Net模型无法自动下载，需要自行训练:
   - 参考: Instance_seg_teeth/notebooks/yolov8+unet/
   - 数据集: UFBA-425 (https://figshare.com/articles/dataset/UFBA-425/29827475)

3. 训练模型的步骤:
   a. 下载UFBA-425数据集
   b. 使用notebooks/yolov8/yolov8_train.ipynb训练YOLOv8
   c. 使用notebooks/yolov8+unet/yolov8+unet_training.ipynb训练U-Net

4. 详细说明请参考: test_instance_seg_teeth/MODEL_DOWNLOAD.md
    """)


if __name__ == "__main__":
    main()
