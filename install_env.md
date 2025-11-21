● 好的！以下是完整的安装指令：

  完整安装指令

  1. 安装NVIDIA驱动 580

  sudo apt update
  sudo apt install -y nvidia-driver-580

  安装完成后必须重启：
  sudo reboot

  重启后验证驱动安装：
  nvidia-smi

  2. 安装CUDA 11.8 和 cuDNN

  TensorFlow 2.12 从 pip 安装时会自动包含所需的CUDA库，不需要单独安装CUDA！

  3. 安装TensorFlow 2.12.0（GPU版本）

  cd ~/Tooth_root_distance_measurement
  source venv/bin/activate

  # 卸载当前版本
  pip uninstall -y tensorflow keras

  # 安装TensorFlow 2.12.0（自动包含GPU支持）
  pip install tensorflow==2.12.0

  4. 验证GPU是否可用

  python3 -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__); print('GPU可用:', 
  tf.config.list_physical_devices('GPU'))"

  5. 测试模型加载

  python3 tooth_cej_root_analyzer.py

  ---
  简化版（一次性执行）

  如果你想一次性执行（除了驱动安装和重启）：

  # 第一步：安装驱动并重启
  sudo apt update && sudo apt install -y nvidia-driver-580
  sudo reboot

  # 重启后执行：
  cd ~/Tooth_root_distance_measurement
  source venv/bin/activate
  pip uninstall -y tensorflow keras
  pip install tensorflow==2.12.0
  python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', 
  tf.config.list_physical_devices('GPU'))"

  ---
  注意事项

  1. 必须重启：安装NVIDIA驱动后必须重启系统才能生效
  2. TensorFlow 2.12 自带CUDA库：不需要单独安装CUDA Toolkit
  3. 验证GPU：运行程序前先用python命令验证GPU是否被识别
