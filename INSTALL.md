# 安装指南

## 问题说明

如果遇到以下错误：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
```

这是因为系统的 matplotlib 与 NumPy 2.x 不兼容。

## 解决方案

### 方案1：使用虚拟环境（推荐）

```bash
# 1. 进入项目目录
cd ~/Tooth_root_distance_measurement

# 2. 创建虚拟环境
python3 -m venv venv

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行程序
python tooth_cej_root_analyzer.py

# 6. 使用完毕后退出虚拟环境
deactivate
```

**下次使用时**只需：
```bash
cd ~/Tooth_root_distance_measurement
source venv/bin/activate
python tooth_cej_root_analyzer.py
```

### 方案2：降级 NumPy（快速但可能影响其他程序）

```bash
pip install "numpy<2.0" --user
```

然后直接运行：
```bash
python3 tooth_cej_root_analyzer.py
```

### 方案3：使用系统包管理器安装

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-numpy python3-opencv python3-matplotlib python3-scipy
```

然后运行：
```bash
python3 tooth_cej_root_analyzer.py
```

## 验证安装

运行以下命令检查依赖是否正确安装：

```bash
python3 -c "import cv2, numpy, matplotlib, scipy; print('所有依赖已正确安装')"
```

## 一键安装脚本

也可以使用提供的安装脚本：

```bash
chmod +x setup.sh
./setup.sh
```

## 依赖说明

本项目**不需要**安装 `Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net` 文件夹中的依赖。

只需要以下核心库：
- **numpy** (<2.0): 数值计算
- **opencv-python-headless**: 图像处理
- **matplotlib**: 可视化
- **scipy**: 科学计算

## 常见问题

**Q: 为什么限制 NumPy < 2.0？**

A: NumPy 2.0 引入了破坏性改变，许多旧版本的库（如系统自带的 matplotlib）需要重新编译才能支持。使用 NumPy 1.x 可以避免兼容性问题。

**Q: 虚拟环境有什么好处？**

A: 虚拟环境将项目依赖与系统环境隔离，避免版本冲突，且不需要 root 权限。

**Q: 能否使用 conda？**

A: 可以！使用 conda 创建环境：
```bash
conda create -n tooth_cej python=3.9 numpy=1.24 opencv matplotlib scipy
conda activate tooth_cej
python tooth_cej_root_analyzer.py
```
