#!/bin/bash
# 牙齿 CEJ 分析系统一键安装脚本

set -e  # 遇到错误立即退出

echo "============================================================"
echo "牙齿 CEJ 分析系统 - 环境配置"
echo "============================================================"
echo ""

# 检查 Python 版本
echo "[1/5] 检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 python3，请先安装 Python 3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python 版本: $(python3 --version)"

# 检查是否已有虚拟环境
if [ -d "venv" ]; then
    echo ""
    echo "⚠️  检测到已存在虚拟环境 venv/"
    read -p "是否删除并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        rm -rf venv
    else
        echo "使用现有环境"
    fi
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/5] 创建虚拟环境..."
    python3 -m venv venv
    echo "✓ 虚拟环境已创建"
else
    echo ""
    echo "[2/5] 使用现有虚拟环境"
fi

# 激活虚拟环境
echo ""
echo "[3/5] 激活虚拟环境..."
source venv/bin/activate
echo "✓ 虚拟环境已激活"

# 升级 pip
echo ""
echo "[4/5] 升级 pip..."
pip install --upgrade pip -q
echo "✓ pip 已升级"

# 安装依赖
echo ""
echo "[5/5] 安装依赖..."
echo "  - numpy (<2.0)"
echo "  - opencv-python-headless"
echo "  - matplotlib"
echo "  - scipy"
echo ""

pip install -r requirements.txt

echo ""
echo "============================================================"
echo "✓ 安装完成！"
echo "============================================================"
echo ""
echo "使用方法："
echo ""
echo "  1. 激活虚拟环境："
echo "     source venv/bin/activate"
echo ""
echo "  2. 运行程序："
echo "     python tooth_cej_root_analyzer.py"
echo ""
echo "  3. 退出虚拟环境："
echo "     deactivate"
echo ""
echo "快速测试："
echo "  python tooth_cej_root_analyzer.py"
echo ""
echo "============================================================"

# 询问是否立即测试
echo ""
read -p "是否立即运行测试？(Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "运行测试..."
    echo ""
    python tooth_cej_root_analyzer.py
fi
