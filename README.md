# Tooth Root Distance Measurement

口腔正畸牙根间距自动测量与风险标注软件

## 项目简介

本项目旨在基于全景X光片自动测量牙根间距，并进行正畸风险标注。通过深度学习技术实现牙齿分割和根部距离的精确测量，为口腔正畸提供辅助诊断工具。

## 技术基础

- **牙齿分割模型**: 基于 [Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net) 开源项目
- **深度学习框架**: U-Net 架构用于全景X光片中的牙齿分割
- **测量算法**: 自动计算相邻牙根间的距离
- **风险评估**: 根据测量结果进行正畸风险标注

## 主要功能

1. **牙齿分割**: 从全景X光片中自动识别和分割单颗牙齿
2. **根部定位**: 精确定位每颗牙齿的根部位置
3. **距离测量**: 自动测量相邻牙根间的距离
4. **风险标注**: 基于测量数据进行正畸风险评估和可视化标注

## 项目结构

```
Tooth_root_distance_measurement/
├── Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net/  # 牙齿分割模型
├── 口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf        # 需求文档
└── README.md                                                   # 项目说明
```

## 开发计划

- [ ] 搭建基础开发环境
- [ ] 集成牙齿分割模型
- [ ] 实现根部检测算法
- [ ] 开发距离测量功能
- [ ] 实现风险标注系统
- [ ] 构建用户界面
- [ ] 测试与优化

## 环境要求

- Python 3.x
- TensorFlow / PyTorch
- OpenCV
- NumPy
- 其他依赖见各模块的 requirements.txt

## 快速开始

```bash
# 克隆项目
git clone git@github.com:wrs-code/Tooth_root_distance_measurement.git
cd Tooth_root_distance_measurement

# 安装依赖
# pip install -r requirements.txt  # 待添加

# 运行示例
# python demo.py  # 待实现
```

## 参考资料

- [技术需求文档](口腔正畸牙根间距自动测量与风险标注软件技术需求文档.pdf)
- [牙齿分割开源项目](https://github.com/SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net)

## 许可证

待定

## 作者

wrs-code

---

*本项目正在开发中，更多功能即将推出*
