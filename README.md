# 小样本学习分类系统 (Few-Shot Learning Classification System)

基于元学习的小样本图像分类系统，支持多种数据集和模型架构，提供可视化分析功能。


## 项目简介

本项目实现了一个灵活的小样本学习（Few-Shot Learning）分类系统，基于元学习（Meta-Learning）框架。系统支持多种主流小样本学习算法和数据集，提供了训练、评估和可视化分析功能。

**核心特点：**
- 支持多种小样本学习模型（ProtoNet、MAML、SAMPN等）
- 支持多个细粒度图像分类数据集
- 强大的可视化分析工具（Grad-CAM注意力可视化）
- 灵活的配置系统
- 实时训练曲线监控
- 早停机制防止过拟合

## 主要特性

- **多模型支持**：ProtoNet、MAML、SAMPN及其变体
- **多骨干网络**：Conv4、ResNet12、ResNet18
- **多数据集**：CUB-200、Flowers102、FGVC-Aircraft
- **可配置的N-way K-shot**：灵活设置小样本学习任务
- **注意力可视化**：基于Grad-CAM的模型解释性分析
- **小样本情景可视化**：针对小样本学习场景的专用可视化工具
- **自动保存**：训练历史、模型权重和可视化结果自动保存

## 支持的模型

| 模型 | 描述 |
|------|------|
| [protonet]| 原型网络（Prototypical Networks），使用欧氏距离度量 |
| [maml]| 模型无关元学习（Model-Agnostic Meta-Learning） |
| [sampn]| **SAMPN**: 相似性-注意力增强多级原型网络。<br>核心组件：<br>1. **AttnRes**: 多尺度注意力残差融合，动态聚合分层特征<br>2. **SAE**: 相似性-注意力增强模块（整合SCR局部结构一致性与CBAM前景重校准)<br>3. **Learnable Mahalanobis**: 基于Cholesky分解的可学习马氏距离度量，适应特征协方差结构 |



## 支持的数据集

### 1. CUB-200-2011 (Caltech-UCSD Birds)
- **类别数**：200种鸟类
- **图像数**：约11,788张
- **特点**：经典的细粒度鸟类分类数据集

### 2. Flowers102 (Oxford 102 Flowers)
- **类别数**：102种花卉
- **图像数**：8,189张
- **特点**：细粒度花卉分类数据集

### 3. FGVC-Aircraft (Fine-Grained Visual Classification of Aircraft)
- **类别数**：102种飞机型号（variant级别）
- **图像数**：10,200张
- **标注级别**：variant、family、manufacturer
- **特点**：细粒度飞机分类数据集

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- Pillow

## 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd github
创建虚拟环境并安装依赖
bash
conda create -n fsl python=3.8
conda activate fsl
pip install torch torchvision
pip install matplotlib numpy Pillow
快速开始
(此处可根据实际代码补充快速启动命令，例如)

bash
python train.py --model sampn --dataset cub --way 5 --shot 1


项目结构
text
.
├── models/             # 模型定义
│   ├── protonet.py
│   ├── maml.py
│   ├── sampn.py        # SAMPN模型实现
│   └── ...
├── datasets/           # 数据处理
├── utils/              # 工具函数
├── visualize_*.py      # 可视化脚本
├── train.py            # 训练入口
├── test.py             # 测试入口
└── README.md


