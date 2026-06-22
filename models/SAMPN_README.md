# SAMPN 模型使用说明

## 模型简介

**SAMPN** (Similar-attention module + AttnRes + Mahalanobis + ProtoNet) 是一个用于小样本学习的高级原型网络。

### 核心特性

1. **多层级特征融合**：整合 ResNet 的 Layer1-Layer4 特征，捕获从低级到高级的语义信息
2. **相似注意力增强**：
   - SCR (Self-Correlation Reconstruction)：自相关重建模块
   - CBAM (Convolutional Block Attention Module)：卷积块注意力模块
3. **AttnRes 注意力融合**：使用伪查询向量和 RMSNorm 进行动态多尺度特征加权
4. **马氏距离度量**：通过可学习的下三角矩阵参数化，保证精度矩阵的对称正定性，实现自适应度量学习

### 数学原理

马氏距离公式：
$$D_M^2(x, \mu) = (x - \mu)^T \Sigma^{-1} (x - \mu) = ||L(x - \mu)||_2^2$$

其中 $\Sigma^{-1} = L^T L$，$L$ 是通过 Cholesky 分解得到的下三角矩阵。

## 使用方法

### 基本训练

```bash
# 5-way 5-shot 训练
python main.py --model_type sampn --n_way 5 --k_shot 5 --epochs 100

# 指定骨干网络
python main.py --model_type sampn --backbone resnet18 --n_way 5 --k_shot 5

# 使用 Flowers102 数据集
python main.py --model_type sampn --dataset_type flowers --n_way 5 --k_shot 5

# 使用 FGVC Aircraft 数据集
python main.py --model_type sampn --dataset_type fgvc_aircraft --n_way 5 --k_shot 5
```

### 可视化分析

```bash
# 启用小样本情景可视化
python main.py --model_type sampn --fewshot_visualize --n_way 5 --k_shot 5

# 标准注意力可视化
python main.py --model_type sampn --visualize --n_way 5 --k_shot 5
```

### 高级配置

```bash
# 自定义学习率和批量大小
python main.py --model_type sampn --lr 0.001 --batch_size 16 --n_way 5 --k_shot 5

# 使用 ResNet18 骨干网络
python main.py --model_type sampn --backbone resnet18 --n_way 5 --k_shot 5
```

## 模型架构

```
Input Image (224x224x3)
    ↓
ResNet Backbone (frozen)
    ├─ Layer1 → Projection → ┐
    ├─ Layer2 → Projection → ├→ AttnRes Fusion
    ├─ Layer3 → Projection → ├→ (6 sources)
    ├─ Layer4 (Identity)    →┘
    ├─ Layer4 + CBAM        →┘
    └─ Layer4 + SCR         →┘
    ↓
Global Average Pooling
    ↓
Projection MLP (512→256)
    ↓
Embeddings (256-dim)
    ↓
Mahalanobis Metric (Learnable L matrix)
    ↓
Classification Logits
```

## 注意事项

1. **显存占用**：由于融合了多层特征和复杂的注意力机制，显存占用较高（建议 ≥ 8GB）
2. **学习率**：建议使用较小的学习率（0.001 或更低），因为骨干网络已冻结
3. **训练时间**：相比基础 ProtoNet，训练时间会增加约 30-50%
4. **收敛性**：马氏距离度量通常需要更多 epoch 才能充分学习协方差结构

## 性能对比

在 CUB-200-2011 数据集上的预期性能（5-way）：
- 1-shot: ~75-80%
- 5-shot: ~85-90%

在 Flowers102 数据集上的预期性能（5-way）：
- 1-shot: ~70-75%
- 5-shot: ~82-87%

## 代码位置

- 模型定义：`models/SAMPN.py`
- 工厂函数：`models/factory.py`
- 主入口：`main.py`


