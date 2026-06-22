# 小样本分类系统配置

# 数据配置
DATA_CONFIG = {
    'data_root': './datasets/CUB_200_2011',
    'image_size': (224, 224),
    'num_classes': 200,
    'train_split_ratio': 0.8
}

# 小样本学习配置
FEW_SHOT_CONFIG = {
    'n_way': 5,      # N-way分类
    'k_shot': 5,     # K-shot样本
    'q_query': 15,   # 查询样本数
    'num_episodes': 100,  # 每轮episode数
    'num_epochs': 100     # 训练轮数
}

# 模型配置
MODEL_CONFIG = {
    'backbone': 'resnet18',  # 骨干网络
    'pretrained': True,      # 是否使用预训练权重
    'embedding_dim': 256,    # 嵌入维度
    'dropout_rate': 0.5      # Dropout比率
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    'scheduler': 'step',
    'step_size': 30,
    'gamma': 0.1
}

# MAML特定配置
MAML_CONFIG = {
    'inner_lr': 0.01,        # 内部学习率
    'meta_lr': 0.001,        # 元学习率
    'inner_steps': 5,        # 内部更新步数
    'first_order': False     # 是否使用一阶近似
}

# ProtoNet特定配置
PROTONET_CONFIG = {
    'distance_metric': 'euclidean'  # 距离度量方式
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'target_layers': ['layer4'],    # 目标可视化层
    'cam_method': 'gradcam',        # CAM方法
    'overlay_alpha': 0.4,           # 叠加透明度
    'colormap': 'jet'               # 热力图颜色映射
}

# 实验配置
EXPERIMENT_CONFIG = {
    'experiment_name': 'cub_fewshot_experiment',
    'seed': 42,#42
    'save_dir': './results',
    'checkpoint_interval': 10,
    'validation_interval': 5
}
