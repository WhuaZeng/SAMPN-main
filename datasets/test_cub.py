import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from cub_dataset import create_dataloaders, get_transforms, CUBDataset, FewShotSampler

# 配置参数
DATA_ROOT = "e:/few_shot_project/datasets/CUB_200_2011"  # 请确保路径正确
N_WAY = 5
K_SHOT = 3
Q_QUERY = 5  # 为了显示方便，每个类只取5个query
NUM_EPISODES = 1

def load_class_names(data_root):
    """加载类别名称映射: index -> name"""
    classes_file = os.path.join(data_root, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found at {classes_file}")
    
    class_names = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                idx = int(parts[0]) - 1  # CUB labels are 1-indexed, convert to 0-indexed
                name = parts[1]
                class_names[idx] = name
    return class_names

def unnormalize_image(img_tensor):
    """
    将标准化后的 Tensor 图像还原为可显示的 numpy 数组 [H, W, C], range [0, 1]
    使用 ImageNet 的 mean 和 std
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # img_tensor: [C, H, W]
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np

def visualize_episode(support_images, support_labels, query_images, query_labels, 
                      global_class_ids, class_names_map, save_path=None):
    """
    可视化一个 episode 的支持集和查询集，优化标签布局
    
    Args:
        global_class_ids: list, 当前episode中局部标签 i 对应的全局类别ID
        class_names_map: dict, 全局类别ID -> 类别名称
    """
    n_way = len(torch.unique(support_labels))
    k_shot = support_images.shape[0] // n_way
    q_query = query_images.shape[0] // n_way
    
    total_cols = k_shot + q_query
    
    # 创建画布，增加左侧边距以容纳类名标签
    fig, axes = plt.subplots(n_way, total_cols, figsize=(3 * total_cols, 4 * n_way))
    
    # 调整子图布局，留出左侧空间给行标签，留出顶部空间给列标题
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.02, wspace=0.05, hspace=0)

    for i in range(n_way):
        # 获取当前局部标签 i 对应的全局类别ID和名称
        global_id = global_class_ids[i]
        class_name = class_names_map.get(global_id, f"Unknown({global_id})")
        
        # --- 1. 在每一行的最左侧显示类别名称 ---
        # 使用 fig.text 在图形坐标系中放置文本 (x, y, text)
        # y 坐标计算：从下往上，每一行的中心位置
        # axes 是 [row, col]，row 0 在最上面。
        # matplotlib figure y=1 是顶部，y=0 是底部。
        # 我们需要计算第 i 行中心的 y 坐标。
        row_height = 1.0 / n_way
        # 由于 subplots_adjust top=0.9, bottom=0.05, 实际绘图区域高度约为 0.85
        # 简单估算：y_pos = top_margin - (i + 0.5) * row_height_scaled
        # 更稳健的方法是使用 axes[i, 0].get_position() 但这里用相对坐标近似即可
        y_pos = 0.90 - (i + 0.5) * (0.85 / n_way) 
        
        fig.text(0.02, y_pos, f"{class_name}\n(ID: {global_id})", 
                 ha='left', va='center', fontsize=18, fontweight='bold', color='darkblue')

        # 找到当前类 i 的 support 和 query 索引
        s_indices = (support_labels == i).nonzero(as_tuple=True)[0]
        q_indices = (query_labels == i).nonzero(as_tuple=True)[0]
        
        # --- 2. 绘制 Support Set ---
        for j, idx in enumerate(s_indices):
            ax = axes[i, j] if n_way > 1 else axes[j]
            img = unnormalize_image(support_images[idx])
            ax.imshow(img)
            ax.axis('off')
            
            # 可选：在图片角落加一个小标记 S
            ax.text(0.05, 0.05, 'S', transform=ax.transAxes, 
                    fontsize=18, color='black', bbox=dict(facecolor='black', alpha=0.5))

        # --- 3. 绘制 Query Set ---
        for j, idx in enumerate(q_indices):
            col_idx = j + k_shot
            ax = axes[i, col_idx] if n_way > 1 else axes[col_idx]
            img = unnormalize_image(query_images[idx])
            ax.imshow(img)
            ax.axis('off')
            
            # 可选：在图片角落加一个小标记 Q
            ax.text(0.05, 0.05, 'Q', transform=ax.transAxes, 
                    fontsize=18, color='white', bbox=dict(facecolor='black', alpha=0.5))

    # --- 4. 添加顶部的 Support 和 Query 全局标题 ---
    # 计算 Support 区域的中心 x 坐标
    # 整个 axes 宽度从 left=0.15 到 right=0.95，总宽 0.8
    # Support 占前 k_shot 列，Query 占后 q_query 列
    
    # 获取第一个和最后一个 axes 的位置来精确定位
    if n_way > 1:
        pos_support_start = axes[0, 0].get_position()
        pos_support_end = axes[0, k_shot-1].get_position()
        pos_query_start = axes[0, k_shot].get_position()
        pos_query_end = axes[0, total_cols-1].get_position()
    else:
        pos_support_start = axes[0].get_position()
        pos_support_end = axes[k_shot-1].get_position()
        pos_query_start = axes[k_shot].get_position()
        pos_query_end = axes[total_cols-1].get_position()

    x_support_center = (pos_support_start.x0 + pos_support_end.x1) / 2
    x_query_center = (pos_query_start.x0 + pos_query_end.x1) / 2
    
    # y 坐标设在顶部 margin 中间
    y_title = 0.96 

    fig.text(x_support_center, y_title, 'Support Set', ha='center', va='bottom', fontsize=14, fontweight='bold')
    fig.text(x_query_center, y_title, 'Query Set', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 主标题
    plt.suptitle(f"Episode Visualization: {N_WAY}-Way {K_SHOT}-Shot", fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    print(f"Loading data from {DATA_ROOT}...")
    
    # 1. 加载类名映射
    class_names_map = load_class_names(DATA_ROOT)
    
    # 2. 创建数据集和采样器
    val_dataset = CUBDataset(
        data_root=DATA_ROOT,
        split='test',
        transform=get_transforms('test')
    )
    
    test_sampler = FewShotSampler(
        dataset=val_dataset,
        n_way=N_WAY,
        k_shot=K_SHOT,
        q_query=Q_QUERY,
        num_episodes=NUM_EPISODES
    )
    
    print("Sampling one episode...")
    # 3. 获取一个 episode
    for episode_data in test_sampler:
        # 【关键修改】解包时接收额外的 episode_classes (全局标签列表)
        if len(episode_data) == 5:
            support_images, support_labels, query_images, query_labels, global_class_ids = episode_data
        else:
            # 兼容旧版本或未修改的 sampler
            support_images, support_labels, query_images, query_labels = episode_data
            global_class_ids = list(range(N_WAY)) # fallback
            print("Warning: Sampler did not return global class IDs. Using dummy IDs.")

        print(f"Support Images Shape: {support_images.shape}") 
        print(f"Support Labels: {support_labels}")
        print(f"Query Images Shape: {query_images.shape}")   
        print(f"Query Labels: {query_labels}")
        print(f"Global Class IDs in this episode: {global_class_ids}")
        
        # 4. 可视化，传入真实的全局ID
        visualize_episode(
            support_images, 
            support_labels, 
            query_images, 
            query_labels,
            global_class_ids=global_class_ids,
            class_names_map=class_names_map,
            save_path="test_episode_visualization.jpg"
        )
        break # 只显示一个 episode

if __name__ == '__main__':
    main()