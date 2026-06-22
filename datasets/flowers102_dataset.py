import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
import random
from collections import defaultdict


class Flowers102Dataset(Dataset):
    """Flowers102数据集类，支持小样本学习"""
    
    def __init__(self, data_root, split='train', transform=None, few_shot=False, n_way=5, k_shot=5):
        """
        初始化Flowers102数据集
        
        Args:
            data_root: 数据集根目录路径
            split: 数据分割 ('train', 'val', 'test')
            transform: 图像变换
            few_shot: 是否使用小样本模式
            n_way: N-way分类任务中的N
            k_shot: K-shot学习中的K
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.few_shot = few_shot
        self.n_way = n_way
        self.k_shot = k_shot
        
        # 加载数据索引
        self.image_paths, self.labels = self._load_data()
        
        # 如果是小样本设置，重新组织数据
        if few_shot:
            self._prepare_few_shot_data()
    
    def _load_data(self):
        """加载图像路径和标签"""
        # 加载标签文件
        labels_mat = loadmat(os.path.join(self.data_root, "imagelabels.mat"))
        labels = labels_mat["labels"][0]  # (8189,)
        
        # 加载数据集划分
        setid = loadmat(os.path.join(self.data_root, "setid.mat"))
        
        # 获取不同分割的索引（转换为0-based）
        train_ids = setid["trnid"][0] - 1
        val_ids = setid["valid"][0] - 1
        test_ids = setid["tstid"][0] - 1
        
        # 根据分割选择对应的索引
        if self.split == 'train':
            indices = train_ids
        elif self.split == 'val':
            indices = val_ids
        else:  # test
            indices = test_ids
        
        # 构建图像路径和标签列表
        image_paths = []
        label_list = []
        
        jpg_dir = os.path.join(self.data_root, "jpg")
        
        for idx in indices:
            # Flowers102的图片命名格式: image_XXXXX.jpg (从1开始)
            img_name = f"image_{idx+1:05d}.jpg"
            img_path = os.path.join(jpg_dir, img_name)
            
            # 检查文件是否存在
            if os.path.exists(img_path):
                image_paths.append(img_path)
                # 标签从1开始，转换为0-based
                label_list.append(int(labels[idx]) - 1)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        print(f"Loaded {len(image_paths)} images for {self.split} split")
        
        return image_paths, label_list
    
    def _prepare_few_shot_data(self):
        """准备小样本学习数据"""
        # 按类别分组
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        
        # 过滤掉样本数不足的类别
        valid_classes = [cls for cls, indices in class_to_indices.items() 
                        if len(indices) >= self.k_shot]
        
        if len(valid_classes) < self.n_way:
            raise ValueError(f"Not enough valid classes. Need {self.n_way}, have {len(valid_classes)}")
        
        # 随机选择n_way个类别
        selected_classes = random.sample(valid_classes, self.n_way)
        
        # 为每个选中的类别选择k_shot个样本
        selected_indices = []
        new_labels = []
        
        for new_label, cls in enumerate(selected_classes):
            indices = random.sample(class_to_indices[cls], self.k_shot)
            selected_indices.extend(indices)
            new_labels.extend([new_label] * self.k_shot)
        
        # 更新数据
        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.labels = new_labels
        
        print(f"Few-shot mode: {self.n_way}-way {self.k_shot}-shot with {len(self.image_paths)} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            with open(image_path, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image: {image_path}")
            print(f"Error message: {str(e)}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Flowers102FewShotSampler:
    """Flowers102小样本采样器，用于生成episodic训练任务"""
    
    def __init__(self, dataset, n_way=5, k_shot=5, q_query=15, num_episodes=100):
        """
        初始化小样本采样器
        
        Args:
            dataset: Flowers102Dataset实例
            n_way: 每个episode的类别数
            k_shot: 每个类别的支持集样本数
            q_query: 每个类别的查询集样本数
            num_episodes: episode总数
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        
        # 按类别组织数据
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.class_indices[label].append(idx)
        
        # 过滤样本数不足的类别
        self.valid_classes = [cls for cls, indices in self.class_indices.items() 
                             if len(indices) >= k_shot + q_query]
        
        print(f"Found {len(self.valid_classes)} valid classes out of {len(self.class_indices)} total classes")
    
    def __len__(self):
        return self.num_episodes
    
    def __iter__(self):
        """生成episodic任务"""
        for episode_idx in range(self.num_episodes):
            # 检查是否有足够的有效类别
            if len(self.valid_classes) < self.n_way:
                print(f"Not enough valid classes. Need {self.n_way}, have {len(self.valid_classes)}")
                break
                
            # 随机选择n_way个类别
            episode_classes = random.sample(self.valid_classes, self.n_way)
            
            support_set = []
            query_set = []
            support_labels = []
            query_labels = []
            
            valid_episode = True
            
            for i, cls in enumerate(episode_classes):
                # 为每个类别选择样本
                class_indices = self.class_indices[cls]
                
                # 确保有足够的样本
                if len(class_indices) < self.k_shot + self.q_query:
                    print(f"Class {cls} has only {len(class_indices)} samples, need {self.k_shot + self.q_query}")
                    valid_episode = False
                    break
                    
                selected_indices = random.sample(class_indices, self.k_shot + self.q_query)
                
                support_indices = selected_indices[:self.k_shot]
                query_indices = selected_indices[self.k_shot:]
                
                # 添加支持集样本
                for idx in support_indices:
                    try:
                        image, _ = self.dataset[idx]
                        support_set.append(image)
                        support_labels.append(i)
                    except Exception as e:
                        print(f"Error loading support image at index {idx}: {str(e)}")
                        # 添加一个占位符张量
                        support_set.append(torch.zeros(3, 224, 224))
                        support_labels.append(i)
                
                # 添加查询集样本
                for idx in query_indices:
                    try:
                        image, _ = self.dataset[idx]
                        query_set.append(image)
                        query_labels.append(i)
                    except Exception as e:
                        print(f"Error loading query image at index {idx}: {str(e)}")
                        # 添加一个占位符张量
                        query_set.append(torch.zeros(3, 224, 224))
                        query_labels.append(i)
            
            # 确保有数据且episode有效
            if valid_episode and len(support_set) > 0 and len(query_set) > 0:
                # 【修改点】返回 episode_classes，以便外部知道局部标签对应的全局类别ID
                yield (torch.stack(support_set), torch.tensor(support_labels),
                       torch.stack(query_set), torch.tensor(query_labels),
                       episode_classes)
            else:
                print(f"Skipping episode {episode_idx} due to insufficient data")


def get_flowers_transforms(split='train'):
    """获取Flowers102数据变换"""
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_flowers_dataloaders(data_root, batch_size=16, n_way=5, k_shot=5, q_query=15, num_episodes=100):
    """
    创建Flowers102数据加载器
    
    Args:
        data_root: 数据集根目录
        batch_size: 批大小
        n_way: N-way
        k_shot: K-shot
        q_query: 查询集每类样本数
        num_episodes: episode数量
    
    Returns:
        train_loader, val_loader, train_sampler, val_sampler
    """
    # 确保数据根目录路径正确
    data_root = os.path.abspath(os.path.normpath(data_root))
    print(f"Using data root: {data_root}")
    
    # 检查必要的文件是否存在
    required_files = ['imagelabels.mat', 'setid.mat']
    for file in required_files:
        file_path = os.path.join(data_root, file)
        if not os.path.exists(file_path):
            raise ValueError(f"Required file not found: {file_path}")
    
    # 检查jpg目录是否存在
    jpg_dir = os.path.join(data_root, 'jpg')
    if not os.path.exists(jpg_dir):
        raise ValueError(f"JPG directory does not exist: {jpg_dir}")
    
    # 基础数据集
    train_dataset = Flowers102Dataset(
        data_root=data_root,
        split='train',
        transform=get_flowers_transforms('train')
    )
    
    val_dataset = Flowers102Dataset(
        data_root=data_root,
        split='val',
        transform=get_flowers_transforms('test')
    )
    
    test_dataset = Flowers102Dataset(
        data_root=data_root,
        split='test',
        transform=get_flowers_transforms('test')
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 检查训练集和验证集的类别分布
    from collections import Counter
    train_label_counts = Counter(train_dataset.labels)
    val_label_counts = Counter(val_dataset.labels)
    
    min_train_samples = min(train_label_counts.values()) if train_label_counts else 0
    min_val_samples = min(val_label_counts.values()) if val_label_counts else 0
    
    print(f"Min samples per class - Train: {min_train_samples}, Val: {min_val_samples}")
    
    # 自动调整 q_query 以适应数据集
    max_q_query = min(min_train_samples, min_val_samples) - k_shot
    if max_q_query < 1:
        print(f"Warning: Not enough samples for {k_shot}-shot learning with current dataset split.")
        print(f"Consider using test set or adjusting k_shot/q_query parameters.")
        max_q_query = 1
    
    actual_q_query = min(q_query, max_q_query)
    if actual_q_query != q_query:
        print(f"Adjusted q_query from {q_query} to {actual_q_query} to fit dataset constraints")
    
    # 训练集小样本采样器
    train_sampler = Flowers102FewShotSampler(
        dataset=train_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=actual_q_query,
        num_episodes=num_episodes
    )
    
    # 验证集小样本采样器
    val_sampler = Flowers102FewShotSampler(
        dataset=val_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=actual_q_query,
        num_episodes=20  # 较少的验证episode
    )
    
    # 基础数据加载器（用于常规训练）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


if __name__ == '__main__':
    # 测试数据集加载
    data_root = "./datasets/flowers-102"  # 根据实际数据集位置调整路径
    
    print("="*50)
    print("Testing Flowers102 Dataset Loading")
    print("="*50)
    
    # 测试基本数据加载
    print("\n1. Testing basic dataset loading...")
    train_dataset = Flowers102Dataset(
        data_root=data_root,
        split='train',
        transform=get_flowers_transforms('train')
    )
    print(f"Train samples: {len(train_dataset)}")
    
    val_dataset = Flowers102Dataset(
        data_root=data_root,
        split='val',
        transform=get_flowers_transforms('test')
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    test_dataset = Flowers102Dataset(
        data_root=data_root,
        split='test',
        transform=get_flowers_transforms('test')
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # 测试单个样本加载
    print("\n2. Testing single sample loading...")
    if len(train_dataset) > 0:
        image, label = train_dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample label: {label}")
    
    # 测试小样本采样器
    print("\n3. Testing few-shot sampler...")
    sampler = Flowers102FewShotSampler(
        dataset=train_dataset,
        n_way=5,
        k_shot=5,
        q_query=15,
        num_episodes=3
    )
    
    for episode_idx, (support_x, support_y, query_x, query_y) in enumerate(sampler):
        print(f"Episode {episode_idx + 1}:")
        print(f"  Support set shape: {support_x.shape}, labels: {support_y.shape}")
        print(f"  Query set shape: {query_x.shape}, labels: {query_y.shape}")
    
    print("\nAll tests completed successfully!")