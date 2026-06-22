import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import defaultdict
import re

class CUBDataset(Dataset):
    """CUB-200-2011数据集类"""
    
    def __init__(self, data_root, split='train', transform=None, few_shot=False, n_way=5, k_shot=5):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.few_shot = few_shot
        self.n_way = n_way
        self.k_shot = k_shot
        
        # 缓存文件系统结构
        self.file_cache = {}
        self._build_file_cache()
        
        # 加载数据索引
        self.image_paths, self.labels = self._load_data()
        
        # 如果是小样本设置，重新组织数据
        if few_shot:
            self._prepare_few_shot_data()
    
    def _build_file_cache(self):
        """构建文件系统缓存，用于快速查找文件"""
        images_dir = os.path.join(self.data_root, 'images')
        print("Building file cache...")
        
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                # 存储小写文件名到实际文件路径的映射
                file_lower = file.lower()
                full_path = os.path.join(root, file)
                
                if file_lower not in self.file_cache:
                    self.file_cache[file_lower] = full_path
                else:
                    # 如果已经有同名文件，记录但使用第一个找到的
                    pass
        
        print(f"File cache built with {len(self.file_cache)} entries")
    
    def _find_file_by_name(self, filename):
        """通过文件名查找文件（不区分大小写）"""
        filename_lower = filename.lower()
        if filename_lower in self.file_cache:
            return self.file_cache[filename_lower]
        return None
    
    def _load_data(self):
        """加载图像路径和标签"""
        # 读取图像列表
        images_df = pd.read_csv(
            os.path.join(self.data_root, 'images.txt'),
            sep=' ', header=None, names=['id', 'filepath']
        )
        
        # 读取标签
        labels_df = pd.read_csv(
            os.path.join(self.data_root, 'image_class_labels.txt'),
            sep=' ', header=None, names=['id', 'label']
        )
        
        # 读取训练测试分割
        split_df = pd.read_csv(
            os.path.join(self.data_root, 'train_test_split.txt'),
            sep=' ', header=None, names=['id', 'is_train']
        )
        
        # 合并数据
        df = images_df.merge(labels_df, on='id').merge(split_df, on='id')
        
        # 根据分割选择数据
        if self.split == 'train':
            df = df[df['is_train'] == 1]
        else:
            df = df[df['is_train'] == 0]
        
        # 构建完整路径
        image_paths = []
        valid_indices = []
        
        for idx, filepath in enumerate(df['filepath'].values):
            # 确保文件路径使用正确的分隔符
            filepath = filepath.replace('/', os.sep)
            full_path = os.path.join(self.data_root, 'images', filepath)
            
            # 检查文件是否存在
            if os.path.exists(full_path):
                image_paths.append(full_path)
                valid_indices.append(idx)
            else:
                # 尝试通过文件名查找
                filename = os.path.basename(filepath)
                found_path = self._find_file_by_name(filename)
                if found_path and os.path.exists(found_path):
                    image_paths.append(found_path)
                    valid_indices.append(idx)
                else:
                    # 文件确实不存在，跳过
                    print(f"Warning: File not found and cannot be located: {filename}")
        
        # 只保留有效的标签
        labels = df.iloc[valid_indices]['label'].values - 1  # 转换为0索引
        
        print(f"Loaded {len(image_paths)} valid images out of {len(df)} total")
        
        return image_paths, labels
    
    def _prepare_few_shot_data(self):
        """准备小样本学习数据"""
        # 按类别分组
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        
        # 过滤掉样本数不足的类别
        valid_classes = [cls for cls, indices in class_to_indices.items() 
                        if len(indices) >= self.k_shot]
        
        # 随机选择n_way个类别
        selected_classes = random.sample(valid_classes, min(self.n_way, len(valid_classes)))
        
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
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 使用更安全的方式加载图像
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

class FewShotSampler:
    """小样本采样器"""
    
    def __init__(self, dataset, n_way=5, k_shot=5, q_query=15, num_episodes=100):
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

def get_transforms(split='train'):
    """获取数据变换"""
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_dataloaders(data_root, batch_size=16, n_way=5, k_shot=5, q_query=15, num_episodes=100):
    """创建数据加载器"""
    
    # 确保数据根目录路径正确
    data_root = os.path.abspath(os.path.normpath(data_root))
    print(f"Using data root: {data_root}")
    
    # 检查数据根目录是否存在
    if not os.path.exists(data_root):
        raise ValueError(f"Data root directory does not exist: {data_root}")
    
    # 检查必要的文件是否存在
    required_files = ['images.txt', 'image_class_labels.txt', 'train_test_split.txt']
    for file in required_files:
        file_path = os.path.join(data_root, file)
        if not os.path.exists(file_path):
            raise ValueError(f"Required file not found: {file_path}")
    
    # 检查images目录是否存在
    images_dir = os.path.join(data_root, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")
    
    # 基础数据集
    train_dataset = CUBDataset(
        data_root=data_root,
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = CUBDataset(
        data_root=data_root,
        split='test',
        transform=get_transforms('test')
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 训练集小样本采样器
    train_few_shot_sampler = FewShotSampler(
        dataset=train_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes
    )
    
    # 验证集小样本采样器
    val_few_shot_sampler = FewShotSampler(
        dataset=val_dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=20  # 较少的验证episode
    )
    
    # 基础数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 设置为0以避免多进程问题
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # 设置为0以避免多进程问题
    )
    
    return train_loader, val_loader, train_few_shot_sampler, val_few_shot_sampler

if __name__ == '__main__':
    # 检查所有图片数据是否均能正常加载
    data_root = "e:/few_shot_project/datasets/CUB_200_2011"  # 根据实际数据集位置调整路径
    
    # 验证训练集
    print("Validating training set...")
    train_dataset = CUBDataset(data_root, split='train', transform=get_transforms('train'))

    
    # 验证测试集
    print("\nValidating test set...")
    test_dataset = CUBDataset(data_root, split='test', transform=get_transforms('test'))
    print(f"Total test samples: {len(test_dataset)}")

    print("\nValidation completed successfully!")