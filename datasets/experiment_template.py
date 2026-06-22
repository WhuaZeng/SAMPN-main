"""
实验数据集模板

这是一个通用的数据集加载器模板，可以根据你的实际需求进行修改。

使用方法:
    1. 复制此文件并重命名（如 my_dataset.py）
    2. 修改类名和配置
    3. 在 main.py 中注册新数据集
    4. 运行训练

示例:
    cp experiment_template.py my_custom_dataset.py
    # 编辑 my_custom_dataset.py
    # 在 main.py 中添加新的 dataset_type 选项
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pathlib import Path


class ExperimentDataset(Dataset):
    """
    通用实验数据集加载器
    
    支持标准的图像分类目录结构：
    dataset_root/
        train/
            class_0/
                img_001.jpg
                ...
            class_1/
                ...
        val/
            ...
        test/
            ...
    """
    
    def __init__(self, data_root, split='train', transform=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 图像变换操作
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # 数据存储
        self.samples = []  # [(image_path, label), ...]
        self.classes = []  # 类别名称列表
        self.class_to_idx = {}  # 类别名到索引的映射
        
        # 加载数据
        self._load_data()
        
        print(f"✓ Loaded {len(self.samples)} samples from '{split}' split")
        print(f"  Classes: {len(self.classes)}")
        if self.classes:
            print(f"  Sample classes: {self.classes[:5]}...")
    
    def _load_data(self):
        """加载数据集"""
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(
                f"Split directory not found: {split_dir}\n"
                f"Available splits: {[d.name for d in self.data_root.iterdir() if d.is_dir()]}"
            )
        
        # 遍历所有类别文件夹
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx
            
            # 遍历该类别的所有图片
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.JPG', '*.PNG']:
                image_files.extend(class_dir.glob(ext))
            
            for img_path in sorted(image_files):
                self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {split_dir}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path, label = self.samples[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 应用变换
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白样本作为占位
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label
    
    def get_class_names(self):
        """获取所有类别名称"""
        return self.classes
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.samples),
            'num_classes': len(self.classes),
            'split': self.split,
            'samples_per_class': {}
        }
        
        # 计算每个类别的样本数
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for _, label in self.samples if label == class_idx)
            stats['samples_per_class'][class_name] = count
        
        return stats


def create_experiment_dataloaders(
    data_root,
    n_way=5,
    k_shot=5,
    q_query=15,
    batch_size=4,
    num_workers=4,
    image_size=224
):
    """
    创建实验数据集的 DataLoader
    
    Args:
        data_root: 数据集根目录
        n_way: 每个 episode 的类别数
        k_shot: 每个类别的支持样本数
        q_query: 每个类别的查询样本数
        batch_size: batch size (用于非 episodic 训练)
        num_workers: 数据加载线程数
        image_size: 图像尺寸
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 创建数据集
    print(f"\nLoading experiment dataset from: {data_root}")
    train_dataset = ExperimentDataset(data_root, split='train', transform=train_transform)
    
    # 检查是否有 val 和 test 集
    val_split = 'val' if (Path(data_root) / 'val').exists() else 'test'
    test_split = 'test' if (Path(data_root) / 'test').exists() else 'val'
    
    val_dataset = ExperimentDataset(data_root, split=val_split, transform=test_transform)
    test_dataset = ExperimentDataset(data_root, split=test_split, transform=test_transform)
    
    # 打印统计信息
    print("\nDataset Statistics:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ==================== 测试代码 ====================

if __name__ == '__main__':
    """测试数据集加载器"""
    
    print("="*70)
    print("Experiment Dataset Template - Test")
    print("="*70)
    
    # 测试参数
    test_data_root = "./data/test_experiment"
    
    # 创建测试数据（如果不存在）
    test_path = Path(test_data_root)
    if not test_path.exists():
        print(f"\nCreating test dataset at: {test_data_root}")
        
        # 创建简单的测试数据结构
        for split in ['train', 'val', 'test']:
            for class_idx in range(3):
                class_dir = test_path / split / f"class_{class_idx}"
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建几个空白图片
                for img_idx in range(5):
                    img = Image.new('RGB', (224, 224), color=(class_idx*50, 100, 150))
                    img.save(class_dir / f"img_{img_idx:03d}.jpg")
        
        print("✓ Test dataset created")
    
    # 测试数据集加载
    try:
        print("\nTesting dataset loading...")
        train_loader, val_loader, test_loader = create_experiment_dataloaders(
            data_root=test_data_root,
            batch_size=2,
            num_workers=0  # Windows 下建议使用 0
        )
        
        # 测试数据迭代
        print("\nTesting data iteration...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"  Batch {batch_idx}: images shape = {images.shape}, labels = {labels.tolist()}")
            
            if batch_idx >= 2:  # 只测试前几个 batch
                break
        
        print("\n✅ All tests passed!")
        print(f"\nNext steps:")
        print(f"  1. Replace '{test_data_root}' with your actual dataset path")
        print(f"  2. Integrate into main.py training pipeline")
        print(f"  3. Run training with: python main.py --dataset_type experiment --data_root YOUR_PATH")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
