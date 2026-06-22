"""
FGVC Aircraft 数据集下载与加载器

FGVC-FGVC Aircraft 是一个细粒度飞机分类数据集，包含：
- 100 个飞机型号（variants）
- 6,667 张训练图像
- 3,333 张验证图像  
- 3,333 张测试图像

数据来源: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

使用方法:
    1. 自动下载: python datasets/fgvc_aircraft_dataset.py --download
    2. 手动下载: 从官网下载后解压到 ./datasets/fgvc-aircraft-2013b
    3. 在 main.py 中使用: --dataset_type fgvc_aircraft
"""

import os
import sys
import urllib.request
import urllib.error
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class FGVCAircraftDataset(Dataset):
    """
    FGVC Aircraft 数据集加载器
    
    支持三种标注级别:
    - variant: 100 个具体型号 (默认)
    - family: 30 个家族类别
    - manufacturer: 30 个制造商类别
    """
    
    def __init__(self, data_root, split='train', annotation_level='variant', transform=None, 
                 n_way=None, k_shot=None, q_query=None, num_episodes=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            annotation_level: 标注级别 ('variant', 'family', 'manufacturer')
            transform: 图像变换
            n_way: 每个 episode 的类别数（小样本模式）
            k_shot: 每个类别的支持样本数（小样本模式）
            q_query: 每个类别的查询样本数（小样本模式）
            num_episodes: Episode 数量（小样本模式）
        """
        self.data_root = Path(data_root)
        self.split = split
        self.annotation_level = annotation_level
        self.transform = transform
        
        # 小样本学习参数
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.is_few_shot = (n_way is not None and k_shot is not None)
        
        # 数据路径 - 适配实际的数据集结构
        self.images_dir = self.data_root / "data" / "images"
        
        # 根据标注级别选择文件
        if annotation_level == 'variant':
            labels_file = self.data_root / "data" / f"images_variant_{split}.txt"
            classes_file = self.data_root / "data" / "variants.txt"
        elif annotation_level == 'family':
            labels_file = self.data_root / "data" / f"images_family_{split}.txt"
            classes_file = self.data_root / "data" / "families.txt"
        elif annotation_level == 'manufacturer':
            labels_file = self.data_root / "data" / f"images_manufacturer_{split}.txt"
            classes_file = self.data_root / "data" / "manufacturers.txt"
        else:
            raise ValueError(f"Unknown annotation level: {annotation_level}")
        
        # 加载数据
        self.image_paths = []  # 图像路径列表
        self.labels = []       # 标签列表
        self.classes = []      # 类别名称列表
        self.class_to_idx = {} # 类别名到索引的映射
        
        self._load_classes(classes_file)
        self._load_samples(labels_file)
        
        print(f"✓ Loaded {len(self.image_paths)} samples from '{split}' split")
        print(f"  Annotation level: {annotation_level}")
        print(f"  Classes: {len(self.classes)}")
        if self.classes:
            print(f"  Sample classes: {self.classes[:5]}...")
    
    def _load_classes(self, classes_file):
        """加载类别列表"""
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
        
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def _load_samples(self, labels_file):
        """加载样本列表"""
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_name = parts[1]
                    
                    if class_name in self.class_to_idx:
                        img_path = self.images_dir / f"{image_name}.jpg"
                        if img_path.exists():
                            self.image_paths.append(str(img_path))
                            self.labels.append(self.class_to_idx[class_name])
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid samples found in {labels_file}")
    
    def __len__(self):
        """返回数据集大小"""
        if self.is_few_shot:
            return self.num_episodes if self.num_episodes else 1000
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本（标准模式）"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回空白样本
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label
    
    def get_episode(self):
        """获取一个小样本 Episode（用于小样本学习模式）"""
        if not self.is_few_shot:
            raise RuntimeError("Not in few-shot mode. Initialize with n_way, k_shot parameters.")
        
        # 按类别分组
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        
        # 过滤掉样本数不足的类别
        min_samples_needed = self.k_shot + self.q_query
        valid_classes = [cls for cls, indices in class_to_indices.items() 
                        if len(indices) >= min_samples_needed]
        
        if len(valid_classes) < self.n_way:
            raise ValueError(f"Not enough valid classes. Need {self.n_way}, have {len(valid_classes)}")
        
        # 随机选择 n_way 个类别
        selected_classes = random.sample(valid_classes, self.n_way)
        
        # 为每个选中的类别选择支持集和查询集
        support_images = []
        support_labels = []
        support_paths = []  # 新增：保存支持集图片路径
        query_images = []
        query_labels = []
        query_paths = []  # 新增：保存查询集图片路径
        
        for new_label, cls in enumerate(selected_classes):
            indices = random.sample(class_to_indices[cls], self.k_shot + self.q_query)
            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:]
            
            for idx in support_indices:
                img_path = self.image_paths[idx]
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    support_images.append(image)
                    support_labels.append(new_label)
                    support_paths.append(img_path)  # 保存路径
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
            
            for idx in query_indices:
                img_path = self.image_paths[idx]
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    query_images.append(image)
                    query_labels.append(new_label)
                    query_paths.append(img_path)  # 保存路径
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
        
        # 转换为 Tensor
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        # 返回全局类别 ID（用于可视化）
        global_cls_ids = selected_classes
        
        return support_images, support_labels, query_images, query_labels, global_cls_ids, support_paths, query_paths
    
    def get_class_names(self):
        """获取所有类别名称"""
        return self.classes
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.image_paths),
            'num_classes': len(self.classes),
            'split': self.split,
            'annotation_level': self.annotation_level,
            'samples_per_class': {}
        }
        
        # 计算每个类别的样本数
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for label in self.labels if label == class_idx)
            stats['samples_per_class'][class_name] = count
        
        return stats


def download_with_progress(url, filepath, chunk_size=8192):
    """
    带进度显示的下载函数
    
    Args:
        url: 下载链接
        filepath: 保存路径
        chunk_size: 每次读取的字节数
    """
    # 检查文件是否已存在
    if filepath.exists():
        existing_size = filepath.stat().st_size
        print(f"⚠️  文件已存在: {filepath}")
        print(f"   当前大小: {existing_size / (1024*1024):.2f} MB")
        
        confirm = input("   是否重新下载? (y/n): ").strip().lower()
        if confirm != 'y':
            return True
    
    try:
        # 创建请求
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        # 打开URL
        response = urllib.request.urlopen(req, timeout=300)  # 5分钟超时
        total_size = int(response.getheader('Content-Length', 0))
        
        print(f"文件大小: {total_size / (1024*1024*1024):.2f} GB")
        print("开始下载...")
        
        downloaded = 0
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                
                f.write(chunk)
                downloaded += len(chunk)
                
                # 显示进度
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024*1024)
                    mb_total = total_size / (1024*1024)
                    print(f"\r进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
                else:
                    mb_downloaded = downloaded / (1024*1024)
                    print(f"\r已下载: {mb_downloaded:.1f} MB", end='', flush=True)
        
        print("\n✓ 下载完成")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n✗ 网络错误: {e.reason}")
        print("\n建议:")
        print("  1. 检查网络连接")
        print("  2. 使用代理或镜像源")
        print("  3. 手动下载后解压")
        return False
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def download_fgvc_aircraft(data_root='./data/fgvc-aircraft'):
    """
    下载 FGVC Aircraft 数据集
    
    Args:
        data_root: 保存路径
    """
    data_path = Path(data_root)
    
    # 检查是否已存在
    if (data_path / "images").exists() and (data_path / "variants.txt").exists():
        print(f"✓ Dataset already exists at: {data_path}")
        print("  Skipping download.")
        return True
    
    # 创建目录
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 下载地址
    url = "http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    tar_path = data_path / "fgvc-aircraft-2013b.tar.gz"
    
    print(f"{'='*70}")
    print(f"Downloading FGVC Aircraft dataset")
    print(f"URL: {url}")
    print(f"Save to: {data_path}")
    print(f"{'='*70}\n")
    
    # 下载文件
    if not download_with_progress(url, tar_path):
        print("\n" + "="*70)
        print("❌ 下载失败")
        print("="*70)
        print("\n请手动下载:")
        print(f"  1. 访问: {url}")
        print(f"  2. 下载 fgvc-aircraft-2013b.tar.gz")
        print(f"  3. 解压到: {data_path}")
        print("\n解压命令:")
        print(f"  tar -xzf fgvc-aircraft-2013b.tar.gz -C {data_path}")
        return False
    
    try:
        # 解压
        print("\n正在解压...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            for i, member in enumerate(members):
                tar.extract(member, path=data_path)
                if i % 100 == 0:
                    print(f"\r解压进度: {i+1}/{len(members)}", end='', flush=True)
        print("\n✓ 解压完成")
        
        # 移动文件到正确位置
        extracted_dir = data_path / "fgvc-aircraft-2013b"
        if extracted_dir.exists():
            print("整理文件结构...")
            for item in extracted_dir.iterdir():
                dest = data_path / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.move(str(item), str(dest))
                    else:
                        item.rename(dest)
            extracted_dir.rmdir()
        
        # 删除压缩包
        print("清理临时文件...")
        tar_path.unlink()
        
        print(f"\n{'='*70}")
        print("✅ Dataset download and extraction completed!")
        print(f"{'='*70}")
        
        # 显示数据结构
        images_count = len(list((data_path / 'images').glob('*.jpg')))
        variants_count = len((data_path / 'variants.txt').read_text().strip().split('\n'))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Images: {images_count} files")
        print(f"  Variants: {variants_count} classes")
        print(f"  Location: {data_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 解压或整理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_fgvc_aircraft_dataloaders(
    data_root='./datasets/fgvc-aircraft-2013b',
    annotation_level='variant',
    n_way=5,
    k_shot=5,
    q_query=15,
    batch_size=4,
    num_workers=4,
    image_size=224,
    num_episodes=100
):
    """
    创建 FGVC Aircraft 数据集的 DataLoader（小样本学习模式）
    
    Args:
        data_root: 数据集根目录
        annotation_level: 标注级别 ('variant', 'family', 'manufacturer')
        n_way: 每个 episode 的类别数
        k_shot: 每个类别的支持样本数
        q_query: 每个类别的查询样本数
        batch_size: batch size（标准模式使用）
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        num_episodes: 每轮 episode 数量
    
    Returns:
        train_loader, val_loader, train_few_shot_sampler, val_few_shot_sampler
    """
    
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
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
    
    # 创建数据集（小样本模式）
    print(f"\nLoading FGVC Aircraft dataset from: {data_root}")
    print(f"Annotation level: {annotation_level}")
    print(f"Few-shot config: {n_way}-way, {k_shot}-shot, {q_query} query")
    
    train_dataset = FGVCAircraftDataset(
        data_root, 
        split='train', 
        annotation_level=annotation_level,
        transform=train_transform,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes
    )
    
    val_dataset = FGVCAircraftDataset(
        data_root, 
        split='val', 
        annotation_level=annotation_level,
        transform=test_transform,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes
    )
    
    # 打印统计信息
    print("\nDataset Statistics:")
    print(f"  Train: {len(train_dataset.image_paths)} samples, {len(train_dataset.classes)} classes")
    print(f"  Val:   {len(val_dataset.image_paths)} samples, {len(val_dataset.classes)} classes")
    
    # 创建标准的 DataLoader（用于非小样本任务，可选）
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
    
    # 创建小样本采样器包装类
    class FewShotSamplerWrapper:
        """将数据集包装为可迭代的采样器"""
        def __init__(self, dataset, num_episodes):
            self.dataset = dataset
            self.num_episodes = num_episodes
        
        def __iter__(self):
            for _ in range(self.num_episodes):
                yield self.dataset.get_episode()
        
        def __len__(self):
            return self.num_episodes
    
    train_sampler = FewShotSamplerWrapper(train_dataset, num_episodes)
    val_sampler = FewShotSamplerWrapper(val_dataset, num_episodes)
    
    # 返回数据集本身作为 sampler（因为数据集已经实现了 get_episode 方法）
    return train_loader, val_loader, train_sampler, val_sampler


# ==================== 命令行接口 ====================

def main():
    """主函数 - 支持命令行下载"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FGVC Aircraft Dataset Tool')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--data_root', type=str, default='./datasets/fgvc-aircraft-2013b', 
                       help='Data root directory')
    parser.add_argument('--test', action='store_true', help='Test dataset loading')
    parser.add_argument('--annotation_level', type=str, default='variant',
                       choices=['variant', 'family', 'manufacturer'],
                       help='Annotation level')
    
    args = parser.parse_args()
    
    if args.download:
        success = download_fgvc_aircraft(args.data_root)
        if success:
            print("\n✅ Dataset ready to use!")
            print(f"Use in training: python main.py --dataset_type fgvc_aircraft --data_root {args.data_root}")
        else:
            print("\n❌ Download failed")
            sys.exit(1)
    
    if args.test:
        print("="*70)
        print("Testing FGVC Aircraft Dataset (Few-Shot Mode)")
        print("="*70)
        
        try:
            train_loader, val_loader, train_sampler, val_sampler = create_fgvc_aircraft_dataloaders(
                data_root=args.data_root,
                annotation_level=args.annotation_level,
                n_way=5,
                k_shot=5,
                q_query=15,
                batch_size=2,
                num_workers=0,
                num_episodes=2
            )
            
            # 测试小样本 Episode 生成
            print("\nTesting few-shot episode generation...")
            for i in range(2):
                episode_data = train_sampler.get_episode()
                
                # 支持新旧版本
                if len(episode_data) == 7:
                    support_images, support_labels, query_images, query_labels, global_cls_ids, support_paths, query_paths = episode_data
                    print(f"  Episode {i+1}:")
                    print(f"    Support: {support_images.shape}, Labels: {support_labels.tolist()}")
                    print(f"    Query: {query_images.shape}, Labels: {query_labels.tolist()}")
                    print(f"    Global class IDs: {global_cls_ids}")
                    print(f"    Support paths: {support_paths[:2]}...")  # 显示前两个路径
                    print(f"    Query paths: {query_paths[:2]}...")
                else:
                    support_images, support_labels, query_images, query_labels, global_cls_ids = episode_data
                    print(f"  Episode {i+1}:")
                    print(f"    Support: {support_images.shape}, Labels: {support_labels.tolist()}")
                    print(f"    Query: {query_images.shape}, Labels: {query_labels.tolist()}")
                    print(f"    Global class IDs: {global_cls_ids}")
            
            print("\n✅ All tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
