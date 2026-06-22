import torch
import argparse
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import create_dataloaders, create_flowers_dataloaders, create_fgvc_aircraft_dataloaders
from models.factory import create_model_and_optimizer
from trainers.meta_trainer import MetaTrainer
from utils.visualization import analyze_model_attention
import warnings
from configs.config import *
warnings.filterwarnings("ignore", category=UserWarning, 
module="torchvision.models._utils")

def load_class_names(data_root, dataset_type='cub'):
    """加载类别名称"""
    if dataset_type == 'flowers':
                                       
        class_names = [f"Flower_{i}" for i in range(102)]
        return class_names
    elif dataset_type == 'fgvc_aircraft':
                                  
        data_path = Path(data_root)
        classes_file = data_path / "data" / "variants.txt"
        
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
        
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        return class_names
    else:
                        
        with open(os.path.join(data_root, 'classes.txt'), 'r') as f:
            lines = f.readlines()
        class_names = [line.strip().split(' ')[1] for line in lines]
        return class_names

def analyze_fewshot_visualization(model, fewshot_sampler, class_names, device, save_dir, n_way=5):
    """基于小样本情景的可视化分析"""
    print("Performing few-shot visualization analysis...")
    
    model.eval()
    model.to(device)
    
    episode_data = next(iter(fewshot_sampler))
    
    if len(episode_data) == 5:
        support_images, support_labels, query_images, query_labels, episode_global_classes = episode_data
    else:
        support_images, support_labels, query_images, query_labels = episode_data
        episode_global_classes = None
    
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)
    
    sample_query_image = query_images[0]
    sample_query_label = query_labels[0].item()
    
    print(f"Analyzing query sample from class {sample_query_label}")
    print(f"Support set contains {len(support_images)} samples from {len(torch.unique(support_labels))} classes")
    
    analysis_result = analyze_model_attention(
        model=model,
        image_tensor=sample_query_image,
        true_label=sample_query_label,
        class_names=class_names
    )
    
    analysis_result.update({
        'fewshot_context': {
            'n_way': n_way,
            'k_shot': len(support_images) // n_way,
            'query_samples': len(query_images),
            'support_classes': [int(cls) for cls in torch.unique(support_labels).cpu().numpy()],
            'support_class_names': [class_names[cls] if cls < len(class_names) else f"Class_{cls}" 
                                  for cls in torch.unique(support_labels).cpu().numpy()]
        },
        'sample_info': {
            'query_class_id': int(sample_query_label),
            'query_class_name': class_names[sample_query_label] if sample_query_label < len(class_names) else f"Class_{sample_query_label}",
            'support_set_size': len(support_images)
        }
    })
    
    analysis_path = os.path.join(save_dir, 'fewshot_attention_analysis.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    plt.savefig(os.path.join(save_dir, 'fewshot_attention_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Few-shot visualization analysis completed!")
    print(f"Results saved to: {analysis_path}")
    
    return analysis_result

def create_model_specific_save_dir(base_save_dir, dataset_type, model_type, backbone, **kwargs):
    """创建按数据集和模型分类的保存目录"""
                  
    dataset_subdir = dataset_type
    
    model_subdir = f"{model_type}_{backbone}"
    
    additional_params = []
    if 'n_way' in kwargs:
        additional_params.append(f"N{kwargs['n_way']}")
    if 'k_shot' in kwargs:
        additional_params.append(f"K{kwargs['k_shot']}")
    if 'lr' in kwargs:
        additional_params.append(f"LR{kwargs['lr']}")
    
    if additional_params:
        model_subdir += "_" + "_".join(additional_params)
    
    full_save_path = os.path.join(base_save_dir, dataset_subdir, model_subdir)
    os.makedirs(full_save_path, exist_ok=True)
    
    print(f"Saving results to: {full_save_path}")
    return full_save_path

def main():
    parser = argparse.ArgumentParser(description='小样本分类系统 (支持CUB、Flowers102和FGVC Aircraft)')
    parser.add_argument('--data_root', type=str, default=None,
                       help='数据集根目录 (如果未指定则使用配置文件中的默认值)')
    parser.add_argument('--dataset_type', type=str, default='cub',
                       choices=['cub', 'flowers', 'fgvc_aircraft'], help='数据集类型')
    parser.add_argument('--annotation_level', type=str, default='variant',
                       choices=['variant', 'family', 'manufacturer'], 
                       help='FGVC Aircraft 标注级别 (仅当 dataset_type=fgvc_aircraft 时有效)')
    parser.add_argument('--model_type', type=str, default='protonet',
                       choices=['protonet', 'maml', 'protonet_scr','protonet_scr_cbam', 'relation_attnres', 'sampn'], help='模型类型')
    parser.add_argument('--backbone', type=str, default=MODEL_CONFIG['backbone'],
                       choices=['conv4','resnet12', 'resnet18'], help='骨干网络')
    parser.add_argument('--n_way', type=int, default=FEW_SHOT_CONFIG['n_way'], help='N-way')
    parser.add_argument('--k_shot', type=int, default=FEW_SHOT_CONFIG['k_shot'], help='K-shot')
    parser.add_argument('--q_query', type=int, default=FEW_SHOT_CONFIG['q_query'], help='查询样本数')
    parser.add_argument('--num_episodes', type=int, default=FEW_SHOT_CONFIG['num_episodes'], help='每轮episode数')
    parser.add_argument('--epochs', type=int, default=FEW_SHOT_CONFIG['num_epochs'], help='训练轮数')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'], help='学习率')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'], help='批量大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--save_dir', type=str, default=EXPERIMENT_CONFIG['save_dir'], help='保存目录')
    parser.add_argument('--visualize', action='store_true', help='是否进行可视化分析')
    parser.add_argument('--fewshot_visualize', action='store_true', help='是否进行基于小样本情景的可视化分析')
    
    args = parser.parse_args()
    if args.data_root is None:
        if args.dataset_type == 'flowers':
            args.data_root = './datasets/flowers-102'
        elif args.dataset_type == 'fgvc_aircraft':
            args.data_root = './datasets/fgvc-aircraft-2013b'
        else:
            args.data_root = DATA_CONFIG['data_root']
    save_dir = create_model_specific_save_dir(
        base_save_dir=args.save_dir,
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        backbone=args.backbone,
        n_way=args.n_way,
        k_shot=args.k_shot,
        lr=args.lr
    )

    print(f"Loading {args.dataset_type.upper()} dataset...")
    if args.dataset_type == 'flowers':
        train_loader, val_loader, train_few_shot_sampler, val_few_shot_sampler = create_flowers_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_episodes=args.num_episodes
        )
    elif args.dataset_type == 'fgvc_aircraft':
        train_loader, val_loader, train_few_shot_sampler, val_few_shot_sampler = create_fgvc_aircraft_dataloaders(
            data_root=args.data_root,
            annotation_level=args.annotation_level,
            batch_size=args.batch_size,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_workers=4,
            num_episodes=args.num_episodes
        )
    else:
        train_loader, val_loader, train_few_shot_sampler, val_few_shot_sampler = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_episodes=args.num_episodes
        )
    print("Data loaded.")
    
    class_names = load_class_names(args.data_root, dataset_type=args.dataset_type)
    
    print(f"Creating {args.model_type} model...")
    model, optimizer = create_model_and_optimizer(
        model_type=args.model_type,
        backbone=args.backbone,
        lr=args.lr
    )
                                      
    trainer = MetaTrainer(model, optimizer, device=args.device)

    print("Starting training...")
    
    plt.ion()          
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 4))
    loss_line, = ax1.plot([], [], 'b-')
    acc_line, = ax2.plot([], [], 'g-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    fig.tight_layout()
        
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 2.5)
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 1)    
                                     
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.epochs):
                     
        trainer.train_one_epoch(train_few_shot_sampler, val_few_shot_sampler)

        current_val_acc = trainer.val_accuracies[-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        n = min(len(trainer.train_losses), 100)
        loss_line.set_data(range(1, n+1), trainer.train_losses[:n])
        acc_line.set_data(range(1, n+1), trainer.val_accuracies[:n])

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model restored with validation accuracy: {best_val_acc:.4f}")

    plt.ioff()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))

    model_path = os.path.join(save_dir, f'{args.model_type}_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    history = {
        'train_losses': trainer.train_losses,
        'val_accuracies': trainer.val_accuracies
    }
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    if args.visualize:
        print("Performing standard visualization analysis...")
                     
        val_dataset = val_loader.dataset
        sample_image, sample_label = val_dataset[0]
        
        analysis_result = analyze_model_attention(
            model=model,
            image_tensor=sample_image,
            true_label=sample_label,
            class_names=class_names
        )
        
        analysis_path = os.path.join(save_dir, 'attention_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        print("Standard visualization analysis completed!")

    if args.fewshot_visualize:
        fewshot_analysis_result = analyze_fewshot_visualization(
            model=model,
            fewshot_sampler=val_few_shot_sampler,
            class_names=class_names,
            device=args.device,
            save_dir=save_dir,
            n_way=args.n_way
        )

if __name__ == '__main__':
    main()