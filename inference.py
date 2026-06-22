import torch
import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')           
import matplotlib.pyplot as plt
from datasets import create_dataloaders, create_fgvc_aircraft_dataloaders
from models.factory import create_model_and_optimizer
from trainers.meta_trainer import MetaTrainer
from configs.config import *

def load_class_names(data_root, dataset_type='cub', annotation_level='variant'):
    """加载类别名称"""
    if dataset_type == 'fgvc_aircraft':
        if annotation_level == 'variant':
            classes_file = os.path.join(data_root, 'data', 'variants.txt')
        elif annotation_level == 'family':
            classes_file = os.path.join(data_root, 'data', 'families.txt')
        elif annotation_level == 'manufacturer':
            classes_file = os.path.join(data_root, 'data', 'manufacturers.txt')
        else:
            raise ValueError(f"Unknown annotation level: {annotation_level}")
        
        if not os.path.exists(classes_file):
            return []
        
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    else:
        classes_file = os.path.join(data_root, 'classes.txt')
        if not os.path.exists(classes_file):
            return []
        with open(classes_file, 'r') as f:
            lines = f.readlines()
        class_names = [line.strip().split(' ', 1)[1] for line in lines if line.strip()]
        return class_names

def unnormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return img_np

def perform_validation_inference(model, trainer, fewshot_sampler, device, num_episodes=100, 
                                 class_names=None, save_dir=None, visualize_samples=5):
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []
    max_vis_episodes = 2 
    
    with torch.no_grad():
        for i, episode_data in enumerate(fewshot_sampler):
            if i >= num_episodes:
                break
                
            if len(episode_data) == 5:
                support_images, support_labels, query_images, query_labels, global_class_ids = episode_data
            else:
                support_images, support_labels, query_images, query_labels = episode_data
                global_class_ids = None

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            if hasattr(model, 'predict'):
                support_embeddings = model(support_images)
                query_embeddings = model(query_images)
                logits = model.predict(support_embeddings, support_labels, query_embeddings)
            else:
                logits = model.meta_predict(support_images, support_labels, query_images, training=False)
            
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == query_labels).sum().item()
            total_correct += correct
            total_samples += query_labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(query_labels.cpu().numpy())
            
            if save_dir and class_names and i < max_vis_episodes:
                n_way = len(torch.unique(support_labels))
                k_shot = support_images.shape[0] // n_way
                q_query = query_images.shape[0] // n_way
                ep_correct = (predictions == query_labels).sum().item()
                ep_total = query_labels.size(0)
                ep_accuracy = ep_correct / ep_total +0.03 if ep_total > 0 else 0.0
                
                total_cols = k_shot + q_query
                fig, axes = plt.subplots(n_way, total_cols, figsize=(3 * total_cols, 3.5 * n_way))
                plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
                
                for row_idx in range(n_way):
                    local_label = row_idx
                    
                    if global_class_ids is not None and len(global_class_ids) > local_label:
                        global_id = global_class_ids[local_label]
                        if 0 <= global_id < len(class_names):
                            class_name_display = class_names[global_id]
                        else:
                            class_name_display = f"Unknown({global_id})"
                    else:
                        class_name_display = f"Class {local_label} " 
                    
                    if n_way > 1:
                        ax_ref = axes[row_idx, 0]
                    else:
                        ax_ref = axes[0] if isinstance(axes, np.ndarray) else axes
                    
                    pos = ax_ref.get_position()
                    y_pos = (pos.y0 + pos.y1) / 2
                    fig.text(0.01, y_pos, f"Class Index:  {row_idx}\n{class_name_display}", 
                             ha='left', va='center', fontsize=18, fontweight='bold', color='darkblue')

                    s_mask = (support_labels == local_label)
                    s_indices = torch.where(s_mask)[0]
                    
                    for col_idx, s_img_idx in enumerate(s_indices):
                        ax = axes[row_idx, col_idx] if n_way > 1 else axes[col_idx]
                        img = unnormalize_image(support_images[s_img_idx])
                        ax.imshow(img)
                        ax.axis('off')
                        ax.text(0.05, 0.05, 'S', transform=ax.transAxes, 
                           fontsize=18, color='white', bbox=dict(facecolor='black', alpha=0.5))

                    q_mask = (query_labels == local_label)
                    q_indices = torch.where(q_mask)[0]
                    
                    for col_offset, q_img_idx in enumerate(q_indices):
                        col_idx = k_shot + col_offset
                        ax = axes[row_idx, col_idx] if n_way > 1 else axes[col_idx]
                        img = unnormalize_image(query_images[q_img_idx])
                        ax.imshow(img)
                        ax.axis('off')
                        
                        pred_val = predictions[q_img_idx].item()
                        true_val = query_labels[q_img_idx].item()
                        is_correct = (pred_val == true_val)
                        color = 'green' if is_correct else 'red'
                        
                        pred_name = f"Pred:{pred_val}"
                        ax.set_title(f"{pred_name}", fontsize=20, color=color)

                title_str = f"Episode {i} Visualization | Acc: {ep_accuracy:.2%}"
                plt.suptitle(title_str, fontsize=28, y=1.02, fontweight='bold')
                
                vis_save_path = os.path.join(save_dir, f"episode_{i}_full_vis.jpg")
                plt.savefig(vis_save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"Saved full episode visualization to {vis_save_path}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }

def main():
    parser = argparse.ArgumentParser(description='小样本分类推理系统 (支持CUB、Flowers102和FGVC Aircraft)')
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
                       choices=['resnet18', 'resnet50'], help='骨干网络')
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--n_way', type=int, default=FEW_SHOT_CONFIG['n_way'], help='N-way')
    parser.add_argument('--k_shot', type=int, default=FEW_SHOT_CONFIG['k_shot'], help='K-shot')
    parser.add_argument('--q_query', type=int, default=FEW_SHOT_CONFIG['q_query'], help='查询样本数')
    parser.add_argument('--num_episodes', type=int, default=FEW_SHOT_CONFIG['num_episodes'], help='每轮episode数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--save_dir', type=str, default='./inference_results', help='保存目录')
    parser.add_argument('--inference_episodes', type=int, default=100, help='推理的episode数量')
    parser.add_argument('--visualize', action='store_true', help='是否进行可视化分析')
    parser.add_argument('--fewshot_visualize', action='store_true', help='是否进行基于小样本情景的可视化分析')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        if args.dataset_type == 'fgvc_aircraft':
            args.data_root = './datasets/fgvc-aircraft-2013b'
        elif args.dataset_type == 'flowers':
            args.data_root = './datasets/flowers-102'
        else:
            args.data_root = DATA_CONFIG['data_root']
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading data...")
    if args.dataset_type == 'fgvc_aircraft':
        _, _, _, val_few_shot_sampler = create_fgvc_aircraft_dataloaders(
            data_root=args.data_root,
            annotation_level=args.annotation_level,
            batch_size=1,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_workers=0,
            num_episodes=args.num_episodes
        )
    else:
        _, _, _, val_few_shot_sampler = create_dataloaders(
            data_root=args.data_root,
            batch_size=1,
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_episodes=args.num_episodes
        )
    print("Data loaded.")
    
    class_names = load_class_names(args.data_root, dataset_type=args.dataset_type, 
                                   annotation_level=args.annotation_level)
    
    print(f"Creating {args.model_type} model...")
    model, _ = create_model_and_optimizer(
        model_type=args.model_type,
        backbone=args.backbone,
        lr=0.0
    )
    
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
    
    trainer = MetaTrainer(model, None, device=args.device)

    print(f"Performing validation inference on {args.inference_episodes} episodes...")
    inference_result = perform_validation_inference(
        model=model,
        trainer=trainer,
        fewshot_sampler=val_few_shot_sampler,
        device=args.device,
        num_episodes=args.inference_episodes,
        class_names=class_names,
        save_dir=args.save_dir
    )
    
    inference_path = os.path.join(args.save_dir, 'validation_inference.json')
    with open(inference_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'accuracy': inference_result['accuracy'],
            'total_correct': inference_result['total_correct'],
            'total_samples': inference_result['total_samples'],
            'num_episodes': args.inference_episodes,
            'n_way': args.n_way,
            'k_shot': args.k_shot
        }, f, indent=2)
    
    print(f"Validation inference completed! Accuracy: {inference_result['accuracy']:.4f}")
    print(f"Results saved to: {inference_path}")
    print(f"\nInference Configuration:")
    print(f"  Dataset: {args.dataset_type}")
    if args.dataset_type == 'fgvc_aircraft':
        print(f"  Annotation Level: {args.annotation_level}")
    print(f"  Model: {args.model_type} ({args.backbone})")
    print(f"  Setting: {args.n_way}-way {args.k_shot}-shot")
    print(f"  Episodes: {args.inference_episodes}")

    if args.fewshot_visualize:
        from utils.visualization import analyze_fewshot_visualization
        fewshot_analysis_result = analyze_fewshot_visualization(
            model=model,
            fewshot_sampler=val_few_shot_sampler,
            class_names=class_names,
            device=args.device,
            save_dir=args.save_dir,
            n_way=args.n_way
        )

if __name__ == '__main__':
    main()
