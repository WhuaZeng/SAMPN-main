import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CAMVisualizer:
    """Class Activation Map可视化器"""
    
    def __init__(self, model, target_layer_name='encoder.layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子函数"""
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层并注册钩子
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in model")
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, image_tensor, target_class=None):
        """生成CAM图"""
        self.model.eval()
        
        # 前向传播
        output = self.model(image_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 计算CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 上采样到原图尺寸
        cam = F.interpolate(cam, size=image_tensor.shape[1:], mode='bilinear', align_corners=False)
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().detach().cpu().numpy(), target_class

class GradCAM:
    """Grad-CAM可视化器"""
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = []
        self.activations = []
        
        # 自动查找合适的层
        if target_layer_name is None:
            self.target_layer_name = self._find_last_conv_layer()
            
        print(f"GradCAM using target layer: {self.target_layer_name}")
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        """自动查找最后一个卷积层"""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
                conv_layers.append(name)
        
        return conv_layers[-1] if conv_layers else 'encoder.layer4'  # 默认值
    
    def _register_hooks(self):
        """注册钩子函数"""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        # 找到目标层
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found")
            
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, image_tensor, target_class=None):
        """生成热力图"""
        self.model.eval()
        self.gradients.clear()
        self.activations.clear()
        
        # 前向传播
        output = self.model(image_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # 获取梯度和激活
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # 全局平均池化梯度
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # 加权激活图
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # 生成热力图
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        return heatmap.detach().cpu().numpy(), target_class

def overlay_heatmap_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """将热力图叠加到原始图像上"""
    # 转换为OpenCV格式
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)
    
    # 调整热力图尺寸
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    
    # 叠加热力图
    overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def analyze_model_attention(model, image_tensor, true_label, class_names=None):
    """分析模型注意力机制"""
    try:
        # 初始化可视化器
        cam_visualizer = CAMVisualizer(model)
        gradcam = GradCAM(model)
        
        # 生成不同类型的注意力图
        cam_map, cam_pred = cam_visualizer.generate_cam(image_tensor)
        gradcam_map, gradcam_pred = gradcam.generate_heatmap(image_tensor)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        original_img = image_tensor.permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # CAM
        axes[0, 1].imshow(cam_map, cmap='jet')
        axes[0, 1].set_title(f'CAM (Pred: {cam_pred})')
        axes[0, 1].axis('off')
        
        # Grad-CAM
        axes[0, 2].imshow(gradcam_map, cmap='jet')
        axes[0, 2].set_title(f'Grad-CAM (Pred: {gradcam_pred})')
        axes[0, 2].axis('off')
        
        # 叠加图
        cam_overlay = overlay_heatmap_on_image(image_tensor, cam_map)
        axes[1, 0].imshow(cam_overlay)
        axes[1, 0].set_title('CAM Overlay')
        axes[1, 0].axis('off')
        
        gradcam_overlay = overlay_heatmap_on_image(image_tensor, gradcam_map)
        axes[1, 1].imshow(gradcam_overlay)
        axes[1, 1].set_title('Grad-CAM Overlay')
        axes[1, 1].axis('off')
        
        # 预测概率分布
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze()
            top_probs, top_indices = torch.topk(probs, 5)
        
        # 显示top-5预测
        pred_text = ""
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = class_names[idx.item()] if class_names else f"Class {idx.item()}"
            pred_text += f"{i+1}. {class_name}: {prob.item():.3f}\n"
        
        axes[1, 2].text(0.1, 0.5, pred_text, fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Top-5 Predictions')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'cam_prediction': cam_pred,
            'gradcam_prediction': gradcam_pred,
            'true_label': true_label,
            'top_predictions': list(zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()))
        }
        
    except Exception as e:
        print(f"Error in attention analysis: {e}")
        # 返回基本分析结果
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze()
            pred = torch.argmax(probs).item()
        
        return {
            'prediction': pred,
            'true_label': true_label,
            'confidence': probs[pred].item(),
            'error': str(e)
        }

# 简化的评估和可视化脚本
def create_evaluation_report(model_path, data_root='./CUB_200_2011'):
    """创建训练结果报告"""
    import json
    from datasets import create_dataloaders
    from models import create_model_and_optimizer, MetaTrainer
    
    print("=== 训练完成报告 ===")
    
    # 加载模型
    model, _ = create_model_and_optimizer(model_type='protonet', backbone='resnet18')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 创建数据加载器进行测试
    _, val_loader, test_sampler = create_dataloaders(
        data_root=data_root,
        batch_size=16,
        n_way=5,
        k_shot=5,
        q_query=15,
        num_episodes=20
    )
    
    # 评估模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = MetaTrainer(model, None, device=device)
    final_accuracy = trainer.validate_meta_epoch(test_sampler, num_episodes=20)
    
    print(f"最终测试准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # 生成报告
    report = {
        'model_path': model_path,
        'final_accuracy': float(final_accuracy),
        'accuracy_percentage': float(final_accuracy * 100),
        'performance_rating': 'Excellent' if final_accuracy > 0.85 else 'Good' if final_accuracy > 0.7 else 'Fair'
    }
    
    # 保存报告
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("报告已保存到 training_report.json")
    return report

if __name__ == '__main__':
    # 使用示例
    print("可视化工具已准备就绪！")