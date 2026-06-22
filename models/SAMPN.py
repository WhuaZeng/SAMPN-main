import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .scr import SCR, SelfCorrelationComputation 
from .cbam import CBAM  
from torchsummary import summary 
from math import sqrt

class RMSNorm(nn.Module):
    """支持自定义维度的RMSNorm归一化层"""
    def __init__(self, dim: int, eps: float = 1e-8, norm_dim: int = -1):
        super().__init__()
        self.eps = eps
        self.norm_dim = norm_dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=self.norm_dim, keepdim=True) + self.eps)
        shape = [1] * x.dim()
        shape[self.norm_dim] = -1
        weight = self.weight.reshape(shape)
        return norm * weight

class AttnResFusion(nn.Module):

    def __init__(self, feature_dim, num_sources=3):
        super(AttnResFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_sources = num_sources
        
        self.Q_pseudo_queries = nn.Parameter(torch.randn(num_sources, feature_dim))

        self.source_norms = nn.ModuleList([
            RMSNorm(feature_dim, norm_dim=-1) for _ in range(num_sources)
        ])
        
        nn.init.xavier_uniform_(self.Q_pseudo_queries)

    def forward(self, *features):

        assert len(features) == self.num_sources, f"Expected {self.num_sources} inputs, got {len(features)}"
        
        if features[0].dim() == 4:

            B, C, H, W = features[0].shape
            
            feat_seqs = [f.flatten(2).transpose(1, 2) for f in features] 
            
            stacked_feats = torch.stack(feat_seqs, dim=2) 

            K_V_stacked = torch.stack([
                norm(feat_seq) for norm, feat_seq in zip(self.source_norms, feat_seqs)
            ], dim=2)
            
            Q = self.Q_pseudo_queries

            logits = torch.einsum('sc, btsc -> bts', Q, K_V_stacked)

            attn_weights = F.softmax(logits, dim=2) 
            
            fused_seq = torch.sum(attn_weights.unsqueeze(-1) * K_V_stacked, dim=2) 

            fused_features = fused_seq.transpose(1, 2).reshape(B, C, H, W)
            
        else:

            stacked_feats = torch.stack(features, dim=1) 
            
            K_V_stacked = torch.stack([
                norm(feat) for norm, feat in zip(self.source_norms, features)
            ], dim=1) 
            
            Q = self.Q_pseudo_queries
            
            logits = torch.einsum('sc, bsc -> bs', Q, K_V_stacked) 
            
            attn_weights = F.softmax(logits, dim=1) 

            fused_features = torch.sum(attn_weights.unsqueeze(-1) * K_V_stacked, dim=1) 
            
        return fused_features

class MahalanobisMetric(nn.Module):
    def __init__(self, feature_dim):
        super(MahalanobisMetric, self).__init__()
                            
        self.tril = nn.Parameter(torch.zeros(feature_dim, feature_dim))
                     
        with torch.no_grad():
            self.tril.data = torch.tril(torch.randn(feature_dim, feature_dim))
                     
            diag_idx = torch.arange(feature_dim)
            self.tril.data[diag_idx, diag_idx] =  0.1 + torch.abs(torch.randn(feature_dim)) * 0.01
    
    def forward(self, query_embeddings, prototypes):
        n_query = query_embeddings.size(0)
        n_way = prototypes.size(0)
        
        L = torch.tril(self.tril)
        
        M = L @ L.t()          
        
        diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)                       
        
        M_diff = torch.einsum('nwd,de->nwe', diff, M)                       
        dist_sq = torch.einsum('nwd,nwd->nw', M_diff, diff)                    
        
        return -dist_sq
class ProtoNet_SCR_CBAM_AttnRes_MultiLevel(nn.Module):
    
    def __init__(self, backbone='resnet18', pretrained=True, use_low_level=True):
        super(ProtoNet_SCR_CBAM_AttnRes_MultiLevel, self).__init__()
        
        self.use_low_level = use_low_level
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=weights)
            self.feature_dim = 512  

            self.layer_channels = {
                'layer1': 64,
                'layer2': 128,
                'layer3': 256,
                'layer4': 512
            }
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=weights)
            self.feature_dim = 2048
            self.layer_channels = {
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024,
                'layer4': 2048
            }
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

        for param in self.parameters():
            param.requires_grad = False

        if self.use_low_level:
            self.low_level_projections = nn.ModuleDict({
                'layer1': nn.Conv2d(self.layer_channels['layer1'], self.feature_dim, kernel_size=1, bias=False),
                'layer2': nn.Conv2d(self.layer_channels['layer2'], self.feature_dim, kernel_size=1, bias=False),
                'layer3': nn.Conv2d(self.layer_channels['layer3'], self.feature_dim, kernel_size=1, bias=False),
            })

            for m in self.low_level_projections.values():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.self_corr = SelfCorrelationComputation(kernel_size=(5, 5), padding=2)
        
        mid_channels = max(32, self.feature_dim // 8)

        self.scr_module = SCR(
            planes=[self.feature_dim, mid_channels, mid_channels, mid_channels, self.feature_dim],
            stride=(1, 1, 1),
            ksize=3,
            do_padding=False,
            bias=False
        )
        
        self.cbam_module = CBAM(in_planes=self.feature_dim)
        
        num_sources = 3 
        if self.use_low_level:
            num_sources += 3
            
        self.attn_fusion = AttnResFusion(feature_dim=self.feature_dim, num_sources=num_sources)
        
        self.concat_feature_dim = self.feature_dim 

        self.embed_dim = 256
        self.projection = nn.Sequential(
            nn.Linear(self.concat_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.embed_dim)
        )
        
        self.metric_module = MahalanobisMetric(feature_dim=self.embed_dim)
        
    def _align_and_project(self, feat, target_h, target_w, proj_layer=None):
        if feat.size()[2:] != (target_h, target_w):
            feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if proj_layer is not None:
            feat = proj_layer(feat)
        return feat

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        identity = l4
        B, C, H, W = identity.shape
        
        fusion_inputs = []
        
        if self.use_low_level:

            l1_aligned = self._align_and_project(l1, H, W, self.low_level_projections['layer1'])
            fusion_inputs.append(l1_aligned)
            
            l2_aligned = self._align_and_project(l2, H, W, self.low_level_projections['layer2'])
            fusion_inputs.append(l2_aligned)
            
            l3_aligned = self._align_and_project(l3, H, W, self.low_level_projections['layer3'])
            fusion_inputs.append(l3_aligned)
        
        fusion_inputs.append(identity)
        
        cbam_features = self.cbam_module(identity)
        if cbam_features.size()[2:] != (H, W):
            cbam_features = F.interpolate(cbam_features, size=(H, W), mode='bilinear', align_corners=False)
        fusion_inputs.append(cbam_features)
        
        corr_features = self.self_corr(identity)
        scr_features = self.scr_module(corr_features)
        if scr_features.size()[2:] != (H, W):
            scr_features = F.interpolate(scr_features, size=(H, W), mode='bilinear', align_corners=False)
        fusion_inputs.append(scr_features)
        
        fused_features = self.attn_fusion(*fusion_inputs)
        
        pooled_features = torch.mean(fused_features, dim=(2, 3))

        embeddings = self.projection(pooled_features)
        
        return embeddings
    
    def compute_prototypes(self, support_embeddings, support_labels):
        n_way = len(torch.unique(support_labels))
        prototypes = torch.zeros(n_way, support_embeddings.size(1)).to(support_embeddings.device)
        
        for i in range(n_way):
            class_mask = (support_labels == i)
            if class_mask.sum() > 0:
                prototypes[i] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def predict(self, support_embeddings, support_labels, query_embeddings):
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        
        logits = self.metric_module(query_embeddings, prototypes)
        return logits
    
class SAMPN(ProtoNet_SCR_CBAM_AttnRes_MultiLevel):                                                          
    """Dual-Attention Metric Fusion Network for Few-Shot Learning"""
    
    def __init__(self, backbone='resnet18', pretrained=True, use_low_level=True):
        super(SAMPN, self).__init__(
            backbone=backbone, 
            pretrained=pretrained, 
            use_low_level=use_low_level
        )
    
    def forward(self, x):
        """前向传播 - Dual-Attention Metric Fusion"""
        return super().forward(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SAMPN(backbone='resnet18', pretrained=True, use_low_level=True).to(device)
    
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    output = model(input_tensor)
    
    print("Output shape:", output.shape)
    assert output.shape == (2, 256), f"Expected (2, 256), got {output.shape}"
    
    n_way = 5
    k_shot = 1
    q_query = 10
    
    support_embeddings = torch.randn(n_way * k_shot, 256).to(device)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot).to(device)
    query_embeddings = torch.randn(q_query * n_way, 256).to(device)
    
    logits = model.predict(support_embeddings, support_labels, query_embeddings)
    
    print("Predicted logits shape:", logits.shape)
    assert logits.shape == (q_query * n_way, n_way)
    
    print("Model with Multi-Level AttnRes Fusion and Mahalanobis Metric created successfully!")
    summary(model, (3, 224, 224)) 