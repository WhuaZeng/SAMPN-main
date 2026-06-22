import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.scr import SCR, SelfCorrelationComputation  
from torchsummary import summary

class GatedResidualFusion(nn.Module):
    
    def __init__(self, feature_dim, reduction_ratio=16):
        super(GatedResidualFusion, self).__init__()
        self.feature_dim = feature_dim
        
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction_ratio, feature_dim),
            nn.Sigmoid()  
        )
        
        self.global_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, enhanced_features, identity_features):

        batch_size = enhanced_features.size(0)
        
        channel_gates = self.gate_network(enhanced_features) 
        channel_gates = channel_gates.view(batch_size, self.feature_dim, 1, 1)
        
        final_gate = self.global_gate * channel_gates + (1 - self.global_gate) * 0.5

        fused_features = final_gate * enhanced_features + (1 - final_gate) * identity_features
        
        return fused_features

class ProtoNet_SCR_CBAM(nn.Module):
    
    def __init__(self, backbone='resnet18', pretrained=True):
        super(ProtoNet_SCR_CBAM, self).__init__()
        
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        if backbone == 'resnet18':
            resnet = models.resnet18(weights=weights)

            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 512  
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048 

        for param in self.encoder.parameters():
            param.requires_grad = False

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
        
        self.scr_fusion = GatedResidualFusion(self.feature_dim)
        self.cbam_fusion = GatedResidualFusion(self.feature_dim)
        
        self.concat_feature_dim = self.feature_dim * 3  
        
        self.projection = nn.Sequential(
            nn.Linear(self.concat_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):

        with torch.no_grad():  
            features = self.encoder(x)
        
        identity = features
        
        cbam_features = self.cbam_module(features)
        
        fused_cbam_features = self.cbam_fusion(cbam_features, identity)

        corr_features = self.self_corr(features) 
        
        scr_features = self.scr_module(corr_features) 
        fused_scr_features = self.scr_fusion(scr_features, identity)

        if fused_scr_features.size()[2:] != identity.size()[2:]:
            fused_scr_features = F.interpolate(fused_scr_features, size=identity.size()[2:], mode='bilinear', align_corners=False)
        
        if fused_cbam_features.size()[2:] != identity.size()[2:]:
            fused_cbam_features = F.interpolate(fused_cbam_features, size=identity.size()[2:], mode='bilinear', align_corners=False)

        concatenated_features = torch.cat([identity, fused_scr_features, fused_cbam_features], dim=1)
        
        pooled_features = torch.mean(concatenated_features, dim=(2, 3))
        
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
    
    def compute_distances(self, query_embeddings, prototypes):

        distances = torch.cdist(query_embeddings, prototypes, p=2)
        return -distances
    
    def predict(self, support_embeddings, support_labels, query_embeddings):

        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        logits = self.compute_distances(query_embeddings, prototypes)
        return logits

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ProtoNet_SCR_CBAM().to(device)
    
    input_tensor = torch.randn(3, 3, 224, 224).to(device)
    output = model(input_tensor)
    
    assert output.shape == (3, 256), f"Expected (3, 256), got {output.shape}"

    support_embeddings = torch.randn(10, 256).to(device)
    support_labels = torch.randint(0, 5, (10,)).to(device)
    query_embeddings = torch.randn(20, 256).to(device)
    
    logits = model.predict(support_embeddings, support_labels, query_embeddings)
    print("Predicted logits shape:", logits.shape)
    print("Model created successfully!")
    summary(model, (3, 224, 224))