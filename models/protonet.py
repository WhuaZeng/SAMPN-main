import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class ProtoNet(nn.Module):
    
    def __init__(self, backbone='resnet18', pretrained=True):
        super(ProtoNet, self).__init__()
        
        if backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'resnet12':
            self.encoder = models.resnet12(pretrained=pretrained)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'conv4':
            self.encoder = Conv4(feature_dim=64)
            self.feature_dim = self.encoder.flatten_dim        

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):

        features = self.encoder(x) 
        embeddings = self.projection(features) 

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
class Conv4(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = feature_dim * 5 * 5 

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ProtoNet().to(device)
    
    input_tensor = torch.randn(3, 3, 225, 225).to(device)
    model.forward(input_tensor)

    summary(model, (3, 225, 225))

    support_embeddings = torch.randn(10, 256).to(device)
    support_labels = torch.randint(0, 5, (10,)).to(device)
    query_embeddings = torch.randn(20, 256).to(device)

    logits = model.predict(support_embeddings, support_labels, query_embeddings)

    print("Predicted logits:", logits)