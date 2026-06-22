import torch.nn as nn
from torch.optim import Adam
from .protonet import ProtoNet
                        
from .protonet_scr import ProtoNet_SCR
from .protonet_scr_cbam import ProtoNet_SCR_CBAM
from .Relation_AttnRest import ProtoNet_SCR_CBAM_AttnRes_MultiLevel
from .SAMPN import SAMPN

def create_model_and_optimizer(model_type='protonet', backbone='resnet18', lr=0.001):
    """创建模型和优化器"""
    if model_type == 'protonet':
        model = ProtoNet(backbone=backbone)
    elif model_type == 'maml':
        model = MAML(backbone=backbone)
    elif model_type == 'protonet_scr':
        model = ProtoNet_SCR(backbone=backbone)
    elif model_type == 'protonet_scr_cbam':
        model = ProtoNet_SCR_CBAM(backbone=backbone)
    elif model_type == 'relation_attnres':
        model = ProtoNet_SCR_CBAM_AttnRes_MultiLevel(backbone=backbone)
    elif model_type == 'sampn':
        model = SAMPN(backbone=backbone)   
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    return model, optimizer