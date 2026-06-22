import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4,ResNet

class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.d = num_channel

        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.resolution = 25 

        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)

        if is_pretraining:
            self.num_cat = num_cat
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   
    
    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        
        if self.resnet:
            feature_map = feature_map/np.sqrt(640)
        
        return feature_map.view(batch_size,self.d,-1).permute(0,2,1).contiguous() 

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):

        reg = support.size(1)/support.size(2)
        
        lam = reg*alpha.exp()+1e-6

        rho = beta.exp()

        st = support.permute(0,2,1) 

        if Woodbury:

            sts = st.matmul(support) 
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() 
            hat = m_inv.matmul(sts) 
        else:

            sst = support.matmul(st)  
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse()  
            hat = st.matmul(m_inv).matmul(support)
        Q_bar = query.matmul(hat).mul(rho) 

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) 
        return dist

    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)

        support = feature_map[:way*shot].view(way, shot*resolution , d)
        query = feature_map[way*shot:].view(way*query_shot*resolution, d)

        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) 
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) 
        
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist

    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index
    
    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta)
        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1)
        
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def forward(self,inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
            
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction, support