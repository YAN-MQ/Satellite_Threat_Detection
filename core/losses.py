"""
Loss Functions Module
====================
提供各类损失函数：Focal Loss、加权交叉熵等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡问题
    
    通过降低易分类样本的权重，使模型更关注难分类样本。
    
    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 类别权重
        gamma: 聚焦参数，默认为2.0
        reduction: 损失聚合方式
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失
    
    根据类别频率自动计算权重，用于处理类别不平衡问题。
    """
    
    def __init__(self, num_classes=4, beta=0.9999):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        
    def forward(self, outputs, targets):
        class_counts = torch.bincount(targets, minlength=self.num_classes).float()
        class_counts = class_counts.clamp(min=1)
        
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        
        return F.cross_entropy(outputs, targets, weight=weights)


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    
    标签平滑正则化，防止模型过度自信。
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class ProximalLoss(nn.Module):
    """
    近端损失 (Proximal Loss)
    
    用于FedProx算法，限制本地模型偏离全局模型。
    
    Args:
        mu: 正则化系数
    """
    
    def __init__(self, mu=0.01):
        super(ProximalLoss, self).__init__()
        self.mu = mu
    
    def forward(self, local_params, global_params):
        loss = 0.0
        for name in local_params:
            loss += torch.sum((local_params[name] - global_params[name]) ** 2)
        return self.mu * loss
