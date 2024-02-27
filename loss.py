import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity

# Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# Self-supervised Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=0.1):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, z_i, z_j):
        sim = cosine_similarity(z_i, z_j, dim=-1) # (cells)
        sim /= self.tau
        exp_sim = torch.exp(sim) # (cells)
        
        positive_samples = torch.diag(exp_sim, diagonal=0) # (cells, cells)
        negative_samples = exp_sim.sum(dim=0) - positive_samples # (cells, cells)
        loss = -torch.log(positive_samples / (positive_samples + negative_samples)).mean()
        return loss