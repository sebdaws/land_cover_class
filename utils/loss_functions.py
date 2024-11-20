import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=19):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, labels):
        labels_one_hot = torch.eye(self.num_classes)[labels].to(logits.device)
        probs = F.softmax(logits, dim=1)
        log_p = torch.log(probs)
        loss = -self.alpha * (1 - probs) ** self.gamma * log_p
        loss = torch.sum(loss * labels_one_hot, dim=1)
        return loss.mean()
    
class KLDivergenceLoss(nn.Module):
    def __init__(self, num_classes, reduction='batchmean'):
        super(KLDivergenceLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=self.reduction)
    
    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = self.kl_loss(log_probs, labels_one_hot)

        return loss

# Dice Loss
def dice_loss(preds, labels, smooth=1e-6):
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice