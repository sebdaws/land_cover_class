import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score
)

class FocalLoss(nn.Module):
    """
    Implements Focal Loss for handling class imbalance problems.
    
    Focal Loss adds a modulating factor to Cross Entropy Loss to focus training
    on hard examples and down-weight easy ones.

    Args:
        alpha (float): Weighting factor, default is 1
        gamma (float): Focusing parameter, default is 2
        num_classes (int): Number of classes in the dataset, default is 19
    """
    def __init__(self, alpha=1, gamma=2, num_classes=19):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, labels):
        """
        Compute the focal loss.

        Args:
            logits (torch.Tensor): Raw predictions from the model (B, C)
            labels (torch.Tensor): Ground truth labels (B,)

        Returns:
            torch.Tensor: Computed focal loss
        """
        labels_one_hot = torch.eye(self.num_classes)[labels].to(logits.device)
        probs = F.softmax(logits, dim=1)
        log_p = torch.log(probs)
        loss = -self.alpha * (1 - probs) ** self.gamma * log_p
        loss = torch.sum(loss * labels_one_hot, dim=1)
        return loss.mean()
    
class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss implementation.
    
    Measures the divergence between predicted probability distributions
    and ground truth distributions.

    Args:
        num_classes (int): Number of classes in the dataset
        reduction (str): Specifies the reduction to apply to the output, default is 'batchmean'
    """
    def __init__(self, num_classes, reduction='batchmean'):
        super(KLDivergenceLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=self.reduction)
    
    def forward(self, logits, labels):
        """
        Compute the KL divergence loss.

        Args:
            logits (torch.Tensor): Raw predictions from the model (B, C)
            labels (torch.Tensor): Ground truth labels (B,)

        Returns:
            torch.Tensor: Computed KL divergence loss
        """
        log_probs = F.log_softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = self.kl_loss(log_probs, labels_one_hot)

        return loss

def dice_loss(preds, labels, smooth=1e-6):
    """
    Compute Dice Loss between predictions and ground truth.
    
    Dice Loss is useful for handling imbalanced segmentation tasks.

    Args:
        preds (torch.Tensor): Predicted probabilities
        labels (torch.Tensor): Ground truth labels
        smooth (float): Smoothing factor to avoid division by zero, default is 1e-6

    Returns:
        torch.Tensor: Computed dice loss
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def get_loss_func(args, num_classes, class_weights, device):
    """
    Factory function to get the specified loss function.

    Args:
        args: Arguments containing loss_func and weights_smooth parameters
        num_classes (int): Number of classes in the dataset
        class_weights (pandas.Series): Weights for each class
        device (torch.device): Device to put the loss function on

    Returns:
        nn.Module: The specified loss function

    Raises:
        ValueError: If the specified loss function is not recognized
    """
    if args.loss_func == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_func == 'weighted_cross_entropy':
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
        smoothed_weights = class_weights + args.weights_smooth
        criterion = nn.CrossEntropyLoss(weight=smoothed_weights)
    elif args.loss_func == 'focal':
        criterion = FocalLoss(num_classes=num_classes)
    elif args.loss_func == 'dice':
        criterion = dice_loss()
    elif args.loss_func == 'kl_div':
        criterion = KLDivergenceLoss(num_classes=num_classes)
    else:
        raise ValueError('Loss function not recognised')
    return criterion

def calculate_metrics(y_true, y_pred, phase='train'):
    """
    Calculate various classification metrics.

    Computes precision, recall, F1 score, and accuracy for the predictions.

    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        phase (str): Current phase ('train' or 'val' or 'test'), default is 'train'

    Returns:
        dict: Dictionary containing computed metrics:
            - 'Precision': Weighted precision score
            - 'Recall': Weighted recall score
            - 'F1': Weighted F1 score
            - 'Accuracy': Overall accuracy
    """
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}