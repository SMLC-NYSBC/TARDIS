from typing import Optional

import torch
import torch.nn as nn


class SigmoidFocalLoss(nn.Module):
    """
    SIGMOID FOCAL LOSS FUNCTION
    As in: doi:10.1088/1742-6596/1229/1/012045

    Input:
        logits of a shape [Batch x Channels x Length x Length]
        target of a shape [Batch x Channels x Length x Length]
    """

    def __init__(self,
                 gamma=0.25,
                 alpha: Optional[int] = None):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor):
        y = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        term1 = (1 - p) ** self.gamma * torch.log(p)
        term2 = p ** self.gamma * torch.log(1 - p)

        if self.alpha is None:
            # SFL(p, y) = -1/n(Σ[y(1-p)^γ*log(p) + (1-y)*p^γ*log(1-p)])
            sfl = torch.mean(y * term1 + ((1 - y) * term2))
            sfl = -(1 / logits.size()[0]) * sfl
        else:
            # SFL(p, y) = -1/n(Σ[α*y(1-p)^γ*log(p) + (1 - α)(1-y)*p^γ*log(1-p)])
            sfl = torch.mean(self.alpha * y * term1 + (1 - self.alpha) * (1 - y) * term2)

        return sfl


class DiceLoss(nn.Module):
    """
    Dice coefficient loss function.
        Dice=2|A∩B||A|+|B| ;
        where |A∩B| represents the common elements between sets A and B
        |A| ann |B| represents the number of elements in set A ans set B
    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                smooth=1):
        logits = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Calculate dice loss
        intersection = (logits * targets).sum()
        dice = (2. * intersection + smooth) / (logits.sum() + targets.sum() + smooth)

        return 1 - dice


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor):

        return self.loss(logits, targets)


class SoftF1(nn.Module):
    def __init__(self,
                 grad: bool):
        super(SoftF1, self).__init__()
        self.grad = grad

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor):
        if self.grad:
            logits = logits.flatten()
            targets = targets.flatten()

            tp = (logits * targets).sum(dim=0)
            fp = (logits * (1 - targets)).sum(dim=0)  # soft
            fn = ((1 - logits) * targets).sum(dim=0)  # soft
            tn = ((1 - logits) * (1 - targets)).sum(dim=0)  # soft

            soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

            cost_class1 = 1 - soft_f1_class1
            cost_class0 = 1 - soft_f1_class0

            return 0.5 * (cost_class1 + cost_class0)
        else:
            with torch.no_grad():
                logits = logits.flatten()
                targets = targets.flatten()

                tp = (logits * targets).sum(dim=0)
                fp = (logits * (1 - targets)).sum(dim=0)  # soft
                fn = ((1 - logits) * targets).sum(dim=0)  # soft
                tn = ((1 - logits) * (1 - targets)).sum(dim=0)  # soft

                """Accuracy Score"""
                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                """Precision Score - tp / (tp + fp)"""
                precision_score_1 = 1 - (tp / (tp + fp + 1e-8))
                precision_score_0 = 1 - (tn / (tn + fp + 1e-8))
                precision = 0.5 * (precision_score_1 + precision_score_0)

                """Recall Score - tp / (tp + tn)"""
                recall_score_1 = 1 - (tp / (tp + fn + 1e-8))
                recall_score_2 = 1 - (tn / (tn + fn + 1e-8))
                recall = 0.5 * (recall_score_1 + recall_score_2)

                """F1 Score """
                soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
                soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

                cost_class1 = 1 - soft_f1_class1
                cost_class0 = 1 - soft_f1_class0
                f1 = 0.5 * (cost_class1 + cost_class0)

            return accuracy.cpu().detach().numpy(), precision.cpu().detach().numpy(), \
                recall.cpu().detach().numpy(), f1.cpu().detach().numpy()
