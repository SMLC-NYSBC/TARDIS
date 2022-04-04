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
        dice = (2. * intersection + smooth) / \
            (logits.sum() + targets.sum() + smooth)

        return 1 - dice


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor):
        BCE = self.loss(logits, targets)

        return BCE
