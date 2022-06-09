import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self,
                 alpha=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = AdaptiveDiceLoss(alpha=alpha)

    def forward(self,
                inputs: torch.Tensor,
                targets: torch. Tensor):
        bce_loss = self.bce(inputs=inputs,
                            targets=targets)

        dice_loss = self.dice(inputs=inputs,
                              targets=targets)
        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor):
        BCE = self.loss(inputs, targets)

        return BCE


class DiceLoss(nn.Module):
    """
    Dice coefficient loss function
        Dice=2|A∩B||A|+|B| ; where
        |A∩B| represents the common elements between sets A and B
        |A| ann |B| represents the number of elements in set A ans set B
    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self,
                 smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self,
                inputs,
                targets):
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate dice loss
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (inputs.square().sum() + targets.square().sum() + self.smooth)

        return 1 - dice


class AdaptiveDiceLoss(nn.Module):
    """
    AdaptiveDice =  2∑[(1-p_i)^a * p_i] * g_i
                   --------------------------
                   ∑[(1-p_i)^a * p_i]  + ∑g_i^2
    Args:
        alpha: Scaling factor
        smooth: Smooth factor to remove division by 0
    """

    def __init__(self,
                 alpha=0.1,
                 smooth=1e-4):
        super(AdaptiveDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self,
                inputs,
                targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs = ((1 - inputs) ** self.alpha) * inputs

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
            (inputs.square().sum() + targets.square().sum() + self.smooth)

        return 1 - dice
