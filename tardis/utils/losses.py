#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional

import torch
import torch.nn as nn


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid focal loss function

    References:
        10.1088/1742-6596/1229/1/012045

    Args:
        gamma (float): Gamma factor used in term1 of SFL.
        alpha (int, optional): Optional alpha factor used for normalizing SPF.
    """

    def __init__(self,
                 gamma=0.25,
                 alpha: Optional[int] = None):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape
                [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape
                [Batch x Channels x Length x Length].
        """
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

    Dice=2(A∩B)(A)+(B);
    where 'A∩B' represents the common elements between sets A and B
    'A' ann 'B' represents the number of elements in set A ans set B

    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                smooth=1e-16) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape
                [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape
                [Batch x Channels x Length x Length].
            smooth (float): Smooth factor to ensure  no 0 division.
        """
        logits = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Calculate dice loss
        intersection = (logits * targets).sum()
        dice = (2 * intersection + smooth) / (logits.square().sum() +
                                              targets.square().sum() +
                                              smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    DICE BCE COMBO LOSS FUNCTION
    """

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            inputs (torch.Tensor): Logits of a shape
                [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape
                [Batch x Channels x Length x Length].
        """
        bce_loss = self.bce(inputs=inputs,
                            targets=targets)

        dice_loss = self.dice(inputs=inputs,
                              targets=targets)
        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class BCELoss(nn.Module):
    """
    STANDARD BINARY CROSS-ENTROPY LOSS FUNCTION

    Args:
        reduction (str, optional): BCE reduction over batch type.
        weight (Optional[float], optional): Optional weigh factor for positive samples.
    """

    def __init__(self,
                 reduction='mean',
                 weight: Optional[float] = None,
                 diagonal=False):
        super(BCELoss, self).__init__()
        self.diagonal = diagonal
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight, reduction=reduction)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape
                [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape
                [Batch x Channels x Length x Length].
        """
        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)
            device = logits.get_device()
            if device == -1:
                device = 'cpu'

            eye = torch.eye(g_len, g_len, device=device)
            logits[:, :, g_range, g_range] = eye[g_range, g_range]
            targets[:, :, g_range, g_range] = eye[g_range, g_range]

        return self.loss(logits, targets)


class CELoss(nn.Module):
    """
    STANDARD CROSS-ENTROPY LOSS FUNCTION

    Args:
        reduction (str, optional):
            BCE reduction over batch type.
                Defaults to 'mean'.
    """

    def __init__(self,
                 reduction='mean'):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor):
                logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor):
                target of a shape [Batch x Channels x Length x Length]
        """

        return self.loss(logits, targets)


class AdaptiveDiceLoss(nn.Module):
    """
    ADAPTIVE DICE LOSS FUNCTION

    AdaptiveDice =  2∑[(1-p_i)^a * p_i] * g_i
                   --------------------------
                   ∑[(1-p_i)^a * p_i]  + ∑g_i^2

    Args:
        alpha (float):
            Optional alpha scaling factor.
                Defaults to 0.1.
    """

    def __init__(self,
                 alpha=0.1):
        super(AdaptiveDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth=1e-16) -> torch.Tensor:
        """
        Forward loos function

        Args:
            inputs (torch.Tensor):
                logits of a shape [Batch x Channels x Length x Length].
            targets (torch.Tensor):
                target of a shape [Batch x Channels x Length x Length].
            smooth (float):
                Smooth factor to ensure  no 0 division.
                    Defaults to 1e-16.
        """
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs = ((1 - inputs) ** self.alpha) * inputs

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.square().sum() +
                                               targets.square().sum() +
                                               smooth)

        return 1 - dice


class SoftF1:
    """
    SOFT F1 LOSS FUNCTION FOR EVALUATION

    Standard F1 loss function operating on probability's
    """

    def __init__(self):
        super(SoftF1, self).__init__()

    def __call__(self,
                 logits: torch.Tensor,
                 targets: torch.Tensor):
        """
        Forward loos function

        Args:
            logits (torch.Tensor):
                logits of a shape [Batch x Channels x Length x Length].
            targets (torch.Tensor):
                target of a shape [Batch x Channels x Length x Length].
        """
        with torch.no_grad():
            logits = torch.sigmoid(logits)
            logits = logits.flatten().cpu().detach().numpy()
            targets = targets.flatten().cpu().detach().numpy()

            tp = (logits * targets).sum()
            fp = (logits * (1 - targets)).sum()  # soft
            fn = ((1 - logits) * targets).sum()  # soft
            tn = ((1 - logits) * (1 - targets)).sum()  # soft

            """Accuracy Score"""
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

            """Precision Score - tp / (tp + fp)"""
            precision_score_1 = 1 - (tp / (tp + fp + 1e-8))
            precision_score_0 = 1 - (tn / (tn + fp + 1e-8))
            precision = 0.5 * (precision_score_1 + precision_score_0)

            """Recall Score - tp / (tp + tn)"""
            recall_score_1 = 1 - (tp / (tp + fn + 1e-8))
            recall_score_0 = 1 - (tn / (tn + fn + 1e-8))
            recall = 0.5 * (recall_score_1 + recall_score_0)

            """F1 Score """
            f1 = 2 * ((precision * recall) / (precision + recall + 1e-8))

        return accuracy, precision, recall, f1
