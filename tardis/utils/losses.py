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
import torch.nn.functional as F


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
                 alpha=0.1,
                 diagonal=False):
        super(AdaptiveDiceLoss, self).__init__()
        self.alpha = alpha
        self.diagonal = diagonal

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                smooth=1e-16) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor):
                logits of a shape [Batch x Channels x Length x Length].
            targets (torch.Tensor):
                target of a shape [Batch x Channels x Length x Length].
            smooth (float):
                Smooth factor to ensure  no 0 division.
                    Defaults to 1e-16.
        """
        logits = torch.sigmoid(logits)

        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        logits = logits.view(-1)
        targets = targets.view(-1)

        logits = ((1 - logits) ** self.alpha) * logits

        intersection = (logits * targets).sum()
        dice = (2. * intersection + smooth) / (logits.square().sum() +
                                               targets.square().sum() +
                                               smooth)

        return 1 - dice


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

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        return self.loss(logits, targets)


class BCEDiceLoss(nn.Module):
    """
    DICE BCE COMBO LOSS FUNCTION
    """

    def __init__(self,
                 diagonal=False):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss(diagonal=diagonal)
        self.dice = DiceLoss(diagonal=diagonal)
        self.diagonal = diagonal

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

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        bce_loss = self.bce(logits, targets)

        dice_loss = self.dice(logits, targets)

        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class CELoss(nn.Module):
    """
    STANDARD CROSS-ENTROPY LOSS FUNCTION

    Args:
        reduction (str, optional):
            BCE reduction over batch type.
                Defaults to 'mean'.
    """

    def __init__(self,
                 reduction='mean',
                 diagonal=False):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        self.diagonal = diagonal

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
        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        return self.loss(logits, targets)


class ClDice(nn.Module):
    """
    Soft skeletonization with DICE loss function
    """
    def __init__(self,
                 _iter=3,
                 smooth=1.0,
                 alpha=0.5,
                 diagonal=False):
        super(ClDice, self).__init__()

        self.iter = _iter
        self.alpha = alpha
        self.smooth = smooth
        self.diagonal = diagonal

        self.soft_dice = DiceLoss(diagonal=self.diagonal)

    @staticmethod
    def _soft_erode(binary_mask):
        if len(binary_mask.shape) == 4:  # 2D
            p1 = -F.max_pool2d(-binary_mask, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-binary_mask, (1, 3), (1, 1), (0, 1))

            return torch.min(p1, p2)
        elif len(binary_mask.shape) == 5:  # 3D
            p1 = -F.max_pool3d(-binary_mask, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-binary_mask, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-binary_mask, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    @staticmethod
    def _soft_dilate(binary_mask):
        if len(binary_mask.shape) == 4:  # 2D
            return F.max_pool2d(binary_mask, (3, 3), (1, 1), (1, 1))
        elif len(binary_mask.shape) == 5:  # 3D
            return F.max_pool3d(binary_mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def _soft_open(self,
                   binary_mask):
        return self._soft_dilate(self._soft_erode(binary_mask))

    def soft_skel(self,
                  binary_mask: torch.Tensor,
                  iter_: int):
        if not isinstance(iter_, int):
            iter_ = int(iter_)

        binary_mask_open = self._soft_open(binary_mask)
        t_skeleton = F.relu(binary_mask - binary_mask_open)

        for j in range(iter_):
            binary_mask = self._soft_erode(binary_mask)
            binary_mask_open = self._soft_open(binary_mask)

            delta = F.relu(binary_mask - binary_mask_open)

        return t_skeleton + F.relu(delta - t_skeleton * delta)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        dice = self.soft_dice(logits, targets)
        sk_logits = self.soft_skel(logits, self.iter)
        sk_targets = self.soft_skel(targets, self.iter)

        t_prec = ((sk_logits * targets).sum() + self.smooth) / (
                    sk_logits.sum() + self.smooth)
        t_sens = ((sk_targets * logits).sum() + self.smooth) / (
                    sk_targets.sum() + self.smooth)

        cl_dice = 2 * (t_prec * t_sens) / (t_prec + t_sens)

        return ((1 - self.alpha) * dice) + (self.alpha * (1 - cl_dice))


class DiceLoss(nn.Module):
    """
    Dice coefficient loss function.

    Dice=2(A∩B)(A)+(B);
    where 'A∩B' represents the common elements between sets A and B
    'A' ann 'B' represents the number of elements in set A ans set B

    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self,
                 diagonal=False):
        super(DiceLoss, self).__init__()
        self.diagonal = diagonal

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
        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

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


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid focal loss function

    References: 10.1088/1742-6596/1229/1/012045

    Args:
        gamma (float): Gamma factor used in term1 of SFL.
        alpha (int, optional): Optional alpha factor used for normalizing SPF.
    """

    def __init__(self,
                 gamma=0.25,
                 alpha: Optional[int] = None,
                 diagonal=False):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.diagonal = diagonal

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

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

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
