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
    """

    def __init__(self, alpha=0.1, smooth=1e-16, diagonal=False):
        """
        Loss initialization

        Args:
            alpha (float):  Optional alpha scaling factor.
            smooth (float): Smooth factor to ensure  no 0 division.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(AdaptiveDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.diagonal = diagonal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): target of a shape [Batch x Channels x Length x Length]
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
        dice = (2.0 * intersection + self.smooth) / (
            logits.square().sum() + targets.square().sum() + self.smooth
        )

        return 1 - dice


class BCELoss(nn.Module):
    """
    STANDARD BINARY CROSS-ENTROPY LOSS FUNCTION
    """

    def __init__(self, reduction="mean", diagonal=False):
        """
        Loss initialization

        Args:
            reduction (str, optional): BCE reduction over batch type.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(BCELoss, self).__init__()
        self.diagonal = diagonal
        self.reduction = reduction

        self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        mask = None

        if self.diagonal:
            g_range = range(logits.shape[2])

            mask = torch.ones_like(targets)
            mask[:, g_range, g_range] = 0.

        if mask is not None:
            self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, weight=mask)

        return self.loss(logits, targets)


class WBCELoss(nn.Module):
    """
    Weighted BINARY CROSS-ENTROPY LOSS FUNCTION
    """
    def __init__(self, reduction="mean", diagonal=False):
        """
        Loss initialization

        Args:
            reduction (str, optional): BCE reduction over batch type.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(WBCELoss, self).__init__()
        self.reduction = reduction
        self.diagonal = diagonal

        assert self.reduction in ['sum', 'mean', 'none']

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        if self.diagonal:
            g_range = range(logits.shape[1])

            logits[:, g_range, g_range] = 1

        # Compute the percentage of positive samples
        positive_samples = torch.sum(targets)
        total_samples = targets.numel()
        positive_ratio = positive_samples / total_samples

        # Calculate class weights
        weight_positive = 1 / positive_ratio
        weight_negative = 1 / (1 - positive_ratio)

        # Apply sigmoid function to the predictions
        y_pred = torch.sigmoid(logits)

        # Compute the binary cross entropy (BCE) loss
        bce_loss = -((weight_positive * targets * torch.log(y_pred + 1e-8)) +
                     (weight_negative * (1 - targets) * torch.log(1 - y_pred + 1e-8)))

        # Average the losses across all samples
        if self.reduction == 'sum':
            return torch.sum(bce_loss)
        elif self.reduction == 'mean':
            return torch.mean(bce_loss)
        else:
            return bce_loss


class BCEDiceLoss(nn.Module):
    """
    DICE BCE COMBO LOSS FUNCTION
    """

    def __init__(self, diagonal=False):
        """
        Loss initialization

        Args:
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss(diagonal=diagonal)
        self.dice = DiceLoss(diagonal=diagonal)
        self.diagonal = diagonal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        if self.diagonal:
            g_range = range(logits.shape[2])

            mask = torch.ones_like(targets)
            mask[:, g_range, g_range] = 0

            self.bce = nn.BCEWithLogitsLoss(weight=mask.float())

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class CELoss(nn.Module):
    """
    STANDARD CROSS-ENTROPY LOSS FUNCTION
    """

    def __init__(self, reduction="mean", diagonal=False):
        """
        Loss initialization

        Args:
            reduction (str): CE reduction over batch type.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(CELoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        self.diagonal = diagonal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): target of a shape [Batch x Channels x Length x Length]
        """
        logits = torch.sigmoid(logits)

        return self.loss(logits, targets)


class DiceLoss(nn.Module):
    """
    Dice coefficient loss function.

    Dice=2(A∩B)(A)+(B);
    where 'A∩B' represents the common elements between sets A and B
    'A' ann 'B' represents the number of elements in set A ans set B

    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self, smooth=1e-16, diagonal=False):
        """
        Loss initialization

        Args:
            diagonal (bool): If True, remove diagonal axis for graph prediction.
            smooth (float): Smooth factor to ensure  no 0 division.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.diagonal = diagonal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        logits = torch.sigmoid(logits)

        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        # Flatten label and prediction tensors
        logits = logits.view(-1)
        targets = targets.view(-1)

        # Calculate dice loss
        intersection = (logits * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            logits.square().sum() + targets.square().sum() + self.smooth
        )

        return 1 - dice


class SoftSkeletonization(nn.Module):
    """
    General soft skeletonization with DICE loss function
    """

    def __init__(self, _iter=5, smooth=1e-16, diagonal=False):
        """
        Loss initialization

        Args:
            _iter (int): Number of skeletonization iterations
            smooth (float): Smoothness factor.
            alpha (float): Alpha loss reduction.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(SoftSkeletonization, self).__init__()

        self.iter = _iter
        self.smooth = smooth
        self.diagonal = diagonal

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

    def _soft_open(self, binary_mask):
        return self._soft_dilate(self._soft_erode(binary_mask))

    def soft_skel(self, binary_mask: torch.Tensor, iter_: int) -> torch.Tensor:
        """
        Soft skeletonization

        Args:
            binary_mask: Binary target mask.
            iter_: Number of iterations for erosion.

        Returns:
            torch.Tensor: Skeleton on-hot mask
        """
        if not isinstance(iter_, int):
            iter_ = int(iter_)

        binary_mask_open = self._soft_open(binary_mask)
        s_skeleton = F.relu(binary_mask - binary_mask_open)

        for j in range(iter_):
            binary_mask = self._soft_erode(binary_mask)
            binary_mask_open = self._soft_open(binary_mask)

            delta = F.relu(binary_mask - binary_mask_open)
            s_skeleton = s_skeleton + F.relu(delta - (s_skeleton * delta))

        return s_skeleton

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        pass


class ClBCE(SoftSkeletonization):
    """
    Soft skeletonization with BCE loss function
    """

    def __init__(self, **kwargs):
        super(ClBCE, self).__init__(**kwargs)
        self.bce = BCELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        # BCE with activation
        bce = self.bce(logits, targets)

        # Activation
        logits = torch.sigmoid(logits)

        # Soft skeletonization
        sk_logits = self.soft_skel(logits, self.iter)
        sk_targets = self.soft_skel(targets, self.iter)

        t_prec = ((sk_logits * targets).sum() + self.smooth) / (
            sk_logits.sum() + self.smooth
        )
        t_sens = ((sk_targets * logits).sum() + self.smooth) / (
            sk_targets.sum() + self.smooth
        )

        # ClBCE
        cl_bce = 2 * (t_prec * t_sens) / (t_prec + t_sens)

        return bce + (1 - cl_bce)


class ClDice(SoftSkeletonization):
    """
    Soft skeletonization with DICE loss function
    """

    def __init__(self, **kwargs):
        super(ClDice, self).__init__(**kwargs)
        self.soft_dice = DiceLoss(diagonal=self.diagonal)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length].
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        # Dice loss with activation
        dice = self.soft_dice(logits, targets)

        # Activation
        logits = torch.sigmoid(logits)

        # Soft skeletonization
        sk_logits = self.soft_skel(logits, self.iter)
        sk_targets = self.soft_skel(targets, self.iter)

        t_prec = ((sk_logits * targets).sum() + self.smooth) / (
            sk_logits.sum() + self.smooth
        )
        t_sens = ((sk_targets * logits).sum() + self.smooth) / (
            sk_targets.sum() + self.smooth
        )

        # CLDice loss
        cl_dice = 2 * (t_prec * t_sens) / (t_prec + t_sens)

        return dice + (1 - cl_dice)


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid focal loss function

    References: 10.1088/1742-6596/1229/1/012045

    Args:
        gamma (float): Gamma factor used in term1 of SFL.
        alpha (int, optional): Optional alpha factor used for normalizing SPF.
    """

    def __init__(self, gamma=0.25, alpha: Optional[int] = None, diagonal=False):
        """
        Loss initialization

        Args:
            gamma (float): Gamma factor used in term1 of SFL.
            alpha (int, optional): Optional alpha factor used for normalizing SPF.
            diagonal (bool): If True, remove diagonal axis for graph prediction.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.diagonal = diagonal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward loos function

        Args:
            logits (torch.Tensor): Logits of a shape [Batch x Channels x Length x Length]
            targets (torch.Tensor): Target of a shape [Batch x Channels x Length x Length]
        """
        p = torch.sigmoid(logits)

        if self.diagonal:
            g_len = logits.shape[2]
            g_range = range(g_len)

            logits[:, g_range, g_range] = 1
            targets[:, g_range, g_range] = 1

        y = targets.unsqueeze(1)
        term1 = (1 - p) ** self.gamma * torch.log(p)
        term2 = p**self.gamma * torch.log(1 - p)

        if self.alpha is None:
            # SFL(p, y) = -1/n(Σ[y(1-p)^γ*log(p) + (1-y)*p^γ*log(1-p)])
            sfl = torch.mean(y * term1 + ((1 - y) * term2))
            sfl = -(1 / logits.size()[0]) * sfl
        else:
            # SFL(p, y) = -1/n(Σ[α*y(1-p)^γ*log(p) + (1 - α)(1-y)*p^γ*log(1-p)])
            sfl = torch.mean(
                self.alpha * y * term1 + (1 - self.alpha) * (1 - y) * term2
            )

        return sfl
