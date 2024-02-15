#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractLoss(nn.Module):
    def __init__(self, smooth=1e-16, reduction="mean", diagonal=False, sigmoid=True):
        """
        Initializes the abstract loss function with the given parameters.

        Args:
            smooth (float): Smoothing factor to ensure no division by 0.
            reduction (str): The reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied.
                'mean': output would be a mean values.
                'sum': the output will be summed.
            diagonal (bool): If True, the diagonal of the adjacency matrix
                is removed in graph predictions.
            sigmoid (bool): If True, compute sigmoid before loss.
        """
        super(AbstractLoss, self).__init__()
        self.smooth = smooth
        self.diagonal = diagonal
        self.sigmoid = sigmoid
        self.reduction = reduction

        assert self.reduction in ["sum", "mean", "none"]

    def activate(self, logits):
        if self.sigmoid:
            return torch.sigmoid(logits)
        return logits

    def ignor_diagonal(self, logits, targets, mask=False):
        if self.diagonal:
            g_range = range(logits.shape[-1])

            if mask:
                mask = torch.ones_like(targets)
                mask[:, g_range, g_range] = 0.0

                logits = logits * mask
                targets = targets * mask
            else:
                logits[:, g_range, g_range] = 1
                targets[:, g_range, g_range] = 1
            return logits, targets
        return logits, targets

    def initialize_tensors(self, logits, targets, mask):
        # Activation and optionally ignore diagonal for graphs
        logits = self.activate(logits)

        return self.ignor_diagonal(logits=logits, targets=targets, mask=mask)

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask=False):
        """

        Args:
            logits (torch.Tensor): The predicted logits. Shape: [Batch x Channels x ...].
            targets (torch.Tensor): The target values. Shape: [Batch x Channels x ...].
            mask (bool): If Ture, output mask diagonal axis.

        Return:
            torch.Tensor: Computed loss function.
        """
        pass


class AdaptiveDiceLoss(AbstractLoss):
    """
    Implements an adaptive Dice loss function, which gives more weight to false negatives.

    The AdaptiveDiceLoss loss function is a variant of the standard Dice loss,
    with an additional adaptive term (1 - logits) ** self.alpha applied to logits.
    This term will give higher weight to false negatives (i.e., the cases where
    the prediction is low but the ground truth is high),
    which can be useful in cases where these are particularly costly.
    """

    def __init__(self, alpha=0.1, **kwargs):
        """
        Initializes the AdaptiveDiceLoss with the given parameters.

        Args:
            alpha (float): The scaling factor for the adaptive term.
                Higher values give more weight to false negatives.
            smooth (float): A small constant to avoid division by zero.
        """
        super(AdaptiveDiceLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Computes the adaptive Dice loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Soft weighted dice
        logits = logits.view(-1)
        targets = targets.view(-1)

        logits = ((1 - logits) ** self.alpha) * logits

        intersection = (logits * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            logits.square().sum() + targets.square().sum() + self.smooth
        )

        return 1 - dice


class BCELoss(AbstractLoss):
    """
    Implements the Binary Cross-Entropy loss function with an option to ignore
    the diagonal elements.

    The BCELoss class can be used for training where pixel-level accuracy is important.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BCELoss with the given parameters.
        """
        super(BCELoss, self).__init__(**kwargs)

        self.loss = nn.BCELoss(reduction=self.reduction)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=True
    ) -> torch.Tensor:
        """
        Computes the BCE loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        return self.loss(logits, targets)


class BCEGraphWiseLoss(AbstractLoss):
    """
    Implements the Binary Cross-Entropy loss function with an option to ignore
    the diagonal elements.

    The BCELoss class can be used for training where pixel-level accuracy is important.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BCELoss with the given parameters.
        """
        super(BCEGraphWiseLoss, self).__init__(**kwargs)

        self.loss = nn.BCELoss(reduction=self.reduction)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=True
    ) -> torch.Tensor:
        """
        Computes the BCE loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)
        idx_1 = torch.where(targets > 0)
        idx_0 = torch.where(targets == 0)

        pos_loss = self.loss(logits[idx_1], targets[idx_1])
        neg_loss = self.loss(logits[idx_0], targets[idx_0])
        return pos_loss + neg_loss


class BCEDiceLoss(AbstractLoss):
    """
    DICE + BCE LOSS FUNCTION
    """

    def __init__(self, **kwargs):
        """
        Loss initialization
        """
        super(BCEDiceLoss, self).__init__(**kwargs)
        self.bce = BCELoss(diagonal=self.diagonal, sigmoid=self.sigmoid)
        self.dice = DiceLoss(diagonal=self.diagonal, sigmoid=self.sigmoid)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loos function
        """
        bce_loss = self.bce(logits, targets, mask)
        dice_loss = self.dice(logits, targets, mask)

        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class CELoss(AbstractLoss):
    """
    STANDARD CROSS-ENTROPY LOSS FUNCTION
    """

    def __init__(self, **kwargs):
        """
        Loss initialization
        """
        super(CELoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loos function
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        return self.loss(logits, targets)


class DiceLoss(AbstractLoss):
    """
    Dice coefficient loss function.

    Dice=2(A∩B)(A)+(B);
    where 'A∩B' represents the common elements between sets A and B
    'A' ann 'B' represents the number of elements in set A ans set B

    This loss effectively zero-out any pixels from our prediction which
    are not "activated" in the target mask.
    """

    def __init__(self, **kwargs):
        """
        Loss initialization

        Args:
            smooth (float): Smooth factor to ensure  no 0 division.
        """
        super(DiceLoss, self).__init__(**kwargs)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loos function
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Flatten label and prediction tensors
        logits = logits[0, :].view(-1)
        targets = targets[0, :].view(-1)

        # Calculate dice loss
        intersection = (logits * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            logits.square().sum() + targets.square().sum() + self.smooth
        )

        return 1 - dice


class LaplacianEigenmapsLoss(AbstractLoss):
    """
    A loss function for deep learning models that computes the mean squared error
    between the first non-zero eigenvectors of the Laplacian matrices of the
    ground truth and predicted adjacency matrices.
    """

    def __init__(self, **kwargs):
        """
        Initializes the LaplacianEigenmapsLoss with an instance of nn.MSELoss
        """
        super(LaplacianEigenmapsLoss, self).__init__(**kwargs)
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    @staticmethod
    def compute_laplacian(A: torch.Tensor) -> torch.Tensor:
        """
        Computes the Laplacian matrix of an adjacency matrix.

        Args:
            A (torch.Tensor): The adjacency matrix.

        Returns:
            The Laplacian matrix.
        """
        D = torch.diag(torch.sum(A, dim=1))

        return D - A

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Computes the Laplacian-Eigenmaps loss between the true and predicted adjacency matrices.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Compute Laplacian matrices
        L_true = self.compute_laplacian(logits)
        L_pred = self.compute_laplacian(targets)

        # Computes the smallest non-zero eigenvector of each Laplacian
        _, eigenvectors_true = torch.linalg.eigh(L_true)
        _, eigenvectors_pred = torch.linalg.eigh(L_pred)

        v_true = eigenvectors_true[:, 1]
        v_pred = eigenvectors_pred[:, 1]

        # computes the mean squared error
        return self.mse_loss(v_true, v_pred)


class SoftSkeletonization(AbstractLoss):
    """
    General soft skeletonization with DICE loss function
    """

    def __init__(self, _iter=5, **kwargs):
        """
        Loss initialization

        Args:
            _iter (int): Number of skeletonization iterations
        """
        super(SoftSkeletonization, self).__init__(**kwargs)

        self.iter = _iter

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

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loos function
        """
        pass


class ClBCELoss(SoftSkeletonization):
    """
    Soft skeletonization with BCE loss function

    Implements a custom version of the Binary Cross Entropy (BCE) loss,
    where an additional term is added to the standard BCE loss.
    This additional term is a kind of F1 score calculated on the soft-skeletonized version
    of the predicted and target masks.
    """

    def __init__(self, **kwargs):
        super(ClBCELoss, self).__init__(**kwargs)
        self.bce = BCELoss(diagonal=self.diagonal, sigmoid=self.sigmoid)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loss function
        """
        # BCE with activation
        bce = self.bce(logits, targets)

        # Activation and optionally omit diagonal
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Soft skeletonization
        sk_logits = self.soft_skel(logits, self.iter)
        sk_targets = self.soft_skel(targets, self.iter)

        t_prec = (sk_logits * targets).sum() + self.smooth
        t_prec /= sk_logits.sum() + self.smooth

        t_sens = (sk_targets * logits).sum() + self.smooth
        t_sens /= sk_targets.sum() + self.smooth

        # ClBCE
        cl_bce = 2 * (t_prec * t_sens) / (t_prec + t_sens)

        return bce + (1 - cl_bce)


class ClDiceLoss(SoftSkeletonization):
    """
    Soft skeletonization with DICE loss function

    Implements a custom version of the Dice loss,
    where an additional term is added to the standard BCE loss.
    This additional term is a kind of F1 score calculated on the soft-skeletonized version
    of the predicted and target masks.
    """

    def __init__(self, **kwargs):
        super(ClDiceLoss, self).__init__(**kwargs)
        self.soft_dice = DiceLoss(diagonal=self.diagonal, sigmoid=self.sigmoid)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Forward loss function
        """
        # Dice loss with activation
        dice = self.soft_dice(logits, targets)

        logits, targets = self.initialize_tensors(logits, targets, mask)
        # Soft skeletonization
        sk_logits = self.soft_skel(logits, self.iter)
        sk_targets = self.soft_skel(targets, self.iter)

        t_prec = (sk_logits * targets).sum() + self.smooth
        t_prec /= sk_logits.sum() + self.smooth

        t_sens = (sk_targets * logits).sum() + self.smooth
        t_sens /= sk_targets.sum() + self.smooth

        # CLDice loss
        cl_dice = 2 * (t_prec * t_sens) / (t_prec + t_sens)

        return dice + (1 - cl_dice)


class SigmoidFocalLoss(AbstractLoss):
    """
    Implements the Sigmoid Focal Loss function with an option to ignore the diagonal elements.

    The SigmoidFocalLoss class implements the Focal Loss, which was proposed
    as a method for focusing the model on hard examples during the training of an object detector.
    It provides an option to ignore the diagonal elements of the input matrices.

    References: 10.1088/1742-6596/1229/1/012045
    """

    def __init__(self, gamma=0.25, alpha=None, **kwargs):
        """
        Initializes the SigmoidFocalLoss with the given parameters.

        Args:
            gamma (float): The gamma factor used in the focal loss computation.
            alpha (float, optional): Optional alpha factor for class balance.
                If not provided, no class balance is performed.
        """
        super(SigmoidFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Computes the sigmoid focal loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Compute focal loss term_1 and term_2
        term1 = (1 - logits) ** self.gamma * torch.log(logits)
        term1 = torch.where(torch.isinf(term1), 0, term1)

        term2 = logits**self.gamma * torch.log(1 - logits)
        term2 = torch.where(torch.isinf(term2), 0, term2)

        # Compute sigmoid focal loss
        y = targets.unsqueeze(1)
        if self.alpha is None:
            sfl = torch.mean(y * term1 + ((1 - y) * term2))
            sfl = -(1 / logits.size()[0]) * sfl
        else:
            sfl = torch.mean(
                self.alpha * y * term1 + (1 - self.alpha) * (1 - y) * term2
            )

        return sfl


class WBCELoss(AbstractLoss):
    """
    Implements a weighted Binary Cross-Entropy loss function with an option
    to ignore the diagonal elements.

    The WBCELoss class can help to balance the contribution of positive and
    negative samples in datasets
    where one class significantly outnumbers the other.
    It provides an option to ignore the diagonal elements of the input matrices,
    which could be useful for applications like graph prediction
    where self-connections might not be meaningful.
    """

    def __init__(self, **kwargs):
        """
        Initializes the WBCELoss with the given parameters.
        """
        super(WBCELoss, self).__init__(**kwargs)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False, pos=1, neg=0.1
    ) -> torch.Tensor:
        """
        Computes the weighted BCE loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Compute the percentage of positive samples
        positive_samples = torch.sum(targets)
        total_samples = targets.numel()
        positive_ratio = positive_samples / total_samples

        # Calculate class weights
        # TODO constant scale for pos and neg.
        # weight_positive = 1 / positive_ratio
        # weight_negative = 1 / (1 - positive_ratio)
        weight_positive = pos
        weight_negative = neg

        # Compute the binary cross entropy (BCE) loss
        bce_loss = -(
            (weight_positive * targets * torch.log(logits + 1e-8))
            + (weight_negative * (1 - targets) * torch.log(1 - logits + 1e-8))
        )

        # Average the losses across all samples
        if self.reduction == "sum":
            return torch.sum(bce_loss)
        elif self.reduction == "mean":
            return torch.mean(bce_loss)
        else:
            return bce_loss


class BCEMSELoss(AbstractLoss):
    """
    Implements the Binary Cross-Entropy over MSE loss function with an option
    to ignore the diagonal elements.

    The BCELoss class can be used for training where pixel-level accuracy is important.
    The MSE loos is used over continues Z slices to ensure smooth segmentation accuracy.
    """

    def __init__(self, mse_weight=0.1, **kwargs):
        """
        Initializes the BCELoss with the given parameters.
        """
        super(BCEMSELoss, self).__init__(**kwargs)

        self.mse_weight = mse_weight

        self.bce_loss = nn.BCELoss(reduction=self.reduction)
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=True
    ) -> torch.Tensor:
        """
        Computes the BCE loss between the logits and targets.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)

        # Avg. MSE to past and future frame
        mse = (
            self.mse_loss(logits[:, :, :-1, ...], targets[:, :, 1:, ...])
            + self.mse_loss(logits[:, :, 1:, ...], targets[:, :, :-1])
        ) / 2

        # Regular BCE
        bce = self.bce_loss(logits, targets)

        return bce + (mse * self.mse_weight)
