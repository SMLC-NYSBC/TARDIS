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
        Initializes an abstract loss function object.

        The constructor defines the essential parameters required for the computation
        of a specific loss function. These parameters include smooth values to avoid
        numerical instability, the type of reduction to be applied on the computed
        loss, whether the diagonal elements should be considered, and whether to
        apply a sigmoid operation on the inputs. It also validates that the reduction
        method is one of the accepted types.

        :param smooth: A small constant value added to avoid division by zero or
            numerical instability.
        :type smooth: float
        :param reduction: Specifies the type of reduction applied to the computed
            loss. Accepted values are: "sum", "mean", or "none".
        :type reduction: str
        :param diagonal: Boolean flag indicating whether diagonal elements are
            considered in the computation of the loss.
        :type diagonal: bool
        :param sigmoid: Boolean flag indicating whether sigmoid activation function
            is applied to the inputs before computing the loss.
        :type sigmoid: bool
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
        """
        Applies processing to logits and targets to ignore the diagonal entries when
        the ``diagonal`` condition is set. The behavior changes based on the value
        of the ``mask`` parameter. If masking is enabled, the diagonal entries of
        logits and targets are nullified. Without masking, the diagonal entries of
        logits and targets are set to 1.

        :param logits: A tensor representing unnormalized log probabilities of
            predicted classes.
        :param targets: A tensor representing the actual target classes.
        :param mask: A boolean indicating whether to apply zero masking
            on the diagonal elements of `logits` and `targets`.
        :return: A tuple containing the processed logits and targets tensors, with
            the diagonal entries adjusted based on the `mask` parameter.
        """
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
        """
        Initializes tensors by applying activation to the logits and optionally ignoring
        diagonal in graph structures.

        :param logits: The input tensor representing logits which undergo activation.
        :type logits: Tensor
        :param targets: A tensor representing target values.
        :type targets: Tensor
        :param mask: A boolean tensor to apply masking for ignoring specific elements,
            such as diagonal values in graphs.
        :type mask: Tensor

        :return: Activated logits tensor with potential diagonal values ignored based
            on the provided mask.
        :rtype: Tensor
        """
        # Activation and optionally ignore diagonal for graphs
        logits = self.activate(logits)

        return self.ignor_diagonal(logits=logits, targets=targets, mask=mask)

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask=False):
        """
        Computes the forward pass of the layer.

        :param logits: The predicted logits represented as a PyTorch Tensor.
        :param targets: The ground-truth values represented as a PyTorch Tensor.
        :param mask: A boolean flag. If set to True, a mask will be applied during
            computation. Defaults to False.

        :return: The computed result as a PyTorch Tensor.
        """
        pass


class AdaptiveDiceLoss(AbstractLoss):
    """
    Computes the Adaptive Dice Loss, a metric commonly used in image segmentation that combines the concept of Dice
    loss with an adaptive exponent to control the weight given to false negatives. The method calculates the similarity
    between predicted outputs (logits) and target values while considering class imbalance.

    This loss function is particularly useful in medical image analysis or other scenarios where the regions of interest
    can be very small compared to the overall image size, making it difficult for traditional loss functions to perform well.
    """

    def __init__(self, alpha=0.1, **kwargs):
        """
        This class implements an adaptive version of the Dice Loss function, commonly
        used in segmentation tasks. It allows dynamically adjusting the trade-off
        parameter, `alpha`, which helps control its sensitivity to the precision
        and recall of predictions.

        :param alpha: A float that determines the weight given to the precision versus
                      the recall in the Dice Loss calculation.
                      Lower values prioritize recall over precision while higher
                      values emphasize precision.
        :param kwargs: Additional keyword arguments to be passed to the base class.
        """
        super(AdaptiveDiceLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask=False
    ) -> torch.Tensor:
        """
        Computes the Soft Weighted Dice Loss between a predicted tensor (logits) and a target
        tensor.
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
    Binary Cross-Entropy Loss (BCELoss) class.

    This class is responsible for computing the binary cross-entropy loss
    between predicted logits and target values. It inherits from the
    AbstractLoss class and uses PyTorch's nn.BCELoss.

    The purpose of this class is to provide a specialized loss computation
    for binary classification tasks or similar problems where binary
    labels are involved. It supports masking functionality to handle
    specific use cases where part of the data needs to be ignored during
    loss calculation.
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
    Handles binary cross-entropy loss computation on a graph-wise level.

    This class extends AbstractLoss and is designed to compute binary cross-entropy
    (BCE) loss with specific functionality for handling graph-based input and output.
    It applies BCE loss separately to positive and negative targets, allowing for
    different treatment of these cases. The loss is calculated using a mask to focus
    only on specific parts of the logits and targets.
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
        Computes and returns the combined loss for positive and negative examples.
        """
        logits, targets = self.initialize_tensors(logits, targets, mask)
        idx_1 = torch.where(targets > 0)
        idx_0 = torch.where(targets == 0)

        pos_loss = self.loss(logits[idx_1], targets[idx_1])
        neg_loss = self.loss(logits[idx_0], targets[idx_0])
        return pos_loss + neg_loss


class BCEDiceLoss(AbstractLoss):
    """
    The BCEDiceLoss class combines Binary Cross-Entropy (BCE) loss and Dice loss
    to support both pixel-wise classification and segmentation tasks. It is designed
    to accommodate weighted or masked losses and is particularly useful in applications
    such as medical image segmentation where overlapping metric-based losses are
    advantageous.
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
        Computes the combined loss by summing Binary Cross-Entropy (BCE) loss and Dice loss.
        """
        bce_loss = self.bce(logits, targets, mask)
        dice_loss = self.dice(logits, targets, mask)

        if dice_loss is None:
            return bce_loss + 1

        return bce_loss + dice_loss


class CELoss(AbstractLoss):
    """
    Implements a Cross-Entropy Loss (CELoss) for neural networks.

    Cross-Entropy loss is commonly used in classification problems as a
    measure of how well the predicted probability distribution aligns with the
    actual distribution. This implementation supports optional masking for
    specific elements during the computation of the loss.
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
    Computes the Dice Loss, primarily used for image segmentation tasks.

    This loss function is frequently used to evaluate the overlap between
    predicted segmentations and ground truth segmentations. It is designed
    to penalize predictions that deviate from the ground truth, especially
    in situations where the classes are imbalanced. Dice Loss is defined
    as 1 - Dice Coefficient and is differentiable, making it useful for
    optimization in deep learning models.
    """

    def __init__(self, **kwargs):
        """
        DiceLoss class is implemented to compute the Dice coefficient loss for evaluating the
        overlap ratio between predicted and ground truth labels.
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
    Implements a loss function based on Laplacian Eigenmaps. This loss function compares
    the smallest non-zero eigenvectors of Laplacian matrices derived from true and
    predicted adjacency matrices. It uses mean squared error to quantify the discrepancy.

    This class leverages `nn.MSELoss` internally to compute the error between eigenvectors.
    The Laplacian matrix is computed as the degree matrix minus the adjacency matrix.
    The loss is primarily designed for graph-structured data.
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
        Computes the Laplacian matrix of a given adjacency matrix.

        :param A: The adjacency matrix represented as a tensor.
                  Each element in the matrix defines the connection
                  weights between nodes in a graph.
        :type A: torch.Tensor

        :return: The Laplacian matrix computed from the given adjacency
                 matrix.
        :rtype: torch.Tensor
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
    Implements a loss function based on soft skeletonization.

    This class provides a mechanism to compute a specific type of loss by applying
    soft morphological operations (erosion, dilation, opening) iteratively to extract
    soft skeletal representations of binary target masks. Primarily used for tasks
    requiring representation of skeletonized structures in binary masks.
    """

    def __init__(self, _iter=5, **kwargs):
        """
        Loss initialization

        :param _iter: Number of skeletonization iterations
        """
        super(SoftSkeletonization, self).__init__(**kwargs)

        self.iter = _iter

    @staticmethod
    def _soft_erode(binary_mask):
        """
        Applies a soft erosion operation on a binary mask using max pooling. The method
        supports both 2D and 3D binary masks. The operation adjusts the binary mask
        by considering the minimum pooling regions and reducing the binary mask values
        accordingly for soft eroding effects.

        :param binary_mask: The input binary mask tensor to be eroded. The tensor can
            either be 2D or 3D with dimensions specified as:
            - 4 dimensions (for 2D): [batch_size, channels, height, width]
            - 5 dimensions (for 3D): [batch_size, channels, depth, height, width]
        :return: The eroded binary mask after performing the soft erosion operation.
        :rtype: torch.Tensor
        """
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
        """
        Applies a soft dilation operation to the input binary mask tensor. The method
        performs max-pooling over the tensor to achieve dilation in a 2D or 3D space,
        depending on the dimensionality of the input. For 2D data, the method uses a
        3x3 kernel, while for 3D data, a 3x3x3 kernel is applied. When the input
        tensor has four dimensions, it is interpreted as 2D data, and when it has
        five dimensions, it is treated as 3D data. The dilation operation increases
        the binary mask regions with a stride of 1, providing a smooth expansion.

        :param binary_mask: A torch tensor representing the binary mask. The input
            tensor must have 4 dimensions (interpreted as 2D data) or 5 dimensions
            (interpreted as 3D data).
        :type binary_mask: torch.Tensor
        :return: A tensor of the same type as the input, with dilated binary mask
            regions. The output tensor shape is consistent with the input shape.
        :rtype: torch.Tensor

        :raises ValueError: If the input tensor does not have a dimensionality
            of 4 or 5.
        """
        if len(binary_mask.shape) == 4:  # 2D
            return F.max_pool2d(binary_mask, (3, 3), (1, 1), (1, 1))
        elif len(binary_mask.shape) == 5:  # 3D
            return F.max_pool3d(binary_mask, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def _soft_open(self, binary_mask):
        """
        Applies soft morphological opening to a binary mask. The operation is composed
        of a soft erosion followed by a soft dilation. Used in image processing or
        segmentation tasks to remove noise from binary masks while preserving shape
        integrity of detected objects.
        """
        return self._soft_dilate(self._soft_erode(binary_mask))

    def soft_skel(self, binary_mask: torch.Tensor, iter_i: int) -> torch.Tensor:
        """
        Extracts a soft skeleton from a binary mask using iterative erosion and morphological
        opening techniques. The algorithm starts by computing a partial skeleton and refines
        it iteratively over a user-defined number of iterations. This operation is used in
        image processing tasks to extract a skeletonized representation of binary masks.

        :param binary_mask: Input binary mask as a tensor on which the soft skeleton
            operation will be performed.
        :param iter_i: Number of iterations for which the skeletonization process will be
            performed. This controls the refinement of the resulting skeleton.
        :return: The resulting soft skeleton as a tensor after processing the input
            binary mask and applying iterative operations.
        """
        if not isinstance(iter_i, int):
            iter_i = int(iter_i)

        binary_mask_open = self._soft_open(binary_mask)
        s_skeleton = F.relu(binary_mask - binary_mask_open)

        for j in range(iter_i):
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
    Computes a combined Binary Cross-Entropy (BCE) loss and soft skeletonization-
    based class-sensitive loss. This class extends SoftSkeletonization and aims
    to address challenges in pixel-based confidence predictions for segmentation tasks.

    The main purpose of this loss function is to enhance performance by integrating
    standard BCE loss with a sensitivity and precision balance derived from skeletonized
    inputs. It calculates a harmonized ClBCE loss by prioritizing critical regions of the
    predictions while maintaining global and structural integrity.
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
    The ClDiceLoss class combines Soft Dice Loss with a topology-preserving loss based
    on soft skeletonization, referred to as clDice loss.

    This class is designed to compute the clDice loss, which aligns with the Dice Loss
    for pixel-wise accuracy while incorporating additional terms to preserve the topology
    of structures in binary segmentation tasks. The clDice loss employs soft skeletonization
    techniques to evaluate the similarity between skeletonized predictions and ground truth.
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
    Implements the Sigmoid Focal Loss function, a type of loss function often
    used for addressing class imbalance problems in classification tasks.

    The function applies a modulating factor to the standard cross-entropy
    criterion to focus learning more on hard-to-classify examples. It includes
    parameters like `gamma` for focusing and an optional `alpha` for
    handling class imbalance.
    """

    def __init__(self, gamma=0.25, alpha=None, **kwargs):
        """
        A class that represents the calculation for the Sigmoid Focal Loss. This class initializes
        parameters required for the loss computation, such as the gamma value (a parameter that
        modulates the weighting of easy and hard examples), and the alpha value (a weight adjustment
        parameter for balancing classes).

        :param gamma: A parameter controlling the weighting of easy and hard examples in the loss function.
        :type gamma: float
        :param alpha: A parameter used as the weights for positive and negative classes during the Focal Loss
            computation. Defaults to None.
        :type alpha: float or None
        :param kwargs: Additional parameters that can be passed to the parent class.
        :type kwargs: dict
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
    Defines the WBCELoss class which calculates a Weighted Binary Cross-Entropy (BCE)
    loss. This loss function is useful in scenarios where there is a significant class
    imbalance, as it allows for custom weighting of positive and negative classes. The
    loss calculation takes into account the logits (predicted probabilities), the target
    values, and optional parameters for masking and adjusting class weights.

    Its primary purpose is to compute a more flexible BCE loss that accommodates
    customizable weight scaling for positive and negative classes, especially in datasets
    with imbalanced distributions.
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
    Combines Binary Cross Entropy (BCE) loss and Mean Squared Error (MSE) loss
    for enhanced predictive modeling.

    This class is designed to compute a loss function that combines BCE
    for binary classification tasks and MSE for additional temporal consistency
    by penalizing differences between adjacent frames. The `mse_weight` parameter
    controls the relative contribution of the MSE to the final loss value.
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
