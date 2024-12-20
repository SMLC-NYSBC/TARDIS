#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Union

import numpy as np

from tardis_em.utils.errors import TardisError


def get_n_params(model):
    """
    Calculate the total number of parameters in a given machine learning model.

    This function computes the total number of learnable parameters for a given
    model by iterating through all its parameters and multiplying the sizes of
    each parameter tensor.

    :param model: The machine learning model whose parameters are to be counted.
        The model must have a 'parameters()' method that provides an
        iterable of tensors representing the parameters.
    :return: Integer value indicating the total number of parameters in the
        provided model.
    :rtype: int
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class EarlyStopping:
    """
    Implements early stopping for training processes.

    This class monitors a specified metric (validation loss or F1 score)
    during a training process and determines when to stop training to
    prevent overfitting or unnecessary computation. Early stopping halts
    the training if the monitored metric stops improving according to
    specified patience and threshold (min_delta) values.
    """

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(
        self, val_loss: Union[float, None] = None, f1_score: Union[float, None] = None
    ):
        """
        Compares current validation loss or F1 score with the previously recorded best value
        and updates it if there is an improvement beyond a minimum delta. If no improvement is
        observed for a consecutive number of evaluations equal to the patience value, triggers
        early stopping.

        The method either updates the current best value of validation loss or F1 score, resets
        the internal counter value when the improvement threshold is met, or increments the counter
        if no significant improvement is seen. Once the counter exceeds the patience threshold,
        `early_stop` is activated.

        :param val_loss: Validation loss from the current evaluation step. If not provided, F1
            score is considered instead.
        :param f1_score: F1 score from the current evaluation step. Used only if `val_loss`
            is not provided.
        :return: None
        """
        if val_loss is None and f1_score is None:
            TardisError(
                "124",
                "tardis_em/utils/utils",
                "Validation loss or F1 score is missing in early stop!",
            )

        if val_loss is not None:
            if self.best_loss is None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                self.counter = 0  # Reset counter if validation loss improves
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1

                if self.counter >= self.patience:
                    print("INFO: Early stopping")
                    self.early_stop = True
        elif f1_score is not None:
            if self.best_loss is None:
                self.best_loss = f1_score
            elif self.best_loss - f1_score < self.min_delta:
                self.best_loss = f1_score
                self.counter = 0  # Reset counter if validation loss improves
            elif self.best_loss - f1_score > self.min_delta:
                self.counter += 1

                if self.counter >= self.patience:
                    print("INFO: Early stopping")
                    self.early_stop = True


def check_uint8(image: np.ndarray):
    """
    Verify and process an input image of type uint8 for specific conditions:
    binary mask, multi-label mask, raw data, or empty image. Based on the image's
    unique pixel values, the function adapts its format accordingly.

    :param image: Input image as a numpy array to be verified and processed
    :type image: numpy.ndarray
    :return: Processed image based on the unique pixel values
    :rtype: numpy.ndarray
    """
    if np.all(np.unique(image) == [0, 1]):
        # Binary mask image
        return image
    elif np.all(np.unique(image) == [0, 254]):
        # Multi-label mask
        return np.array(np.where(image > 1, 1, 0), dtype=np.uint8)
    elif len(np.unique(image) > 2):
        # Raw image data
        return image.astype(np.uint8)
    elif np.all(image == 0):
        # Image is empty
        return image
