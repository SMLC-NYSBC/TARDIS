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
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Args:
        patience (int): how many epochs to wait before stopping when loss is
               not improving.
        min_delta (int): minimum difference between new loss and old loss for
               new loss to be considered as an improvement.
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
    Simple check for uint8 array.

    If array is not compliant, then data are examine for a type of data (binary
    mask, image with int8 etc.) and converted back to uint8.

    Args:
        image (np.ndarray): Image for evaluation.
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
