import numpy as np


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self,
                 patience=10,
                 min_delta=0):
        """
        patience: how many epochs to wait before stopping when loss is
               not improving
        min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def check_uint8(image: np.ndarray):
    if np.all(np.unique(image) == [0, 1]):
        return image
    elif np.all(np.unique(image) == [0, 254]) or np.all(np.unique(image) == [0, 255]):
        return np.array(np.where(image > 1, 1, 0), dtype=np.int8)
    else:
        raise TypeError('Given file is not uint8 or int8')
