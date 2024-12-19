#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import numpy as np
import torch

from tardis_em.utils.metrics import calculate_f1
from tardis_em.utils.trainer import BasicTrainer


class CNNTrainer(BasicTrainer):
    """
    Class for training and validation of a Convolutional Neural Network (CNN).

    Handles the entire process of training and validating a CNN model, including
    data loading, forward and backward passes, evaluation, progress tracking, and
    parameter updates. Designed to work with various configurations such as
    classification tasks, learning rate scheduling, and early stopping based on
    evaluation metrics.
    """

    def __init__(self, **kwargs):
        super(CNNTrainer, self).__init__(**kwargs)

    def _train(self):
        """
        Manages the training loop for a CNN model, including forward pass,
        backward pass, loss computation, optimizer step, and learning rate tracking.
        Handles intermediate evaluations, progress monitoring, and loss tracking
        for each training batch.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc="Training: (loss 1.000)", idx=0)

        # Run training for CNN model
        for idx, (i, m) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            """Training"""
            i, m = i.to(self.device), m.to(self.device)
            self.optimizer.zero_grad()

            if self.classification:
                i, _ = self.model(i)  # one forward pass
            else:
                i = self.model(i)  # one forward pass

            # Back-propagate
            loss = self.criterion(i, m)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            # Store training loss metric
            loss_value = loss.item()
            self.training_loss.append(loss_value)

            # Store and update learning rate
            if self.lr_scheduler:
                self.lr = self.optimizer.get_lr_scale()
            self.learning_rate.append(self.lr)

            # Update progress bar
            self._update_progress_bar(
                loss_desc=f"Training: (loss {loss_value:.4f};" f" LR: {self.lr:.5f})",
                idx=idx,
            )

    def _validate(self):
        """
        Validates the performance of the model on the validation dataset during training.

        This method computes various evaluation metrics such as accuracy, precision, recall,
        F1-score, and thresholding performance for each batch in the validation dataset. It
        uses the model to predict outputs on validation data, calculates the loss using the
        specified criterion, and updates the appropriate metrics for performance analysis.
        Additionally, it maintains and compares evaluation loss over epochs to decide on early
        stopping.
        """
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []

        SM = torch.nn.Softmax(1)

        for idx, (img, mask) in enumerate(self.validation_DataLoader):
            img, mask = img.to(self.device), mask.to(self.device)

            with torch.no_grad():
                img = self.model(img)
                loss = self.criterion(img, mask)

                if img.shape[1] != 1:
                    img = SM(img)[0, :, :].flatten()
                    mask = mask[0, :, :].flatten()
                else:
                    img = torch.sigmoid(img)[0, 0, :]
                    mask = mask[0, 0, :]

                img = np.where(img.cpu().detach().numpy() >= 0.5, 1, 0)
                mask = mask.cpu().detach().numpy()
                acc, prec, recall, f1 = calculate_f1(
                    logits=img, targets=mask, best_f1=False
                )

                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(0.5)
                valid = (
                    f"Validation: (loss {loss.item():.4f} "
                    f"Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})"
                )

                # Update progress bar
                self._update_progress_bar(loss_desc=valid, idx=idx, train=False)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=np.mean(F1_mean))
