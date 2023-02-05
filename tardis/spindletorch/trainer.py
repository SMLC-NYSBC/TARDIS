#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import numpy as np
import torch

from tardis.utils.metrics import eval_graph_f1
from tardis.utils.trainer import BasicTrainer


class CNNTrainer(BasicTrainer):
    """
    GENERAL CNN TRAINER
    """

    def __init__(self,
                 **kwargs):
        super(CNNTrainer, self).__init__(**kwargs)

    def _train(self):
        """
        Run model training.
        """
        # Update progress bar
        self._update_progress_bar(loss_desc='Training: (loss 1.000)',
                                  idx=0)

        # Run training for CNN model
        for idx, (i, m) in enumerate(self.training_DataLoader):
            """Mid-training eval"""
            self._mid_training_eval(idx=idx)

            """Training"""
            i, m = i.to(self.device), m.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

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

            # Update progress bar
            self._update_progress_bar(loss_desc=f'Training: (loss {loss_value:.4f})',
                                      idx=idx)

    def _validate(self):
        """
        Test model against validation dataset.
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

                acc, prec, recall, f1, th = eval_graph_f1(logits=img, targets=mask)
                # Avg. precision score
                valid_losses.append(loss.item())
                accuracy_mean.append(acc)
                precision_mean.append(prec)
                recall_mean.append(recall)
                F1_mean.append(f1)
                threshold_mean.append(th)
                valid = f'Validation: (loss {loss.item():.4f} Prec: {prec:.2f} Rec: {recall:.2f} F1: {f1:.2f})'

                # Update progress bar
                self._update_progress_bar(loss_desc=valid,
                                          idx=idx,
                                          train=False)

        # Reduce eval. metric with mean
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.threshold.append(np.mean(threshold_mean))
        self.f1.append(np.mean(F1_mean))

        # Check if average evaluation loss dropped
        self.early_stopping(f1_score=self.f1[-1:][0])
