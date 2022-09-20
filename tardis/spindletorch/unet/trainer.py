from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis.utils.logo import Tardis_Logo, printProgressBar
from tardis.utils.metrics import calculate_F1
from tardis.utils.utils import EarlyStopping
from torch import nn


class Trainer:
    """
    WRAPPER FOR TRAINER

     Args:
        model: Model with loaded pre-trained weights
        device: Device on which to predict
        criterion: Loss function
        optimizer: Optimizer
        training_DataLoader: DataLoader object with data sets for training
        validation_DataLoader: DataLoader object with data sets for training
        lr_scheduler: learning rete scheduler
        epochs: Number of epoches for training
        early_stop_rate: Number of epoches without improvement for early stop
        tqdm: If True plot progress bar in Jupyter else console
        checkpoint_name: Name for saving checkpoint
        classification: If True Unet3Plus use classification before loss evaluation
     """

    def __init__(self,
                 model,
                 type: dict,
                 device: str,
                 criterion,
                 optimizer,
                 training_DataLoader,
                 validation_DataLoader=None,
                 lr_scheduler=False,
                 epochs=100,
                 early_stop_rate=10,
                 checkpoint_name="Unet",
                 classification=False):
        self.model = model.to(device)
        self.type = type
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.early_stop_rate = early_stop_rate
        self.checkpoint_name = checkpoint_name
        self.classification = classification
        if classification:
            self.criterion_cls = nn.BCEWithLogitsLoss()

        self.progress_epoch = Tardis_Logo()
        self.progress_train = Tardis_Logo()

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.threshold = []

    def run_trainer(self):
        self.progress_epoch(title='CNN training module',
                            text_2='Epoch:',
                            text_3=printProgressBar(0, self.epochs))

        if isdir('cnn_checkpoint'):
            rmtree('cnn_checkpoint')
            mkdir('cnn_checkpoint')
        else:
            mkdir('cnn_checkpoint')

        early_stopping = EarlyStopping(patience=self.early_stop_rate, min_delta=0)
        for id in range(self.epochs):
            """Training block"""
            if id == 0:
                epoch_desc = 'Epochs: stop counter 0; best F1: NaN'
            else:
                epoch_desc = f'Epochs: stop counter {early_stopping.counter}; best F1 {round(np.max(self.f1), 3)}'

            self.progress_epoch(title='CNN training module',
                                text_2=epoch_desc,
                                text_3=printProgressBar(id, self.epochs))

            self._train(epoch_desc,
                        id)

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate(epoch_desc,
                               id)
                early_stopping(f1_score=self.f1[len(self.f1) - 1])

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            """ Save training metrics """
            np.savetxt(join(getcwd(), 'cnn_checkpoint', 'training_losses.csv'),
                       self.training_loss,
                       delimiter=';')
            np.savetxt(join(getcwd(), 'cnn_checkpoint', 'validation_losses.csv'),
                       self.validation_loss,
                       delimiter=',')
            np.savetxt(join(getcwd(), 'cnn_checkpoint', 'accuracy.csv'),
                       np.column_stack([self.accuracy,
                                        self.precision,
                                        self.recall,
                                        self.f1]),
                       delimiter=',')

            """ Save current model weights"""
            """ If F1 is higher then save checkpoint """
            if (np.array(self.f1)[:len(self.f1) - 1] < self.f1[len(self.f1) - 1]).all():
                torch.save({'model_struct_dict': self.type,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           join(getcwd(),
                                'cnn_checkpoint',
                                f'checkpoint_{self.checkpoint_name}.pth'))

            torch.save({'model_struct_dict': self.type,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       join(getcwd(), 'cnn_checkpoint', 'model_weights.pth'))

            if early_stopping.early_stop:
                break

    def _train(self,
               epoch_desc,
               progress_epoch):

        self.progress_train(title='CNN training module',
                            text_2=epoch_desc,
                            text_3=printProgressBar(progress_epoch, self.epochs),
                            text_4='Training: (loss 1.000)',
                            text_5=printProgressBar(0, self.training_DataLoader.__len__()))

        for idx, (x, y) in enumerate(self.training_DataLoader):
            self.model.train()
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.classification:
                out, _ = self.model(input)  # one forward pass
            else:
                out = self.model(input)  # one forward pass

            loss = self.criterion(out, target)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            loss_value = loss.item()
            self.training_loss.append(loss_value)

            self.progress_train(title='CNN training module',
                                text_2=epoch_desc,
                                text_3=printProgressBar(progress_epoch, self.epochs),
                                text_4=f'Training: (loss {loss_value:.4f})',
                                text_5=printProgressBar(idx, self.training_DataLoader.__len__()))

        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

    def _validate(self,
                  epoch_desc,
                  progress_epoch):
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses
        accuracy_mean = []  # accumulate the accuracy
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []

        SM = torch.nn.Softmax(1)

        for _, (x, y) in enumerate(self.validation_DataLoader):
            input, target = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()

                if out.shape[1] != 1:
                    out = SM(out)
                    out = out[0, 1, :]
                    target = target[0, 1, :]
                else:
                    out = torch.sigmoid(out)
                    out = out[0, 0, :]
                    target = target[0, 0, :]

                accuracy_score, precision_score, recall_score, \
                    F1_score, threshold = calculate_F1(input=out,
                                                       target=target,
                                                       best_f1=True)

                self.progress_train(title='CNN training module',
                                    text_2=epoch_desc,
                                    text_3=printProgressBar(progress_epoch, self.epochs),
                                    text_4=f'Validation: (loss {loss_value:.4f})',
                                    text_5=printProgressBar(0, self.validation_DataLoader.__len__()))

                # Avg. precision score
                valid_losses.append(loss_value)
                accuracy_mean.append(accuracy_score)
                precision_mean.append(precision_score)
                recall_mean.append(recall_score)
                F1_mean.append(F1_score)
                threshold_mean.append(threshold)

        self.threshold.append(np.mean(threshold_mean))
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))
