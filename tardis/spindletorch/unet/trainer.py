import numpy as np
import torch
from tardis.spindletorch.utils.metrics import calculate_F1
from torch import nn
from os import mkdir, getcwd
from os.path import isdir, join
from shutil import rmtree
from tardis.spindletorch.utils.utils import EarlyStopping


class Trainer:
    """
    Wrapper for training

     Args:
         model: Model with loaded pretrained weights
         device: Device on which to predict
         criterion: Loss function
         optimizer: Optimizer
         training_DataLoader: DataLoader object with data sets for training
         validation_DataLoader: DataLoader object with data sets for training
         lr_scheduler: learning rete scheduler
         epochs: Number of epoches for training
         notebook: If True plot progress bar in Jupyter else console

     """

    def __init__(self,
                 model,
                 device: str,
                 criterion,
                 optimizer,
                 training_DataLoader,
                 validation_DataLoader=None,
                 lr_scheduler=False,
                 epochs=100,
                 notebook=False,
                 checkpoint_name="Unet",
                 classification=False):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.notebook = notebook
        self.checkpoint_name = checkpoint_name
        self.classification = classification
        if classification:
            self.criterion_cls = nn.BCEWithLogitsLoss()

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.threshold = []

    def run_trainer(self):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange

        if isdir('temp'):
            rmtree('temp')
            mkdir('temp')
        else:
            mkdir('temp')

        early_stoping = EarlyStopping(patience=10, min_delta=0)
        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
                early_stoping(
                    val_loss=self.validation_loss[len(self.validation_loss) - 1])

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            """ Save training metrics """
            np.savetxt(join(getcwd(), 'temp', 'training_losses.csv'),
                       self.training_loss,
                       delimiter=';')
            np.savetxt(join(getcwd(), 'temp', 'validation_losses.csv'),
                       self.validation_loss,
                       delimiter=',')
            np.savetxt(join(getcwd(), 'temp', 'accuracy.csv'),
                       np.column_stack([self.accuracy,
                                        self.precision,
                                        self.recall,
                                        self.f1]),
                       delimiter=',')

            """ Save current model weights"""
            """ If F1 is higher then save checkpoint """
            if (np.array(self.f1)[:len(self.f1) - 1] < self.f1[len(self.f1) - 1]).all():
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           join(getcwd(), 'temp', 'checkpoint_{}.pth'.format(self.checkpoint_name)))
                print(
                    f'Saved model checkpoint no. {i} for F1 {self.f1[len(self.f1) - 1]:.2f}')

            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       join(getcwd(), 'temp', 'model_weights.pth'))

            if early_stoping.early_stop:
                break

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        batch_iter = tqdm(enumerate(self.training_DataLoader),
                          'Training',
                          total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            self.model.train()
            input, target = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.classification:
                # TODO: Include double loss function for classes and final output
                out, out_cls = self.model(input)  # one forward pass
            else:
                out = self.model(input)  # one forward pass
                loss = self.criterion(out, target)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            loss_value = loss.item()
            self.training_loss.append(loss_value)

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses
        accuracy_mean = []  # accumulate the accuracy
        precision_mean = []
        recall_mean = []
        F1_mean = []
        threshold_mean = []

        batch_iter = enumerate(self.validation_DataLoader)
        SM = torch.nn.Softmax(1)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()

                if out.shape[1] != 1:
                    out = SM(out)
                    # out = torch.argmax(out, 1)
                    out = out[0, 1, :]
                    target = target[0, 1, :]
                else:
                    out = torch.sigmoid(out)
                    # out = torch.where(out[0, 0, :] > 0.5, 1, 0)
                    out = out[0, 0, :]
                    target = target[0, 0, :]

                accuracy_score, precision_score, recall_score, \
                    F1_score, threshold = calculate_F1(input=out,
                                                       target=target,
                                                       best_f1=True)

                # Avg. precision score
                valid_losses.append(loss_value)
                accuracy_mean.append(accuracy_score)
                precision_mean.append(precision_score)
                recall_mean.append(recall_score)
                F1_mean.append(F1_score)
                threshold_mean.append(threshold)

        print(f'Validation: (loss {np.mean(valid_losses):.4f}) & '
              f'F1: {np.mean(F1_mean):.2f} '
              f'at Threshold {np.mean(threshold_mean):.2f}')

        self.threshold.append(np.mean(threshold_mean))
        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))