from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis.utils.utils import EarlyStopping
from tqdm import tqdm as tq


class Trainer:
    """
    MAIN MODULE THAT BUILD TRAINER FOR GRAPHFORMER

    Args:
        model: GraphFormer model.
        node_input: If True image patches are used for training
        device: Device name on which model is train.
        criterion: Loss function for training
        optimizer: Optimizer object for training
        training_DataLoader: DataLoader with training dataset
        validation_DataLoader: DataLoader with validation dataset
        validation_step: Number of step after each validation is performed
        downsampling: Threshold number of points in point cloud before downsampling
        lr_scheduler: Learning rate scheduler if used.
        epochs: Number of epochs used for training.
        checkpoint_name: Checkpoint prefix name.
    """

    def __init__(self,
                 model,
                 node_input: bool,
                 device: str,
                 criterion,
                 optimizer,
                 training_DataLoader,
                 validation_DataLoader,
                 epochs: int,
                 checkpoint_name: str,
                 lr_scheduler=None,
                 tqdm=True):
        self.f1 = []
        self.recall = []
        self.precision = []
        self.accuracy = []
        self.learning_rate = []
        self.validation_loss = []
        self.training_loss = []

        self.model = model
        self.model.to(device)

        self.node_input = node_input
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.checkpoint_name = checkpoint_name
        self.tqdm = tqdm

    @ staticmethod
    def calculate_F1(logits,
                     targets):
        """ Calculate confusion matrix """
        confusion_vector = logits / targets
        tp = torch.sum(confusion_vector == 1).item()
        fp = torch.sum(confusion_vector == float('inf')).item()
        tn = torch.sum(torch.isnan(confusion_vector)).item()
        fn = torch.sum(confusion_vector == 0).item()

        """Accuracy Score - (tp + tn) / (tp + tn + fp + fn)"""
        accuracy_score = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        """Precision Score - tp / (tp + fp)"""
        precision_score = tp / (tp + fp + 1e-8)

        """Recall Score - tp / (tp + tn)"""
        recall_score = tp / (tp + fn + 1e-8)

        """F1 Score - 2 * [(Prec * Rec) / (Prec + Rec)]"""
        F1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score + 1e-8))

        return accuracy_score, precision_score, recall_score, F1_score

    def run_training(self):
        if self.tqdm:
            epoch_progress = tq(range(self.epochs),
                                'Epochs:',
                                total=self.epochs,
                                leave=True, ascii=True)
        else:
            epoch_progress = range(self.epochs)

        early_stoping = EarlyStopping(patience=50, min_delta=0)

        if isdir('GF_checkpoint'):
            rmtree('GF_checkpoint')
            mkdir('GF_checkpoint')
        else:
            mkdir('GF_checkpoint')

        for i in epoch_progress:
            """For each Epoch load be t model from previous run"""
            self.model.train()
            self.train()

            self.model.eval()
            self.validate()

            early_stoping(
                val_loss=self.validation_loss[len(self.validation_loss) - 1])
            np.savetxt('GF_checkpoint/training_losses.csv',
                       self.training_loss,
                       delimiter=',')
            np.savetxt('GF_checkpoint/validation_losses.csv',
                       self.validation_loss,
                       delimiter=',')
            np.savetxt('GF_checkpoint/lr_rates.csv',
                       self.learning_rate,
                       delimiter=',')
            np.savetxt('GF_checkpoint/accuracy.csv',
                       np.column_stack([self.accuracy,
                                        self.precision,
                                        self.recall,
                                        self.f1]),
                       delimiter=',')

            """ If F1 is higher then save checkpoint """
            if (np.array(self.f1)[:len(self.f1) - 1] < self.f1[len(self.f1) - 1]).all():
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           'GF_checkpoint/checkpoint_{}.pth'.format(self.checkpoint_name))
                print(
                    f'Saved model checkpoint no. {i} for F1 {self.f1[len(self.f1) - 1]:.2f}')

            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       join(getcwd(), 'GF_checkpoint', 'model_weights.pth'))

            epoch_progress.set_description(
                f'Epochs: stop counter {early_stoping.counter}, best F1 {np.max(self.f1)}')

            if early_stoping.early_stop:
                break

    def train(self):
        if self.tqdm:
            train_progress = tq(enumerate(self.training_DataLoader),
                                'Training:',
                                total=len(self.training_DataLoader),
                                leave=True, ascii=True)
        else:
            train_progress = enumerate(self.training_DataLoader)

        for _, (x, y, z, _) in train_progress:
            for c, i, g in zip(x, y, z):
                c, g = c.to(self.device), g.to(self.device)

                self.optimizer.zero_grad()
                if self.node_input:
                    i = i.to(self.device)
                    out = self.model(coords=c,
                                     node_features=i,
                                     padding_mask=None)
                else:
                    out = self.model(coords=c,
                                     node_features=None,
                                     padding_mask=None)

                loss = self.criterion(out[0, :],
                                      g)
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters

                loss_value = loss.item()
                self.training_loss.append(loss_value)

                train_progress.set_description(
                    f'Training: loss {loss.item():.4f}')

        """ Save current learning rate """
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        """Learning rate scheduler block"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validate(self):
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []

        with torch.no_grad():
            for x, y, z, _ in self.validation_DataLoader:
                for c, i, g in zip(x, y, z):
                    c, g = c.to(self.device), g.to(self.device)
                    if self.node_input:
                        i = i.to(self.device)
                        out = self.model(coords=c,
                                         node_features=i,
                                         padding_mask=None)
                    else:
                        out = self.model(coords=c,
                                         node_features=None,
                                         padding_mask=None)

                    target = g
                    out = out[0, :]
                    loss = self.criterion(out,
                                          target)
                    out = torch.where(torch.sigmoid(out) > 0.5, 1, 0)

                    acc, prec, recall, f1 = self.calculate_F1(logits=out,
                                                              targets=target)
                    # Avg. precision score
                    valid_losses.append(loss.item())
                    accuracy_mean.append(acc)
                    precision_mean.append(prec)
                    recall_mean.append(recall)
                    F1_mean.append(f1)

        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))
