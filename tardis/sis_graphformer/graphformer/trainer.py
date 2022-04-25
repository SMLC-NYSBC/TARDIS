from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis.utils.utils import EarlyStopping


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
                 batch: int,
                 criterion,
                 optimizer,
                 training_DataLoader,
                 validation_DataLoader,
                 validation_step: int,
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
        self.node_input = node_input
        self.device = device
        self.batch = batch
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.validation_step = validation_step * batch
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
            from tqdm import tqdm

            progressbar = tqdm(range(self.epochs),
                               'Epochs:',
                               total=self.epochs,
                               leave=False)
        else:
            progressbar = range(self.epochs)

        early_stoping = EarlyStopping(patience=25, min_delta=0)

        if isdir('GF_checkpoint'):
            rmtree('GF_checkpoint')
            mkdir('GF_checkpoint')
        else:
            mkdir('GF_checkpoint')

        for i in progressbar:
            """ For each Epoch load be t model from previous run"""
            self.train()

            self.validate()
            early_stoping(
                val_loss=self.validation_loss[len(self.validation_loss) - 1])

            np.savetxt('GF_checkpoint/validation_losses.csv',
                       self.validation_loss,
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

            if early_stoping.early_stop:
                break

    def train(self):
        if self.tqdm:
            from tqdm import tqdm

            train_progress = tqdm(enumerate(self.training_DataLoader),
                                  'Training:',
                                  total=len(self.training_DataLoader),
                                  leave=False)
        else:
            train_progress = enumerate(self.training_DataLoader)

        self.model.train()

        for i, (x, y, z, _) in train_progress:
            for j in range(len(x)):
                self.optimizer.zero_grad(set_to_none=True)

                if self.node_input:
                    logits = self.model(coords=x[j].to(self.device),
                                        node_features=y[j].to(self.device),
                                        padding_mask=None)
                else:
                    logits = self.model(coords=x[j].to(self.device),
                                        node_features=None,
                                        padding_mask=None)

                loss = self.criterion(logits.permute(0, 3, 1, 2)[0, :],
                                      z[j].to(self.device))

                loss.backward()
                self.optimizer.step()

                self.training_loss.append(loss.item())
                np.savetxt('GF_checkpoint/training_losses.csv',
                           self.training_loss,
                           delimiter=',')

                train_progress.set_description(
                    f'Training: (loss {loss.item():.4f})')

        """ Save current learning rate """
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        np.savetxt('GF_checkpoint/lr_rates.csv',
                   self.learning_rate,
                   delimiter=',')

        """Learning rate scheduler block"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        train_progress.close()

    def validate(self):
        self.model.eval()
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []

        validate_progress = enumerate(self.validation_DataLoader)
        self.model.eval()

        for i, (x, y, z, _) in validate_progress:
            with torch.no_grad():
                for j in range(len(x)):
                    if self.node_input:
                        logits = self.model(coords=x[j].to(self.device),
                                            node_features=y[j].to(self.device),
                                            padding_mask=None)
                    else:
                        logits = self.model(coords=x[j].to(self.device),
                                            node_features=None,
                                            padding_mask=None)

                    loss = self.criterion(logits.permute(0, 3, 1, 2)[0, :],
                                          z[j].to(self.device))

                    logits = torch.sigmoid(logits.permute(0, 3, 1, 2)[0, :])
                    logits = torch.where(logits > 0.5, 1, 0)
                    acc, prec, recall, f1 = self.calculate_F1(logits=logits,
                                                              targets=z[j].to(self.device))
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
