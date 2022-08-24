from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis.utils.logo import Tardis_Logo, printProgressBar
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
        downsampling: Threshold number of points in point cloud before
            downsampling
        lr_scheduler: Learning rate scheduler if used.
        epochs: Number of epochs used for training.
        checkpoint_name: Checkpoint prefix name.
    """

    def __init__(self,
                 model,
                 type: str,
                 node_input: bool,
                 device: str,
                 criterion,
                 optimizer,
                 training_DataLoader,
                 validation_DataLoader,
                 epochs: int,
                 checkpoint_name: str,
                 print_setting: tuple,
                 lr_scheduler=None):
        self.f1 = []
        self.recall = []
        self.precision = []
        self.accuracy = []
        self.learning_rate = []
        self.validation_loss = []
        self.training_loss = []

        self.progress_epoch = Tardis_Logo()
        self.progress_train = Tardis_Logo()

        self.model = model.to(device)

        self.node_input = node_input
        self.type = type

        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.checkpoint_name = checkpoint_name
        self.print_setting = print_setting

        self.gpu_info = ""

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
        self.progress_epoch(title='DIST training module',
                            text_2='Epoch:',
                            text_3=printProgressBar(0, self.epochs))

        early_stopping = EarlyStopping(patience=50, min_delta=0)

        if isdir('GF_checkpoint'):
            rmtree('GF_checkpoint')
            mkdir('GF_checkpoint')
        else:
            mkdir('GF_checkpoint')

        for id in range(self.epochs):
            """For each Epoch load be t model from previous run"""
            if id == 0:
                epoch_desc = 'Epochs: stop counter 0; best F1: NaN'
            else:
                epoch_desc = f'Epochs: stop counter {early_stopping.counter}; best F1 {round(np.max(self.f1), 3)}'

            if self.device.type == 'cuda':
                import nvidia_smi

                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.device.index)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_info = "Device {}: {}, Memory : {:.2f}% free".format(self.device.index,
                                                                              nvidia_smi.nvmlDeviceGetName(handle),
                                                                              100 * info.free / info.total)
                nvidia_smi.nvmlShutdown()
            else:
                self.gpu_info = " "

            self.progress_epoch(title='DIST training module',
                                text_1=self.print_setting[0],
                                text_2=self.print_setting[1],
                                text_3=self.print_setting[2],
                                text_4=self.print_setting[3],
                                text_5=self.gpu_info,
                                text_7=epoch_desc,
                                text_8=printProgressBar(id, self.epochs))

            self.model.train()
            self.train(epoch_desc, id)

            self.model.eval()
            self.validate(epoch_desc, id)

            early_stopping(val_loss=self.validation_loss[len(self.validation_loss) - 1])

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

            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       join(getcwd(), 'GF_checkpoint', 'model_weights.pth'))

            if early_stopping.early_stop:
                break

    def train(self,
              epoch_desc,
              progress_epoch):
        self.progress_train(title='DIST training module',
                            text_1=self.print_setting[0],
                            text_2=self.print_setting[1],
                            text_3=self.print_setting[2],
                            text_4=self.print_setting[3],
                            text_5=self.gpu_info,
                            text_7=epoch_desc,
                            text_8=printProgressBar(progress_epoch, self.epochs),
                            text_9='Training: (loss 1.000)',
                            text_10=printProgressBar(0, self.training_DataLoader.__len__()))

        if self.type == 'instance':
            for idx, (x, y, z, _, _) in enumerate(self.training_DataLoader):
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

                    loss = self.criterion(out[0, :], g)

                    loss.backward()  # one backward pass
                    self.optimizer.step()  # update the parameters

                    loss_value = loss.item()
                    self.training_loss.append(loss_value)

                    self.progress_train(title='DIST training module',
                                        text_1=self.print_setting[0],
                                        text_2=self.print_setting[1],
                                        text_3=self.print_setting[2],
                                        text_4=self.print_setting[3],
                                        text_5=self.gpu_info,
                                        text_7=epoch_desc,
                                        text_8=printProgressBar(progress_epoch, self.epochs),
                                        text_9=f'Training: (loss {loss_value:.4f})',
                                        text_10=printProgressBar(idx, self.training_DataLoader.__len__()))
        elif self.type == 'semantic':
            for idx, (x, y, z, _, cls_g) in enumerate(self.training_DataLoader):
                for c, i, g, cls in zip(x, y, z, cls_g):
                    c, g = c.to(self.device), g.to(self.device)
                    self.optimizer.zero_grad()

                    out, out_cls = self.model(coords=c,
                                              padding_mask=None)

                    cls = cls.to(self.device)
                    loss = self.criterion(out[0, :], g) + self.criterion(out_cls, cls)

                    loss.backward()  # one backward pass
                    self.optimizer.step()  # update the parameters

                    loss_value = loss.item()
                    self.training_loss.append(loss_value)

                    self.progress_train(title='DIST training module',
                                        text_1=self.print_setting[0],
                                        text_2=self.print_setting[1],
                                        text_3=self.print_setting[2],
                                        text_4=self.print_setting[3],
                                        text_5=self.gpu_info,
                                        text_7=epoch_desc,
                                        text_8=printProgressBar(progress_epoch, self.epochs),
                                        text_9=f'Training: (loss {loss_value:.4f})',
                                        text_10=printProgressBar(idx, self.training_DataLoader.__len__()))
        """ Save current learning rate """
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        """Learning rate scheduler block"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def validate(self,
                 epoch_desc,
                 progress_epoch):
        valid_losses = []
        accuracy_mean = []
        precision_mean = []
        recall_mean = []
        F1_mean = []

        with torch.no_grad():
            if self.type == 'instance':
                for idx, (x, y, z, _, _) in enumerate(self.validation_DataLoader):
                    for c, i, g in zip(x, y, z):
                        c, target = c.to(self.device), g.to(self.device)

                        if self.node_input:
                            i = i.to(self.device)
                            out = self.model(coords=c,
                                             node_features=i,
                                             padding_mask=None)
                        else:
                            out = self.model(coords=c,
                                             node_features=None,
                                             padding_mask=None)

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

                    self.progress_train(title='DIST training module',
                                        text_1=self.print_setting[0],
                                        text_2=self.print_setting[1],
                                        text_3=self.print_setting[2],
                                        text_4=self.print_setting[3],
                                        text_5=self.gpu_info,
                                        text_7=epoch_desc,
                                        text_8=printProgressBar(progress_epoch, self.epochs),
                                        text_9=f'Validation: (loss {loss.item():.4f})',
                                        text_10=printProgressBar(idx, self.validation_DataLoader.__len__()))
            elif self.type == 'semantic':
                for idx, (x, y, z, _, cls_g) in enumerate(self.validation_DataLoader):
                    for c, i, g, cls in zip(x, y, z, cls_g):
                        c, target = c.to(self.device), g.to(self.device)

                        out, out_cls = self.model(coords=c,
                                                  padding_mask=None)

                        cls = cls.to(self.device)
                        loss = self.criterion(out[0, :], target) + self.criterion(out_cls, cls)

                        acc, \
                            prec, recall, \
                            f1 = self.calculate_F1(logits=torch.where(torch.sigmoid(out[0, :]) > 0.5, 1, 0),
                                                   targets=target)

                        # Avg. precision score
                        valid_losses.append(loss.item())
                        accuracy_mean.append(acc)
                        precision_mean.append(prec)
                        recall_mean.append(recall)
                        F1_mean.append(f1)

                    self.progress_train(title='DIST training module',
                                        text_1=self.print_setting[0],
                                        text_2=self.print_setting[1],
                                        text_3=self.print_setting[2],
                                        text_4=self.print_setting[3],
                                        text_5=self.gpu_info,
                                        text_7=epoch_desc,
                                        text_8=printProgressBar(progress_epoch, self.epochs),
                                        text_9=f'Validation: (loss {loss.item():.4f})',
                                        text_10=printProgressBar(idx, self.validation_DataLoader.__len__()))

        self.validation_loss.append(np.mean(valid_losses))
        self.accuracy.append(np.mean(accuracy_mean))
        self.precision.append(np.mean(precision_mean))
        self.recall.append(np.mean(recall_mean))
        self.f1.append(np.mean(F1_mean))
