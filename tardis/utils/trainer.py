#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch

from tardis.utils.logo import print_progress_bar, TardisLogo
from tardis.utils.utils import EarlyStopping


class BasicTrainer:
    """
    BASIC MODEL TRAINER FOR DIST AND CNN

    Args:
        model (nn.Module): ML model build with nn.Module or nn.sequential.
        structure (dict): Model structure as dictionary.
        device (torch.device): Device for training.
        criterion (nn.loss): Loss function type.
        optimizer (torch.optim): Optimizer type.
        training_DataLoader (torch.DataLoader): DataLoader with training dataset.
        validation_DataLoader (torch.DataLoader, optional): DataLoader with test dataset.
        print_setting (tuple): Model property to display in TARDIS progress bar.
        lr_scheduler (torch.StepLR, optional): Optional Learning rate schedular.
        epochs (int): Max number of epoch's.
        early_stop_rate (int): Number of epoch's without improvement after which
            Trainer stop training.
        checkpoint_name (str): Name of the checkpoint.
    """

    def __init__(self,
                 model,
                 structure: dict,
                 device: torch.device,
                 criterion,
                 optimizer,
                 print_setting: tuple,
                 training_DataLoader,
                 validation_DataLoader=None,
                 lr_scheduler=None,
                 epochs=100,
                 early_stop_rate=10,
                 checkpoint_name="DIST",
                 classification=False):
        super(BasicTrainer, self).__init__()

        self.early_stopping = None
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.early_stop_rate = early_stop_rate
        self.checkpoint_name = checkpoint_name
        self.structure = structure

        if 'cnn_type' in self.structure:
            self.classification = classification
            self.nn_name = self.structure['cnn_type']
        elif 'dist_type' in self.structure:
            self.nn_name = self.structure['dist_type']

            if 'node_input' in structure:
                self.node_input = structure['node_input']

        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader

        # Set-up progress bar
        self.progress_epoch = TardisLogo()
        self.progress_train = TardisLogo()
        self.print_setting = print_setting

        self.id = 0
        self.epoch_desc = ''
        self.gpu_info = ""

        # Storage for all training metrics
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

        # Storage for all evaluation metrics
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.threshold = []

    @staticmethod
    def _update_desc(stop_count: int,
                     f1: list) -> str:
        """
        Utility function to update progress bar description.

        Args:
            stop_count (int): Early stop count.
            f1 (list): Best f1 score.

        Returns:
            str: Updated progress bar status.
        """
        desc = f'Epochs: stop counter {stop_count}; best F1 {f1[0]:.2f}; last f1: {f1[1]:.2f}'
        return desc

    def _update_progress_bar(self,
                             loss_desc: str,
                             idx: int,
                             train=True):
        """
        Update entire Tardis progress bar.

        Args:
            loss_desc (str): Description for loss function current state.
            idx (int): Number of the current epoch step.
            train (bool): If true, count progressbar for training dataset, else
                progressbar for validation
        """
        if train:
            data_set_len = len(self.training_DataLoader)
        else:
            data_set_len = len(self.validation_DataLoader)

        self.progress_train(title=f'{self.checkpoint_name} training module',
                            text_1=self.print_setting[0],
                            text_2=self.print_setting[1],
                            text_3=self.print_setting[2],
                            text_4=self.print_setting[3],
                            text_7=self.epoch_desc,
                            text_8=print_progress_bar(self.id, self.epochs),
                            text_9=loss_desc,
                            text_10=print_progress_bar(idx, data_set_len))

    def _mid_training_eval(self,
                           idx):
        if idx % (len(self.training_DataLoader) // 4) == 0:
            # Do not validate at first idx and last 10%
            if idx != 0 or idx >= int(len(self.training_DataLoader) * 0.9):
                self._validate()

                self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                    [round(np.max(self.f1), 3),
                                                     self.f1[-1:][0]])

                # Update checkpoint weights if validation loss dropped
                if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
                    torch.save({'model_struct_dict': self.structure,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                               join(getcwd(),
                                    f'{self.checkpoint_name}_checkpoint',
                                    f'{self.checkpoint_name}_checkpoint.pth'))

    def run_trainer(self):
        """
        Main training loop.
        """
        # Initialize progress bar.
        self.progress_epoch(title=f'{self.checkpoint_name} training module.',
                            text_2='Epoch: 0',
                            text_3=print_progress_bar(0, self.epochs))

        # Initialize early stop check.
        self.early_stopping = EarlyStopping(patience=self.early_stop_rate)

        # Build training directory.
        if isdir(f'{self.checkpoint_name}_checkpoint'):
            rmtree(f'{self.checkpoint_name}_checkpoint')
            mkdir(f'{self.checkpoint_name}_checkpoint')
        else:
            mkdir(f'{self.checkpoint_name}_checkpoint')

        for id in range(self.epochs):
            """Initialized training"""
            self.id = id

            # For each Epoch load be t model from previous run
            if self.id == 0:
                self.epoch_desc = 'Epochs: stop counter 0; best F1: NaN'
            else:
                self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                    [round(np.max(self.f1), 3), 0.0])

            self.progress_epoch(title=f'{self.checkpoint_name} training module',
                                text_1=self.print_setting[0],
                                text_2=self.print_setting[1],
                                text_3=self.print_setting[2],
                                text_4=self.print_setting[3],
                                text_7=self.epoch_desc,
                                text_8=print_progress_bar(self.id, self.epochs))

            """Training block"""
            self.model.train()
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self.model.eval()
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

                # Save current learning rate
                self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

            """ Save training metrics """
            if len(self.training_loss) > 0:
                np.savetxt(join(getcwd(),
                                f'{self.checkpoint_name}_checkpoint',
                                'training_losses.csv'),
                           self.training_loss,
                           delimiter=';')
            if len(self.validation_loss) > 0:
                np.savetxt(join(getcwd(),
                                f'{self.checkpoint_name}_checkpoint',
                                'validation_losses.csv'),
                           self.validation_loss,
                           delimiter=',')
            if len(self.f1) > 0:
                np.savetxt(join(getcwd(),
                                f'{self.checkpoint_name}_checkpoint',
                                'eval_metric.csv'),
                           np.column_stack([self.accuracy,
                                            self.precision,
                                            self.recall,
                                            self.threshold,
                                            self.f1]),
                           delimiter=',')

            """ Save current model weights"""
            # If mean evaluation loss is higher than save checkpoint
            if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
                torch.save({'model_struct_dict': self.structure,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           join(getcwd(),
                                f'{self.checkpoint_name}_checkpoint',
                                f'{self.checkpoint_name}_checkpoint.pth'))

            torch.save({'model_struct_dict': self.structure,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       join(getcwd(),
                            f'{self.checkpoint_name}_checkpoint',
                            'model_weights.pth'))

            if self.early_stopping.early_stop:
                break

    def _train(self):
        pass

    def _validate(self):
        pass
