from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree

import numpy as np
import torch
from tardis.utils.logo import Tardis_Logo, printProgressBar
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
        validation_DataLoader (torch.DataLoader, optional): DataLoader with test dataset..
        print_setting (tuple): Model property to display in TARDIS progress bar.
        lr_scheduler (torch.StepLR, optional): Optional Learning rate schedular.
        epochs (int): Max number of epoches.
        early_stop_rate (int): Number of epoches without improvement after which
            Trainer stop training.
        checkpoint_name (str): Name of the checkpoint.
        cnn (bool): If True expect CNN model parameters.
    """

    def __init__(self,
                 model,
                 structure: dict,
                 device: str,
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
        elif 'dist_type' in self.structure:
            if 'node_input' in structure:
                self.node_input = structure['node_input']

        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader

        # Set-up progress bar
        self.progress_epoch = Tardis_Logo()
        self.progress_train = Tardis_Logo()
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

    @ staticmethod
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
        desc = f'Epochs: stop counter {stop_count}; best F1 {f1[0]}; last f1: {f1[1]}'
        return desc

    def _update_progress_bar(self,
                             loss_desc: str,
                             idx: int):
        """
        _update_progress_bar _summary_

        Args:
            loss_desc (str): _description_
            idx (int): _description_
        """
        self.progress_train(title='DIST training module',
                            text_1=self.print_setting[0],
                            text_2=self.print_setting[1],
                            text_3=self.print_setting[2],
                            text_4=self.print_setting[3],
                            text_5=self.gpu_info,
                            text_7=self.epoch_desc,
                            text_8=printProgressBar(self.id,
                                                    self.epochs),
                            text_9=loss_desc,
                            text_10=printProgressBar(idx,
                                                     len(self.training_DataLoader)))

    def _mid_training_eval(self,
                           idx):
        if idx % (len(self.training_DataLoader) // 4) == 0:
            self._validate()

            self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                [round(np.max(self.f1), 3),
                                                 self.f1[-1:]])

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
                            text_3=printProgressBar(0, self.epochs))

        # Initialize early stop check.
        self.early_stopping = EarlyStopping(patience=self.early_stop_rate,
                                            min_delta=0)

        # Build training directory.
        if isdir(f'{self.checkpoint_name}_checkpoint'):
            rmtree(f'{self.checkpoint_name}_checkpoint')
            mkdir(f'{self.checkpoint_name}_checkpoint')
        else:
            mkdir(f'{self.checkpoint_name}_checkpoint')

        for id in range(self.epochs):
            """Initialized training"""
            self.id = id

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

            # For each Epoch load be t model from previous run
            if self.id == 0:
                self.epoch_desc = 'Epochs: stop counter 0; best F1: NaN'
            else:
                self.epoch_desc = self._update_desc(self.early_stopping.counter,
                                                    [round(np.max(self.f1), 3),
                                                     0.0])

            self.progress_epoch(title=f'{self.checkpoint_name} training module',
                                text_1=self.print_setting[0],
                                text_2=self.print_setting[1],
                                text_3=self.print_setting[2],
                                text_4=self.print_setting[3],
                                text_5=self.gpu_info,
                                text_7=self.epoch_desc,
                                text_8=printProgressBar(self.id, self.epochs))

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
            # If mean evaluation loss is higher then save checkpoint
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