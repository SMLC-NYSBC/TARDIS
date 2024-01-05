#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from os import getcwd, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Union

import numpy as np
import torch
from torch import optim

from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.utils import EarlyStopping


class ISR_LR:
    """
    Costume Inverse Square Root Learning Rate Schedular
    """

    def __init__(
        self, optimizer: optim.Adam, lr_mul: float, warmup_steps=1000, scale=100
    ):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.scale = scale
        self.param_groups = self._optimizer.param_groups

    def load_state_dict(self, checkpoint: dict):
        """
        Wrapper for loading Optimizer state dictionary

        Args:
            checkpoint (dict): Dictionary with optimizer state.
        """
        self._optimizer.load_state_dict(checkpoint)

    def state_dict(self):
        """Wrapper for retrieving Optimizer state dictionary"""
        return self._optimizer.state_dict()

    def step(self):
        """Step with the inner optimize"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad(set_to_none=True)

    def get_lr_scale(self):
        """Compute scaler for LR"""
        n_steps, n_warmup_steps = self.steps, self.warmup_steps
        return (self.scale**-0.5) * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.steps += 1
        lr = self.lr_mul * self.get_lr_scale()

        for g in self._optimizer.param_groups:
            g["lr"] = lr


class BasicTrainer:
    """
    BASIC MODEL TRAINER FOR DIST AND CNN

    Args:
        model (nn.Module): ML model build with nn.Module or nn.sequential.
        structure (dict): Model structure as dictionary.
        device (torch.device): Device for training.
        criterion (nn.loss): Loss function type.
        optimizer (optim.Adam, ISR_LR): Optimizer type.
        training_DataLoader (torch.DataLoader): DataLoader with training dataset.
        validation_DataLoader (torch.DataLoader, optional): DataLoader with test dataset.
        print_setting (tuple): Model property to display in TARDIS progress bar.
        lr_scheduler (bool): Optional Learning rate schedular.
        epochs (int): Max number of epoch's.
        early_stop_rate (int): Number of epoch's without improvement after which
            Trainer stop training.
        checkpoint_name (str): Name of the checkpoint.
    """

    def __init__(
        self,
        model,
        structure: dict,
        device: torch.device,
        criterion,
        optimizer: Union[ISR_LR, optim.Adam],
        print_setting: tuple,
        training_DataLoader,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        instance_cov=2,
        checkpoint_name="DIST",
        classification=False,
    ):
        super(BasicTrainer, self).__init__()

        self.early_stopping = None
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if lr_scheduler:
            self.lr = 1.0
        else:
            self.lr = self.optimizer.param_groups[0]["lr"]
        self.epochs = epochs
        self.early_stop_rate = early_stop_rate
        self.checkpoint_name = checkpoint_name
        self.structure = structure
        self.instance_cov = instance_cov

        if "cnn_type" in self.structure:
            self.classification = classification
            self.nn_name = self.structure["cnn_type"]
        elif "dist_type" in self.structure:
            self.nn_name = self.structure["dist_type"]

            if "node_input" in structure:
                self.node_input = structure["node_input"]

        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader

        # Set-up progress bar
        self.progress_epoch = TardisLogo()
        self.progress_train = TardisLogo()
        self.print_setting = print_setting

        self.id = 0
        self.epoch_desc = ""
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
    def _update_desc(stop_count: int, metric: list) -> str:
        """
        Utility function to update progress bar description.

        Args:
            stop_count (int): Early stop count.
            metric (list): Best f1 score.

        Returns:
            str: Updated progress bar status.
        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
        # For each Epoch load be t model from previous run
        if self.id == 0:
            self.epoch_desc = "Epochs: early_stop: 0; best F1: NaN"
        else:
            self.epoch_desc = self._update_desc(
                self.early_stopping.counter,
                [
                    np.max(self.f1) if len(self.f1) > 0 else 0.0,
                    self.f1[-1:][0] if len(self.f1) > 0 else 0.0,
                ],
            )

    def _update_progress_bar(self, loss_desc: str, idx: int, train=True, task=""):
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

        self.progress_train(
            title=f"{self.checkpoint_name} training module",
            text_1=self.print_setting[0],
            text_2=self.print_setting[1],
            text_3=self.print_setting[2],
            text_4=self.print_setting[3],
            text_6=task,
            text_7=self.epoch_desc,
            text_8=print_progress_bar(self.id, self.epochs),
            text_9=loss_desc,
            text_10=print_progress_bar(idx, data_set_len),
        )

    def _save_metric(self) -> bool:
        """Save training metrics"""
        if len(self.training_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "training_losses.csv",
                ),
                self.training_loss,
                delimiter=",",
            )

        if len(self.validation_loss) > 0:
            np.savetxt(
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    "validation_losses.csv",
                ),
                self.validation_loss,
                delimiter=",",
            )

        if len(self.f1) > 0:
            np.savetxt(
                join(getcwd(), f"{self.checkpoint_name}_checkpoint", "eval_metric.csv"),
                np.column_stack(
                    [
                        self.accuracy,
                        self.precision,
                        self.recall,
                        self.threshold,
                        self.f1,
                    ]
                ),
                delimiter=",",
            )

        if len(self.learning_rate) > 0:
            np.savetxt(
                join(
                    getcwd(), f"{self.checkpoint_name}_checkpoint", "learning_rate.csv"
                ),
                self.learning_rate,
                delimiter=",",
            )

        """ Save current model weights"""
        # If mean evaluation loss is higher than save checkpoint
        if all(self.f1[-1:][0] >= i for i in self.f1[:-1]):
            torch.save(
                {
                    "model_struct_dict": self.structure,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                join(
                    getcwd(),
                    f"{self.checkpoint_name}_checkpoint",
                    f"{self.checkpoint_name}_checkpoint.pth",
                ),
            )

        torch.save(
            {
                "model_struct_dict": self.structure,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            join(getcwd(), f"{self.checkpoint_name}_checkpoint", "model_weights.pth"),
        )

        if self.early_stopping.early_stop:
            return True
        return False

    def _mid_training_eval(self, idx):
        if idx % (len(self.training_DataLoader) // 4) == 0:
            if idx != 0 or self.id != 0:  # do not compute at trainer initialization
                # Do not validate at first idx and last 10%
                if idx != 0 and idx <= int(len(self.training_DataLoader) * 0.75):
                    self.model.eval()  # Enter Validation
                    self._validate()

                    self._update_epoch_desc()
                    self._save_metric()

                    self.model.train()  # Move back to training

    def run_trainer(self):
        """
        Main training loop.
        """
        # Initialize progress bar.
        self.progress_epoch(
            title=f"{self.checkpoint_name} training module.",
            text_2="Epoch: 0",
            text_3=print_progress_bar(0, self.epochs),
        )

        # Initialize early stop check.
        self.early_stopping = EarlyStopping(patience=self.early_stop_rate)

        # Build training directory.
        if isdir(f"{self.checkpoint_name}_checkpoint"):
            rmtree(f"{self.checkpoint_name}_checkpoint")
            mkdir(f"{self.checkpoint_name}_checkpoint")
        else:
            mkdir(f"{self.checkpoint_name}_checkpoint")

        for id_ in range(self.epochs):
            """Initialized training"""
            self.id = id_

            self._update_epoch_desc()
            self.progress_epoch(
                title=f"{self.checkpoint_name} training module",
                text_1=self.print_setting[0],
                text_2=self.print_setting[1],
                text_3=self.print_setting[2],
                text_4=self.print_setting[3],
                text_7=self.epoch_desc,
                text_8=print_progress_bar(self.id, self.epochs),
            )

            """Validation block"""
            if self.validation_DataLoader is not None and self.id != 0:
                self.model.eval()
                self._validate()

            """Training block"""
            self.model.train()
            self._train()

            stop = self._save_metric()
            if stop:
                break

    def _train(self):
        pass

    def _validate(self):
        pass
