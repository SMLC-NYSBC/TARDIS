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
    Handles learning rate scheduling with warmup and scaling for an inner optimizer.

    This class provides a custom learning rate scheduling strategy, blending a warmup
    period followed by scaled, decreasing learning rates. It also wraps core optimizer
    functionalities like stepping, zeroing gradients, saving, and loading the optimizer's
    state. The scheduling ensures a dynamic learning rate control to improve model
    training stability and efficiency.
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
        Loads the state_dict of the optimizer from a given checkpoint.

        :param checkpoint: A dictionary containing the state_dict of the optimizer.
        :return: None
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
    This class initializes a trainer for machine learning models, with the capability
    to handle various neural network structures and configurations. It includes train
    and validation data loaders, optional learning rate scheduling, early stopping, and
    metric tracking. It is targeted for both classification and distributed computation
    tasks.
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
        """
        Initializes the BasicTrainer class with the necessary model, hyperparameters, and
        dataset information. Prepares the trainer for executing training and validation
        operations with optional early stopping, learning rate scheduling, and instance
        coverage settings.

        :param model: The deep learning model instance that will be trained.
        :param structure: Dictionary defining the structure and configuration specifics
            of the model (e.g., type, node input size if applicable).
        :param device: The torch.device defining whether to use CPU or GPU for training.
        :param criterion: The loss function used for optimization.
        :param optimizer: The optimizer applied during training to update model parameters.
        :param print_setting: Tuple containing settings to configure print-related
            details such as progress bars.
        :param training_DataLoader: Dataloader for the training dataset.
        :param validation_DataLoader: Optional. Dataloader for the validation dataset. Default is None.
        :param lr_scheduler: Optional. Indicates whether a learning rate scheduler
            is used. Default is False.
        :param epochs: Optional. Number of training epochs. Default is 100.
        :param early_stop_rate: Optional. Number of consecutive epochs to suffer no improvement
            in validation loss before early stopping. Default is 10.
        :param instance_cov: Optional. Setting for instance coverage. Default is 2.
        :param checkpoint_name: Optional. String defining the checkpoint name for saving
            intermediate training states. Default is "DIST".
        :param classification: Optional. Boolean indicating whether a classification
            task is performed. Default is False.
        """
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
        Generates and returns a description string summarizing the early stopping count
        and F1 metrics. This method is static and does not depend on instance
        attributes or methods.

        :param stop_count: The count of epochs where early stopping conditions
            have been evaluated.
        :type stop_count: int
        :param metric: A list representing the F1 metrics with both precision and
            recall percentages as floating-point values.
        :type metric: list
        :return: A formatted description string that includes the early stopping
            count and F1 metrics with two decimal precision.
        :rtype: str
        """
        desc = (
            f"Epochs: early_stop: {stop_count}; "
            f"F1: [{metric[0]:.2f}; {metric[1]:.2f}]"
        )
        return desc

    def _update_epoch_desc(self):
        """
        Updates the epoch description based on the specifics of the current training state.

        For the first epoch (id == 0), sets a default description. For subsequent epochs,
        updates the description to include information about the early stopping state and
        best F1 scores from the current and previous states if applicable.

        :param self: Instance of the class containing the method.
        :type self: ClassName
        """
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
        Updates the progress bar during the training or validation phase. This method
        determines whether the operation is in training or validation mode, calculates
        the dataset length accordingly, and updates the progress bar display using the
        specified parameters.

        :param loss_desc: A string description representing the current loss value or
            status of the process.
        :type loss_desc: str
        :param idx: Index of the current batch or step being processed.
        :type idx: int
        :param train: Flag indicating whether the progress is for training
            (True) or validation (False). Defaults to True.
        :type train: bool
        :param task: Optional string specifying the current task name or description.
            Defaults to an empty string.
        :type task: str
        :return: None
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
        """
        Saves performance metrics, learning rates, and model weights during training
        and validation processes. This function handles various checkpoints such as
        training losses, validation losses, evaluation metrics, learning rates,
        and model weights. It ensures the most recent and best-performing model weights
        are saved based on the evaluation metrics such as F1 score.

        :raises: None

        :return:
            A boolean indicating whether early stopping has been triggered.
        :rtype: bool
        """
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

        # torch.save(
        #     self.model,
        #     join(getcwd(), f"{self.checkpoint_name}_checkpoint", "model_weights.pth"),
        # )

        if self.early_stopping.early_stop:
            return True
        return False

    def _mid_training_eval(self, idx):
        """
        Evaluate and validate model performance during the middle of the training phase. This method is triggered at
        specific intervals during the training process to ensure the model is progressing as expected. Evaluation is
        not performed at the very beginning of training or during the last 10% of the training process to maintain
        training efficiency. The method temporarily switches the model to evaluation mode for validation, performs
        the necessary validation steps, updates training metrics, saves metrics, and then restores the training
        environment.

        :param idx: Current index of the training iteration.
        :type idx: int
        :return: None
        """
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
        Executes the training process for a machine learning model. This method handles
        the initialization of necessary components such as progress bars, early stopping,
        and training directories. It iteratively trains and validates the model over a
        specified number of epochs, updating progress and metrics. The process supports
        early stopping to terminate training if a certain condition is met.

        :raises FileExistsError: If errors occur while handling directory setup.

        :return: None
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
