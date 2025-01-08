#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import sys
from os import getcwd
from typing import Optional

from torch import optim

from tardis_em.cnn.cnn import build_cnn_network
from tardis_em.cnn.trainer import CNNTrainer
from tardis_em.cnn.utils.utils import check_model_dict
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.losses import *
from tardis_em.utils.trainer import ISR_LR

# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train_cnn(
    train_dataloader,
    test_dataloader,
    model_structure: dict,
    checkpoint: Optional[str] = None,
    loss_function="bce",
    learning_rate=1,
    learning_rate_scheduler=False,
    early_stop_rate=10,
    device="gpu",
    warmup=100,
    epochs=1000,
):
    """
    This function trains a Convolutional Neural Network (CNN) using the provided
    data loaders, model structure, and training parameters. It initializes the
    model, sets up the training pipeline, and manages checkpoints, learning rate
    schedulers, and optimization steps. The function supports retraining from
    checkpoints and includes a variety of loss functions for customization.

    :param train_dataloader: A DataLoader used for training the CNN, containing
        the training dataset.
    :type train_dataloader: torch.utils.data.DataLoader

    :param test_dataloader: A DataLoader used for validation/testing the CNN,
        containing the test dataset.
    :type test_dataloader: torch.utils.data.DataLoader

    :param model_structure: A dictionary defining the model's structure, including
        network type, input/output channels, and CNN configurations.
    :type model_structure: dict

    :param checkpoint: Optional path to a checkpoint file for retraining a
        pre-existing model.
    :type checkpoint: Optional[str]

    :param loss_function: Specifies the loss function to be used during training.
        Defaults to "bce" (Binary Cross-Entropy). Available options include
        AdaptiveDiceLoss, BCELoss, and others.
    :type loss_function: str

    :param learning_rate: The initial learning rate for the optimizer. Defaults
        to 1. Higher or lower values can impact model convergence.
    :type learning_rate: float

    :param learning_rate_scheduler: Boolean flag indicating whether to use a
        learning rate scheduler during training. Defaults to False.
    :type learning_rate_scheduler: bool

    :param early_stop_rate: The number of consecutive epochs of non-improvement
        in validation metrics before early stopping occurs. Defaults to 10.
    :type early_stop_rate: int

    :param device: The computational device to run the training on. Can be "gpu",
        "cpu", or a torch.device instance. Defaults to "gpu".
    :type device: Union[str, torch.device]

    :param warmup: The number of warmup steps for the optimizer, used when
        a learning rate scheduler is enabled. Defaults to 100.
    :type warmup: int

    :param epochs: The maximum number of training epochs. Defaults to 1000.
    :type epochs: int

    :return: None
    """
    """Losses"""
    loss_functions = [
        AdaptiveDiceLoss,
        BCELoss,
        WBCELoss,
        BCEDiceLoss,
        CELoss,
        DiceLoss,
        ClDiceLoss,
        ClBCELoss,
        SigmoidFocalLoss,
        LaplacianEigenmapsLoss,
        BCEMSELoss,
    ]
    losses_f = {f.__name__: f() for f in loss_functions}

    """Check input variable"""
    if checkpoint is not None:
        save_train = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model_struct_dict" in save_train.keys():
            model_structure = save_train["model_struct_dict"]

    model_structure = check_model_dict(model_structure)
    if not isinstance(device, torch.device) and isinstance(device, str):
        device = get_device(device)

    """Build CNN model"""
    try:
        model = build_cnn_network(
            network_type=model_structure["cnn_type"],
            structure=model_structure,
            img_size=model_structure["img_size"],
            prediction=False,
        )
    except:
        TardisError(
            "14",
            "tardis_em/cnn/train.py",
            f"CNNModelError: Model type: {type} was not build correctly!",
        )
        sys.exit()

    """Build TARDIS progress bar output"""
    print_setting = [
        f"Training is started for {model_structure['cnn_type']}:",
        f"Local dir: {getcwd()}",
        f"Training for {model_structure['cnn_type']} with "
        f"No. of Layers: {model_structure['num_conv_layers']} with "
        f"{model_structure['in_channel']} input and "
        f"{model_structure['out_channel']} output channel",
        f"Layers are build of {model_structure['layer_components']} modules, "
        f"train on {model_structure['img_size']} pixel images, "
        f"with {model_structure['conv_scaler']} up/down sampling "
        "channel scaler.",
    ]

    """Optionally: Load checkpoint for retraining"""
    if checkpoint is not None:
        model.load_state_dict(save_train["model_state_dict"])
    model = model.to(device)

    """Define loss function for training"""
    loss_fn = losses_f["BCELoss"]
    if loss_function in losses_f:
        loss_fn = losses_f[loss_function]

    """Build training optimizer"""
    if learning_rate_scheduler:
        optimizer = optim.NAdam(
            params=model.parameters(),
            betas=(0.9, 0.999),
            lr=learning_rate,
            eps=1e-8,
            momentum_decay=4e-3,
        )
    else:
        optimizer = optim.NAdam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            momentum_decay=4e-3,
        )

    """Optionally: Build learning rate scheduler"""
    if learning_rate_scheduler:
        optimizer = ISR_LR(
            optimizer, lr_mul=learning_rate, warmup_steps=warmup, scale=1
        )

    """Optionally: Checkpoint model"""
    if checkpoint is not None:
        optimizer.load_state_dict(save_train["optimizer_state_dict"])
        del save_train

    """Build trainer"""
    train = CNNTrainer(
        model=model,
        structure=model_structure,
        device=device,
        criterion=loss_fn,
        optimizer=optimizer,
        print_setting=print_setting,
        training_DataLoader=train_dataloader,
        validation_DataLoader=test_dataloader,
        lr_scheduler=learning_rate_scheduler,
        epochs=epochs,
        early_stop_rate=early_stop_rate,
        checkpoint_name=model_structure["cnn_type"],
    )

    """Train"""
    train.run_trainer()
