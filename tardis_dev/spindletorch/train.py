import sys
from os import getcwd
from typing import Optional

import torch
from tardis_dev.spindletorch.spindletorch import build_network
from tardis_dev.spindletorch.trainer import CNNTrainer
from tardis_dev.spindletorch.utils.utils import check_model_dict
from tardis_dev.utils.device import get_device
from tardis_dev.utils.logo import Tardis_Logo
from tardis_dev.utils.losses import BCELoss, CELoss, DiceLoss
from torch import optim
from torch.optim.lr_scheduler import StepLR


# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train_cnn(train_dataloader,
              test_dataloader,
              model_structure: dict,
              checkpoint: Optional[str] = None,
              loss_function='bce',
              learning_rate=0.001,
              learning_rate_scheduler=False,
              early_stop_rate=10,
              device='gpu',
              epochs=1000):
    """
    Wrapper for CNN models.

    Args:
        train_dataloader (torch.DataLoader): DataLoader with train dataset.
        test_dataloader (torch.DataLoader): DataLoader with test dataset.
        model_structure (dict): Dictionary with model setting.
        checkpoint (None, optional): Optional, CNN model checkpoint.
        loss_function (str): Type of loss function.
        learning_rate (float): Learning rate.
        learning_rate_scheduler (bool): If True, StepLR is used with training.
        early_stop_rate (int): Define max. number of epoches without improvements
        after which training is stopped.
        device (torch.device): Device on which model is trained.
        epochs (int): Max number of epoches.
    """
    """Check input variable"""
    model_structure = check_model_dict(model_structure)

    if not isinstance(device, torch.device) and isinstance(device, str):
        device = get_device(device)

    """Build DIST model"""
    try:
        model = build_network(network_type=model_structure['cnn_type'],
                              classification=model_structure['classification'],
                              in_channel=model_structure['in_channel'],
                              out_channel=model_structure['out_channel'],
                              img_size=model_structure['img_size'],
                              dropout=model_structure['dropout'],
                              num_conv_layers=model_structure['num_conv_layers'],
                              conv_scaler=model_structure['conv_scaler'],
                              conv_kernel=model_structure['conv_kernel'],
                              conv_padding=model_structure['conv_padding'],
                              maxpool_kernel=model_structure['maxpool_kernel'],
                              layer_components=model_structure['layer_components'],
                              num_group=model_structure['num_group'],
                              prediction=False)
    except:
        tardis_logo = Tardis_Logo()
        tardis_logo(text_1=f'CNNModelError: Model type: {type} was not build correctly!')
        sys.exit()

    """Build TARDIS progress bar output"""
    print_setting = [f"Training is started on {device}",
                     f"Local dir: {getcwd()}",
                     f"Training for {model_structure['cnn_type']} with "
                     f"No. of Layers: {model_structure['num_conv_layers']} with "
                     f"{model_structure['in_channel']} input and "
                     f"{model_structure['out_channel']} output channel",
                     f"Layers are build of {model_structure['layer_components']} modules, "
                     f"train on {model_structure['img_size']} pixel images, "
                     f"with {model_structure['conv_scaler']} up/down sampling "
                     "channel scaler."]

    """Optionally: Load checkpoint for retraining"""
    if checkpoint is not None:
        save_train = torch.load(checkpoint, map_location=device)

        if 'model_struct_dict' in save_train.keys():
            model_dict = save_train['model_struct_dict']
            globals().update(model_dict)

        model.load_state_dict(save_train['model_state_dict'])

    model = model.to(device)

    """Define loss function for training"""
    if loss_function == "dice":
        loss_fn = DiceLoss()
    elif loss_function == "bce":
        if model_structure['out_channel'] > 1:
            loss_fn = CELoss()
        else:
            loss_fn = BCELoss()

    """Build training optimizer"""
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate)

    """Optionally: Checkpoint model"""
    if checkpoint is not None:
        optimizer.load_state_dict(save_train['optimizer_state_dict'])

        save_train = None
        del save_train

    """Optionally: Build learning rate scheduler"""
    if learning_rate_scheduler:
        learning_rate_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    else:
        learning_rate_scheduler = None

    """Build trainer"""
    train = CNNTrainer(model=model,
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
                       checkpoint_name=model_structure['cnn_type'])

    """Train"""
    train.run_trainer()