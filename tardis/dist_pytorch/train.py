import sys
from os import getcwd
from typing import Optional

import torch
from tardis.dist_pytorch.dist import C_DIST, DIST
from tardis.dist_pytorch.trainer import C_DistTrainer, DistTrainer
from tardis.dist_pytorch.utils.utils import check_model_dict
from tardis.utils.device import get_device
from tardis.utils.logo import Tardis_Logo
from tardis.utils.losses import BCELoss, CELoss, DiceLoss
from torch import optim
from torch.optim.lr_scheduler import StepLR

# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train_dist(train_dataloader,
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
    Wrapper for DIST or C_DIST models.

    Args:
        train_dataloader (torch.DataLoader): DataLoader with train dataset.
        test_dataloader (torch.DataLoader): DataLoader with test dataset.
        model_structure (dict): Dictionary with model setting.
        checkpoint (None, optional): Optional, DIST model checkpoint.
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
    if model_structure['dist_type'] == 'instance':
        model = DIST(n_out=model_structure['n_out'],
                     node_input=model_structure['node_input'],
                     node_dim=model_structure['node_dim'],
                     edge_dim=model_structure['edge_dim'],
                     num_layers=model_structure['num_layers'],
                     num_heads=model_structure['num_heads'],
                     coord_embed_sigma=model_structure['coord_embed_sigma'],
                     dropout_rate=model_structure['dropout_rate'],
                     structure=model_structure['structure'],
                     predict=False)
    elif model_structure['dist_type'] == 'semantic':
        model = C_DIST(n_out=model_structure['n_out'],
                       node_input=model_structure['node_input'],
                       node_dim=model_structure['node_dim'],
                       edge_dim=model_structure['edge_dim'],
                       num_layers=model_structure['num_layers'],
                       num_heads=model_structure['num_heads'],
                       num_cls=model_structure['num_cls'],
                       coord_embed_sigma=model_structure['coord_embed_sigma'],
                       dropout_rate=model_structure['dropout_rate'],
                       structure=model_structure['structure'],
                       predict=False)
    else:
        tardis_logo = Tardis_Logo()
        tardis_logo(text_1=f'ValueError: Model type: {type} is not supported!')
        sys.exit()

    """Build TARDIS progress bar output"""
    print_setting = [f"Training is started on {device}",
                     f"Local dir: {getcwd()}",
                     f"Training for {model_structure['dist_type']} with "
                     f"No. of Layers: {model_structure['num_layers']} with "
                     f"{model_structure['num_heads']} heads",
                     f"Layers are build of {model_structure['node_dim']} nodes, "
                     f"{model_structure['edge_dim']} edges, "
                     f"{model_structure['coord_embed_sigma']} sigma"]

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
        loss_fn = BCELoss()
    elif loss_function == 'ce':
        loss_fn = CELoss()

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
    if model_structure['dist_type'] == 'instance':
        train = DistTrainer(model=model,
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
                            checkpoint_name=model_structure['dist_type'])
    elif model_structure['dist_type'] == 'semantic':
        train = C_DistTrainer(model=model,
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
                              checkpoint_name=model_structure['dist_type'])

    """Train"""
    train.run_trainer()