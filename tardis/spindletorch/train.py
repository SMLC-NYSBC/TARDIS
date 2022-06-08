from os import mkdir
from os.path import isdir, join
from typing import Optional

import torch
from tardis.spindletorch.unet.losses import (AdaptiveDiceLoss, BCEDiceLoss,
                                             BCELoss, DiceLoss)
from tardis.spindletorch.unet.trainer import Trainer
from tardis.spindletorch.utils.build_network import build_network
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          img_size=64,
          cnn_type='unet',
          classification=False,
          dropout: Optional[float] = None,
          convolution_layer=5,
          convolution_multiplayer=64,
          convolution_structure='gcl',
          cnn_checkpoint: Optional[str] = None,
          loss_function='bce',
          loss_alpha: Optional[float] = None,
          learning_rate=0.001,
          learning_rate_scheduler=False,
          early_stop_rate=10,
          tqdm=True,
          device='gpu',
          epochs=100):
    """
    CNN MAIN TRAINING WRAPPER

    Args:
        train_dataloader: DataLoader with train dataset
        test_dataloader: DataLoader with validation dataset
        img_size: Image patch size, needed to calculate CNN layers
        cnn_type: Name of CNN type
        classification: If True Unet3Plus use classification before loss evaluation
        dropout: Dropout value, if None dropout is not used
        convolution_layer: Number of convolution layers
        convolution_multiplayer: Number of output channels in first CNN layer
        convolution_structure: Structure and order of CNN layer components
        cnn_checkpoint: If not None, indicate dir. with a checpoint weights
        loss_function: Name of loss function used for evaluation
        loss_alpha: Alpha value for Adaptive_Dice loss function
        learning_rate: Float value of learning rate
        learning_rate_scheduler: If True, build optimizer with lr scheduler
        early_stop_rate: Number of epoches without improvement needed for early stop of the training
        tqdm: If True, build Trainer with progress bar
        device: Device ID used for training
        epochs: Number of epoches
    """
    img, mask = next(iter(train_dataloader))
    print(f'x = shape: {img.shape}; '
          f'type: {img.dtype}')
    print(f'x = min: {img.min()}; '
          f'max: {img.max()}')
    print(f'y = shape: {mask.shape}; '
          f'class: {mask.unique()}; '
          f'type: {mask.dtype}')

    """Build CNN"""
    model = build_network(network_type=cnn_type,
                          classification=classification,
                          in_channel=1,
                          out_channel=1,
                          img_size=img_size,
                          dropout=dropout,
                          no_conv_layers=convolution_layer,
                          conv_multiplayer=convolution_multiplayer,
                          layer_components=convolution_structure,
                          no_groups=8,
                          prediction=False)

    """Load checkpoint for retraining"""
    if cnn_checkpoint is not None:
        save_train = join(cnn_checkpoint)

        save_train = torch.load(join(save_train))
        model.load_state_dict(save_train['model_state_dict'])

    """Define loss function for training"""
    if loss_function == "dice":
        loss_fn = DiceLoss()
    if loss_function == "bce":
        loss_fn = BCELoss()
    if loss_function == "adaptive_dice":
        loss_fn = AdaptiveDiceLoss(alpha=loss_alpha)
    if loss_function == "hybrid":
        loss_fn = BCEDiceLoss(alpha=loss_alpha)

    """Define Optimizer for training"""
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate)
    if cnn_checkpoint is not None:
        optimizer.load_state_dict(save_train['optimizer_state_dict'])

        save_train = None
        del(save_train)

    """Learning rate for the optimizer"""
    if learning_rate_scheduler:
        learning_rate_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    else:
        learning_rate_scheduler = None

    """Output build CNN specification"""
    print(f"Training is started on {device}, with: "
          f"Loss function = {loss_function} "
          f"LR = {optimizer.param_groups[0]['lr']}, and "
          f"LRS = {learning_rate_scheduler}")

    print('The Network was build:')
    print(f"Network: {cnn_type}, "
          f"No. of Layers: {convolution_layer} with {convolution_multiplayer} multiplayer, "
          f"Each layer is build of {convolution_structure}, "
          f"Image patch size: {img.shape[2:]}, ")

    if dropout is None:
        print('No dropout layer are used.')
    else:
        print(
            f'Dropout with {dropout} propability is used for each conv. layer')

    trainer = Trainer(model=model.to(device),
                      device=device,
                      criterion=loss_fn,
                      optimizer=optimizer,
                      training_DataLoader=train_dataloader,
                      validation_DataLoader=test_dataloader,
                      epochs=epochs,
                      lr_scheduler=learning_rate_scheduler,
                      early_stop_rate=early_stop_rate,
                      tqdm=tqdm,
                      checkpoint_name=cnn_type,
                      classification=classification)

    trainer.run_trainer()

    if not isdir('model'):
        mkdir('model')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               join('model', 'model_weights.pth'))
