from os import mkdir
from os.path import isdir, join
from typing import Optional

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tardis.spindletorch.utils.dataset_loader import VolumeDataset
from tardis.spindletorch.utils.build_network import build_network
from tardis.spindletorch.unet.losses import BCELoss, BCEDiceLoss, DiceLoss, \
    AdaptiveDiceLoss
from tardis.spindletorch.unet.trainer import Trainer


# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def train(data_dir: Optional[str] = None,
          img_size=64,
          batch_size=10,
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
          device='gpu',
          epochs=100):
    """Build data directory"""
    if data_dir is None:
        img_train = join('data', 'train', 'imgs')
        mask_train = join('data', 'train', 'mask')
        img_test = join('data', 'test', 'imgs')
        mask_test = join('data', 'test', 'mask')
    else:
        img_train = join(data_dir, 'train', 'imgs')
        mask_train = join(data_dir, 'train', 'mask')
        img_test = join(data_dir, 'test', 'imgs')
        mask_test = join(data_dir, 'test', 'mask')

    """Build train and test dataset"""
    train_dataloader = DataLoader(VolumeDataset(img_dir=img_train,
                                                mask_dir=mask_train,
                                                size=img_size,
                                                mask_suffix='_mask',
                                                normalize="simple",
                                                transform=True,
                                                out_channels=1),
                                  shuffle=True,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=4)
    test_dataloader = DataLoader(VolumeDataset(img_dir=img_test,
                                               mask_dir=mask_test,
                                               size=img_size,
                                               mask_suffix='_mask',
                                               normalize="simple",
                                               transform=False,
                                               out_channels=1),
                                 shuffle=False,
                                 batch_size=1,
                                 pin_memory=True)

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
    if cnn_checkpoint:
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
                      notebook=False,
                      checkpoint_name=cnn_type,
                      classification=classification)

    trainer.run_trainer()
    if not isdir('model'):
        mkdir('model')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               join('model', 'model_weights.pth'))


if __name__ == '__main__':
    train()