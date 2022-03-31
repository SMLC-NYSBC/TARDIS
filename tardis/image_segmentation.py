from os import listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Optional

from torch.utils.data import DataLoader

from tardis.slcpy_data_processing.build_training_dataset import BuildDataSet
from tardis.spindletorch.train import train
from tardis.spindletorch.utils.dataset_loader import VolumeDataset


def main(training_dataset: str,
         image_size: int,
         cnn_type: str,
         cnn_out_channel: int,
         taining_batch_size: int,
         cnn_layers: int,
         cnn_multiplayer: int,
         cnn_structure: str,
         cnn_loss: str,
         loss_lr_rate: float,
         lr_rate_schedule: bool,
         device: str,
         epochs: int,
         train_test_ratio: Optional[int] = 10,
         loss_alpha: Optional[float] = None,
         cnn_checkpoint: Optional[str] = None,
         dropout_rate: Optional[float] = None):
    """Set environment"""
    train_imgs_dir = join(training_dataset, 'train', 'imgs')
    train_masks_dir = join(training_dataset, 'train', 'masks')
    test_imgs_dir = join(training_dataset, 'train', 'imgs')
    test_masks_dir = join(training_dataset, 'train', 'masks')
    dataset_test = False

    # Check if dir has train/test folder and if folder have data
    if isdir(join(training_dataset, 'train')) and isdir(join(training_dataset, 'test')):
        dataset_test = True

        # Check if train img and mask exist and have same files
        if isdir(train_imgs_dir) and isdir(train_masks_dir):
            if len([f for f in listdir(train_imgs_dir) if f.endswith('.tif')]) == \
                    len([f for f in listdir(train_masks_dir) if f.endswith('.tif')]):
                if len([f for f in listdir(train_imgs_dir) if f.endswith('.tif')]) == 0:
                    dataset_test = False
            else:
                dataset_test = False

        # Check if test img and mask exist and have same files
        if isdir(test_imgs_dir) and isdir(test_masks_dir):
            if len([f for f in listdir(test_imgs_dir) if f.endswith('.tif')]) == \
                    len([f for f in listdir(test_masks_dir) if f.endswith('.tif')]):
                if len([f for f in listdir(test_imgs_dir) if f.endswith('.tif')]) == 0:
                    dataset_test = False
            else:
                dataset_test = False

    # If any incompatibility and data exist, build dataset
    if not dataset_test:
        assert len([f for f in listdir(train_imgs_dir) if f.endswith('.tif')]) > 0, \
            'Indicated folder for training do not have any compatible data or ' \
            'one of the following folders: '\
            'test/imgs; test/masks; train/imgs; train/masks'
        if isdir(join(training_dataset, 'train')):
            rmtree(join(training_dataset, 'train'))

            mkdir(join(training_dataset, 'train'))
            mkdir(train_imgs_dir)
            mkdir(train_masks_dir)

        if isdir(join(training_dataset, 'test')):
            rmtree(join(training_dataset, 'test'))

            mkdir(join(training_dataset, 'test'))
            mkdir(test_imgs_dir)
            mkdir(test_masks_dir)

        """Build train/test DataSets if they don't exist"""
        dataset_builder = BuildDataSet(dataset_dir=training_dataset,
                                       circle_size=250,
                                       multi_layer=False,
                                       tqdm=True)
        dataset_builder.__builddataset__(trim_xy=image_size,
                                         trim_z=image_size)

    """Build training and test dataset 2D/3D"""
    train_DL = DataLoader(dataset=VolumeDataset(img_dir=train_imgs_dir,
                                                mask_dir=train_masks_dir,
                                                size=image_size,
                                                mask_suffix='_mask',
                                                normalize='simple',
                                                transform=True,
                                                out_channels=cnn_out_channel),
                          batch_size=taining_batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)

    test_DL = DataLoader(dataset=VolumeDataset(img_dir=test_imgs_dir,
                                               mask_dir=test_masks_dir,
                                               size=image_size,
                                               mask_suffix='_mask',
                                               normalize='simple',
                                               transform=True,
                                               out_channels=cnn_out_channel),
                         batch_size=1,
                         shuffle=True,
                         num_workers=8,
                         pin_memory=True)

    """Run Training loop"""
    train(train_dataloader=train_DL,
          test_dataloader=test_DL,
          img_size=image_size,
          cnn_type=cnn_type,
          classification=False,
          dropout=dropout_rate,
          convolution_layer=cnn_layers,
          convolution_multiplayer=cnn_multiplayer,
          convolution_structure=cnn_structure,
          cnn_checkpoint=cnn_checkpoint,
          loss_function=cnn_loss,
          loss_alpha=loss_alpha,
          learning_rate=loss_lr_rate,
          learning_rate_scheduler=lr_rate_schedule,
          device=device,
          epochs=epochs)


if __name__ == '__main__':
    main()
