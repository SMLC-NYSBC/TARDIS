#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Optional, Union

import click
import torch
from torch.utils.data import DataLoader

from tardis_em.cnn.datasets.build_dataset import build_train_dataset
from tardis_em.cnn.datasets.dataloader import CNNDataset
from tardis_em.cnn.train import train_cnn
from tardis_em.utils.dataset import build_test_dataset
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.logo import TardisLogo
from tardis_em.utils.setup_envir import check_dir
from tardis_em._version import version


@click.command()
@click.option(
    "-dir",
    "--path",
    default=getcwd(),
    type=str,
    help="Directory with train, test folder or folder with dataset "
    "to be used for training.",
    show_default=True,
)
@click.option(
    "-ps",
    "--patch_size",
    default=64,
    type=int,
    help="Image size used for prediction.",
    show_default=True,
)
@click.option(
    "-px",
    "--pixel_size",
    default=None,
    type=float,
    help="Pixel size in [A] to which all images are resize.",
    show_default=True,
)
@click.option(
    "-ms",
    "--mask_size",
    default=150,
    type=int,
    help="Size of drawn mask in A. If you are using .csv files as labels.",
    show_default=True,
)
@click.option(
    "-cnn",
    "--cnn_type",
    default="unet",
    type=click.Choice(
        ["unet", "resunet", "unet3plus", "big_unet", "fnet", "fnet_attn"]
    ),
    help="Type of NN used for training.",
    show_default=True,
)
@click.option(
    "-co",
    "--cnn_out_channel",
    default=1,
    type=int,
    help="Number of output channels for the NN.",
    show_default=True,
)
@click.option(
    "-b",
    "--training_batch_size",
    default=25,
    type=int,
    help="Batch size.",
    show_default=True,
)
@click.option(
    "-cl",
    "--cnn_layers",
    default=5,
    type=int,
    help="Number of convolution layer for NN.",
    show_default=True,
)
@click.option(
    "-cm",
    "--cnn_scaler",
    default=32,
    type=int,
    help="Convolution multiplayer for CNN layers.",
    show_default=True,
)
@click.option(
    "-cs",
    "--cnn_structure",
    default="3gcl",
    type=str,
    help="Define structure of the convolution layer."
    "2 or 3 - dimension in 2D or 3D"
    "c - convolution"
    "g - group normalization"
    "b - batch normalization"
    "r - ReLU"
    "l - LeakyReLU"
    "e - GeLu"
    "p - PReLu",
    show_default=True,
)
@click.option(
    "-ck",
    "--conv_kernel",
    default=3,
    type=int,
    help="Kernel size for 2D or 3D convolution.",
    show_default=True,
)
@click.option(
    "-cp",
    "--conv_padding",
    default=1,
    type=int,
    help="Padding size for convolution.",
    show_default=True,
)
@click.option(
    "-cmpk",
    "--pool_kernel",
    default=2,
    type=int,
    help="Max_pooling kernel.",
    show_default=True,
)
@click.option(
    "-l",
    "--cnn_loss",
    default="BCELoss",
    type=str,
    help="Loss function use for training.",
    show_default=True,
)
@click.option(
    "-lr",
    "--loss_lr_rate",
    default=0.0005,
    type=float,
    help="Learning rate for NN.",
    show_default=True,
)
@click.option(
    "-lrs",
    "--lr_rate_schedule",
    default=False,
    type=bool,
    help="If True learning rate scheduler is used.",
    show_default=True,
)
@click.option(
    "-dv",
    "--device",
    default=0,
    type=str,
    help="Define which device use for training: "
    "gpu: Use ID 0 gpus"
    "cpu: Usa CPU"
    "mps: Apple silicon"
    "0-9 - specified gpu device id to use",
    show_default=True,
)
@click.option(
    "-w",
    "--warmup",
    default=100,
    type=int,
    help="Number of warmup steps.",
    show_default=True,
)
@click.option(
    "-e",
    "--epochs",
    default=10000,
    type=int,
    help="Number of epoches.",
    show_default=True,
)
@click.option(
    "-es",
    "--early_stop",
    default=1000,
    type=int,
    help="Number of epoches without improvement after which early stop "
    "is initiated. Default should is 10% of the total number of epochs.",
    show_default=True,
)
@click.option(
    "-cch",
    "--cnn_checkpoint",
    default=None,
    type=str,
    help="If indicated, dir to training checkpoint to reinitialized training.",
    show_default=True,
)
@click.option(
    "-dp",
    "--dropout_rate",
    default=0.5,
    type=float,
    help="If indicated, value of dropout for CNN.",
    show_default=True,
)
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(
    path: str,
    patch_size: int,
    pixel_size: Union[float, None],
    mask_size: int,
    cnn_type: str,
    cnn_out_channel: int,
    training_batch_size: int,
    cnn_layers: int,
    cnn_scaler: int,
    conv_kernel: int,
    conv_padding: int,
    pool_kernel: int,
    cnn_structure: str,
    cnn_loss: str,
    loss_lr_rate: float,
    lr_rate_schedule: bool,
    device: str,
    warmup: int,
    epochs: int,
    early_stop: int,
    cnn_checkpoint: Optional[str] = None,
    dropout_rate: Optional[float] = None,
    test_click=False,
):
    """
    MAIN MODULE FOR TRAINING CNN UNET/RESUNET/UNET3PLUS MODELS
    """
    """Initialize TARDIS progress bar"""
    tardis_logo = TardisLogo()
    tardis_logo(title="CNN training module")

    """Set environment"""
    TRAIN_IMAGE_DIR = join(path, "train", "imgs")
    TRAIN_MASK_DIR = join(path, "train", "masks")
    TEST_IMAGE_DIR = join(path, "test", "imgs")
    TEST_MASK_DIR = join(path, "test", "masks")

    IMG_FORMAT = (".tif", ".am", ".mrc", ".rec", ".map")

    """Check if dir has train/test folder and if folder have compatible data"""
    DATASET_TEST = check_dir(
        dir_s=path,
        with_img=True,
        train_img=TRAIN_IMAGE_DIR,
        train_mask=TRAIN_MASK_DIR,
        img_format=IMG_FORMAT,
        test_img=TEST_IMAGE_DIR,
        test_mask=TEST_MASK_DIR,
        mask_format=(
            "_mask.am",
            ".CorrelationLines.am",
            "_mask.mrc",
            "_mask.tif",
            "_mask.csv",
        ),
    )

    """Optionally: Set-up environment if not existing"""
    if not DATASET_TEST:
        # Check and set-up environment
        if not len([f for f in listdir(path) if f.endswith(IMG_FORMAT)]) > 0:
            TardisError(
                "100",
                "tardis_em/train_cnn.py",
                "Indicated folder for training do not have any compatible "
                "data or one of the following folders: "
                "test/imgs; test/masks; train/imgs; train/masks",
            )

        if isdir(join(path, "train")):
            rmtree(join(path, "train"))

        mkdir(join(path, "train"))
        mkdir(TRAIN_IMAGE_DIR)
        mkdir(TRAIN_MASK_DIR)

        if isdir(join(path, "test")):
            rmtree(join(path, "test"))

        mkdir(join(path, "test"))
        mkdir(TEST_IMAGE_DIR)
        mkdir(TEST_MASK_DIR)

        # Build train and test dataset
        build_train_dataset(
            dataset_dir=path,
            circle_size=mask_size,
            resize_pixel_size=pixel_size,
            trim_xy=patch_size,
            trim_z=patch_size,
        )

        no_dataset = int(len([f for f in listdir(path) if f.endswith(IMG_FORMAT)]) / 2)
        build_test_dataset(dataset_dir=path, dataset_no=no_dataset)

    """Build training and test dataset 2D/3D"""
    if not test_click:
        train_DL = DataLoader(
            dataset=CNNDataset(
                img_dir=TRAIN_IMAGE_DIR,
                mask_dir=TRAIN_MASK_DIR,
                size=patch_size,
                out_channels=cnn_out_channel,
            ),
            batch_size=training_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        test_DL = DataLoader(
            dataset=CNNDataset(
                img_dir=TEST_IMAGE_DIR,
                mask_dir=TEST_MASK_DIR,
                size=patch_size,
                out_channels=cnn_out_channel,
            ),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    if cnn_out_channel > 1:
        cnn_loss = "CELoss"

    if loss_lr_rate == 1.0:
        lr_rate_schedule = True

    """Get device"""
    device = get_device(device)

    """Model structure dictionary"""
    """Optionally: pre-load model structure from checkpoint"""
    model_dict = None
    if cnn_checkpoint is not None:
        save_train = torch.load(cnn_checkpoint, map_location=device, weights_only=False)

        if "model_struct_dict" in save_train.keys():
            model_dict = save_train["model_struct_dict"]
            globals().update(model_dict)
    if model_dict is None:
        model_dict = {
            "cnn_type": cnn_type,
            "classification": False,
            "in_channel": 1,
            "out_channel": cnn_out_channel,
            "img_size": patch_size,
            "dropout": dropout_rate,
            "num_conv_layers": cnn_layers,
            "conv_scaler": cnn_scaler,
            "conv_kernel": conv_kernel,
            "conv_padding": conv_padding,
            "maxpool_kernel": pool_kernel,
            "layer_components": cnn_structure,
            "attn_features": True if cnn_type == "fnet_attn" else False,
            "num_group": 8,
            "prediction": False,
        }

    """Run Training loop"""
    if not test_click:
        train_cnn(
            train_dataloader=train_DL,
            test_dataloader=test_DL,
            model_structure=model_dict,
            checkpoint=cnn_checkpoint,
            loss_function=cnn_loss,
            learning_rate=loss_lr_rate,
            learning_rate_scheduler=lr_rate_schedule,
            early_stop_rate=early_stop,
            device=device,
            warmup=warmup,
            epochs=epochs,
        )
    else:
        rmtree(join(path, "train"))
        rmtree(join(path, "test"))


if __name__ == "__main__":
    main()
