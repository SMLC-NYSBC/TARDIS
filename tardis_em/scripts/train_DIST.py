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
from os import getcwd, listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import Union

import click
import torch

from tardis_em.dist_pytorch.datasets.dataloader import build_dataset
from tardis_em.dist_pytorch.train import train_dist
from tardis_em.utils.dataset import build_test_dataset, move_train_dataset
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
    "-dt",
    "--dataset_type",
    default="filament",
    type=str,
    help="Define training dataset type.",
    show_default=True,
)
@click.option(
    "-no",
    "--n_out",
    default=1,
    type=int,
    help="Number of output channels in DIST.",
    show_default=True,
)
@click.option(
    "-nd",
    "--node_dim",
    default=0,
    type=int,
    help="Number embedding channels for nodes.",
    show_default=True,
)
@click.option(
    "-ed",
    "--edge_dim",
    default=128,
    type=int,
    help="Number embedding channels for edges.",
    show_default=True,
)
@click.option(
    "-eg",
    "--edge_angles",
    default=False,
    type=bool,
    help="Perfomer angle embedding.",
    show_default=True,
)
@click.option(
    "-nk",
    "--num_knn",
    default=None,
    type=int,
    help="Number of KNN used for building sparse cdist matrix.",
    show_default=True,
)
@click.option(
    "-ly",
    "--layers",
    default=6,
    type=int,
    help="Number of DIST layers",
    show_default=True,
)
@click.option(
    "-hd",
    "--heads",
    default=8,
    type=int,
    help="Number of DIST heads in MHA",
    show_default=True,
)
@click.option(
    "-dp",
    "--dropout",
    default=0,
    type=float,
    help="If 0, dropout is turn-off. Else indicate dropout rate.",
    show_default=True,
)
@click.option(
    "-nsg",
    "--node_sigma",
    default=1.0,
    type=float,
    help="Sigma value for RGB node embedding.",
    show_default=True,
)
@click.option(
    "-esg",
    "--edge_sigma",
    default=2.0,
    type=str,
    help="Sigma value for distance edges embedding.",
    show_default=True,
)
@click.option(
    "-st",
    "--structure",
    default="triang",
    type=click.Choice(["full", "full_af", "self_attn", "triang", "dualtriang", "quad"]),
    help="Structure of the DIST layers.",
    show_default=True,
)
@click.option(
    "-ds",
    "--dist_structure",
    default="instance",
    type=click.Choice(["instance", "instance-sparse", "semantic"]),
    help="Type of DIST model prediction.",
    show_default=True,
)
@click.option(
    "-sp",
    "--scale_pc",
    default="v_0.05",
    type=str,
    help="Point cloud downsampling factor.",
    show_default=True,
)
@click.option(
    "-ps",
    "--pc_sampling",
    default=500,
    type=int,
    help="Max number of points per patch.",
    show_default=True,
)
@click.option(
    "-lo",
    "--loss",
    default="BCELoss",
    type=str,
    help="Type of loss function use for training.",
    show_default=True,
)
@click.option(
    "-lr",
    "--loss_lr",
    default=1.0,
    type=float,
    help="Learning rate.",
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
    "-ch",
    "--checkpoint",
    default=None,
    type=str,
    help="If not None, directory to checkpoint.",
    show_default=True,
)
@click.option(
    "-dv",
    "--device",
    default=0,
    type=str,
    help="Define which device use for training: "
    "gpu: Use ID 0 gpus "
    "cpu: Usa CPU "
    "0-9 - specified gpu device id to use",
    show_default=True,
)
@click.option(
    "-ep",
    "--epochs",
    default=100,
    type=int,
    help="Number of epoches.",
    show_default=True,
)
@click.option(
    "-er",
    "--early_stop",
    default=10,
    type=int,
    help="Number or epoch's without improvement, " "after which training is stopped.",
    show_default=True,
)
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(
    path: str,
    dataset_type: str,
    n_out: int,
    node_dim: int,
    edge_dim: int,
    edge_angles: bool,
    num_knn: int,
    layers: int,
    heads: int,
    dropout: float,
    node_sigma: float,
    edge_sigma: Union[float, str],
    structure: str,
    dist_structure: str,
    pc_sampling: int,
    scale_pc: float,
    loss: str,
    loss_lr: float,
    lr_rate_schedule: bool,
    checkpoint,
    device: str,
    epochs: int,
    early_stop: int,
    test_click=False,
):
    """Initialize TARDIS progress bar"""
    tardis_logo = TardisLogo()
    tardis_logo(title="DIST training module")

    """Stor all temp. directories"""
    TRAIN_IMAGE_DIR = join(path, "train", "imgs")
    TRAIN_COORD_DIR = join(path, "train", "masks")
    TEST_IMAGE_DIR = join(path, "test", "imgs")
    TEST_COORD_DIR = join(path, "test", "masks")

    COORD_FORMAT = ".txt"
    if dataset_type not in ["stanford", "stanford_rgb"]:
        COORD_FORMAT = (".CorrelationLines.am", ".npy", ".csv", ".ply")
    if dataset_type.startswith("simulate_"):
        COORD_FORMAT = None

    """Check if dir has train/test folder and if f  older have compatible data"""
    DATASET_TEST = check_dir(
        dir_s=path,
        train_img=TRAIN_IMAGE_DIR,
        train_mask=TRAIN_COORD_DIR,
        img_format=(),
        test_img=TEST_IMAGE_DIR,
        test_mask=TEST_COORD_DIR,
        mask_format=COORD_FORMAT,
        with_img=False,
    )

    """Optionally: Set-up environment if not existing"""
    if not DATASET_TEST:
        if COORD_FORMAT is not None:
            # Check and set-up environment
            if not len([f for f in listdir(path) if f.endswith(COORD_FORMAT)]) > 0:
                if dataset_type not in ["stanford", "stanford_rgb"]:
                    TardisError(
                        "12",
                        "tardis_em/train_DIST.py",
                        "Indicated folder for training do not have any compatible "
                        "data or one of the following folders: "
                        "test/imgs; test/masks; train/imgs; train/masks",
                    )

            if isdir(join(path, "train")):
                rmtree(join(path, "train"))
            mkdir(join(path, "train"))
            mkdir(TRAIN_IMAGE_DIR)
            mkdir(TRAIN_COORD_DIR)

            if isdir(join(path, "test")):
                rmtree(join(path, "test"))
            mkdir(join(path, "test"))
            mkdir(TEST_IMAGE_DIR)
            mkdir(TEST_COORD_DIR)

            # Build train and test dataset
            move_train_dataset(dir_s=path, coord_format=COORD_FORMAT, with_img=False)

            no_dataset = int(
                len([f for f in listdir(path) if f.endswith(COORD_FORMAT)]) / 2
            )

            if dataset_type in ["stanford", "stanford_rgb"]:
                build_test_dataset(
                    dataset_dir=path, dataset_no=no_dataset, stanford=True
                )
            else:
                build_test_dataset(dataset_dir=path, dataset_no=no_dataset)

    if dataset_type.startswith("simulate_"):
        dataset_type = dataset_type.split("_")

    """Pre-setting for building DataLoader"""
    # Check for general dataset
    if dataset_type == "general":
        tardis_logo(
            text_1=f"General DataSet loader is not supported in TARDIS {version}"
        )
        sys.exit()

    """Build DataLoader for training/validation"""
    dl_train_graph, dl_test_graph = build_dataset(
        dataset_type=dataset_type,
        dirs=[TRAIN_COORD_DIR, TEST_COORD_DIR],
        max_points_per_patch=pc_sampling,
        downscale=scale_pc,
    )

    """Setup training"""
    device = get_device(device)

    if not isinstance(dataset_type, list):
        if dataset_type.endswith("rgb"):
            if node_dim == 0:
                TardisError(
                    "161",
                    "tardis_em/train_DIST.py",
                    "Model initiated with node feasters as RGB but "
                    f"node_dim is {node_dim}.",
                )
                sys.exit()
    if node_dim > 0:
        node_input = 3
    else:
        node_input = 0
    if isinstance(edge_sigma, str):
        edge_sigma = edge_sigma.split("_")
        edge_sigma = [float(x) for x in edge_sigma]

        if len(edge_sigma) == 1:
            edge_sigma = edge_sigma[0]

    if dist_structure in ["instance", "instance-sparse"]:
        num_cls = None
    elif dist_structure == "semantic":
        num_cls = 200
    else:
        tardis_logo(text_1=f"ValueError: Wrong DIST type {dist_structure}!")
        sys.exit()

    if loss_lr == 1.0:
        lr_rate_schedule = True

    """Optionally: pre-load model structure from checkpoint"""
    model_dict = {}
    if checkpoint is not None:
        save_train = torch.load(
            join(checkpoint), map_location=device, weights_only=False
        )

        if "model_struct_dict" in save_train.keys():
            model_dict = save_train["model_struct_dict"]
            globals().update(model_dict)

    if len(model_dict) == 0:
        model_dict = {
            "dist_type": dist_structure,
            "n_out": n_out,
            "node_input": node_input,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "num_knn": num_knn,
            "num_cls": num_cls,
            "num_layers": layers,
            "num_heads": heads,
            "rgb_embed_sigma": node_sigma,
            "coord_embed_sigma": edge_sigma,
            "dropout_rate": dropout,
            "structure": structure,
        }

    if not test_click:
        train_dist(
            dataset_type=dataset_type,
            edge_angles=edge_angles,
            train_dataloader=dl_train_graph,
            test_dataloader=dl_test_graph,
            model_structure=model_dict,
            checkpoint=checkpoint,
            loss_function=loss,
            learning_rate=loss_lr,
            lr_scheduler=lr_rate_schedule,
            early_stop_rate=early_stop,
            device=device,
            epochs=epochs,
        )
    else:
        rmtree(join(path, "train"))
        rmtree(join(path, "temp_train"))
        rmtree(join(path, "test"))
        rmtree(join(path, "temp_test"))


if __name__ == "__main__":
    main()
