#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import torch.nn as nn
from torch import optim

from tardis_em.dist_pytorch.trainer import CDistTrainer, DistTrainer
from tardis_em.cnn.trainer import CNNTrainer
from tardis_em.utils.trainer import BasicTrainer


def test_trainer_init():
    BasicTrainer(
        model=nn.Conv1d(1, 2, 3),
        structure={},
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        lr_scheduler=False,
        print_setting=(),
        training_DataLoader=None,
        checkpoint_name="test",
    )

    CNNTrainer(
        model=nn.Conv1d(1, 2, 3),
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        print_setting=[],
        training_DataLoader=None,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        checkpoint_name="test",
        structure={"cnn_type": "unet"},
        classification=False,
    )

    DistTrainer(
        model=nn.Conv1d(1, 2, 3),
        structure={"dist_type": "instance", "node_input": 0},
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        print_setting=[],
        training_DataLoader=None,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        checkpoint_name="test",
    )

    CDistTrainer(
        model=nn.Conv1d(1, 2, 3),
        structure={"node_input": 0, "dist_type": "semantic"},
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        print_setting=[],
        training_DataLoader=None,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        checkpoint_name="test",
    )

    CDistTrainer(
        model=nn.Conv1d(1, 2, 3),
        structure={"node_input": 0, "dist_type": "instance"},
        device="cpu",
        criterion=None,
        optimizer=optim.Adam(nn.Conv1d(1, 2, 3).parameters(), 0.001),
        print_setting=[],
        training_DataLoader=None,
        validation_DataLoader=None,
        lr_scheduler=False,
        epochs=100,
        early_stop_rate=10,
        checkpoint_name="test",
    )
