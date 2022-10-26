import shutil

import torch
import torch.nn as nn
from tardis_dev.dist_pytorch.trainer import C_DistTrainer, DistTrainer
from tardis_dev.spindletorch.trainer import CNNTrainer
from tardis_dev.utils.device import get_device
from tardis_dev.utils.trainer import BasicTrainer
from torch.utils.data import DataLoader


def test_trainer_init():
    dl = BasicTrainer(model=nn.Conv1d(1, 2, 3),
                      structure={},
                      device='cpu',
                      criterion=None,
                      optimizer=None,
                      print_setting=[],
                      training_DataLoader=None,
                      validation_DataLoader=None,
                      lr_scheduler=None,
                      epochs=100,
                      early_stop_rate=10,
                      checkpoint_name='test')

    dl = CNNTrainer(model=nn.Conv1d(1, 2, 3),
                    device='cpu',
                    criterion=None,
                    optimizer=None,
                    print_setting=[],
                    training_DataLoader=None,
                    validation_DataLoader=None,
                    lr_scheduler=None,
                    epochs=100,
                    early_stop_rate=10,
                    checkpoint_name='test',
                    structure={'cnn_type': 'unet'},
                    classification=False)

    dl = DistTrainer(model=nn.Conv1d(1, 2, 3),
                     structure={'dist_type': 'instance',
                                'node_input': 0},
                     device='cpu',
                     criterion=None,
                     optimizer=None,
                     print_setting=[],
                     training_DataLoader=None,
                     validation_DataLoader=None,
                     lr_scheduler=None,
                     epochs=100,
                     early_stop_rate=10,
                     checkpoint_name='test')

    dl = C_DistTrainer(model=nn.Conv1d(1, 2, 3),
                       structure={'node_input': 0,
                                  'dist_type': 'semantic'},
                       device='cpu',
                       criterion=None,
                       optimizer=None,
                       print_setting=[],
                       training_DataLoader=None,
                       validation_DataLoader=None,
                       lr_scheduler=None,
                       epochs=100,
                       early_stop_rate=10,
                       checkpoint_name='test')

    dl = C_DistTrainer(model=nn.Conv1d(1, 2, 3),
                       structure={'node_input': 0,
                                  'dist_type': 'instance'},
                       device='cpu',
                       criterion=None,
                       optimizer=None,
                       print_setting=[],
                       training_DataLoader=None,
                       validation_DataLoader=None,
                       lr_scheduler=None,
                       epochs=100,
                       early_stop_rate=10,
                       checkpoint_name='test')


def test_trainer_utils():
    dl = BasicTrainer(model=nn.Conv1d(1, 2, 3),
                      structure={},
                      device=get_device('cpu'),
                      criterion=None,
                      optimizer=None,
                      print_setting=['test', 'test', 'test', 'test', 'test'],
                      training_DataLoader=DataLoader(dataset=torch.rand(1, 10, 10)),
                      validation_DataLoader=DataLoader(dataset=torch.rand(1, 10, 10)),
                      lr_scheduler=None,
                      epochs=100,
                      early_stop_rate=10,
                      checkpoint_name='test')

    dl._update_desc(stop_count=1,
                    f1=[0.1, 0.2])

    dl._update_progress_bar(loss_desc='test text',
                            idx=5)

    # Test training path till torch.save model state
    try:
        dl.run_trainer()
    except AttributeError:
        shutil.rmtree('./test_checkpoint')
        pass
