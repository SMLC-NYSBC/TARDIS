#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from os import listdir
from os.path import expanduser, join
from typing import Union

import click
import torch

from tardis.benchmarks.predictor import CnnBenchmark, DISTBenchmark
from tardis.utils.aws import get_benchmark_aws, get_model_aws
from tardis.utils.device import get_device
from tardis.utils.errors import TardisError
from tardis.utils.logo import TardisLogo
from tardis.utils.predictor import Predictor
from tardis.version import version


@click.command()
@click.option('-ds', '--data_set',
              type=str,
              help='Data set name used for testing.',
              show_default=True)
@click.option('-ch', '--model_checkpoint',
              type=str,
              help='Dir or https:/.. for NN with model weight and structure.',
              show_default=True)
@click.option('-th', '--nn_threshold',
              default=0.5,
              type=float,
              help='Threshold use for NN prediction.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=None,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-sg', '--sigma',
              default=None,
              type=float,
              help='Sigma value for distance embedding.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Number of point per voxel.',
              show_default=True)
@click.option('-d', '--device',
              default='0',
              type=str,
              help='Define which device use for training: '
                   'gpu: Use ID 0 GPUs '
                   'cpu: Usa CPU '
                   '0-9 - specified GPU device id to use',
              show_default=True)
@click.version_option(version=version)
def main(data_set: str,
         model_checkpoint: str,
         nn_threshold: float,
         patch_size: Union[None, int],
         sigma: Union[None, float],
         points_in_patch: Union[None, int],
         device: str):
    """
    Standard benchmark for DIST on medical and standard point clouds

    Benchmark to the following:
    - Identified standard location for test data and sanity_checks
    - Retrieve json with best CNN metric from S3 bucket
    - Build NN from checkpoint or accept model
    - Run benchmark on standard data
    - For each data calculate F1, AP-25, AP-50, AP-75
    - Get mean value for each metric

    ToDo: Check if json have metric for tested dataset
    ToDo: Check if metrics are higher. If yes update json

    ToDo: If metric higher, sent json and save .pth with model structure at standard dir
    """
    """Global setting"""
    tardis_progress = TardisLogo()
    title = 'TARDIS - NN Benchmark'
    tardis_progress(title=title)

    DIR_ = join(expanduser('~') + '/../../data/rkiewisz/Benchmarks')

    """Get model for benchmark"""
    if model_checkpoint.startswith('http'):
        model_checkpoint = get_model_aws(model_checkpoint).content

    model = torch.load(model_checkpoint,
                       map_location=device)

    """Best model list from S3"""
    if [True for x in model['model_struct_dict'] if x.startswith('cnn')]:
        network = 'cnn'
        DIR_NN = join(DIR_, 'Best_model_CNN', data_set)
        DIR_EVAL = join(DIR_, 'Eval_CNN', data_set)
    else:
        network = 'dist'
        DIR_NN = join(DIR_, 'Best_model_DIST', data_set)
        DIR_EVAL = join(DIR_, 'Eval_DIST', data_set)

    if data_set not in listdir(DIR_NN):
        TardisError(id='',
                    py='tardis/benchmarks/benchmarks.py',
                    desc=f'Given data set {data_set} is not supporter! '
                    f'Expected one of {listdir(DIR_NN)}')

    BEST_SCORE = get_benchmark_aws(network)

    model = Predictor(checkpoint=model,
                      img_size=patch_size,
                      sigma=sigma,
                      device=get_device(device))

    """Build DataLoader"""
    if network == 'cnn':
        predictor_bch = CnnBenchmark(model=model,
                                     dataset=data_set,
                                     dir_=DIR_EVAL,
                                     threshold=nn_threshold,
                                     patch_size=patch_size)
    else:
        predictor_bch = DISTBenchmark(model=model,
                                      dataset=data_set,
                                      dir_=DIR_EVAL,
                                      threshold=nn_threshold,
                                      points_in_patch=points_in_patch)

    """Compared with best models"""
    
    """Benchmark Summary"""
    benchmark_result = predictor_bch()

    """Sent updated json and model to S3"""
    pass


if __name__ == '__main__':
    main()
