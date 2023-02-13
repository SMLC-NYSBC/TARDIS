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

import torch

from tardis.utils.aws import get_benchmark_aws, get_model_aws
from tardis.utils.device import get_device
from tardis.utils.errors import TardisError
from tardis.utils.predictor import Predictor


def main(data_set: str,
         model_checkpoint: str,
         img_size: Union[None, int],
         sigma: Union[None, float],
         device: str):
    """
    Standard benchmark for DIST on medical and standard point clouds

    Done: Identified standard location for test data and sanity_checks
    Done: Retrieve json with best CNN metric from S3 bucket
    Done: Build CNN from checkpoint or accept model

    ToDo: Run benchmark on standard data
    ToDo: For each data calculate F1, AP-25, AP-50, AP-75
    ToDo: Get mean value for each metric

    ToDo: Check if json have metric for tested dataset
    ToDo: Check if metrics are higher. If yes update json

    ToDo: If metric higher, sent json and save .pth with model structure at standard dir

    Args:
        data_set (str): Dataset name
        model_checkpoint (str): Directory for dict with model checkpoint with
         model structure.
        img_size (None, int):
        sigma (None, float):
        device (str):
    """
    """Global setting"""
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
                      img_size=img_size,
                      sigma=sigma,
                      device=get_device(device))

    """Predict Dataset"""

    """Run Benchmark"""

    """Compared with best models"""

    """Sent updated json and model to S3"""
    pass


if __name__ == '__main__':
    main()
