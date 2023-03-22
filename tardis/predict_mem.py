#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import warnings
from os import getcwd

import click

from tardis.utils.predictor import DataSetPredictor
from tardis.version import version

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option('-dir', '--dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-out', '--output_format',
              default='tif',
              type=click.Choice(['None_amSG', 'am_amSG', 'tif_amSG', 'mrc_amSG',
                                 'None_csv', 'am_csv', 'tif_csv', 'mrc_csv',
                                 'None_mrcM', 'mrc_mrcM', 'mrc_mrcM', 'mrc_mrcM']),
              help='Type of output.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=128,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.15,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-dt', '--dist_threshold',
              default=0.95,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Number of point per voxel.',
              show_default=True)
@click.option('-in', '--instances',
              default=False,
              type=bool,
              help='If True, try predict instances from semantic binary labels.',
              show_default=True)
@click.option('-dv', '--device',
              default=0,
              type=str,
              help='Define which device use for training: '
                   'gpu: Use ID 0 gpus'
                   'cpu: Usa CPU'
                   'mps: Apple silicon'
                   '0-9 - specified gpu device id to use',
              show_default=True)
@click.option('-db', '--debug',
              default=False,
              type=bool,
              help='If True, save output from each step for debugging.',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         output_format: str,
         patch_size: int,
         cnn_threshold: float,
         dist_threshold: float,
         points_in_patch: int,
         instances: bool,
         device: str,
         debug: bool):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    predictor = DataSetPredictor(predict='Microtubule',
                                 dir=dir,
                                 output_format=output_format,
                                 patch_size=patch_size,
                                 cnn_threshold=cnn_threshold,
                                 dist_threshold=dist_threshold,
                                 points_in_patch=points_in_patch,
                                 predict_with_rotation=False,
                                 amira_prefix=None,
                                 filter_by_length=None,
                                 connect_splines=None,
                                 connect_cylinder=None,
                                 amira_compare_distance=None,
                                 amira_inter_probability=None,
                                 instances=instances,
                                 device=device,
                                 debug=debug)
    predictor()


if __name__ == '__main__':
    main()
