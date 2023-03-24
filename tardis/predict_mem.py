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
              default='mrc_None',
              type=click.Choice(['None_amSG', 'am_amSG', 'mrc_amSG', 'tif_amSG',
                                 'None_mrcM', 'am_mrcM', 'mrc_mrcM', 'tif_mrcM',
                                 'None_tifM', 'am_tifM', 'mrc_tifM', 'tif_tifM',
                                 'None_mrcM', 'am_csv', 'mrc_csv', 'tif_csv',
                                 'am_None', 'mrc_None', 'tif_None']),
              help='Type of output files. The First optional output file is the binary mask '
                   'which can be of type None [no output], am [Amira], mrc or tif. '
                   'Second output is instance segmentation of objects, which can be '
                   'output as amSG [Amira], mrcM [mrc mask], tifM [tif mask], '
                   'csv coordinate file [ID, X, Y, Z] or None [no instance prediction].',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=128,
              type=int,
              help='Size of image patch used for prediction. This will break '
                   'the tomogram volumes into 3D patches where each patch will be '
                   'separately predicted and then stitched back together '
                   'with 25% overlap.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.15,
              type=float,
              help='Threshold used for CNN prediction..',
              show_default=True)
@click.option('-dt', '--dist_threshold',
              default=0.95,
              type=float,
              help='Threshold used for instance prediction.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Size of the cropped point cloud, given as a max. number of points '
                   'per crop. This will break generated from the binary mask '
                   'point cloud into smaller patches with overlap.',
              show_default=True)
@click.option('-dv', '--device',
              default=0,
              type=str,
              help='Define which device to use for training: '
                   'gpu: Use ID 0 GPU'
                   'cpu: Usa CPU'
                   'mps: Apple silicon'
                   '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-db', '--debug',
              default=False,
              type=bool,
              help=' If True, save the output from each step for debugging.',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         output_format: str,
         patch_size: int,
         cnn_threshold: float,
         dist_threshold: float,
         points_in_patch: int,
         device: str,
         debug: bool):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    out = output_format.split('_')
    if out[1] == 'None':
        instances = False
    else:
        instances = True

    predictor = DataSetPredictor(predict='Membrane',
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
