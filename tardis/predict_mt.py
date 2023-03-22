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
              default='None_amSG',
              type=click.Choice(['None_amSG', 'am_amSG', 'mrc_amSG', 'tif_amSG',
                                 'None_mrcM', 'am_mrcM', 'mrc_mrcM', 'tif_mrcM',
                                 'None_tifM', 'am_tifM', 'mrc_tifM', 'tif_tifM',
                                 'None_mrcM', 'am_csv', 'mrc_csv', 'tif_csv']),
              help='Type of output.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=128,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.5,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-dt', '--dist_threshold',
              default=0.75,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Number of point per voxel.',
              show_default=True)
@click.option('-ap', '--amira_prefix',
              default='.CorrelationLines',
              type=str,
              help='Prefix name for amira files.',
              show_default=True)
@click.option('-fl', '--filter_by_length',
              default=500,
              type=int,
              help='Filter out splines with length shorter then given length [A].',
              show_default=True)
@click.option('-cs', '--connect_splines',
              default=2500,
              type=int,
              help='Connect splines that are facing the same direction and are at'
                   'given max distance [A].',
              show_default=True)
@click.option('-cr', '--connect_cylinder',
              default=250,
              type=int,
              help='Cylinder radius used to for searching of the near spline [A].',
              show_default=True)
@click.option('-acd', '--amira_compare_distance',
              default=175,
              type=int,
              help='Distance threshold used to evaluate similarity between two '
                   'splines based on its coordinates [A].',
              show_default=True)
@click.option('-aip', '--amira_inter_probability',
              default=0.25,
              type=float,
              help='Interaction threshold used to evaluate reject splines that are'
                   'similar below that threshold.',
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
         filter_by_length: float,
         connect_splines: int,
         connect_cylinder: int,
         amira_prefix: str,
         amira_compare_distance: int,
         amira_inter_probability: float,
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
                                 amira_prefix=amira_prefix,
                                 filter_by_length=filter_by_length,
                                 connect_splines=connect_splines,
                                 connect_cylinder=connect_cylinder,
                                 amira_compare_distance=amira_compare_distance,
                                 amira_inter_probability=amira_inter_probability,
                                 instances=True,
                                 device=device,
                                 debug=debug)

    predictor()


if __name__ == '__main__':
    main()
