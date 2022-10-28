import warnings
from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import open3d as o3d
import tifffile.tifffile as tif

from tardis_dev.dist_pytorch.datasets.patches import PatchDataSet
from tardis_dev.dist_pytorch.utils.build_point_cloud import ImageToPointCloud
from tardis_dev.dist_pytorch.utils.segment_point_cloud import GraphInstanceV2
from tardis_dev.dist_pytorch.utils.utils import pc_median_dist
from tardis_dev.spindletorch.data_processing.stitch import StitchImages
from tardis_dev.spindletorch.data_processing.trim import (scale_image,
                                                          trim_with_stride)
from tardis_dev.spindletorch.datasets.dataloader import PredictionDataset
from tardis_dev.spindletorch.predictor import Predictor
from tardis_dev.utils.device import get_device
from tardis_dev.utils.export_data import NumpyToAmira
from tardis_dev.utils.load_data import load_image, import_am
from tardis_dev.utils.logo import Tardis_Logo, printProgressBar
from tardis_dev.utils.setup_envir import build_temp_dir, clean_up
from tardis_dev.utils.utils import check_uint8
from tardis_dev.version import version

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option('-dir', '--dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=96,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-cnn', '--cnn_network',
              default='unet_32',
              type=str,
              help='CNN network name.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.1,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-dt', '--dist_threshold',
              default=0.5,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Number of point per voxal.',
              show_default=True)
@click.option('-ch', '--checkpoints',
              default=False,
              type=bool,
              help='If True, include offline checkpoints',
              show_default=True)
@click.option('-cch', '--cnn_checkpoint',
              default=None,
              type=str,
              help='If not None, str checkpoints for Big_Unet',
              show_default=True)
@click.option('-dch', '--dist_checkpoint',
              default=None,
              type=str,
              help='If not None, str checkpoints for DIST',
              show_default=True)
@click.option('-d', '--device',
              default='0',
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 GPUs '
              'cpu: Usa CPU '
              '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-db', '--debug',
              default=False,
              type=bool,
              help='If True, save output from each step for debugging.',
              show_default=True)
@click.option('-v', '--visualizer',
              default=None,
              type=click.Choice(['f', 'p']),
              help='If not None, output visualization of the prediction'
              'f: Output as filaments'
              'p: Output as segmented point cloud',
              show_default=True)
@click.version_option(version=version)
def main(dir: str,
         patch_size: int,
         cnn_network: str,
         cnn_threshold: float,
         dist_threshold: float,
         points_in_patch: int,
         checkpoints: bool,
         device: str,
         debug: bool,
         visualizer: Optional[str] = None,
         cnn_checkpoint: Optional[str] = None,
         dist_checkpoint: Optional[str] = None):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    """Initial Setup"""
    if debug:
        str_debug = '<Debugging Mode>'
    else:
        str_debug = ''

    tardis_progress = Tardis_Logo()
    tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}')

    # Searching for available images for prediction
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(dir, 'temp', 'Predictions')
    am_output = join(dir, 'Predictions')

    predict_list = [f for f in listdir(dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given directory!'
    tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                    text_1=f'Found {len(predict_list)} images to predict!',
                    text_5='Point Cloud: In processing...',
                    text_7='Current Task: Set-up environment...')

    # Build handler's
    image_stitcher = StitchImages()
    post_processer = ImageToPointCloud()
    BuildAmira = NumpyToAmira()

    # Build CNN from checkpoints
    if checkpoints:
        checkpoints = (cnn_checkpoint, dist_checkpoint)
    else:
        checkpoints = (None, None)

    device = get_device(device)
    cnn_network = cnn_network.split('_')
    assert len(cnn_network) == 2, 'CNN type should be formatted as `unet_32`'

    predict_cnn = Predictor(checkpoint=checkpoints[0],
                            network=cnn_network[0],
                            subtype=int(cnn_network[1]),
                            device=device)

    predict_dist = Predictor(checkpoint=checkpoints[2],
                             network='dist',
                             subtype='without_img',
                             model_type='microtubules',
                             device=device)

    """Process each image with CNN and GF"""
    tardis_progress = Tardis_Logo()
    for id, i in enumerate(sorted(predict_list)):
        """Pre-Processing"""
        if i.endswith('CorrelationLines.am'):
            continue

        out_format = 0
        if i.endswith(('.tif', '.mrc', '.rec')):
            out_format = 4
        elif i.endswith('.tiff'):
            out_format = 5
        elif i.endswith('.am'):
            out_format = 3

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4='Pixel size: Nan A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Preprocessing for CNN...')

        # Build temp dir
        build_temp_dir(dir=dir)

        # Cut image for smaller image
        if i.endswith('.am'):
            image, px, _, transformation = import_am(am_file=join(dir, i))
        else:
            image, px = load_image(join(dir, i))

        scale_factor = px / 25
        org_shape = image.shape
        scale_shape = np.multiply(org_shape, scale_factor).astype(np.int16)

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7=f'Current Task: Sub-dividing images for {patch_size} size')

        trim_with_stride(image=image,
                         scale=scale_factor,
                         trim_size_xy=patch_size,
                         trim_size_z=patch_size,
                         output=join(dir, 'temp', 'Patches'),
                         image_counter=0,
                         clean_empty=False,
                         stride=10,
                         prefix='')
        image = None
        del image

        # Setup for predicting image patches
        patches_DL = PredictionDataset(img_dir=join(dir, 'temp', 'Patches'),
                                       out_channels=1)

        """CNN prediction"""
        for j in range(patches_DL.__len__()):
            if j // 100:
                tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                                text_1=f'Found {len(predict_list)} images to predict!',
                                text_3=f'Image: {i}',
                                text_4=f'Pixel size: {px} A; Image re-sample to 25 A',
                                text_5='Point Cloud: In processing...',
                                text_7='Current Task: CNN prediction...',
                                text_8=printProgressBar(j, patches_DL.__len__()))

            # Pick image['s]
            input, name = patches_DL.__getitem__(j)

            # Predict & Threshold
            out = np.where(predict_cnn._predict(input[None, :]) >= cnn_threshold, 1, 0)

            # Save
            assert out.min() == 0 and out.max() == 1
            tif.imwrite(join(output, f'{name}.tif'),
                        np.array(out, dtype=np.uint8))

        """Post-Processing"""
        # Stitch predicted image patches
        scale_factor = 25 / px

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Stitching...')

        image = check_uint8(image_stitcher(image_dir=output,
                                           output=None,
                                           mask=True,
                                           prefix='',
                                           dtype=np.int8)[:scale_shape[0] + 1,
                                                          :scale_shape[1] + 1,
                                                          :scale_shape[2] + 1])
        image, _ = scale_image(image=image,
                               mask=None,
                               scale=scale_factor)

        assert image.shape == org_shape, 'Error while converting from 2.5 nm pixel size. ' \
            f'Org. shape {org_shape} is not the same as converted shape {image.shape}'
        assert image.min() == 0 and image.max() == 1

        # Check if predicted image is not empty
        if debug:
            tif.imwrite(join(am_output, f'{i[:-out_format]}_CNN.tif'),
                        image)

        # If prediction fail aka no prediction was produces continue with next image
        if image is None:
            continue

        # Post-process predicted image patches
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Image Postprocessing...')

        point_cloud = post_processer(image=image,
                                     euclidean_transform=True,
                                     label_size=3,
                                     down_sampling_voxal_size=None)

        # Transform for xyz and pixel size for coord
        image = None
        del image

        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_raw_pc.npy'),
                    point_cloud)

        """DIST Prediction"""
        # Find downsampling value by 5 to reduce noise
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=5).points)

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                        text_7='Current Task: Preparing for MT segmentation...')

        # Normalize point cloud KNN distance
        dist = pc_median_dist(point_cloud, avg_over=True)

        # Build patches dataset with
        VD = PatchDataSet(coord=point_cloud / dist,
                          label_cls=None,
                          rbg=None,
                          patch_3d=False,
                          max_number_of_points=points_in_patch,
                          init_patch_size=0,
                          drop_rate=1,
                          graph=False,
                          tensor=True)

        coords_df, _, output_idx, _ = VD.voxalize_dataset(mesh=False,
                                                          dist_th=None)
        coords_df = [c / pc_median_dist(c) for c in coords_df]

        # Predict point cloud
        graphs = []
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                        text_7='Current Task: DIST prediction...',
                        text_8=printProgressBar(0, len(coords_df)))

        for id, coord in enumerate(coords_df):
            if id // 50:
                tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                                text_1=f'Found {len(predict_list)} images to predict!',
                                text_3=f'Image: {i}',
                                text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                                text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                                text_7='Current Task: DIST prediction...',
                                text_8=printProgressBar(id, len(coords_df)))

            graph = predict_dist._predict(x=coord[None, :])
            graphs.append(graph)

        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_graph_voxal.npy'),
                    graphs,
                    allow_pickle=True)

        """Graphformer post-processing"""
        if format in ['amira', 'mrc']:
            tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image: {i}',
                            text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                            text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                            text_7='Current Task: MT Segmentation...',
                            text_8='MTs segmentation is fitted to:',
                            text_9=f'pixel size: {px}; transformation: {transformation}')

            point_cloud = point_cloud * px
            point_cloud[:, 0] = point_cloud[:, 0] + transformation[0]
            point_cloud[:, 1] = point_cloud[:, 1] + transformation[1]
            point_cloud[:, 2] = point_cloud[:, 2] + transformation[2]
        else:
            tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image: {i}',
                            text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                            text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                            text_7='Current Task: MT Segmentation...')

        GraphToSegment = GraphInstanceV2(threshold=dist_threshold,
                                         connection=2,
                                         smooth=True)

        segments = GraphToSegment.patch_to_segment(graph=graphs,
                                                   coord=point_cloud,
                                                   idx=output_idx,
                                                   prune=5,
                                                   sort=True,
                                                   visualize=visualizer)

        """Threshold drop for hard datasets"""
        if 100 - ((segments.shape[0] * 100) / point_cloud.shape[0]) > 25:
            GraphToSegment = GraphInstanceV2(threshold=dist_threshold / 2,
                                             connection=2,
                                             smooth=True)
            segments = GraphToSegment.patch_to_segment(graph=graphs,
                                                       coord=point_cloud,
                                                       idx=output_idx,
                                                       prune=5,
                                                       sort=True,
                                                       visualize=visualizer)
        if 100 - ((segments.shape[0] * 100) / point_cloud.shape[0]) > 25:
            GraphToSegment = GraphInstanceV2(threshold=dist_threshold / 5,
                                             connection=2,
                                             smooth=True)
            segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                       coord=point_cloud,
                                                       idx=output_idx,
                                                       prune=5,
                                                       sort=True,
                                                       visualize=visualizer)

        # Save debugging check point
        if debug:
            if device == 'cpu':
                np.save(join(am_output,
                             f'{i[:-out_format]}_coord_voxal.npy'),
                        point_cloud,
                        allow_pickle=True)
                np.save(join(am_output,
                             f'{i[:-out_format]}_idx_voxal.npy'),
                        output_idx,
                        allow_pickle=True)
            else:
                np.save(join(am_output,
                             f'{i[:-out_format]}_coord_voxal.npy'),
                        point_cloud,
                        allow_pickle=True)
                np.save(join(am_output,
                             f'{i[:-out_format]}_idx_voxal.npy'),
                        output_idx,
                        allow_pickle=True)

        if debug:
            np.save(join(am_output,
                         f'{i[:-out_format]}_segments.npy'),
                    segments)

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; {np.max(segments[:, 0])} Segments',
                        text_7='Current Task: Segmentation finished!')

        """Save as .am"""
        BuildAmira.export_amira(coord=segments,
                                file_dir=join(am_output, f'{i[:-out_format]}_SpatialGraph.am'))

        """Clean-up temp dir"""
        clean_up(dir=dir)


if __name__ == '__main__':
    main()
