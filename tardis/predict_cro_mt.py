#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import sys
import time
import warnings
from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import open3d as o3d
import tifffile.tifffile as tif

from tardis.dist_pytorch.datasets.patches import PatchDataSet
from tardis.dist_pytorch.utils.build_point_cloud import ImageToPointCloud
from tardis.dist_pytorch.utils.segment_point_cloud import (FilterSpatialGraph,
                                                           GraphInstanceV2)
from tardis.dist_pytorch.utils.utils import pc_median_dist
from tardis.spindletorch.data_processing.stitch import StitchImages
from tardis.spindletorch.data_processing.trim import scale_image, trim_with_stride
from tardis.spindletorch.datasets.augment import MinMaxNormalize, RescaleNormalize
from tardis.spindletorch.datasets.dataloader import PredictionDataset
from tardis.utils.device import get_device
from tardis.utils.export_data import NumpyToAmira, to_mrc
from tardis.utils.load_data import import_am, load_image
from tardis.utils.logo import print_progress_bar, TardisLogo
from tardis.utils.predictor import Predictor
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.version import version

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
              default='fnet_32',
              type=str,
              help='CNN network name.',
              show_default=True)
@click.option('-cch', '--cnn_checkpoint',
              default=None,
              type=str,
              help='If not None, str checkpoints for CNN',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.2,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-dch', '--dist_checkpoint',
              default=None,
              type=str,
              help='If not None, str checkpoints for DIST',
              show_default=True)
@click.option('-dt', '--dist_threshold',
              default=0.5,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_patch',
              default=1000,
              type=int,
              help='Number of point per voxel.',
              show_default=True)
@click.option('-f', '--filter_mt',
              default=0,
              type=int,
              help='Remove MT that are shorter then given A value '
                   'NOT SUPPORTED FOR .TIF FILE FORMAT '
                   'There are two filtering mechanisms: '
                   '- Remove short segments (aka. Segments shorter then XX A. '
                   '- Connect segments that are closer then 17.5 nm',
              show_default=True)
@click.option('-d', '--device',
              default='0',
              type=str,
              help='Define which device use for training: '
                   'gpu: Use ID 0 GPUs '
                   'cpu: Usa CPU '
                   '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-o', '--output',
              default=None,
              type=click.Choice(['csv', 'mrc']),
              help='Define output format type.',
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
         filter_mt: float,
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

    tardis_progress = TardisLogo()
    tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}')

    # Searching for available images for prediction
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(dir, 'temp', 'Predictions')
    am_output = join(dir, 'Predictions')

    predict_list = [f for f in listdir(dir) if f.endswith(available_format)]

    # Tardis progress bar update
    if len(predict_list) == 0:
        tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_5='Point Cloud: Nan', text_7='Current Task: NaN',
                        text_8='Tardis Error: Wrong directory:',
                        text_9=f'Given {dir} is does not contain any recognizable file formats!', )
        sys.exit()
    else:
        tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Set-up environment...', )

    # Hard fix for dealing with tif file lack of pixel sizes...
    tif_px = None
    if np.any([True for x in predict_list if x.endswith(('.tif', '.tiff'))]):
        tif_px = click.prompt('Detected .tif files, please provide pixel size:',
                              type=float)

    # Build handler's
    normalize = RescaleNormalize(clip_range=(1, 99))  # Normalize histogram
    minmax = MinMaxNormalize()

    image_stitcher = StitchImages()
    post_processes = ImageToPointCloud()
    build_amira_file = NumpyToAmira()
    patch_pc = PatchDataSet(max_number_of_points=points_in_patch,
                            graph=False)
    GraphToSegment = GraphInstanceV2(threshold=dist_threshold,
                                     smooth=True)
    filter_segments = FilterSpatialGraph(connect_seg_if_closer_then=filter_mt)

    # Build CNN from checkpoints
    checkpoints = (cnn_checkpoint, dist_checkpoint)

    device = get_device(device)
    cnn_network = cnn_network.split('_')
    if not len(cnn_network) == 2:
        tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_5='Point Cloud: Nan', text_7='Current Task: NaN',
                        text_8='Tardis Error: Given CNN type is wrong!:',
                        text_9=f'Given {cnn_network} but should be e.g. `unet_32`', )
        sys.exit()

    # Build CNN network with loaded pre-trained weights
    predict_cnn = Predictor(checkpoint=checkpoints[0],
                            network=cnn_network[0],
                            subtype=cnn_network[1],
                            model_type='microtubules',
                            img_size=patch_size,
                            device=device, )

    # Build DIST network with loaded pre-trained weights
    predict_dist = Predictor(checkpoint=checkpoints[1],
                             network='dist',
                             subtype='triang',
                             model_type='microtubules',
                             device=device)

    """Process each image with CNN and DIST"""
    tardis_progress = TardisLogo()
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

        # Tardis progress bar update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4='Pixel size: Nan A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Preprocessing for CNN...', )

        # Build temp dir
        build_temp_dir(dir=dir)

        # Cut image for smaller image
        if i.endswith('.am'):
            image, px, _, transformation = import_am(am_file=join(dir, i))
        else:
            image, px = load_image(join(dir, i))
            transformation = [0, 0, 0]

        if tif_px is not None:
            px = tif_px

        if px == 0:
            px = click.prompt(
                f"Image file has pixel size {px}, that's obviously wrong... "
                "What is the correct value:",
                type=float)
        if px == 1:
            px = click.prompt(
                f"Image file has pixel size {px}, that's maybe wrong... "
                'What is the correct value:',
                default=px, type=float, )

        # Check image structure and normalize histogram
        if image.min() > 5 or image.max() < 250:  # Rescale image intensity
            image = normalize(image)
        if not image.min() >= 0 or not image.max() <= 1:  # Normalized between 0 and 1
            image = minmax(image)

        if not image.dtype == np.float32:
            tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                            text_5='Point Cloud: Nan A', text_7='Current Task: NaN',
                            text_8=f'Tardis Error: Error while loading image {i}:',
                            text_9=f'Image loaded correctly, but output format {image.dtype} is not float32!', )
            sys.exit()

        # Calculate parameters for normalizing image pixel size
        scale_factor = px / 25
        org_shape = image.shape
        scale_shape = tuple(np.multiply(org_shape, scale_factor).astype(np.int16))

        # Tardis progress bar update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Pixel size: {px} A',
                        text_5='Point Cloud: In processing...',
                        text_7=f'Current Task: Sub-dividing images for {patch_size} size')

        # Cut image for fix patch size and normalizing image pixel size
        trim_with_stride(image=image.astype(np.float32),
                         scale=scale_shape,
                         trim_size_xy=patch_size,
                         trim_size_z=patch_size,
                         output=join(dir, 'temp', 'Patches'),
                         image_counter=0,
                         clean_empty=False,
                         stride=10)
        del image

        # Setup CNN dataloader
        patches_DL = PredictionDataset(img_dir=join(dir, 'temp', 'Patches'))

        """CNN prediction"""
        iter_time = 1
        for j in range(len(patches_DL)):
            if j % iter_time == 0:
                # Tardis progress bar update
                tardis_progress(
                    title=f'Fully-automatic MT segmentation module  {str_debug}',
                    text_1=f'Found {len(predict_list)} images to predict!',
                    text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                    text_4=f'Pixel size: {px} A', text_5='Point Cloud: In processing...',
                    text_7='Current Task: CNN prediction...',
                    text_8=print_progress_bar(j, len(patches_DL)), )

            # Pick image['s]
            input, name = patches_DL.__getitem__(j)

            if j == 0:
                start = time.time()

                # Predict & Threshold
                input = predict_cnn.predict(input[None, :])

                end = time.time()
                iter_time = 10 // (end - start)  # Scale progress bar refresh to 10s
                if iter_time <= 1:
                    iter_time = 1
            else:
                # Predict & Threshold
                input = predict_cnn.predict(input[None, :])

            if cnn_threshold != 0:
                input = np.where(input >= cnn_threshold, 1, 0)

            tif.imwrite(join(output, f'{name}.tif'), np.array(input, dtype=input.dtype))

        """Post-Processing"""
        # Tardis progress bar update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Original pixel size: {px} A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Stitching...', )

        # Stitch predicted image patches
        image = image_stitcher(image_dir=output,
                               mask=True,
                               dtype=input.dtype)[: org_shape[0],
                                                  : org_shape[1],
                                                  : org_shape[2]]

        # Restored original image pixel size
        image, _ = scale_image(image=image,
                               scale=org_shape)

        if cnn_threshold == 0:
            """Clean-up temp dir"""
            tif.imwrite(join(am_output, f'{i[:-out_format]}_CNN.tif'), image)
            clean_up(dir=dir)
            continue

        else:
            # Threshold image
            image = np.where(image >= cnn_threshold, 1, 0).astype(np.uint8)

        # Check if predicted image
        if not image.shape == org_shape:
            tardis_progress(title=f'Fully-automatic MT segmentation module {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                            text_4=f'Original pixel size: {px} A',
                            text_5='Point Cloud: NaN.',
                            text_7='Last Task: Stitching/Scaling/Make correction...',
                            text_8=f'Tardis Error: Error while converting to {px} A pixel size.',
                            text_9=f'Org. shape {org_shape} is not the same as converted shape {image.shape}', )
            sys.exit()

        # If prediction fail aka no prediction was produces continue with next image
        if image is None:
            continue

        if debug:  # Debugging checkpoint
            tif.imwrite(join(am_output, f'{i[:-out_format]}_CNN.tif'), image)
        if output == 'mrc':
            to_mrc(data=image, file_dir=join(am_output, f'{i[:-out_format]}_CNN.mrc'))

        if not image.min() == 0 and not image.max() == 1:
            continue

        # Tardis progress bar update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Original pixel size: {px} A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Image Postprocessing...', )

        # Post-process predicted image patches
        point_cloud = post_processes(image=image,
                                     label_size=3)

        if point_cloud.shape[0] < 100:
            point_cloud = post_processes(image=image,
                                         label_size=0.5)

        if point_cloud.shape[0] < 100:
            continue

        # Transform for xyz and pixel size for coord
        del image

        if debug:  # Debugging checkpoint
            np.save(join(am_output, f'{i[:-out_format]}_raw_pc.npy'), point_cloud)

        """DIST Prediction"""
        # Find down-sampling value by voxel size 5 to reduce noise
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=5).points)

        # Tardis progress bar update
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Original pixel size: {px} A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                        text_7='Current Task: Preparing for MT segmentation...', )

        # Normalize point cloud KNN distance !Soon depreciated!
        dist = pc_median_dist(point_cloud, avg_over=True)

        # Build patches dataset
        coords_df, _, output_idx, _ = patch_pc.patched_dataset(coord=point_cloud / dist)

        # Predict point cloud
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Original pixel size: {px} A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                        text_7='Current Task: DIST prediction...',
                        text_8=print_progress_bar(0, len(coords_df)), )

        """DIST prediction"""
        iter_time = 1
        graphs = []
        for id_dist, coord in enumerate(coords_df):
            if id_dist % iter_time == 0:
                tardis_progress(
                    title=f'Fully-automatic MT segmentation module  {str_debug}',
                    text_1=f'Found {len(predict_list)} images to predict!',
                    text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                    text_4=f'Original pixel size: {px} A',
                    text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                    text_7='Current Task: DIST prediction...',
                    text_8=print_progress_bar(id, len(coords_df)), )

            if id_dist == 0:
                start = time.time()

                graph = predict_dist.predict(x=coord[None, :])

                end = time.time()
                iter_time = 10 // (end - start)  # Scale progress bar refresh to 10s
                if iter_time <= 1:
                    iter_time = 1
            else:
                graph = predict_dist.predict(x=coord[None, :])

            graphs.append(graph)
        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_graph_voxel.npy'), graphs)

        """DIST post-processing"""
        if i.endswith(('.am', '.rec', '.mrc')):
            tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                            text_4=f'Original pixel size: {px} A',
                            text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                            text_7='Current Task: MT Segmentation...',
                            text_8='MTs segmentation is fitted to:',
                            text_9=f'pixel size: {px}; transformation: {transformation}', )

            point_cloud = point_cloud * px
            point_cloud[:, 0] = point_cloud[:, 0] + transformation[0]
            point_cloud[:, 1] = point_cloud[:, 1] + transformation[1]
            point_cloud[:, 2] = point_cloud[:, 2] + transformation[2]
        else:
            tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                            text_1=f'Found {len(predict_list)} images to predict!',
                            text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                            text_4=f'Original pixel size: {px} A',
                            text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                            text_7='Current Task: MT Segmentation...', )

        segments = GraphToSegment.patch_to_segment(graph=graphs,
                                                   coord=point_cloud,
                                                   idx=output_idx,
                                                   prune=5,
                                                   visualize=visualizer)

        segments_filter = filter_segments(segments)

        # Save debugging check point
        if debug:
            if device == 'cpu':
                np.save(join(am_output, f'{i[:-out_format]}_coord_voxel.npy'),
                        point_cloud)
                np.save(join(am_output, f'{i[:-out_format]}_idx_voxel.npy'),
                        output_idx)
            else:
                np.save(join(am_output, f'{i[:-out_format]}_coord_voxel.npy'),
                        point_cloud)
                np.save(join(am_output, f'{i[:-out_format]}_idx_voxel.npy'),
                        output_idx)

        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_segments.npy'), segments)

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image {id + 1}/{len(predict_list)}: {i}',
                        text_4=f'Original pixel size: {px} A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; {np.max(segments[:, 0])} Segments',
                        text_7='Current Task: Segmentation finished!', )

        """Save as .am"""
        build_amira_file.export_amira(coord=segments,
                                      file_dir=join(am_output,
                                                    f'{i[:-out_format]}_SpatialGraph.am'))
        build_amira_file.export_amira(coord=segments_filter,
                                      file_dir=join(am_output,
                                                    f'{i[:-out_format]}_SpatialGraph_filter.am'))
        if output == 'csv':
            np.savetxt(join(am_output, f'{i[:-out_format]}' '_SpatialGraph.csv'),
                       segments,
                       delimiter=",")
            np.savetxt(join(am_output, f'{i[:-out_format]}' '_SpatialGraph_filter.csv'),
                       segments_filter,
                       delimiter=",")

        """Clean-up temp dir"""
        clean_up(dir=dir)


if __name__ == '__main__':
    main()
