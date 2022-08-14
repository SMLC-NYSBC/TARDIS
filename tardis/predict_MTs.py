import warnings
from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import open3d as o3d
import tifffile.tifffile as tif

from tardis.dist_pytorch.transformer.network import DIST
from tardis.dist_pytorch.utils.voxal import VoxalizeDataSetV2
from tardis.slcpy.image_postprocess import ImageToPointCloud
from tardis.slcpy.utils.export_data import NumpyToAmira
from tardis.slcpy.utils.load_data import import_am, import_mrc, import_tiff
from tardis.slcpy.utils.segment_point_cloud import GraphInstanceV2
from tardis.slcpy.utils.stitch import StitchImages
from tardis.slcpy.utils.trim import trim_with_stride
from tardis.spindletorch.unet.predictor import Predictor
from tardis.spindletorch.utils.build_network import build_network
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.utils.device import get_device
from tardis.utils.logo import Tardis_Logo, printProgressBar
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.utils.utils import check_uint8, pc_median_dist
from tardis.version import version

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option('-dir', '--prediction_dir',
              default=getcwd(),
              type=str,
              help='Directory with images for prediction with CNN model.',
              show_default=True)
@click.option('-ps', '--patch_size',
              default=96,
              type=int,
              help='Size of image size used for prediction.',
              show_default=True)
@click.option('-ct', '--cnn_threshold',
              default=0.35,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-gt', '--gt_threshold',
              default=0.5,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_voxal',
              default=500,
              type=int,
              help='Number of point per voxal.',
              show_default=True)
@click.option('-ch', '--checkpoints',
              default=False,
              type=bool,
              help='If True, include offline checkpoints',
              show_default=True)
@click.option('-chu', '--checkpoints_big_unet',
              default=None,
              type=str,
              help='If not None, str checkpoints for Big_Unet',
              show_default=True)
@click.option('-chg', '--checkpoints_gf',
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
              type=str,
              help='If not None, output visualization of the prediction'
              'f: Output as filaments'
              'p: Output as segmented point cloud',
              show_default=True)
@click.version_option(version=version)
def main(prediction_dir: str,
         patch_size: int,
         cnn_threshold: float,
         gt_threshold: float,
         points_in_voxal: int,
         checkpoints: bool,
         device: str,
         debug: bool,
         visualizer: Optional[str] = None,
         checkpoints_big_unet: Optional[str] = None,
         checkpoints_gf: Optional[str] = None):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    """Initial Setup"""
    if debug:
        str_debug = 'Debugging Mode'
    else:
        str_debug = ''

    tardis_progress = Tardis_Logo()
    tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}')

    # Searching for available images for prediction
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(prediction_dir, 'temp', 'Predictions')
    am_output = join(prediction_dir, 'Predictions')

    predict_list = [f for f in listdir(prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given directory!'
    tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                    text_1=f'Found {len(predict_list)} images to predict!')

    # Build handler's
    stitcher = StitchImages(tqdm=False)
    post_processer = ImageToPointCloud(tqdm=False)
    BuildAmira = NumpyToAmira()

    # Build CNN from checkpoints
    if checkpoints:
        checkpoints = (checkpoints_big_unet, checkpoints_gf)
    else:
        checkpoints = (None, None, None)

    device = get_device(device)

    predict = Predictor(model=build_network(network_type='fnet',
                                            classification=False,
                                            in_channel=1,
                                            out_channel=1,
                                            img_size=patch_size,
                                            dropout=None,
                                            no_conv_layers=5,
                                            conv_multiplayer=32,
                                            layer_components='3gcl',
                                            no_groups=8,
                                            prediction=True),
                        checkpoint=checkpoints[0],
                        network='fnet',
                        subtype=str(32),
                        device=device)

    """Process each image with CNN and GF"""
    tardis_progress = Tardis_Logo()
    for id, i in enumerate(sorted(predict_list)):
        """Pre-Processing"""
        if i.endswith('CorrelationLines.am'):
            continue
        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4='Pixel size: Nan A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Preprocessing for CNN...')

        # Build temp dir
        build_temp_dir(dir=prediction_dir)

        # Cut image for smaller image
        if i.endswith(('.tif', '.tiff')):
            image, px = import_tiff(img=join(prediction_dir, i),
                                    dtype=np.uint8)
            if i.endswith('.tif'):
                out_format = 4
            else:
                out_format = 5
            format = 'tif'
        elif i.endswith(('.mrc', '.rec')):
            image, px = import_mrc(img=join(prediction_dir, i))
            out_format = 4
            format = 'mrc'
        elif i.endswith('.am'):
            image, px, _, transformation = import_am(img=join(prediction_dir, i))
            out_format = 3
            format = 'amira'

        scale_factor = px / 25
        org_shape = image.shape

        trim_with_stride(image=image,
                         scale=scale_factor,
                         trim_size_xy=patch_size,
                         trim_size_z=patch_size,
                         output=join(prediction_dir, 'temp', 'Patches'),
                         image_counter=0,
                         clean_empty=False,
                         stride=10,
                         prefix='')
        image = None
        del(image)

        # Setup for predicting image patches
        patches_DL = PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                       size=patch_size,
                                       out_channels=1)

        """CNN prediction"""
        for j in range(patches_DL.__len__()):
            if j // 10:
                tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                                text_1=f'Found {len(predict_list)} images to predict!',
                                text_3=f'Image: {i}',
                                text_4=f'Pixel size: {px} A; Image re-sample to 25 A',
                                text_5='Point Cloud: In processing...',
                                text_7='Current Task: CNN prediction...',
                                text_8=printProgressBar(j, patches_DL.__len__()))

            input, name = patches_DL.__getitem__(j)

            """Predict & Threshold"""
            out = np.where(predict._predict(input[None, :]) >= cnn_threshold, 1, 0)

            """Save"""
            tif.imwrite(join(output, f'{name}.tif'),
                        np.array(out, dtype=np.int8))

        """Post-Processing"""
        # Stitch predicted image patches
        scale_factor = 25 / px

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5='Point Cloud: In processing...',
                        text_7='Current Task: Stitching...')

        image = check_uint8(stitcher(image_dir=output,
                                     output=None,
                                     mask=True,
                                     scale=scale_factor,
                                     prefix='',
                                     dtype=np.int8)[:org_shape[0],
                                                    :org_shape[1],
                                                    :org_shape[2]])

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
        point_cloud = point_cloud * scale_factor

        # Transform for xyz and pixel size for coord
        image = None
        del(image)

        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_raw_pc.npy'),
                    point_cloud)

        """DIST Prediction"""
        # Post-process predicted image patches
        down_sample = 2.5
        if point_cloud.shape[0] > 25000:
            down_sample = 1.5
        if point_cloud.shape[0] > 30000:
            down_sample = 2.5
        if point_cloud.shape[0] > 40000:
            down_sample = 3
        if point_cloud.shape[0] > 50000:
            down_sample = 4
        if point_cloud.shape[0] > 80000:
            down_sample = 5

        # Find downsampling value by 5 to reduce noise
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = np.asarray(pcd.voxel_down_sample(voxel_size=down_sample).points)

        tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                        text_1=f'Found {len(predict_list)} images to predict!',
                        text_3=f'Image: {i}',
                        text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                        text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                        text_7='Current Task: Preparing for MT segmentation...')

        # Normalize point cloud KNN distance
        dist = pc_median_dist(point_cloud, avg_over=True)

        # Build voxalized dataset with
        VD = VoxalizeDataSetV2(coord=point_cloud / dist,
                               downsampling_rate=None,
                               init_voxal_size=0,
                               drop_rate=1,
                               downsampling_threshold=points_in_voxal,
                               graph=False)

        coords_df, _, output_idx = VD.voxalize_dataset(prune=10, out_idx=True)
        coords_df = [c / pc_median_dist(c) for c in coords_df]

        # Predict point cloud
        predict_gf = Predictor(model=DIST(n_out=1,
                                          node_input=None,
                                          node_dim=256,
                                          edge_dim=128,
                                          num_layers=6,
                                          num_heads=8,
                                          dropout_rate=0,
                                          coord_embed_sigma=2,
                                          structure='triang',
                                          dist_embed=True,
                                          predict=True),
                               checkpoint=checkpoints[2],
                               network='graphformer',
                               subtype='without_img',
                               model_type='microtubules',
                               device=device)

        if debug:
            if device == 'cpu':
                np.save(join(am_output,
                             f'{i[:-out_format]}_coord_voxal.npy'),
                        np.array([c.detach().numpy() for c in coords_df]),
                        allow_pickle=True)
                np.save(join(am_output,
                             f'{i[:-out_format]}_idx_voxal.npy'),
                        output_idx,
                        allow_pickle=True)
            else:
                np.save(join(am_output,
                             f'{i[:-out_format]}_coord_voxal.npy'),
                        np.array([c.cpu().detach().numpy() for c in coords_df]),
                        allow_pickle=True)
                np.save(join(am_output,
                             f'{i[:-out_format]}_idx_voxal.npy'),
                        output_idx,
                        allow_pickle=True)

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
            if id // 25:
                tardis_progress(title=f'Fully-automatic MT segmentation module  {str_debug}',
                                text_1=f'Found {len(predict_list)} images to predict!',
                                text_3=f'Image: {i}',
                                text_4=f'Original pixel size: {px} A; Image re-sample to 25 A',
                                text_5=f'Point Cloud: {point_cloud.shape[0]} Nodes; NaN Segments',
                                text_7='Current Task: DIST prediction...',
                                text_8=printProgressBar(id, len(coords_df)))

            graph = predict_gf._predict(x=coord[None, :])
            graphs.append(graph)

        if debug:
            np.save(join(am_output, f'{i[:-out_format]}_graph_voxal.npy'),
                    graphs,
                    allow_pickle=True)

        """Graphformer post-processing"""
        if format == 'amira':
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

        GraphToSegment = GraphInstanceV2(threshold=gt_threshold,
                                         connection=2,
                                         prune=5)

        segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                   coord=point_cloud,
                                                   idx=output_idx,
                                                   visualize=visualizer)

        """Threshold drop for hard datasets"""
        if 100 - ((segments.shape[0] * 100) / point_cloud.shape[0]) > 25:
            GraphToSegment = GraphInstanceV2(threshold=0.25,
                                             connection=2,
                                             prune=5)
            segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                       coord=point_cloud,
                                                       idx=output_idx,
                                                       visualize=visualizer)
        if 100 - ((segments.shape[0] * 100) / point_cloud.shape[0]) > 25:
            GraphToSegment = GraphInstanceV2(threshold=0.1,
                                             connection=2,
                                             prune=5)
            segments = GraphToSegment.voxal_to_segment(graph=graphs,
                                                       coord=point_cloud,
                                                       idx=output_idx,
                                                       visualize=visualizer)
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
        clean_up(dir=prediction_dir)


if __name__ == '__main__':
    main()
