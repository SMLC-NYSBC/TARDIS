from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import tifffile.tifffile as tif
import open3d as o3d
from tardis.sis_graphformer.graphformer.network import CloudToGraph
from tardis.sis_graphformer.utils.utils import cal_node_input
from tardis.sis_graphformer.utils.voxal import VoxalizeDataSetV2
from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira
from tardis.slcpy_data_processing.utils.load_data import (import_am,
                                                          import_mrc,
                                                          import_tiff)
from tardis.slcpy_data_processing.utils.segment_point_cloud import GraphInstanceV2
from tardis.slcpy_data_processing.utils.stitch import StitchImages
from tardis.slcpy_data_processing.utils.trim import trim_image
from tardis.spindletorch.unet.predictor import Predictor
from tardis.spindletorch.utils.build_network import build_network
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.utils.utils import check_uint8, pc_median_dist
from tardis.utils.device import get_device
from tardis.version import version


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
              default=0.2,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-gt', '--gt_threshold',
              default=0.5,
              type=float,
              help='Threshold use for graph segmentation.',
              show_default=True)
@click.option('-pv', '--points_in_voxal',
              default=1000,
              type=int,
              help='Number of point per voxal.',
              show_default=True)
@click.option('-ch', '--checkpoints',
              default=False,
              type=bool,
              help='If True, include offline checkpoints',
              show_default=True)
@click.option('-chu', '--checkpoints_unet',
              default=None,
              type=str,
              help='If not None, str checkpoints for Unet',
              show_default=True)
@click.option('-chup', '--checkpoints_unetplus',
              default=None,
              type=str,
              help='If not None, str checkpoints for Unet3Plus',
              show_default=True)
@click.option('-chg', '--checkpoints_gf',
              default=None,
              type=str,
              help='If not None, str checkpoints for graphormer',
              show_default=True)
@click.option('-d', '--device',
              default='0',
              type=str,
              help='Define which device use for training: '
              'gpu: Use ID 0 GPUs '
              'cpu: Usa CPU '
              '0-9 - specified GPU device id to use',
              show_default=True)
@click.option('-tq', '--tqdm',
              default=True,
              type=bool,
              help='If True, build with progressbar.',
              show_default=True)
@click.option('-db', '--debug',
              default=False,
              type=bool,
              help='If True, save output from each step for debugging.',
              show_default=True)
@click.version_option(version=version)
def main(prediction_dir: str,
         patch_size: int,
         cnn_threshold: float,
         gt_threshold: float,
         points_in_voxal: int,
         checkpoints: bool,
         device: str,
         tqdm: bool,
         debug: bool,
         checkpoints_unet: Optional[str] = None,
         checkpoints_unetplus: Optional[str] = None,
         checkpoints_gf: Optional[str] = None,):
    """
    MAIN MODULE FOR PREDICTION MT WITH TARDIS-PYTORCH
    """
    """Initial setup"""
    # Searching for available images for prediction
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(prediction_dir, 'temp', 'Predictions')
    am_output = join(prediction_dir, 'Predictions')

    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    # Build handler's
    stitcher = StitchImages(tqdm=False)
    post_processer = ImageToPointCloud(tqdm=False)
    GraphToSegment = GraphInstanceV2(threshold=gt_threshold)
    BuildAmira = NumpyToAmira()

    # Build CNN from checkpoints
    if checkpoints:
        checkpoints = (checkpoints_unet, checkpoints_unetplus, checkpoints_gf)
    else:
        checkpoints = (None, None, None)

    device = get_device(device)

    predict_unet = Predictor(model=build_network(network_type='unet',
                                                 classification=False,
                                                 in_channel=1,
                                                 out_channel=1,
                                                 img_size=patch_size,
                                                 dropout=None,
                                                 no_conv_layers=5,
                                                 conv_multiplayer=32,
                                                 layer_components='gcl',
                                                 no_groups=8,
                                                 prediction=True),
                             checkpoint=checkpoints[0],
                             network='unet',
                             subtype=str(32),
                             device=device,
                             tqdm=tqdm)

    predict_unet3plus = Predictor(model=build_network(network_type='unet3plus',
                                                      classification=False,
                                                      in_channel=1,
                                                      out_channel=1,
                                                      img_size=patch_size,
                                                      dropout=None,
                                                      no_conv_layers=5,
                                                      conv_multiplayer=32,
                                                      layer_components='gcl',
                                                      no_groups=8,
                                                      prediction=True),
                                  checkpoint=checkpoints[1],
                                  network='unet3plus',
                                  subtype=str(32),
                                  device=device,
                                  tqdm=tqdm)

    if tqdm:
        from tqdm import tqdm as tq

        batch_iter = tq(predict_list)
    else:
        batch_iter = predict_list

    """Process each image with CNN and GF"""
    for i in batch_iter:
        """CNN prediction"""
        if i.endswith('CorrelationLines.am'):
            continue

        if tqdm:
            batch_iter.set_description(f'Preprocessing for CNN {i}')

        # Build temp dir
        build_temp_dir(dir=prediction_dir)

        # Cut image for smaller image
        if i.endswith(('.tif', '.tiff')):
            image, _ = import_tiff(img=join(prediction_dir, i),
                                   dtype=np.uint8)
            if i.endswith('.tif'):
                out_format = 4
            else:
                out_format = 5
        elif i.endswith(('.mrc', '.rec')):
            image, _ = import_mrc(img=join(prediction_dir, i))
            out_format = 4
        elif i.endswith('.am'):
            image, _ = import_am(img=join(prediction_dir, i))
            out_format = 3

        org_shape = image.shape

        trim_image(image=image,
                   trim_size_xy=patch_size,
                   trim_size_z=patch_size,
                   output=join(prediction_dir, 'temp', 'Patches'),
                   image_counter=0,
                   clean_empty=False,
                   prefix='')

        image = None
        del(image)

        # Setup for predicting image patches
        patches_DL = PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                       size=patch_size,
                                       out_channels=1)
        dl_len = patches_DL.__len__()

        if tqdm:
            dl_iter = tq(range(dl_len),
                         'Images',
                         leave=False)

            batch_iter.set_description(f'CNN prediction for {i}')
        else:
            dl_iter = range(dl_len)

        # Predict image patches
        for j in dl_iter:
            input, name = patches_DL.__getitem__(j)

            """Predict"""
            out_unet = predict_unet._predict(input[None, :])

            out_unet3plus = predict_unet3plus._predict(input[None, :])

            out = (out_unet + out_unet3plus) / 2

            """Threshold"""
            out = np.where(out >= cnn_threshold, 1, 0)

            """Save"""
            tif.imwrite(join(output, f'{name}.tif'),
                        np.array(out, dtype=np.int8))

        """CNN post-process"""
        # Stitch predicted image patches
        if tqdm:
            batch_iter.set_description(f'Stitching for {i}')

        image = check_uint8(stitcher(image_dir=output,
                                     output=None,
                                     mask=True,
                                     prefix='',
                                     dtype=np.int8)[:org_shape[0],
                                                    :org_shape[1],
                                                    :org_shape[2]])

        # Check if predicted image is not empty
        if debug:
            tif.imwrite(join(am_output,
                             f'{i[:-out_format]}_CNN.tif'),
                        image)

        if image is None:
            continue

        # Post-process predicted image patches
        if tqdm:
            batch_iter.set_description(f'Postprocessing for {i}')

        point_cloud = post_processer(image=image,
                                     euclidean_transform=True,
                                     label_size=3,
                                     down_sampling_voxal_size=None)
        image = None
        del(image)

        if debug:
            np.save(join(am_output,
                         f'{i[:-out_format]}_raw_pc.npy'),
                    point_cloud)

        """Graphformer prediction"""
        if tqdm:
            batch_iter.set_description(f'Building voxal for {i}')

        # Find downsampling value
        dist = pc_median_dist(pc=point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        point_cloud = np.asarray(pcd.voxel_down_sample(dist * 5).points)

        # Build voxalized dataset with
        VD = VoxalizeDataSetV2(coord=point_cloud,
                               image=None,
                               init_voxal_size=5000,
                               drop_rate=50,
                               downsampling_threshold=points_in_voxal,
                               downsampling_rate=None,
                               graph=False)
        coords_df, _, output_idx, = VD.voxalize_dataset(out_idx=True)

        # Calculate sigma for graphformer from mean of nearest point dist
        if tqdm:
            batch_iter.set_description(f'Compute sigma for {i}')

        sigma = pc_median_dist(GraphToSegment._stitch_coord(coords_df,
                                                            output_idx))

        # Predict point cloud
        predict_gf = Predictor(model=CloudToGraph(n_out=1,
                                                  node_input=cal_node_input(
                                                      (10, 10, 10)),
                                                  node_dim=128,
                                                  edge_dim=64,
                                                  num_layers=3,
                                                  num_heads=4,
                                                  dropout_rate=0,
                                                  coord_embed_sigma=sigma,
                                                  predict=True),
                               checkpoint=checkpoints[2],
                               network='graphformer',
                               subtype='without_img',
                               model_type='microtubules',
                               device=device,
                               tqdm=tqdm)

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
                        np.array([c.cpu().detach().numpy()
                                 for c in coords_df]),
                        allow_pickle=True)
                np.save(join(am_output,
                             f'{i[:-out_format]}_idx_voxal.npy'),
                        output_idx,
                        allow_pickle=True)

        # Predict point cloud
        if tqdm:
            dl_iter = tq(coords_df,
                         'Voxals',
                         leave=False)

            batch_iter.set_description(
                f'GF prediction for {i} with sigma {sigma}')
        else:
            dl_iter = coords_df

        graphs = []
        coords = []
        for coord in dl_iter:
            graph = predict_gf._predict(x=coord[None, :])
            graphs.append(graph)
            coords.append(coord.cpu().detach().numpy())

        if debug:
            np.save(join(am_output,
                         f'{i[:-out_format]}_graph_voxal.npy'),
                    graphs,
                    allow_pickle=True)
            np.save(join(am_output,
                         f'{i[:-out_format]}_coord_voxal.npy'),
                    coords,
                    allow_pickle=True)

        """Graphformer post-processing"""
        if tqdm:
            batch_iter.set_description(f'Graph segmentation for {i}')

        segments = GraphToSegment.graph_to_segments(graph=graphs,
                                                    coord=coords,
                                                    idx=output_idx)

        if debug:
            np.save(join(am_output,
                         f'{i[:-out_format]}_segments.npy'),
                    segments)

        """Save as .am"""
        if tqdm:
            n_ele = np.max(segments[:, 0])
            batch_iter.set_description(f'Saving .am {i} with {n_ele}')

        BuildAmira.export_amira(coord=segments,
                                file_dir=join(am_output,
                                              f'{i[:-out_format]}_SpatialGraph.am'))

        """Clean-up temp dir"""
        if tqdm:
            batch_iter.set_description(f'Clean-up temp for {i}')

        clean_up(dir=prediction_dir)


if __name__ == '__main__':
    main()
