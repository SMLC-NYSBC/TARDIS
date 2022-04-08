from os import getcwd, listdir
from os.path import join
from shutil import rmtree

import click
import numpy as np
import tifffile.tifffile as tif

from tardis.sis_graphformer.graphformer.network import CloudToGraph
from tardis.sis_graphformer.utils.utils import cal_node_input
from tardis.sis_graphformer.utils.voxal import VoxalizeDataSetV2
from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira
from tardis.slcpy_data_processing.utils.load_data import (import_am,
                                                          import_mrc,
                                                          import_tiff)
from tardis.slcpy_data_processing.utils.segment_point_cloud import GraphInstance
from tardis.slcpy_data_processing.utils.stitch import StitchImages
from tardis.slcpy_data_processing.utils.trim import trim_image
from tardis.spindletorch.unet.predictor import Predictor
from tardis.spindletorch.utils.build_network import build_network
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.utils.setup_envir import build_temp_dir, clean_up
from tardis.utils.utils import check_uint8
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
              default=0.3,
              type=float,
              help='Threshold use for model prediction.',
              show_default=True)
@click.option('-ch', '--checkpoints',
              default=(None, None, None),
              type=tuple,
              help='If not None, str for Unet and Unet3Plus checkpoints',
              show_default=True)
@click.option('-d', '--device',
              default=0,
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
@click.version_option(version=version)
def main(prediction_dir: str,
         patch_size: int,
         cnn_threshold: float,
         checkpoints: tuple,
         device: str,
         tqdm: bool):
    """
    MAIN MODULE FOR PREDICTION MT with Tardis
    """
    """Searching for available images for prediction"""
    available_format = ('.tif', '.mrc', '.rec', '.am')
    output = join(prediction_dir, 'temp', 'Predictions')

    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    stitcher = StitchImages(tqdm=False)
    post_processer = ImageToPointCloud(tqdm=False)
    GraphToSegment = GraphInstance(max_interactions=2,
                                   threshold=0.1)
    BuildAmira = NumpyToAmira()

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

    predict_gf = Predictor(model=CloudToGraph(n_out=1,
                                              node_input=cal_node_input(
                                                  (10, 10, 10)),
                                              node_dim=128,
                                              edge_dim=64,
                                              num_layers=3,
                                              num_heads=4,
                                              dropout_rate=0,
                                              coord_embed_sigma=16,
                                              predict=True),
                           checkpoint=checkpoints[2],
                           network='graphformer',
                           subtype='without_img',
                           model_type='microtubules',
                           device=device,
                           tqdm=tqdm)
    if tqdm:
        from tqdm import tqdm as tq

        batch_iter = tq(predict_list,
                        leave=True)
    else:
        batch_iter = predict_list

    for i in batch_iter:
        if tqdm:
            batch_iter.set_description(f'Predicting image {i} ...')

        """Build temp dir"""
        build_temp_dir(dir=prediction_dir)

        """Voxalize image"""
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
            if i.endswith('CorrelationLines.am'):
                continue

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

        """Predict image patches"""
        patches_DL = PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                       size=patch_size,
                                       out_channels=1)
        dl_len = patches_DL.__len__()

        if tqdm:
            dl_iter = tq(range(dl_len),
                         'Predicting images: ')
        else:
            dl_iter = range(dl_len)

        for j in dl_iter:
            input, name = patches_DL.__getitem__(j)

            """Predict"""
            out_unet = predict_unet._predict(input[None, :])

            out_unet3plus = predict_unet3plus._predict(input[None, :])

            out = (out_unet + out_unet3plus) / 2

            """Threshold"""
            out = np.where(out >= cnn_threshold, 1, 0)

            """Save"""
            tif.imwrite(file=join(output, f'{name}.tif'),
                        data=np.array(out, dtype=np.int8))

        """Stitch patches and post-process"""
        image = check_uint8(stitcher(image_dir=output,
                                     output=None,
                                     mask=True,
                                     prefix='',
                                     dtype=np.int8)[:org_shape[0],
                                                    :org_shape[1],
                                                    :org_shape[2]])

        """Check if predicted image is not empty"""
        if image is None:
            continue

        point_cloud = post_processer(image=image,
                                     euclidean_transform=True,
                                     label_size=3,
                                     down_sampling_voxal_size=None)
        image = None
        del(image)

        """Build data loader for point cloud"""
        if tqdm:
            batch_iter.set_description(
                f'Image postprocess and building voxal ...')

        VD = VoxalizeDataSetV2(coord=point_cloud,
                               image=None,
                               init_voxal_size=5000,
                               drop_rate=100,
                               downsampling_threshold=500,
                               downsampling_rate=16,
                               graph=False)
        coords_df, _, output_idx, = VD.voxalize_dataset(out_idx=True)

        """Predict point cloud"""
        if tqdm:
            batch_iter.set_description(f'Predicting point cloud {i} ...')
            pc_iter = tq(coords_df,
                         'Predicting graph: ')
        else:
            pc_iter = coords_df

        graphs = []
        coords = []
        for coord in pc_iter:
            graph = predict_gf._predict(x=coord[None, :])
            graphs.append(graph[0, :])
            coords.append(coord.cpu().detach().numpy())

        """Graph  to segmented point cloud"""
        # TODO Rebuild Graph segmentation and clean-up
        segments = clean_segments(GraphToSegment.segment_voxals(graph_voxal=graphs,
                                                                coord_voxal=coords))
        """Save as .am"""
        # TODO add additional check-up to validate if file will open in Amira
        BuildAmira._write_to_amira(data=segments,
                                   file_dir=join(output, f'{i[:-out_format]}_SpatialGraph.am'))

        """Clean-up temp dir"""
        clean_up(dir=prediction_dir)
