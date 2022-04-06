from os import listdir
from os.path import join
from shutil import rmtree
import numpy as np

from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.load_data import import_tiff, import_mrc, \
    import_am
from tardis.slcpy_data_processing.utils.trim import trim_image
from tardis.slcpy_data_processing.utils.stitch import StitchImages
from tardis.spindletorch.predict import predict
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from tardis.utils.setup_envir import build_temp_dir, clean_up
from torch.utils.data import DataLoader

from tardis.utils.utils import check_uint8


def main(prediction_dir: str,
         patch_size: int,
         cnn_threshold: float,
         tqdm: bool):
    """
    MAIN MODULE FOR PREDICTION MT with Tardis
    """
    """Searching for available images for prediction"""
    available_format = ['.tif', '.mrc', '.rec', '.am']
    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    stitcher = StitchImages(tqdm=tqdm)
    post_processer = ImageToPointCloud(tqdm=True)

    for i in predict_list:
        """Build temp dir"""
        build_temp_dir(dir=prediction_dir)

        """Voxalize image"""
        if i.endswith('.tif'):
            image, _ = import_tiff(img=join(prediction_dir, i),
                                   dtype=np.uint8)
        elif i.endswith(['.mrc', '.rec']):
            image, _ = import_mrc(img=join(prediction_dir, i))
        elif i.endswith('.am'):
            image, _ = import_am(img=join(prediction_dir, i))

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
        patches_DL = DataLoader(dataset=PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                                          size=patch_size,
                                                          out_channels=1),
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=5)

        predict(image_DL=patches_DL,
                output=join(prediction_dir, 'temp', 'Prediction'),
                cnn_type=['unet', 'unet3plus'],
                cnn_in_channel=1,
                cnn_out_channel=1,
                image_patch_size=patch_size,
                cnn_layers=5,
                cnn_multiplayer=32,
                cnn_composition='gcl',
                tqdm=tqdm,
                threshold=cnn_threshold,
                cnn_dropout=None)

        """Stitch patches and post-process"""
        image = check_uint8(stitcher(image_dir=join(prediction_dir, 'temp', 'Prediction'),
                                     output=None,
                                     mask=True,
                                     prefix='',
                                     dtype=np.int8)[:org_shape[0],
                                                    :org_shape[1],
                                                    :org_shape[2]])
        _ = post_processer(image=image,
                           euclidean_transform=True,
                           label_size=300,
                           down_sampling_voxal_size=None)

        """Build Graphformer and predict point cloud"""

        """Graph representation to segmented point cloud"""

        """Save as .am"""

        """Clean-up temp dir"""
        rmtree(join(prediction_dir, 'temp'))
        clean_up(dir=prediction_dir)
