from os import listdir
from os.path import join
from shutil import rmtree
from typing import Optional

import numpy as np

from tardis.spindletorch.predict import predict
from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.spindletorch.utils.dataset_loader import PredictionDataSet
from torch.utils.data import DataLoader
import tifffile.tifffile as tif


def main(prediction_dir: str,
         patch_size: int,
         cnn_type: tuple,
         in_channel: int,
         out_channel: int,
         cnn_layers: int,
         cnn_multiplayer: int,
         tqdm: bool,
         output: str,
         threshold: Optional[float] = None,
         dropout: Optional[float] = None):
    """

    Args:
                prediction_dir:
                tqdm:
    """
    available_format = ['.tif', '.mrc', '.rec', '.am']
    predict_list = [f for f in listdir(
        prediction_dir) if f.endswith(available_format)]
    assert len(predict_list) > 0, 'No file found in given direcotry!'

    post_process = ImageToPointCloud(tqdm=tqdm)

    for i in predict_list:
        """Build temp dir"""

        """Voxalize image"""

        """Predict image patches"""
        patches_DL = DataLoader(dataset=PredictionDataSet(img_dir=join(prediction_dir, 'temp', 'Patches'),
                                                          size=patch_size,
                                                          out_channels=out_channel),
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=5)

        predict(image_DL=patches_DL,
                output=join(prediction_dir, 'temp', 'Prediction'),
                cnn_type=cnn_type,
                cnn_in_channel=in_channel,
                cnn_out_channel=out_channel,
                image_patch_size=patch_size,
                cnn_layers=cnn_layers,
                cnn_multiplayer=cnn_multiplayer,
                tqdm=tqdm,
                threshold=threshold,
                cnn_dropout=dropout)

        """Stitch patches from temp dir"""
        stitched_image = []

        """Save point cloud, (Optional) image"""

        tif.imsave(file=join(prediction_dir, 'Prediction', i),
                   data=np.array(stitched_image, dtype=np.int8))

        """Clean-up temp dir"""
        rmtree(join(prediction_dir, 'temp'))


if __name__ == '__main__':
    main()
