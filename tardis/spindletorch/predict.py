from os.path import join
from typing import Optional

import numpy as np
import tifffile.tifffile as tif
import torch
from tardis.spindletorch.unet.predictor import Predictor
from tardis.spindletorch.utils.build_network import build_network
from torch.utils.data import DataLoader

# Setting for stable release to turn off all debug APIs
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(mode=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)


def predict(image_DL: DataLoader,
            output: str,
            cnn_type: tuple,
            cnn_in_channel: int,
            cnn_out_channel: int,
            image_patch_size: int,
            cnn_layers: int,
            cnn_multiplayer: int,
            cnn_composition: str,
            device: str,
            tqdm: bool,
            checkpoints: Optional[tuple] = None,
            threshold: Optional[float] = None,
            cnn_dropout: Optional[float] = None):
    """
    PREDICTION MODULE

    For each given dataloader with images

    Args:
        image_DL: DataLoader class with image set
        output: Directory where predicted images are saved
        cnn_type: Type of CNN use for prediction as tuple. Up to 2 CNN.
        cnn_in_channel: In channel for CNN
        cnn_out_channel: Out channel for CNN
        image_patch_size: Image patch size
        cnn_layers: Number of layers for CNN
        cnn_multiplayer: Layer multiplayer for CNN
        cnn_composition: Structure of each layer for CNN
        tqdm: If True, build with progressbar
        checkpoints: If not None, dir. name with checkpoint weights
        threshold: Threshold for image prediction
        cnn_dropout: If not None, float use as dropout rate in CNN
    """
    """Build NN"""
    image_predict = Predictor(model=build_network(network_type=cnn_type[0],
                                                  classification=False,
                                                  in_channel=cnn_in_channel,
                                                  out_channel=cnn_out_channel,
                                                  img_size=image_patch_size,
                                                  dropout=cnn_dropout,
                                                  no_conv_layers=cnn_layers,
                                                  conv_multiplayer=cnn_multiplayer,
                                                  layer_components=cnn_composition,
                                                  no_groups=8,
                                                  prediction=True),
                              checkpoint=checkpoints[0],
                              network=cnn_type[0],
                              subtype=str(cnn_multiplayer),
                              device=device,
                              tqdm=tqdm)

    if len(cnn_type) == 2:
        image_predict_2 = Predictor(model=build_network(network_type=cnn_type[1],
                                                        classification=False,
                                                        in_channel=cnn_in_channel,
                                                        out_channel=cnn_out_channel,
                                                        img_size=image_patch_size,
                                                        dropout=cnn_dropout,
                                                        no_conv_layers=cnn_layers,
                                                        conv_multiplayer=cnn_multiplayer,
                                                        layer_components=cnn_composition,
                                                        no_groups=8,
                                                        prediction=True),
                                    checkpoint=checkpoints[1],
                                    network=cnn_type[1],
                                    subtype=str(cnn_multiplayer),
                                    device=device,
                                    tqdm=tqdm)
    else:
        image_predict_2 = None

    """Prediction"""
    dl_len = image_DL.__len__()
    if tqdm:
        from tqdm import tqdm
        dl_iter = tqdm(range(dl_len),
                       'Predicting images: ',
                       ascii=True,
                       leave=True)
    else:
        dl_iter = range(dl_iter)

    for i in dl_iter:
        input, name = image_DL.__getitem__(i)

        """Predict"""
        out = image_predict._predict(input[None, :])

        if len(cnn_type) == 2:
            out_support = image_predict_2._predict(input[None, :])

            out = (out + out_support) / 2
        else:
            out_support = None

        """Threshold"""
        out = np.where(out >= threshold, 1, 0)

        """Save"""
        tif.imsave(file=join(output, f'{name}.tif'),
                   data=np.array(out, dtype=np.int8))

    """Clean-up env."""
    image_predict = None
    image_predict_2 = None
    out = None
