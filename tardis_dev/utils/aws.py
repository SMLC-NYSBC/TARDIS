import io
from os import getcwd, mkdir
from os.path import isdir, join
from typing import Optional

import requests


def get_weights_aws(network: str,
                    subtype: str,
                    model: Optional[str] = '',
                    save_weights=True) -> io.BytesIO:
    """
    Module to download pre-train weights from S3 AWS bucket.

    TODO: Operate on the fixed temp folder and compared with existed weights file
        and download only if a new weight is available.

    Args:
        network (str): Type of network for which weights are requested.
        subtype (str): Sub-name of the network or sub-parameter for the network.
        model (str): Additional dataset name used for the DIST.
        save_weights (bool): If the True model is saved in the temp repository.
    """
    if network in ['unet', 'unet3plus', 'big_unet', 'fnet']:
        assert subtype in ['32', '64'], \
            'For DIST, pre train model must be selected!'

        weight = requests.get(f'https://tardis-weigths.s3.amazonaws.com/{network}_{subtype}/model_weights.pth')
    elif network == 'dist':
        assert model in ['cryo_membrane', 'microtubules'], \
            'For DIST, pre train model must be selected!'
        assert subtype in ['with_img', 'without_img'], \
            'For DIST, pre train subtype of model must be selected!'

        weight = requests.get(f'https://tardis-weigths.s3.amazonaws.com/{network}/{model}/{subtype}/model_weights.pth')

    if save_weights:
        dir = join(getcwd(), 'model')

        if not isdir('model'):
            mkdir('model')
        open('model/model_weights.pth', 'wb').write(weight.content)

        print(f'Pre-Trained model download from S3 and saved in {dir}')
        return join(dir, 'model_weights.pth')

    return io.BytesIO(weight.content)
