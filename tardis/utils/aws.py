import io
from os import getcwd, mkdir
from os.path import isdir, join
from typing import Optional

import requests


def get_weights_aws(network: str,
                    subtype: str,
                    model: Optional[str] = '',
                    save_weights=True):
    """
    Module to download pre-train weights from S3 aws bucket
    Args:
        network: Type of network for which weight are requested
        subtype: Sub-name of the network or sub parameter for network
        model: Additional dataset name use for the graphformer
        save_weights: If True model is saved in temp repository
    """
    if network in ['unet', 'unet3plus']:
        assert subtype in ['32', '64'], \
            'For Graphformer, pre train model must be selected!'

        weight = requests.get(f'https://tardis-weigths.s3.amazonaws.com/{network}_{subtype}/model_weights.pth')
    elif network == 'graphformer':
        assert model in ['cryo_membrane', 'microtubules'], \
            'For Graphformer, pre train model must be selected!'
        assert subtype in ['with_img', 'without_img'], \
            'For Graphformer, pre train subtype of model must be selected!'

        weight = requests.get(
            f'https://tardis-weigths.s3.amazonaws.com/{network}/{model}/{subtype}/model_weights.pth')

    if save_weights:
        dir = join(getcwd(), 'model')

        if not isdir('model'):
            mkdir('model')
        open('model/model_weights.pth', 'wb').write(weight.content)

        print(f'Pre-Trained model download from S3 and saved in {dir}')
        return join(dir, 'model_weights.pth')

    return io.BytesIO(weight.content)
