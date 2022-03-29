import io
from os import getcwd, mkdir
from os.path import isdir, join

import requests


def get_weights_aws(save_weights=True):
    """

    Args:
        save_weights: If True model is saved in temp repository
    """
    weight = requests.get(
        'https://spindletorch-weights.s3.amazonaws.com/unet3plus/model_weights.pth')

    if save_weights:
        dir = join(getcwd(), 'model')

        if not isdir('model'):
            mkdir('model')
        open('model/model_weights.pth', 'wb').write(weight.content)

        print(f'Pre-Trained model download from S3 and saved in {dir}')
        return join(dir, 'model_weights.pth')

    return io.BytesIO(weight.content)
