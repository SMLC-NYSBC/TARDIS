import io
from os import mkdir
from os.path import expanduser, isdir, isfile, join
from typing import Optional

import requests
from tardis.utils.utils import MD5Count


def get_weights_aws(network: str,
                    subtype: str,
                    model: Optional[str] = ''):
    """
    Module to download pre-train weights from S3 AWS bucket.

    Args:
        network (str): Type of network for which weights are requested.
        subtype (str): Sub-name of the network or sub-parameter for the network.
        model (str): Additional dataset name used for the DIST.
        save_weights (bool): If the True model is saved in the temp repository.
    """
    """Get weights for CNN"""
    if network in ['unet', 'unet3plus', 'fnet']:
        assert subtype in ['16', '32', '64'], \
            'For TARDIS, pre train model must be selected correctly!'

        dir = join(expanduser('~'), '.tardis_pytorch', f'{network}_{subtype}')
        if aws_check_with_temp(model_name=f'{network}/{subtype}'):
            return join(dir, 'model_weights.pth')
        else:
            weight = requests.get(
                f'https://tardis-weigths.s3.amazonaws.com/{network}_{subtype}/model_weights.pth'
            )
    elif network == 'dist':
        """Get weights for DIST"""
        assert model in ['cryo_membrane', 'microtubules'], \
            'For DIST, pre train model must be selected!'
        assert subtype in ['with_img', 'without_img'], \
            'For DIST, pre train subtype of model must be selected!'

        dir = join(expanduser('~'), '.tardis_pytorch', f'{network}_{model}_{subtype}')
        if aws_check_with_temp(model_name=f'{network}/{model}/{subtype}'):
            return join(dir, 'model_weights.pth')
        else:
            weight = requests.get(
                f'https://tardis-weigths.s3.amazonaws.com/{network}/{model}/{subtype}/model_weights.pth'
            )

    """Save temp weights"""
    if not isdir(join(expanduser('~'), '.tardis_pytorch')):
        mkdir(join(expanduser('~'), '.tardis_pytorch'))

    if not isdir(dir):
        mkdir(dir)

    open(join(dir, 'model_weights.pth'), 'wb').write(weight.content)
    print(f'Pre-Trained model download from S3 and saved/updated in {dir}')

    return io.BytesIO(weight.content)


def aws_check_with_temp(model_name: str) -> bool:
    """Check if temp dir exist"""
    if not isdir(join(expanduser('~'), '.tardis_pytorch')):
        return False

    """Check MD5 for stored file in ~/tardis_pytorch/..."""
    save_md5 = None
    m = model_name.split('/')

    if len(m) == 2:
        if not isfile(join(expanduser('~'),
                           '.tardis_pytorch',
                           f'{m[0]}_{m[1]}',
                           'model_weights.pth')):
            return False
        else:
            md5 = MD5Count(open(join(expanduser('~'),
                                     '.tardis_pytorch',
                                     f'{m[0]}_{m[1]}',
                                     'model_weights.pth'), 'rb'))
            save_md5 = md5.hexdigest()

    elif len(m) == 3:
        if not isfile(join(expanduser('~'),
                           '.tardis_pytorch',
                           f'{m[0]}_{m[1]}_{m[2]}',
                           'model_weights.pth')):
            return False
        else:

            md5 = MD5Count(open(join(expanduser('~'),
                                     '.tardis_pytorch',
                                     f'{m[0]}_{m[1]}_{m[2]}',
                                     'model_weights.pth'), 'rb'))
            save_md5 = md5.hexdigest()

    aws_md5 = None
    if save_md5 is None:
        return False
    else:
        if len(m) == 2:
            md5 = MD5Count(requests.get(
                f'https://tardis-weigths.s3.amazonaws.com/{m[0]}_{m[1]}/model_weights.pth',
                stream=True
            ))
            aws_md5 = md5.hexdigest()
        elif len(m) == 3:
            md5 = MD5Count(requests.get(
                f'https://tardis-weigths.s3.amazonaws.com/{m[0]}/{m[2]}/{m[1]}/model_weights.pth',
                           stream=True
                           ))
            aws_md5 = md5.hexdigest()

    if save_md5 == aws_md5:
        return True
    else:
        return False
