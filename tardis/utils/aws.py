#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import io
import json
from os import makedirs, mkdir
from os.path import expanduser, isdir, isfile, join
from typing import Optional

import requests

from tardis.utils.errors import TardisError


def get_benchmark_aws() -> dict:
    """
    Retrieve best benchmarking score for given NN type

    Args:
        network (str): Benchmarking network type [dist or cnn]

    Returns:
        dict: Dictionary with keys[network name] and values[list of scores]
    """
    network_benchmark = requests.get('https://tardis-weigths.s3.amazonaws.com/'
                                     f'benchmark/best_scores.json')

    if network_benchmark.status_code == 200:
        network_benchmark = json.loads(network_benchmark.content.decode('utf-8'))

    return network_benchmark


def put_benchmark_aws(data: dict,
                      network: Optional[str] = '',
                      model=None) -> bool:
    """
    Upload new or update dictionary stored on S3

    Args:
        data (dict): Dictionary with network the best metrics
        network (Optional, str): Benchmarking network name [e.g. fnet_32_microtubules_id].
        model (Optional, str): Optional dictionary to model.

    Returns:
        bool: True if save correctly
    """
    r = requests.put('https://tardis-weigths.s3.amazonaws.com/'
                     f'benchmark/best_scores.json',
                     json.dumps(data, indent=2, default=str))

    if model is not None and r.status_code == 200:
        with open(model, 'rb') as data:
            r_m = requests.put('https://tardis-weigths.s3.amazonaws.com/'
                               f'benchmark/models/{network}.pth', data=data)

        return r_m.status_code == 200
    return r.status_code == 200


def get_model_aws(https: str):
    return requests.get(https)


def get_weights_aws(network: str,
                    subtype: str,
                    model: Optional[str] = None):
    """
    Module to download pre-train weights from S3 AWS bucket.

    Model weight stored on S3 bucket with the naming convention
    network_subtype/model/model_weights.pth
    References.:
    - fnet_32/microtubules/model_weights.pth
    - dist_triang/microtubules/model_weights.pth

    Weights are stored in ~/.tardis_pytorch with the same convention and .txt
    file with file header information to identified update status for local file
    if the network connection can be established.

    Args:
        network (str): Type of network for which weights are requested.
        subtype (str): Sub-name of the network or sub-parameter for the network.
        model (str): Additional dataset name used for the DIST.
    """
    """Get weights for CNN"""
    dir = join(expanduser('~'), '.tardis_pytorch', f'{network}_{subtype}', f'{model}')

    if network not in ['unet', 'unet3plus', 'fnet', 'dist']:
        TardisError('19',
                    'tardis/utils/aws.py',
                    f'Incorrect CNN network selected {network}_{subtype}')
    if subtype not in ['16', '32', '64', '96', '128', 'triang', 'full']:
        TardisError('19',
                    'tardis/utils/aws.py',
                    f'Incorrect CNN subtype selected {network}_{subtype}')

    if model not in ['microtubules', 'cryo_mem']:
        TardisError('19',
                    'tardis/utils/aws.py',
                    f'Incorrect CNN model selected {model}')

    if aws_check_with_temp(model_name=[network, subtype, model]):
        if isfile(join(dir, 'model_weights.pth')):
            return join(dir, 'model_weights.pth')
        else:
            TardisError('19',
                        'tardis/utils/aws.py',
                        'No weights found')
    else:
        weight = get_model_aws('https://tardis-weigths.s3.amazonaws.com/'
                               f'{network}_{subtype}/'
                               f'{model}/model_weights.pth')

    """Save temp weights"""
    if not isdir(join(expanduser('~'), '.tardis_pytorch')):
        mkdir(join(expanduser('~'), '.tardis_pytorch'))

    if not isdir(dir):
        makedirs(dir)

    # Save weights
    open(join(dir, 'model_weights.pth'), 'wb').write(weight.content)

    # Save header
    with open(join(dir, 'model_header.json'), 'w') as f:
        json.dump(dict(weight.headers), f)

    print(f'Pre-Trained model download from S3 and saved/updated in {dir}')

    weight = weight.content
    if 'AccessDenied' in str(weight[:100]):
        return join(dir, 'model_weights.pth')
    return io.BytesIO(weight)


def aws_check_with_temp(model_name: list) -> bool:
    """
    Module to check aws up-to data status.

    Quick check if local file if exist is up-to data with aws server.

    Args:
        model_name (list): Name of the NN model.

    Returns:
        bool: If True, local file is up-to-date.
    """
    """Check if temp dir exist"""
    if not isdir(join(expanduser('~'), '.tardis_pytorch')):
        return False  # No weight, first Tardis run, download from aws

    """Check for stored file header in ~/.tardis_pytorch/..."""
    if not isfile(join(expanduser('~'),
                       '.tardis_pytorch',
                       f'{model_name[0]}_{model_name[1]}',
                       f'{model_name[2]}',
                       'model_weights.pth')):
        return False  # Define network was never used with tardis, download from aws
    else:
        if not isfile(join(expanduser('~'),
                           '.tardis_pytorch',
                           f'{model_name[0]}_{model_name[1]}',
                           f'{model_name[2]}',
                           'model_header.json')):
            return False  # Weight found but no json, download from aws
        else:
            try:
                save = json.load(open(join(expanduser('~'),
                                           '.tardis_pytorch',
                                           f'{model_name[0]}_{model_name[1]}',
                                           f'{model_name[2]}',
                                           'model_header.json')))
            except:
                save = None

    """Compare stored file with file stored on aws"""
    if save is None:
        print('Network cannot be checked! Connect to the internet next time!')
        return False  # Error loading json, download from aws
    else:
        try:
            weight = requests.get('https://tardis-weigths.s3.amazonaws.com/'
                                  f'{model_name[0]}_{model_name[1]}/'
                                  f'{model_name[2]}/model_weights.pth', stream=True)
            aws = dict(weight.headers)
        except:
            print('Network cannot be checked! Connect to the internet next time!')
            return True  # Found saved weight but cannot connect to aws

    try:
        aws_data = aws['Last-Modified']
    except KeyError:
        aws_data = aws['Date']

    try:
        save_data = save['Last-Modified']
    except KeyError:
        save_data = save['Date']

    if save_data == aws_data:
        return True  # Up-to data weight, load from local dir
    else:
        return False  # There is new version on aws, download from aws
