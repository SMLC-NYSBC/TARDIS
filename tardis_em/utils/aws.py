#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import io
import json
from os import makedirs, mkdir
from os.path import expanduser, isdir, isfile, join
from typing import Optional

import requests

from tardis_em.utils.errors import TardisError


def get_benchmark_aws() -> dict:
    """
    Retrieve best benchmarking score for given NN type

    Returns:
        dict: Dictionary with keys[network name] and values[list of scores]
    """
    network_benchmark = requests.get(
        "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/benchmark/best_scores.json",
        timeout=(5, None),
    )

    if network_benchmark.status_code == 200:
        network_benchmark = json.loads(network_benchmark.content.decode("utf-8"))

    return network_benchmark


def put_benchmark_aws(data: dict, network: Optional[str] = "", model=None) -> bool:
    """
    Upload new or update dictionary stored on S3

    Args:
        data (dict): Dictionary with network the best metrics
        network (Optional, str): Benchmarking network name [e.g. fnet_32_microtubules_id].
        model (Optional, str): Optional dictionary to model.

    Returns:
        bool: True if save correctly
    """
    r = requests.put(
        "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
        "benchmark/best_scores.json",
        json.dumps(data, indent=2, default=str),
        timeout=(5, None),
    )

    if model is not None and r.status_code == 200:
        with open(model, "rb") as data:
            r_m = requests.put(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
                f"benchmark/models/{network}.pth",
                data=data,
                timeout=(5, None),
            )

        return r_m.status_code == 200
    return r.status_code == 200


def get_model_aws(https: str):
    return requests.get(
        https,
        timeout=(5, None),
    )


def get_weights_aws(network: str, subtype: str, model: Optional[str] = None):
    """
    Module to download pre-train weights from S3 AWS bucket.

    Model weight stored on S3 bucket with the naming convention
    network_subtype/model/model_weights.pth
    References.:
    - fnet_32/microtubules_3d/model_weights.pth
    - dist_triang/microtubules_3d/model_weights.pth

    Weights are stored in ~/.tardis_em with the same convention and .txt
    file with file header information to identified update status for local file
    if the network connection can be established.

    Args:
        network (str): Type of network for which weights are requested.
        subtype (str): Sub-name of the network or sub-parameter for the network.
        model (str): Additional dataset name used for the DIST.
    """
    ALL_MODELS = ["unet", "unet3plus", "fnet_attn", "dist"]
    ALL_SUBTYPE = ["16", "32", "64", "96", "128", "triang", "full"]
    CNN = ["unet", "unet3plus", "fnet", "fnet_attn"]
    CNN_DATASET = ["microtubules_3d", "microtubules_2d", "membrane_3d", "membrane_2d"]
    DIST_DATASET = ["microtubules", "s3dis", "membrane_2d", "2d", "3d"]

    """Chech dir"""
    if not isdir(join(expanduser("~"), ".tardis_em")):
        mkdir(join(expanduser("~"), ".tardis_em"))
    if not isdir(join(expanduser("~"), ".tardis_em", f"{network}_{subtype}")):
        mkdir(join(expanduser("~"), ".tardis_em", f"{network}_{subtype}"))
    if not isdir(
        join(expanduser("~"), ".tardis_em", f"{network}_{subtype}", f"{model}")
    ):
        mkdir(join(expanduser("~"), ".tardis_em", f"{network}_{subtype}", f"{model}"))

    """Get weights for CNN"""
    dir_ = join(expanduser("~"), ".tardis_em", f"{network}_{subtype}", f"{model}")

    if network not in ALL_MODELS:
        TardisError(
            "19",
            "tardis_em/utils/aws.py",
            f"Incorrect CNN network selected {network}_{subtype}",
        )
    if subtype not in ALL_SUBTYPE:
        TardisError(
            "19",
            "tardis_em/utils/aws.py",
            f"Incorrect CNN subtype selected {network}_{subtype}",
        )

    if network in CNN:
        if model not in CNN_DATASET:
            TardisError(
                "19",
                "tardis_em/utils/aws.py",
                f"Incorrect CNN model selected {model} but expected {CNN_DATASET}",
            )

    if network == "dist":
        if model not in DIST_DATASET:
            TardisError(
                "19",
                "tardis_em/utils/aws.py",
                f"Incorrect DIST model selected {model} but expected {DIST_DATASET}",
            )

    if aws_check_with_temp(model_name=[network, subtype, model]):
        if isfile(join(dir_, "model_weights.pth")):
            return join(dir_, "model_weights.pth")
        else:
            TardisError("19", "tardis_em/utils/aws.py", "No weights found")
    else:
        weight = get_model_aws(
            "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
            f"{network}_{subtype}/"
            f"{model}/model_weights.pth"
        )
        # Save weights
        open(join(dir_, "model_weights.pth"), "wb").write(weight.content)

        # Save header
        with open(join(dir_, "model_header.json"), "w") as f:
            json.dump(dict(weight.headers), f)

        """Save temp weights"""
        if not isdir(join(expanduser("~"), ".tardis_em")):
            mkdir(join(expanduser("~"), ".tardis_em"))

        if not isdir(dir_):
            makedirs(dir_)

        print(f"Pre-Trained model download from S3 and saved/updated in {dir_}")

        weight = weight.content
        if "AccessDenied" in str(weight[:100]):
            return join(dir_, "model_weights.pth")
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
    if not isdir(join(expanduser("~"), ".tardis_em")):
        return False  # No weight, first Tardis run, download from aws

    """Check for stored file header in ~/.tardis_em/..."""
    if not isfile(
        join(
            expanduser("~"),
            ".tardis_em",
            f"{model_name[0]}_{model_name[1]}",
            f"{model_name[2]}",
            "model_weights.pth",
        )
    ):
        return False  # Define network was never used with tardis_em, download from aws
    else:
        if not isfile(
            join(
                expanduser("~"),
                ".tardis_em",
                f"{model_name[0]}_{model_name[1]}",
                f"{model_name[2]}",
                "model_header.json",
            )
        ):
            return False  # Weight found but no json, download from aws
        else:
            try:
                save = json.load(
                    open(
                        join(
                            expanduser("~"),
                            ".tardis_em",
                            f"{model_name[0]}_{model_name[1]}",
                            f"{model_name[2]}",
                            "model_header.json",
                        )
                    )
                )
            except:
                save = None

    """Compare stored file with file stored on aws"""
    if save is None:
        print("Network cannot be checked! Connect to the internet next time!")
        return False  # Error loading json, download from aws
    else:
        try:
            weight = requests.get(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
                f"{model_name[0]}_{model_name[1]}/"
                f"{model_name[2]}/model_weights.pth",
                stream=True,
                timeout=(5, None),
            )
            aws = dict(weight.headers)
        except:
            print("Network cannot be checked! Connect to the internet next time!")
            return True  # Found saved weight but cannot connect to aws

    try:
        aws_data = aws["Last-Modified"]
    except KeyError:
        aws_data = aws["Date"]

    try:
        save_data = save["Last-Modified"]
    except KeyError:
        save_data = save["Date"]

    if save_data == aws_data:
        return True  # Up-to data weight, load from local dir
    else:
        return False  # There is new version on aws, download from aws


def aws_check_pkg_with_temp() -> bool:
    """
    Module to check aws up-to data status for OTA-Update.

    Quick check if local pkg file exist and is up-to data with aws server.

    Returns:
        bool: If True, local file is up-to-date.
    """
    """Check if temp dir exist"""
    if not isdir(join(expanduser("~"), ".tardis_em")):
        return False  # No pkg, first Tardis run, download from aws

    """Check for stored file header in ~/.tardis_em/pkg_header.json."""
    if not isfile(
        join(
            expanduser("~"),
            ".tardis_em/tardis_em-x.x.x-py3-none-any.whl",
        )
    ):
        return False  # PKG was never download, download from aws
    else:
        if not isfile(
            join(
                expanduser("~"),
                ".tardis_em/pkg_header.json",
            )
        ):
            return False  # PKG found but no header, download from aws
        else:
            try:
                save = json.load(
                    open(
                        join(
                            expanduser("~"),
                            ".tardis_em/pkg_header.json",
                        )
                    )
                )
            except:
                save = None

    """Compare stored file with file stored on aws"""
    if save is None:
        print("Tardis_e pkg cannot be checked! Connect to the internet next time!")
        return False  # Error loading json, download from aws
    else:
        try:
            pkg = requests.get(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/"
                "tardis_em/tardis_em-x.x.x-py3-none-any.whl",
                timeout=(5, None),
            )
            aws = dict(pkg.headers)
        except:
            print("Tardis_em pkg cannot be checked! Connect to the internet next time!")
            return True  # Found saved weight but cannot connect to aws

    try:
        aws_data = aws["Last-Modified"]
    except KeyError:
        aws_data = aws["Date"]

    try:
        save_data = save["Last-Modified"]
    except KeyError:
        save_data = save["Date"]

    if save_data == aws_data:
        return True  # Up-to data weight, load from local dir
    else:
        return False  # There is new version on aws, download from aws
