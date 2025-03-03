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
    Fetch benchmark data from a specified AWS endpoint.

    This function sends a GET request to the specified AWS S3 URL to fetch the latest network
    benchmark data. If the request is successful (HTTP status code 200), it decodes the
    JSON content from the response and returns it as a Python dictionary. The function
    handles the request with a specific timeout configuration.

    :return: A dictionary containing the benchmark data fetched from the AWS endpoint if
        the request is successful and content is valid JSON.
    :rtype: dict
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
    Sends benchmark data and optional model weights to an AWS S3 bucket.

    This function uploads provided benchmark data as a JSON file to a predefined
    location in an AWS S3 bucket. Optionally, if a model file is provided, it
    uploads the model weights to a corresponding directory in the bucket based on
    the network name. The function ensures appropriate HTTP status codes are checked
    to confirm successful uploads.

    :param data: The benchmark data to upload. It should be a dictionary containing
        the relevant information.
    :param network: The name of the network associated with the uploaded model. This
        serves as part of the model file name. Defaults to an empty string if not
        provided.
    :param model: The path to the model's file to upload. If provided, the function
        will attempt to upload the model to the AWS S3 bucket.

    :return: A boolean value indicating whether the data upload (and optionally,
        the model upload) was successful.
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


def get_all_version_aws(network: str, subtype: str, model: str):
    """
    Fetches all available version strings of a specific model from the AWS Tardis weights bucket.

    This function communicates with an AWS S3 bucket to retrieve version information
    for specified network, subtype, and model. It parses the response to extract
    all relevant versions that match the expected format.

    :param network: The name of the network being queried.
    :type network: str
    :param subtype: The specific subtype of the network.
    :type subtype: str
    :param model: The model identifier within the specified network and subtype.
    :type model: str

    :return: A list of version strings that match the query parameters, starting with "V".
    :rtype: list[str]
    """
    r = requests.get(
        "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/",
    )
    r = r.content.decode("utf-8")
    all_version = [
        f[1:-3].split("/")[-1]
        for f in r.split("Key")
        if f.startswith(f">tardis_em/{network}_{subtype}/{model}/V_")
    ]
    return [v for v in all_version if v.startswith("V_")]


def get_model_aws(https: str):
    """
    Fetches a model from AWS by sending a GET request to the provided URL.

    :param https: The HTTPS URL string to fetch the model from.
    :type https: str

    :return: The response object from the GET request.
    :rtype: requests.Response
    """
    return requests.get(
        https,
        timeout=(5, None),
    )


def get_weights_aws(
    network: str,
    subtype: str,
    model: Optional[str] = None,
    version: Optional[int] = None,
):
    """
    Retrieve the weights of a specified deep learning model architecture and subtype,
    optionally specifying the model and version, from AWS or locally cached storage if
    available. The function handles directory setup, input validation, and either fetching
    or downloading the required weights.

    :param network: Name of the neural network model to retrieve. Must be one of the
        predefined valid model types from 'ALL_MODELS'.
    :type network: str
    :param subtype: Specific subtype or configuration of the network. Must be one of
        the predefined valid subtype values from 'ALL_SUBTYPE'.
    :type subtype: str
    :param model: Dataset or model specification for the given network and subtype. Defaults
        to None. Relevant only for certain network types.
    :type model: Optional[str]
    :param version: Version of the model to retrieve. Defaults to None, which automatically
        selects the latest version available for the given network, subtype, and model.
    :type version: Optional[int]

    :return: Path to the weights file if available locally or a file-like buffer containing
        the downloaded weights if fetched from AWS.
    """
    ALL_MODELS = ["unet", "unet3plus", "fnet_attn", "dist"]
    ALL_SUBTYPE = ["16", "32", "64", "96", "128", "triang", "full"]
    CNN = ["unet", "unet3plus", "fnet", "fnet_attn"]
    CNN_DATASET = [
        "microtubules_tirf",
        "microtubules_3d",
        "microtubules_2d",
        "membrane_3d",
        "membrane_2d",
        "actin_3d",
    ]
    DIST_DATASET = ["microtubules", "s3dis", "membrane_2d", "2d", "3d"]

    """Check dir"""
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

    all_version = get_all_version_aws(network, subtype, model)

    if len(all_version) == 0:
        version = None
    else:
        if version is not None:
            version_assert = [v for v in all_version if v == f"V_{version}"]
            if not len(version_assert) == 1:
                TardisError(
                    "19", "tardis_em/utils/aws.py", f"{model} of V{version} not found"
                )
            else:
                version = f"{version_assert[0]}"
        else:
            version = f"V_{max([int(v.split('_')[1]) for v in all_version])}"

    if aws_check_with_temp(model_name=[network, subtype, model, version]):
        print(f"Loaded temp weights for: {network}_{subtype} {version}...")

        if isfile(join(dir_, "model_weights.pth")):
            return join(dir_, "model_weights.pth")
        else:
            TardisError("19", "tardis_em/utils/aws.py", "No weights found")
    else:
        print(f"Downloading new weights for: {network}_{subtype} {version}...")

        if version is None:
            weight = get_model_aws(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
                f"{network}_{subtype}/"
                f"{model}/model_weights.pth"
            )
        else:
            weight = get_model_aws(
                "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
                f"{network}_{subtype}/"
                f"{model}/{version}/model_weights.pth"
            )

        # Save weights
        open(join(dir_, "model_weights.pth"), "wb").write(weight.content)

        # Save header
        with open(join(dir_, "model_header.json"), "w") as f:
            json.dump(dict(weight.headers), f)

        # Save version
        with open(join(dir_, "model_version.json"), "w") as f:
            json.dump({'version': version}, f)

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
    Check the availability and validity of local model weights and headers compared
    to the version stored on AWS.

    :param model_name: A list containing identifiers for the model. It is expected
        to contain at least four elements where each corresponds to specific attributes
        or subdirectories of the model.
    :return: A boolean value indicating whether the locally stored model weights
        are up-to-date (True) or require a download/update (False).
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
            return False  # Weight found but no JSON, download from aws
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
        return False  # Error loading JSON, download from aws
    else:
        try:
            if model_name[3] == None:
                weight = requests.get(
                    "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
                    f"{model_name[0]}_{model_name[1]}/"
                    f"{model_name[2]}/model_weights.pth",
                    stream=True,
                    timeout=(5, None),
                )
                aws = dict(weight.headers)
            else:
                weight = requests.get(
                    "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
                    f"{model_name[0]}_{model_name[1]}/"
                    f"{model_name[2]}/{model_name[3]}/model_weights.pth",
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
        return False  # There is a new version on aws, download from aws


def aws_check_pkg_with_temp() -> bool:
    """
    Checks the existence and validity of a local Tardis package and compares it with
    the latest version available on AWS.

    :return: A boolean indicating whether the local package exists and is up-to-date.
        If False is returned, the caller is expected to download the package from AWS.
    :rtype: bool
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
        return False  # Error loading JSON, download from aws
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
        return False  # There is a new version on aws, download from aws
