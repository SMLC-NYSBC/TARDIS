#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import time
from os import listdir
from os.path import expanduser, join
from typing import Optional

import click
import torch

from tardis_em.benchmarks.predictor import CnnBenchmark, DISTBenchmark
from tardis_em.utils.aws import get_benchmark_aws, put_benchmark_aws
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.logo import TardisLogo
from tardis_em.utils.metrics import compare_dict_metrics
from tardis_em.utils.predictor import Predictor
from tardis_em._version import version


@click.command()
@click.option(
    "-dir",
    "--local_directory",
    default=None,
    type=str,
    help="Optionally, specified benchmark directory if different then"
    "SMLC local server.",
    show_default=True,
)
@click.option(
    "-ds",
    "--data_set",
    type=str,
    help="Data set name used for testing, should be the same as dataset"
    "storage folder name for given dataset.",
    show_default=True,
)
@click.option(
    "-ch",
    "--model_checkpoint",
    type=str,
    help="Directory for model pre-trained weight and structure dictionary.",
    show_default=True,
)
@click.option(
    "-th",
    "--nn_threshold",
    default=0.5,
    type=float,
    help="Threshold use for NN prediction.",
    show_default=True,
)
@click.option(
    "-ps",
    "--patch_size",
    default=None,
    type=int,
    help="Optional: Size of image size used for prediction.",
    show_default=True,
)
@click.option(
    "-sg",
    "--sigma",
    default=None,
    type=float,
    help="Optional: Sigma value for distance embedding.",
    show_default=True,
)
@click.option(
    "-pv",
    "--points_in_patch",
    default=1000,
    type=int,
    help="Optional: Number of point per voxel.",
    show_default=True,
)
@click.option(
    "-sv",
    "--save",
    default=True,
    type=bool,
    help="Flag to switch of saving model weights and metric on S3.",
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    default="0",
    type=str,
    help="Define which device use for training: "
    "0-9: Use positive numeric value to use specific GPU ID "
    "-1: Usa to use CPU "
    "mps: Use to use Apple Silicon",
    show_default=True,
)
@click.version_option(version=version)
def main(
    local_directory: Optional[str],
    data_set: str,
    model_checkpoint: str,
    nn_threshold: float,
    patch_size: Optional[int],
    sigma: Optional[float],
    points_in_patch: Optional[int],
    save: bool,
    device: str,
):
    """
    Standard benchmark for DIST on medical and standard point clouds
    """
    """Global setting"""
    tardis_progress = TardisLogo()
    title = "TARDIS - NN Benchmark"
    tardis_progress(title=title)

    # Specified local directory for benchmark dataset
    if local_directory is None:  # Default NYSBC-SMLC internal server
        DIR_ = join(expanduser("~") + "/../../data/rkiewisz/Benchmarks")
    else:
        DIR_ = local_directory

    """Get model for benchmark"""
    model = torch.load(
        model_checkpoint, map_location=get_device(device), weights_only=False
    )

    """Best model list from S3"""
    rgb = False
    if data_set.endswith("rgb"):
        data_set = data_set[:-4]
        rgb = True

    if [True for x in model["model_struct_dict"] if x.startswith("cnn")]:
        network = "cnn"
        DIR_NN = join(DIR_, "Best_model_CNN", data_set)
        DIR_EVAL = join(DIR_, "Eval_CNN", data_set)
    else:
        network = "dist"
        DIR_NN = join(DIR_, "Best_model_DIST", data_set)
        DIR_EVAL = join(DIR_, "Eval_DIST", data_set)

    if data_set not in listdir(join(DIR_, "Eval_DIST")):
        TardisError(
            id_="",
            py="tardis_em/benchmarks/benchmarks.py",
            desc=f"Given data set {data_set} is not supporter! "
            f'Expected one of {listdir(join(DIR_, "Eval_DIST"))}',
        )

    if rgb:
        data_set = f"{data_set}_rgb"

    BEST_SCORE = get_benchmark_aws()

    m_name = model["model_struct_dict"]

    predictor = Predictor(
        checkpoint=model, img_size=patch_size, sigma=sigma, device=get_device(device)
    )

    """Build DataLoader"""
    if network == "cnn":
        m_name = f'{m_name["cnn_type"]}_{m_name["conv_scaler"]}'

        predictor_bch = CnnBenchmark(
            model=predictor,
            dataset=data_set,
            dir_s=DIR_EVAL,
            threshold=nn_threshold,
            patch_size=patch_size,
        )
    else:
        m_name = f'dist_{m_name["dist_type"]}_{m_name["structure"]}'

        predictor_bch = DISTBenchmark(
            model=predictor,
            dataset=data_set,
            dir_s=DIR_EVAL,
            threshold=nn_threshold,
            points_in_patch=points_in_patch,
        )
    nbm = predictor_bch()

    """Compared with best models"""
    new_is_best = False
    model_best_time = "None"
    cbm = None
    model_id = 0

    # Pick last best model
    try:
        model_best = BEST_SCORE[m_name][data_set]
        model_id = len(model_best)
        model_best = model_best[-1]  # Pick last/best model from the list
        model_best_time = model_best[0]  # Pick benchmark time
        cbm = model_best[3]  # Pick the best metric dictionary
    except KeyError:
        pass

    """Benchmark Summary"""
    metric_keys = (
        ["IoU", "AUC", "mCov", "mWCov"]
        if network != "cnn"
        else ["F1", "AUC", "IoU", "AP"]
    )

    if model_best_time != "None":
        # Take mean of all metrics and check which one performs better
        new_is_best = compare_dict_metrics(last_best_dict=cbm, new_dict=nbm)

        best_metric = "; ".join([f"{key}: {cbm.get(key, '')}" for key in metric_keys])
    else:
        best_metric = "There is no existing model to compare with."

    new_metric = "; ".join([f"{key}: {nbm.get(key, '')}" for key in metric_keys])

    tardis_progress(
        title="TARDIS - NN Benchmark - Results",
        text_1=f"New model is better: {new_is_best}",
        text_3=f"Benchmark results for model: {m_name}: ",
        text_4=f"Model dataset benchmarked on: {data_set}",
        text_6=f"Last best model from [{model_best_time}]:",
        text_7=best_metric,
        text_9=f"Current benchmark model from [{time.asctime()}]:",
        text_10=new_metric,
    )

    """Sent updated json and model to S3"""
    if new_is_best:
        id_save = f"{m_name}_{data_set}_{model_id}"
    else:
        id_save = f"{m_name}_{data_set}_0"
    link = (
        "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/benchmark/models/"
        f"{id_save}"
    )

    if save:
        if m_name in BEST_SCORE:
            if data_set in BEST_SCORE[m_name]:
                BEST_SCORE[m_name][data_set].append(
                    [time.asctime(), link, round(sum(nbm.values()) / len(nbm), 2), nbm]
                )
            else:
                BEST_SCORE[m_name][data_set] = [
                    [time.asctime(), link, round(sum(nbm.values()) / len(nbm), 2), nbm]
                ]
        else:
            BEST_SCORE[m_name] = {f"{data_set}": []}
            BEST_SCORE[m_name][data_set].append(
                [time.asctime(), link, round(sum(nbm.values()) / len(nbm), 2), nbm]
            )

        """Upload model and json"""
        if model_best_time == "None":
            put_benchmark_aws(data=BEST_SCORE, network=id_save, model=model_checkpoint)
            torch.save(model_checkpoint, join(DIR_NN, id_save))
        elif new_is_best:
            put_benchmark_aws(data=BEST_SCORE, network=id_save, model=model_checkpoint)
            torch.save(model_checkpoint, join(DIR_NN, id_save))


if __name__ == "__main__":
    main()
