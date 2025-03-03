#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import warnings
from os import getcwd, mkdir
from os.path import join, isdir

import click

from tardis_em._version import version
from tardis_em.utils.aws import get_all_version_aws, get_model_aws
from tardis_em.utils.errors import TardisError

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option(
    "-dir",
    "--save_dir",
    default=getcwd(),
    type=str,
    help="Directory where weights will be saved.",
    show_default=True,
)
@click.option(
    "-mc",
    "--model_cnn",
    default=None,
    type=click.Choice(
        [
            "microtubules_tirf",
            "microtubules_3d",
            "microtubules_2d",
            "membrane_3d",
            "membrane_2d",
            "actin_3d",
        ]
    ),
    help="Name of the model weight",
    show_default=True,
)
@click.option(
    "-cv",
    "--cnn_version",
    default=None,
    type=int,
    help="Optional CNN version of the model from 1 to inf.",
    show_default=True,
)
@click.option(
    "-md",
    "--model_dist",
    default=None,
    type=click.Choice(["2d", "3d"]),
    help="Name of the model weight",
    show_default=True,
)
@click.option(
    "-dv",
    "--dist_version",
    default=None,
    type=int,
    help="Optional DIST version of the model from 1 to inf.",
    show_default=True,
)
@click.version_option(version=version)
def main(
    save_dir: str,
    model_cnn: str,
    cnn_version: str,
    model_dist: str,
    dist_version: str,
):
    """
    MAIN MODULE FOR FETCHING WEIGHTS FILES
    """
    if not isdir(save_dir):
        TardisError(
            "19",
            "tardis_em/utils/aws.py",
            f"{save_dir} not found",
        )

    if model_cnn is not None:
        if not isdir(save_dir):
            mkdir(save_dir)
        if not isdir(
            join(
                save_dir,
                "fnet_attn_32",
            )
        ):
            mkdir(join(save_dir, "fnet_attn_32"))
        if not isdir(join(save_dir, "fnet_attn_32", model_cnn)):
            mkdir(join(save_dir, "fnet_attn_32", model_cnn))

    """Save CNN Weights in selected directory"""
    if model_cnn is not None:
        all_version = get_all_version_aws("fnet_attn", "32", model_cnn)
        if cnn_version is not None:
            version_assert = [v for v in all_version if v == f"V_{cnn_version}"]
            if not len(version_assert) == 1:
                TardisError(
                    "19",
                    "tardis_em/utils/aws.py",
                    f"{model_cnn} of V{cnn_version} not found",
                )
            else:
                cnn_version = f"{version_assert[0]}"
        else:
            cnn_version = f"V_{max([int(v.split('_')[1]) for v in all_version])}"

        if not isdir(join(save_dir, "fnet_attn_32", model_cnn, cnn_version)):
            mkdir(join(save_dir, "fnet_attn_32", model_cnn, cnn_version))

        weight_cnn = get_model_aws(
            "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
            "fnet_attn_32/"
            f"{model_cnn}/{cnn_version}//model_weights.pth"
        )

        open(
            join(save_dir, "fnet_attn_32", model_cnn, cnn_version, "model_weights.pth"),
            "wb",
        ).write(weight_cnn.content)

    """Save DIST Weights in selected directory"""
    if model_dist is not None:
        if not isdir(save_dir):
            mkdir(save_dir)
        if not isdir(
            join(
                save_dir,
                "dist_triang",
            )
        ):
            mkdir(join(save_dir, "dist_triang"))
        if not isdir(join(save_dir, "dist_triang", model_dist)):
            mkdir(join(save_dir, "dist_triang", model_dist))

        all_version = get_all_version_aws("dist", "triang", model_dist)
        if dist_version is not None:
            version_assert = [v for v in all_version if v == f"V_{dist_version}"]
            if not len(version_assert) == 1:
                TardisError(
                    "19",
                    "tardis_em/utils/aws.py",
                    f"{model_dist} of V{dist_version} not found",
                )
            else:
                dist_version = f"{version_assert[0]}"
        else:
            dist_version = f"V_{max([int(v.split('_')[1]) for v in all_version])}"

        if not isdir(join(save_dir, "dist_triang", model_dist, dist_version)):
            mkdir(join(save_dir, "dist_triang", model_dist, dist_version))
        weight_dist = get_model_aws(
            "https://tardis-weigths.s3.dualstack.us-east-1.amazonaws.com/tardis_em/"
            "dist_triang/"
            f"{model_dist}/{dist_version}/model_weights.pth"
        )
        open(
            join(
                save_dir, "dist_triang", model_dist, dist_version, "model_weights.pth"
            ),
            "wb",
        ).write(weight_dist.content)


if __name__ == "__main__":
    main()
