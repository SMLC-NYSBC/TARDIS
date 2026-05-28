#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  Robert Kiewisz                                                     #
#  MIT License 2021 - 2026                                            #
#######################################################################
import warnings
from os import getcwd

import click

from tardis_em._version import version
from tardis_em.utils.logo import print_error
from tardis_em_analysis.serial_stitch.cli import run_stitch

warnings.simplefilter("ignore", UserWarning)


@click.command()
@click.option(
    "-dir",
    "--path",
    default=getcwd(),
    type=str,
    help="Directory with serial-section tomograms (.am) and their "
    "*_spatialGraph.am microtubule graphs.",
    show_default=True,
)
@click.option(
    "-out",
    "--output",
    default=None,
    type=str,
    help="Output directory (default <dir>/stitched_output).",
    show_default=True,
)
@click.option(
    "-omega",
    "--warp_omega",
    default=0.3,
    type=float,
    help="Warp vorticity bound; lower = smoother deformation, fewer whirlpools.",
    show_default=True,
)
@click.option(
    "--image_fill/--no_image_fill",
    default=True,
    help="Align MT-free regions from image content (needs volumes).",
    show_default=True,
)
@click.option(
    "-m",
    "--method",
    default="mi",
    type=click.Choice(["mi", "grad", "ncc"]),
    help="Image-fill matching metric (mi = mutual information; robust to "
    "contrast/blur).",
    show_default=True,
)
@click.option(
    "-ds",
    "--downscale",
    default=1,
    type=int,
    help="Volume decimation factor (1 = full resolution).",
    show_default=True,
)
@click.option(
    "--cpu",
    is_flag=True,
    default=False,
    help="Force CPU warp (disable GPU CUDA/MPS).",
    show_default=True,
)
@click.option(
    "-w",
    "--workers",
    default=None,
    type=int,
    help="CPU processes for image-fill matching (default: cpu_count - 2).",
    show_default=True,
)
@click.option("-test_click", "--test_click", default=False, hidden=True)
@click.version_option(version=version)
def main(
    path: str,
    output: str,
    warp_omega: float,
    image_fill: bool,
    method: str,
    downscale: int,
    cpu: bool,
    workers: int,
    test_click: bool,
):
    """
    Stitch a folder of serial-section tomograms into one volume + merged
    microtubules. Runs fully automatically with defaults;
    GPU (CUDA/MPS) is used when available. Writes the stitched volume, merged
    spatial graph, and a detailed log to the output directory.
    """
    if test_click:
        return

    try:
        run_stitch(
            path,
            output,
            image_fill=image_fill,
            method=method,
            warp_omega=warp_omega,
            downscale=downscale,
            use_gpu=False if cpu else None,
            workers=workers,
        )
    except (FileNotFoundError, ValueError) as e:
        # Expected, user-facing problems (bad/empty folder, no usable sections,
        # nothing to stitch): show the message inside the TARDIS logo box, no
        # Python traceback.
        print_error(str(e), title="serial-section stitcher — cannot stitch")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
