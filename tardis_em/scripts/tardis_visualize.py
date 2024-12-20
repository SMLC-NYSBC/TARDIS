#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import click
import numpy as np
from tardis_em.utils.visualize_pc import VisualizeFilaments, VisualizePointCloud
from tardis_em.utils.load_data import ImportDataFromAmira, load_mrc_file, load_am
import tifffile.tifffile as tif
from tardis_em._version import version


@click.command()
@click.option(
    "-dir",
    "--path",
    type=str,
    help="File directory to visualize.",
    show_default=True,
)
@click.option(
    "-2d",
    "--_2d",
    type=bool,
    default=False,
    help="True if 2D data",
    show_default=True,
)
@click.option(
    "-t",
    "--type_s",
    type=click.Choice(["f", "p"]),
    default="p",
    help="Visualize filaments or points",
    show_default=True,
)
@click.option(
    "-a",
    "--animate",
    type=bool,
    default=False,
    help="Animate if True.",
    show_default=True,
)
@click.option(
    "-wn",
    "--with_node",
    type=bool,
    default=False,
    help="If visualizing filaments, show color codded filaments with nodes.",
    show_default=True,
)
@click.option(
    "-c",
    "--color",
    type=str,
    default=None,
    help="Optional color. For example '1 1 1' define white color in RGB.",
    show_default=True,
)
@click.version_option(version=version)
def main(
    path: str,
    _2d: bool,
    type_s: str,
    animate: bool,
    with_node: bool,
    color: tuple,
):
    if color is not None:
        color = color.split(" ")
        color = [float(c) for c in color]

    pc = None

    if path.endswith(".csv"):
        pc = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    elif path.endswith("am"):
        try:
            pc = ImportDataFromAmira(path).get_segmented_points()
        except:
            pc, _, _, _ = load_am(path)
            pc = build_pc_from_img(pc)
    elif path.endswith(".npy"):
        pc = np.load(path)
    else:
        if path.endswith((".mrc", ".rec")):
            pc, _ = load_mrc_file(path)
        elif path.endswith((".tif", ".tiff")):
            pc = tif.imread(path)

        pc = build_pc_from_img(pc)

    if pc is not None:
        if type_s == "p":
            if pc.shape[1] == 4 or pc.shape[1] == 3 and _2d:
                VisualizePointCloud(pc, segmented=True, animate=animate)
            elif pc.shape[1] > 4:
                VisualizePointCloud(
                    pc[:, :3], segmented=False, rgb=pc[:, 3:], animate=animate
                )
            elif pc.shape[1] == 3:
                VisualizePointCloud(pc, segmented=False, animate=animate)
            else:
                return
        else:
            assert (
                pc.shape[1] == 4
            ), "Filament visualization require segmented point cloud"
            VisualizeFilaments(
                pc, filament_color=color, animate=animate, with_node=with_node
            )


def build_pc_from_img(img: np.ndarray):
    pc = None

    if img.min() == 0 and img.max() == 1:
        pc = np.array(np.where(img > 0)).T
    else:
        idx_n = np.unique(img)

        px_df = []
        for i in idx_n:
            if i == 0:
                continue

            i_ = np.array(np.where(img == i)).T
            px_df.append(np.hstack((np.repeat(i, len(i_))[:, None], i_)))

        pc = np.vstack(px_df)

    return pc


if __name__ == "__main__":
    main()
