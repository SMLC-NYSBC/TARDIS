#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:
    pass
from tardis_em.utils import SCANNET_COLOR_MAP_20


def _dataset_format(coord: np.ndarray, segmented: bool) -> Tuple[np.ndarray, bool]:
    """
    Silently check for an array format and correct 2D datasets to 3D.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.

    Returns:
        Tuple[np.ndarray, bool]: Checked and corrected coord array with boolean
        statement if array is compatible.
    """
    check = True

    if segmented:
        if coord.shape[1] not in [3, 4]:
            check = False
            print("Coord data must be 2D/3D with labels (4D/5D)")

        # Correct 2D to 3D
        if coord.shape[1] == 3:
            coord = np.vstack(
                (coord[:, 0], coord[:, 1], coord[:, 2], np.zeros((coord.shape[0],)))
            ).T
    else:
        if coord.shape[1] not in [2, 3]:
            check = False
            print("Coord data must be 2D/3D with labels (2D/3D)")

        # Correct 2D to 3D
        if coord.shape[1] == 2:
            coord = np.vstack((coord[:, 0], coord[:, 1], np.zeros((coord.shape[0],)))).T

    return coord, check


def _rgb(coord: np.ndarray, segmented: bool, ScanNet=False, color=False) -> np.ndarray:
    """
    Convert float to RGB classes.

    Use predefined Scannet V2 RBG classes or random RGB classes.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.
        ScanNet (bool): If True output scannet v2 classes.

    Returns:
        np.ndarray: 3D array with RGB values for each point.
    """
    rgb = np.zeros((coord.shape[0], 3), dtype=np.float64)

    if segmented:
        if ScanNet:
            for id_, i in enumerate(coord[:, 0]):
                color = SCANNET_COLOR_MAP_20.get(i, SCANNET_COLOR_MAP_20[0])
                rgb[id_, :] = [x / 255 for x in color]
        else:
            unique_ids = np.unique(coord[:, 0])
            if color:
                rgb_list = [np.array((0, 0, 0)) for _ in unique_ids]
            else:
                rgb_list = [
                    np.array((np.random.rand(), np.random.rand(), np.random.rand()))
                    for _ in unique_ids
                ]
            id_to_rgb = {idx: color for idx, color in zip(unique_ids, rgb_list)}

            for id_, i in enumerate(coord[:, 0]):
                df = id_to_rgb[i]
                rgb[id_, :] = df
    else:
        rgb[:] = [1, 0, 0]

    return rgb


def segment_to_graph(coord: np.ndarray) -> list:
    """
    Build filament vector lines for open3D.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].

    Returns:
        list: list of segments converted for open3D
    """
    graph_list = []
    stop = 0

    for i in np.unique(coord[:, 0]):
        id_ = np.where(coord[:, 0] == i)[0]
        id_ = coord[id_]

        x = 0  # Iterator checking if current point is a first on in the list
        start = stop
        stop += len(id_)

        if x == 0:
            graph_list.append([start, start + 1])

        length = stop - start  # Number of point in a segment
        for j in range(1, length - 1):
            graph_list.append([start + (x + 1), start + x])

            if j != (stop - 1):
                graph_list.append([start + (x + 1), start + (x + 2)])
            x += 1
        graph_list.append([start + (x + 1), start + x])

    return graph_list


def VisualizePointCloud(
    coord: np.ndarray, segmented: bool, rgb: Optional[np.ndarray] = None, animate=False
):
    """
    Visualized point cloud.

    Output color coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.
        rgb (np.ndarray): Optional, indicate rgb values.
        animate (bool): Optional trigger to turn off animated rotation.
    """
    coord, check = _dataset_format(coord=coord, segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)

        if rgb is not None:
            if np.max(rgb) > 1:
                rgb = rgb / 255
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented))

        if animate:
            o3d.visualization.draw_geometries_with_animation_callback(
                [pcd], rotate_view
            )
        else:
            o3d.visualization.draw_geometries_with_animation_callback(
                [pcd], background_view
            )


def rotate_view(vis):
    """
    Optional viewing parameter for open3D to constantly rotate scene.
    Args:
        vis: Open3D view control setting.
    """
    ctr = vis.get_view_control()
    ctr.rotate(2.0, 0.0)

    return False


def background_view(vis):
    """
    Optional viewing parameter for open3D to constantly rotate scene.
    Args:
        vis: Open3D view control setting.
    """
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False


def VisualizeFilaments(coord: np.ndarray, animate=True, with_node=True):
    """
    Visualized filaments.

    Output color coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        animate (bool): Optional trigger to turn off animated rotation.
        with_node (bool): Optional, If True show points on filament
    """
    coord, check = _dataset_format(coord=coord, segmented=True)

    if check:
        if with_node:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
            pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, True))

        graph = segment_to_graph(coord=coord)
        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(coord[:, 1:])
        line_set.lines = o3d.utility.Vector2iVector(graph)
        line_set.colors = o3d.utility.Vector3dVector(_rgb(coord, True, True))

        if animate:
            if with_node:
                o3d.visualization.draw_geometries_with_animation_callback(
                    [pcd, line_set], rotate_view
                )
            else:
                o3d.visualization.draw_geometries_with_animation_callback(
                    [line_set], rotate_view
                )
        else:
            if with_node:
                o3d.visualization.draw_geometries_with_animation_callback(
                    [pcd, line_set], background_view
                )
            else:
                o3d.visualization.draw_geometries([line_set])


def VisualizeScanNet(coord: np.ndarray, segmented: True):
    """
    Visualized scannet scene

    Output color-coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.
    """
    coord, check = _dataset_format(coord=coord, segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented, True))

        o3d.visualization.draw_geometries([pcd])
