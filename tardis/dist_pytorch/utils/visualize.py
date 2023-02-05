#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Tuple

import numpy as np
import open3d as o3d


def _dataset_format(coord: np.ndarray,
                    segmented: bool) -> Tuple[np.ndarray, bool]:
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
            print('Coord data must be 2D/3D with labels (4D/5D)')

        # Correct 2D to 3D
        if coord.shape[1] == 3:
            coord = np.vstack((coord[:, 0], coord[:, 1], coord[:, 2],
                               np.zeros((coord.shape[0], )))).T
    else:
        if coord.shape[1] not in [2, 3]:
            check = False
            print('Coord data must be 2D/3D with labels (2D/3D)')

        # Correct 2D to 3D
        if coord.shape[1] == 2:
            coord = np.vstack(
                (coord[:, 0], coord[:, 1], np.zeros((coord.shape[0], )))).T

    return coord, check


def _rgb(coord: np.ndarray,
         segmented: bool,
         ScanNet=False) -> np.ndarray:
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

    SCANNET_COLOR_MAP_20 = {
        0: (0., 0., 0.),
        1: (174., 199., 232.),
        2: (152., 223., 138.),
        3: (31., 119., 180.),
        4: (255., 187., 120.),
        5: (188., 189., 34.),
        6: (140., 86., 75.),
        7: (255., 152., 150.),
        8: (214., 39., 40.),
        9: (197., 176., 213.),
        10: (148., 103., 189.),
        11: (196., 156., 148.),
        12: (23., 190., 207.),
        14: (247., 182., 210.),
        15: (66., 188., 102.),
        16: (219., 219., 141.),
        17: (140., 57., 197.),
        18: (202., 185., 52.),
        19: (51., 176., 203.),
        20: (200., 54., 131.),
        21: (92., 193., 61.),
        22: (78., 71., 183.),
        23: (172., 114., 82.),
        24: (255., 127., 14.),
        25: (91., 163., 138.),
        26: (153., 98., 156.),
        27: (140., 153., 101.),
        28: (158., 218., 229.),
        29: (100., 125., 154.),
        30: (178., 127., 135.),
        32: (146., 111., 194.),
        33: (44., 160., 44.),
        34: (112., 128., 144.),
        35: (96., 207., 209.),
        36: (227., 119., 194.),
        37: (213., 92., 176.),
        38: (94., 106., 211.),
        39: (82., 84., 163.),
        40: (100., 85., 144.),
    }

    if segmented:
        if ScanNet:
            for id, i in enumerate(coord[:, 0]):
                color = SCANNET_COLOR_MAP_20.get(i, SCANNET_COLOR_MAP_20[0])
                rgb[id, :] = [x / 255 for x in color]
        else:
            unique_ids = np.unique(coord[:, 0])
            rgb_list = [np.array((np.random.rand(), np.random.rand(), np.random.rand()))
                        for _ in unique_ids]
            id_to_rgb = {idx: color for idx, color in zip(unique_ids, rgb_list)}
            for id, i in enumerate(coord[:, 0]):
                df = id_to_rgb[i]
                rgb[id, :] = df
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
        id = np.where(coord[:, 0] == i)[0]
        id = coord[id]

        x = 0  # Iterator checking if current point is a first on in the list
        start = stop
        stop += len(id)

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


def VisualizePointCloud(coord: np.ndarray,
                        segmented: True):
    """
    Visualized point cloud.

    Output color coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.
    """
    coord, check = _dataset_format(coord=coord,
                                   segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented))

        o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                                  rotate_view)


def rotate_view(vis):
    """
    Optional viewing parameter for open3D to constantly rotate scene.
    Args:
        vis: Open3D view control setting.
    """
    ctr = vis.get_view_control()
    ctr.rotate(1.0, 0.0)
    return False


def VisualizeFilaments(coord: np.ndarray):
    """
    Visualized filaments.

    Output color coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
    """
    coord, check = _dataset_format(coord=coord,
                                   segmented=True)

    if check:
        graph = segment_to_graph(coord=coord)
        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(coord[:, 1:])
        line_set.lines = o3d.utility.Vector2iVector(graph)

        o3d.visualization.draw_geometries([line_set])


def VisualizeScanNet(coord: np.ndarray,
                     segmented: True):
    """
    Visualized scannet scene

    Output color-coded point cloud. Color values indicate individual segments.

    Args:
        coord (np.ndarray): 2D or 3D array of shape [(s) x X x Y x Z] or [(s) x X x Y].
        segmented (bool): If True expect (s) in a data format as segmented values.
    """
    coord, check = _dataset_format(coord=coord,
                                   segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented, True))

        o3d.visualization.draw_geometries([pcd])
