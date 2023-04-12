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
import open3d as o3d


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
            coord = np.vstack((coord[:, 0], coord[:, 1], coord[:, 2], np.zeros((coord.shape[0],)))).T
    else:
        if coord.shape[1] not in [2, 3]:
            check = False
            print("Coord data must be 2D/3D with labels (2D/3D)")

        # Correct 2D to 3D
        if coord.shape[1] == 2:
            coord = np.vstack((coord[:, 0], coord[:, 1], np.zeros((coord.shape[0],)))).T

    return coord, check


def _rgb(coord: np.ndarray, segmented: bool, ScanNet=False) -> np.ndarray:
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
        0: (0.0, 0.0, 0.0),
        1: (174.0, 199.0, 232.0),
        2: (152.0, 223.0, 138.0),
        3: (31.0, 119.0, 180.0),
        4: (255.0, 187.0, 120.0),
        5: (188.0, 189.0, 34.0),
        6: (140.0, 86.0, 75.0),
        7: (255.0, 152.0, 150.0),
        8: (214.0, 39.0, 40.0),
        9: (197.0, 176.0, 213.0),
        10: (148.0, 103.0, 189.0),
        11: (196.0, 156.0, 148.0),
        12: (23.0, 190.0, 207.0),
        14: (247.0, 182.0, 210.0),
        15: (66.0, 188.0, 102.0),
        16: (219.0, 219.0, 141.0),
        17: (140.0, 57.0, 197.0),
        18: (202.0, 185.0, 52.0),
        19: (51.0, 176.0, 203.0),
        20: (200.0, 54.0, 131.0),
        21: (92.0, 193.0, 61.0),
        22: (78.0, 71.0, 183.0),
        23: (172.0, 114.0, 82.0),
        24: (255.0, 127.0, 14.0),
        25: (91.0, 163.0, 138.0),
        26: (153.0, 98.0, 156.0),
        27: (140.0, 153.0, 101.0),
        28: (158.0, 218.0, 229.0),
        29: (100.0, 125.0, 154.0),
        30: (178.0, 127.0, 135.0),
        32: (146.0, 111.0, 194.0),
        33: (44.0, 160.0, 44.0),
        34: (112.0, 128.0, 144.0),
        35: (96.0, 207.0, 209.0),
        36: (227.0, 119.0, 194.0),
        37: (213.0, 92.0, 176.0),
        38: (94.0, 106.0, 211.0),
        39: (82.0, 84.0, 163.0),
        40: (100.0, 85.0, 144.0),
    }

    if segmented:
        if ScanNet:
            for id, i in enumerate(coord[:, 0]):
                color = SCANNET_COLOR_MAP_20.get(i, SCANNET_COLOR_MAP_20[0])
                rgb[id, :] = [x / 255 for x in color]
        else:
            unique_ids = np.unique(coord[:, 0])
            rgb_list = [np.array((np.random.rand(), np.random.rand(), np.random.rand())) for _ in unique_ids]
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


def VisualizePointCloud(coord: np.ndarray, segmented: bool, rgb: Optional[np.ndarray] = None, animate=True):
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
            o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)
        else:
            o3d.visualization.draw_geometries([pcd])


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
    coord, check = _dataset_format(coord=coord, segmented=True)

    if check:
        graph = segment_to_graph(coord=coord)
        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(coord[:, 1:])
        line_set.lines = o3d.utility.Vector2iVector(graph)

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
