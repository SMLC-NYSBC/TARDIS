import open3d as o3d
import numpy as np


def _DataSetFormat(coord: np.ndarray,
                   segmented: bool):
    check = True

    if segmented:
        if coord.shape[1] not in [3, 4]:
            check = False
            print('Coord data must be 2D/3D with labels (4D/5D)')

        # Correct 2D to 3D
        if coord.shape[1] == 3:
            coord = np.vstack((coord[:, 0], coord[:, 1], coord[:, 2], np.zeros((coord.shape[0], )))).T
    else:
        if coord.shape[1] not in [2, 3]:
            check = False
            print('Coord data must be 2D/3D with labels (2D/3D)')

        # Correct 2D to 3D
        if coord.shape[1] == 2:
            coord = np.vstack((coord[:, 0], coord[:, 1], np.zeros((coord.shape[0], )))).T

    return coord, check


def _rgb(coord: np.ndarray,
         segmented: bool):
    rgb = np.zeros((coord.shape[0], 3), dtype=np.float64)
    if segmented:
        rgb_list = [np.array((np.random.rand(),
                            np.random.rand(),
                            np.random.rand())) for _ in np.unique(coord[:, 0])]

        for id, _ in enumerate(rgb):
            rgb[id, :] = rgb_list[int(coord[id, 0])]
    else:
        rgb_list = [[1, 0, 0]]

        for id, _ in enumerate(rgb):
            rgb[id, :] = rgb_list[0]

    return rgb


def SegmentToGraph(coord: np.ndarray):
    graph_list = []
    start = 0
    stop = 0

    for i in np.unique(coord[:, 0]):
        id = np.where(coord[:, 0] == i)[0]
        id = coord[id]

        itter = 0
        start = stop
        stop += len(id)

        if itter == 0:
            graph_list.append([start, start + 1])
        length = stop - start
        for j in range(1, length - 1):
            graph_list.append([start + (itter + 1), start + itter])

            if j != (stop - 1):
                graph_list.append([start + (itter + 1), start + (itter + 2)])
            itter += 1

        graph_list.append([start + (itter + 1), start + itter])

    return graph_list


def VisualizePointCloud(coord: np.ndarray,
                        segmented: True):
    coord, check = _DataSetFormat(coord=coord, 
                                  segmented=segmented)

    if check:
        pcd = o3d.geometry.PointCloud()

        if segmented:
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])
        else:
            pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, segmented))

        o3d.visualization.draw_geometries([pcd],)


def VisualizeFilaments(coord: np.ndarray):
    coord, check = _DataSetFormat(coord=coord, 
                                  segmented=True)

    if check:
        graph = SegmentToGraph(coord=coord)
        line_set = o3d.geometry.LineSet()

        line_set.points = o3d.utility.Vector3dVector(coord[:, 1:])
        line_set.lines = o3d.utility.Vector2iVector(graph)

        o3d.visualization.draw_geometries([line_set])
