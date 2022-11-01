import itertools
from math import sqrt
from typing import Optional

import numpy as np
import torch
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from tardis.dist_pytorch.utils.visualize import VisualizeFilaments, VisualizePointCloud


class GraphInstanceV2:
    def __init__(self,
                 threshold=float,
                 connection=2,
                 smooth=False):
        self.threshold = threshold

        self.connection = connection
        self.smooth = smooth

    @staticmethod
    def _stitch_graph(graph_pred: list,
                      idx: list):
        """
        STITCHER FOR GRAPH REPRESENTATION

        Args:
            graph_pred: Patches of graph predictions
            idx: Idx for each node in patches
        """
        # Build empty graph
        graph = max([max(f) for f in idx]) + 1
        graph = np.zeros((graph, graph),
                         dtype=np.float32)

        for idx_patch, graph_patch in zip(idx, graph_pred):
            for k, _ in enumerate(idx_patch):
                row = graph_patch[k, :]
                row_v = [row[id] if graph[i, idx_patch[k]] == 0
                         else np.mean((graph[i, idx_patch[k]], row[id]))
                         for id, i in enumerate(idx_patch)]

                column = graph_patch[:, k]
                column_v = [row[id] if graph[i, idx_patch[k]] == 0
                            else np.mean((graph[i, idx_patch[k]], column[id]))
                            for id, i in enumerate(idx_patch)]

                graph[list(idx_patch), idx_patch[k]] = row_v
                graph[idx_patch[k], list(idx_patch)] = column_v

        return graph

    @staticmethod
    def _stitch_coord(coord: list,
                      idx: list):
        """
        Stitcher for coord in patches

        Args:
            coord: Coords in each patch
            idx: Idx for each node in patches
        """
        # Conversion to Torch
        if isinstance(coord[0], torch.Tensor):
            coord = [c.cpu().detach().numpy() for c in coord]

        # Build empty coord array
        dim = coord[0].shape[1]
        coord_df = max([max(f) for f in idx]) + 1
        coord_df = np.zeros((coord_df, dim),
                            dtype=np.float32)

        for coord_patch, idx_patch in zip(coord, idx):
            for value, id in zip(coord_patch, idx_patch):
                coord_df[id, :] = value

        return coord_df

    @staticmethod
    def _stitch_cls(cls: list,
                    idx: list):
        """
        Stitcher for nodes in patches

        Args:
            cls: Predicted class in each patch
            idx: Idx for each node in patches
        """
        # Conversion to Torch
        if isinstance(cls[0], torch.Tensor):
            cls = [c.cpu().detach().numpy() for c in cls]

        # Build empty coord array
        cls_df = max([max(f) for f in idx]) + 1
        cls_df = np.zeros((cls_df),
                          dtype=np.float32)

        for cls_patch, idx_patch in zip(cls, idx):
            # cls_patch = [np.where(i == 1)[0][0] for i in cls_patch]
            for value, id in zip(cls_patch, idx_patch):
                cls_df[id] = value

        return cls_df

    def adjacency_list(self,
                       graph: np.ndarray):
        """
        # Stitch coord list and graph
        # Get Point ID
        # For each point find ID of 2 interactions with highest prop
            # For each possible interaction
            # Check for example: if Point ID 0 - 1 and 1 - 0 are highest prop for each other
                # if yes add prop
                # if not, check if there is other pair that has highest prop for each other
                    # If not remove
                    # If yes add
        """
        adjacency_list_id = []
        adjacency_list_prop = []

        all_prop = []
        for id, i in enumerate(graph):
            prop = sorted(zip(i[np.where(i > self.threshold)[0]],
                              np.where(i > self.threshold)[0]), reverse=True)

            all_prop.append([i for i in prop if i[1] != id])

        # Current ID edge for which interaction is search
        for id, i in enumerate(all_prop):
            edges = 0  # Number of edges found
            prop_id = 0  # Id in order list, take 1st, 2nd etc highest value
            prop_id_limit = 5

            while edges != self.connection and prop_id != prop_id_limit:
                if len(i) != 0:
                    node_id = [id, i[prop_id][1]]
                    node_prop = i[prop_id][0]

                    """Search if the interaction has highest prop for each connected points"""
                    if np.any([True for p in all_prop[node_id[1]][:2] if p[1] == id]):
                        if node_id[::-1] not in adjacency_list_id:
                            adjacency_list_id.append(node_id)
                            adjacency_list_prop.append(node_prop)

                        edges += 1

                    prop_id += 1
                    if prop_id + 1 > len(i):
                        prop_id = 5
                else:
                    break

        return adjacency_list_id, adjacency_list_prop

    def _adjacency_matrix(self,
                          graphs: list,
                          coord: np.ndarray,
                          output_idx: Optional[None] = list):
        """
        Builder of adjacency matrix from stitched coord and and graph voxels
        The output of the adjacency matrix is list containing:
        [id][coord][interactions][interaction probability]

        Args:
            graphs: graph patch output from DIST
            coord: stitched coord output from DIST
        """
        all_prop = [[id, list(i), [], []] for id, i in enumerate(coord)]

        if output_idx is None:
            output_idx = np.arange(graphs[0].shape[0])

        for g, o in zip(graphs, output_idx):
            df = np.where(g >= self.threshold)

            col = list(df[0])
            row = list(df[1])
            int = [[c, r] for c, r in zip(col, row) if c != r]
            prop = [g[i[0], i[1]] for i in int]
            int = [[o[i[0]], o[i[1]]] for i in int]

            for i, p in zip(int, prop):
                all_prop[i[0]][2].append(i[1])
                all_prop[i[0]][3].append(p)

        # Take mean value of all repeated interactions
        for p_id, i in enumerate(all_prop):
            inter = i[2]
            prop = i[3]

            if len(inter) > 1:
                all_prop[p_id][2] = list(np.unique(inter))
                all_prop[p_id][3] = [np.median([x for idx, x in enumerate(prop)
                                                if idx in [id for id, j in enumerate(inter)
                                                           if j == k]])
                                     for k in np.unique(inter)]

        # Sort each interactions based on the probability
        # Highest value are first
        for id, a in enumerate(all_prop):
            # Sort only, if there are more then 1 interaction
            if len(a[2]) > 1:
                # Sort
                prop, inter = zip(*sorted(zip(a[3], a[2]), reverse=True))

                # Replace for sorted value
                all_prop[id][2] = list(inter)
                all_prop[id][3] = list(prop)

        return all_prop

    def _find_segment_matrix(self,
                             adj_matrix: list):
        """
        Iterative search mechanism that search for connected points in the
        adjacency list.

        Args:
            adj_matrix: adjacency matrix of graph connections
        """
        idx_df = [0]
        x = 0

        # Find initial point
        while len(idx_df) == 1:
            idx_df = adj_matrix[x][2][:2]
            idx_df.append(x)
            x += 1

            if x > len(adj_matrix):
                break

        x -= 1
        new = new_df = adj_matrix[x][2][:self.connection]

        # Pick all point associated with the initial point
        # Check 1: Check if node is not already on the list
        # Check 2: Check if new interaction show up secondary interaction
        # Check 3: Check if found idx's were not associated to previous instance
        while len(new_df) != 0:
            # check only new elements
            new_df = []
            for i in new:
                # Pick secondary interaction for i
                reverse_int = adj_matrix[i][2][:self.connection]

                new_df = new_df + [j for j in reverse_int
                                   if j not in idx_df and  # Check 1
                                   i in adj_matrix[j][2][:self.connection]]  # Check 2
                new_df = list(np.unique(new_df))

            new = new_df
            idx_df = idx_df + new

        idx_df = np.unique(idx_df)

        idx_df = [i for i in idx_df if adj_matrix[i][2] != []]  # Check 3
        return idx_df

    def _smooth_segments(self,
                         coord: np.ndarray):
        smooth_spline = []
        tortuosity_spline = []

        # Smooth spline
        for i in np.unique(coord[:, 0]):
            x = coord[np.where(coord[:, 0] == int(i))[0], :]

            if len(x) > 4:
                tck, _ = splprep([x[:, 1], x[:, 2], x[:, 3]])

                u_fine = np.linspace(0, 1, int(len(x) * 0.5))  # Output half number of len(x)
                x_fine, y_fine, z_fine = splev(u_fine, tck)
                filament = np.vstack((x_fine, y_fine, z_fine)).T

                id = np.zeros((len(filament), 1))
                id += i
                df = np.hstack((id, filament))
                smooth_spline.append(df)
            else:
                smooth_spline.append(x)

        coord_segment_smooth = np.concatenate(smooth_spline)

        for i in np.unique(coord_segment_smooth[:, 0]):
            filament = coord_segment_smooth[np.where(coord_segment_smooth[:, 0] == int(i))[0], :]
            tortuosity_spline.append(tortuosity(filament))

        # Remove errors with hight tortuosity
        error = [id for id, i in enumerate(tortuosity_spline) if i > 1.1]
        segments = np.stack([i for i in coord_segment_smooth if i[0] not in error])

        return reorder_segments_id(segments)

    def patch_to_segment(self,
                         graph: list,
                         coord: np.ndarray,
                         idx: list,
                         prune: int,
                         sort=True,
                         visualize: Optional[str] = None):
        """
        Point cloud instance segmenter from graph representation

        From each point cloud (patch) segmenter first build adjacency matrix.
        Matrix is then used to iteratively search for new segments.
        For each initial node algorithm search for 2 edges with highest prop.
        with given threshold. For each found edges, algorithm check if found
        node creates an edge with previous node within 2 highest probability.
        If not alg. search for new edge that fulfill the statement.

        E.g.
        0
        |
        151_
        |   \
        515_  0
        |   \
        600  151
        |
        ...

        Args:
            graph: Graph patch output from Dist
            coord: Coordinates for each unsorted point idx
            idx: idx of point included in the segment
            sort: If True sort output
            visualize: If not None, visualize output with open3D
        """
        """Check data"""
        if isinstance(graph, np.ndarray):
            graph = [graph]
        elif isinstance(graph, torch.Tensor):
            graph = [graph.cpu().detach().numpy()]

        if isinstance(idx, np.ndarray):
            idx = [idx]
        elif isinstance(graph, torch.Tensor):
            idx = [idx.cpu().detach().numpy()]

        assert isinstance(coord, np.ndarray), \
            'Coord must be an array of all nodes!'

        """Build Adjacency list from graph representation"""
        adjacency_matrix = self._adjacency_matrix(graphs=graph,
                                                  coord=coord,
                                                  output_idx=idx)

        coord_segment = []
        stop = False
        segment_id = 0

        while not stop:
            idx = self._find_segment_matrix(adjacency_matrix)

            """Select segment longer then 3 points"""
            if len(idx) >= prune:
                # Sort points in segment
                if sort:
                    segment = sort_segment(coord=coord[idx])
                else:
                    segment = coord[idx]

                if segment.shape[1] == 3:
                    coord_segment.append(np.stack((np.repeat(segment_id, segment.shape[0]),
                                                   segment[:, 0],
                                                   segment[:, 1],
                                                   segment[:, 2])).T)
                elif segment.shape[1] == 2:
                    coord_segment.append(np.stack((np.repeat(segment_id, segment.shape[0]),
                                                   segment[:, 0],
                                                   segment[:, 1],
                                                   np.zeros((segment.shape[0], )))).T)
                segment_id += 1

            # Mask point assigned to the instance
            for id in idx:
                adjacency_matrix[id][1], adjacency_matrix[id][2], adjacency_matrix[id][3] = [], [], []

            if sum([sum(i[2]) for i in adjacency_matrix]) == 0:
                stop = True

        segments = np.vstack(coord_segment)
        if self.smooth:
            segments = self._smooth_segments(segments)

        if visualize is not None:
            assert visualize in ['f', 'p'], \
                'To visualize output use "f" for filament or "p" for point cloud!'

            if visualize == 'p':
                VisualizePointCloud(segments, True)
            elif visualize == 'f':
                VisualizeFilaments(segments)

        return segments


class FilterSpatialGraph:
    def __init__(self,
                 filter_unconnected_segments=True,
                 filter_short_spline=True):
        self.filter_connections = filter_unconnected_segments
        self.filter_short_segment = filter_short_spline

    def __call__(self,
                 segments: np.ndarray):
        if self.filter_connections:
            # Gradually connect segments with ends close to each other
            segments = filter_connect_near_segment(segments, 50)
            segments = filter_connect_near_segment(segments, 100)
            segments = filter_connect_near_segment(segments, 150)
            segments = filter_connect_near_segment(segments, 175)

        if self.filter_short_segment:
            length = []
            for i in np.unique(segments[:, 0]):
                length.append(total_length(segments[np.where(segments[:, 0] == int(i))[0], 1:]))

            length = [id for id, i in enumerate(length) if i > 1000]

            new_seg = []
            for i in length:
                new_seg.append(segments[np.where(segments[:, 0] == i), :])

            segments = np.hstack(new_seg)[0, :]

        return segments


def reorder_segments_id(coord: np.ndarray,
                        order_range: Optional[list] = None):
    df = np.unique(coord[:, 0])

    if order_range is None:
        df_range = np.asarray(range(0, len(df)), dtype=df.dtype)
    else:
        df_range = np.asarray(range(order_range[0], order_range[1]), dtype=df.dtype)

    for id, i in enumerate(coord[:, 0]):
        coord[id, 0] = df_range[np.where(df == i)[0][0]]

    return coord


def sort_segment(coord: np.ndarray):
    """
    Sorting of the point cloud based on number of connections followed by
    searching of the closest point with cdist.

    Args:
        coord: Coordinates for each unsorted point idx
        idx: idx of point included in the segment
    """
    new_c = []
    for i in range(len(coord) - 1):
        if i == 0:
            id = np.where([sum(i) for i in cdist(coord, coord)] == max(
                [sum(i) for i in cdist(coord, coord)]))[0]

            new_c.append(coord[id[0]])
            coord = np.delete(coord, id[0], 0)

        kd = KDTree(coord)
        points = kd.query(np.expand_dims(new_c[len(new_c) - 1], 0), 1)[1][0][0]

        new_c.append(coord[points])
        coord = np.delete(coord, points, 0)
    return np.stack(new_c)


def total_length(coord: np.ndarray):
    length = 0
    c_len = len(coord) - 1

    for id, _ in enumerate(coord):
        if id == c_len:
            break
        # sqrt((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)
        length += sqrt((coord[id][0] - coord[id + 1][0]) ** 2 +
                       (coord[id][1] - coord[id + 1][1]) ** 2 +
                       (coord[id][2] - coord[id + 1][2]) ** 2)

    return length


def tortuosity(coord: np.ndarray):
    length = total_length(coord)
    end_length = sqrt((coord[0][0] - coord[-1][0]) ** 2 +
                      (coord[0][1] - coord[-1][1]) ** 2 +
                      (coord[0][2] - coord[-1][2]) ** 2)

    return (length + 1e-16) / (end_length + 1e-16)


def filter_connect_near_segment(segments: np.ndarray,
                                dist_th=200):
    seg_list = [segments[np.where(segments[:, 0] == i), :][0] for i in np.unique(segments[:, 0])]
    filtered_seg = []

    # Find all segments ends
    ends = [s[0] for s in seg_list] + [s[-1] for s in seg_list]
    ends_coord = [s[0, 1:] for s in seg_list] + [s[-1, 1:] for s in seg_list]

    kd = cdist(ends_coord, ends_coord)

    # Find segments pair with ends in dist_th distance
    df = []
    for i in kd:
        df.append(np.where(i < dist_th)[0])
    idx_connect = [[int(ends[i[0]][0]), int(ends[i[1]][0])] for i in df if len(i) > 1]

    idx_connect.sort()
    idx_connect = list(k for k, _ in itertools.groupby(idx_connect))

    s = set()
    a1 = []
    for t in idx_connect:
        if tuple(t) not in s:
            a1.append(t)
            s.add(tuple(t)[::-1])
    idx_connect = list(s)

    # Select segments without any pair
    new_seg = []
    for i in [int(id) for id in np.unique(segments[:, 0]) if id not in np.unique(np.concatenate(idx_connect))]:
        new_seg.append(segments[np.where(segments[:, 0] == i), :])
    new_seg = np.hstack(new_seg)[0, :]

    # Fix breaks in spline numbering
    new_seg = reorder_segments_id(new_seg)

    connect_seg = []
    for i in [int(id) for id in np.unique(segments[:, 0]) if id in np.unique(np.concatenate(idx_connect))]:
        connect_seg.append(segments[np.where(segments[:, 0] == i), :])
    connect_seg = np.hstack(connect_seg)[0, :]

    assert len(new_seg) + len(connect_seg) == len(segments)

    # Connect selected segments pairs
    df_connect_seg = []
    idx = 1000000
    for i in idx_connect:
        for j in i:
            df = np.where(connect_seg[:, 0] == j)[0]
            connect_seg[df, 0] = idx
        idx += 1
    assert len(new_seg) + len(connect_seg) == len(segments), f'{len(new_seg)}, {len(connect_seg)}, {len(segments)}'

    # Fix breaks in spline numbering
    connect_seg = reorder_segments_id(connect_seg,
                                      order_range=[int(np.max(new_seg[:, 0])) + 1,
                                                   int(np.max(new_seg[:, 0])) + 1 +
                                                   len(np.unique(connect_seg[:, 0]))])

    connect_seg_sort = []
    for i in np.unique(connect_seg[:, 0]):
        connect_seg_sort.append(sort_segment(connect_seg[np.where(connect_seg[:, 0] == int(i)), :][0]))
    connect_seg = np.vstack(connect_seg_sort)

    new_seg = np.concatenate((new_seg, connect_seg))

    return new_seg
