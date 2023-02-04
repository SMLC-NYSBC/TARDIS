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
import torch

from tardis.dist_pytorch.utils.visualize import VisualizeFilaments, VisualizePointCloud
from tardis.utils.errors import TardisError
from tardis.utils.spline_metric import smooth_spline, sort_segment


class GraphInstanceV2:
    """
    GRAPH CUT

    Perform graph cut on predicted point cloud graph representation using in-coming
    and out-coming edges probability.

    Args:
        threshold (float): Edge connection threshold.
        connection (int): Max allowed number of connections per node.
        smooth (bool): If True, smooth splines.
    """

    def __init__(self,
                 threshold=float,
                 connection=2,
                 smooth=False):
        self.threshold = threshold
        self.connection = connection
        self.smooth = smooth

    @staticmethod
    def _stitch_graph(graph_pred: list,
                      idx: list) -> np.ndarray:
        """
        Stitcher for graph representation

        Args:
            graph_pred (list): Patches of graph predictions
            idx (list): Idx for each node in patches

        Returns:
             np.ndarray: Stitched graph.
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
                      idx: list) -> np.ndarray:
        """
        Stitcher for coord in patches.

        Args:
            coord (list): Coords in each patch.
            idx (list): Idx for each node in patches.

        Returns:
             np.ndarray: Stitched coordinates.
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
                    idx: list) -> np.ndarray:
        """
        Stitcher for nodes in patches.

        Args:
            cls (list): Predicted class in each patch.
            idx (list): Idx for each node in patches.

        Returns:
             np.ndarray: Stitched classed id's.
        """
        # Conversion to Torch
        if isinstance(cls[0], torch.Tensor):
            cls = [c.cpu().detach().numpy() for c in cls]

        # Build empty coord array
        cls_df = max([max(f) for f in idx]) + 1
        cls_df = np.zeros(cls_df,
                          dtype=np.float32)

        for cls_patch, idx_patch in zip(cls, idx):
            # cls_patch = [np.where(i == 1)[0][0] for i in cls_patch]
            for value, id in zip(cls_patch, idx_patch):
                cls_df[id] = value

        return cls_df

    def adjacency_list(self,
                       graph: np.ndarray) -> Tuple[list, list]:
        """
        # Stitch coord list and graph
        # Get Point ID
        # For each point find ID of 2 interactions with the highest prop
            # For each possible interaction
            # Check for example: if Point ID 0 - 1 and 1 - 0 are the highest prop.
                for  each other:
                # if yes add prop
                # if not, check if there is other pair that has the highest prop.
                    for each other:
                    # If not remove
                    # If yes add

        Args:
            graph (np.ndarray): Stitched graph from DIST.

        Returns:
             list, list: Adjacency list of node id's and corresponding edge probability.
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

                    """
                    Search if the interaction has highest prop for each connected
                    points
                    """
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
                          output_idx: Optional[list] = None) -> list:
        """
        Builder of adjacency matrix from stitched coord and graph voxels
        The output of the adjacency matrix is list containing:
        [id][coord][interactions][interaction probability]

        Args:
            graphs (list): Graph patch output from DIST.
            coord (np.ndarray): Stitched coord output from DIST.
            output_idx (list, None): Index lists from DIST.

        Returns:
            list: Adjacency list of all bind graph connections.
        """
        all_prop = [[id, list(i), [], []] for id, i in enumerate(coord)]

        if output_idx is None:
            output_idx = np.arange(graphs[0].shape[0])

        for g, o in zip(graphs, output_idx):
            df = np.where(g >= self.threshold)

            col = list(df[0])
            row = list(df[1])
            interaction_id = [[c, r] for c, r in zip(col, row) if c != r]
            prop = [g[i[0], i[1]] for i in interaction_id]
            interaction_id = [[o[i[0]], o[i[1]]] for i in interaction_id]

            for i, p in zip(interaction_id, prop):
                all_prop[i[0]][2].append(i[1])
                all_prop[i[0]][3].append(p)

        # Take mean value of all repeated interactions
        for p_id, i in enumerate(all_prop):
            inter = i[2]
            prop = i[3]

            if len(inter) > 1:
                all_prop[p_id][2] = list(np.unique(inter))
                all_prop[p_id][3] = [np.median([x for idx, x in enumerate(prop)
                                                if idx in [id for id, j
                                                           in enumerate(inter)
                                                           if j == k]])
                                     for k in np.unique(inter)]

        # Sort each interaction based on the probability
        # The Highest value are first
        for id, a in enumerate(all_prop):
            # Sort only, if there are more than 1 interaction
            if len(a[2]) > 1:
                # Sort
                prop, inter = zip(*sorted(zip(a[3], a[2]), reverse=True))

                # Replace for sorted value
                all_prop[id][2] = list(inter)
                all_prop[id][3] = list(prop)

        return all_prop

    def _find_segment_matrix(self,
                             adj_matrix: list) -> list:
        """
        Iterative search mechanism that search for connected points in the
        adjacency list.

        Args:
            adj_matrix (list): adjacency matrix of graph connections

        Returns:
            list: List of connected nodes.
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

    @staticmethod
    def _smooth_segments(coord: np.ndarray) -> np.ndarray:
        """
        Smooth spline by fixed factor.

        Args:
            coord (np.ndarray): Splines in array [ID, X, Y, (Z)].

        Returns:
            np.ndarray: Smooth splines.
        """
        splines = []

        # Smooth spline
        for i in np.unique(coord[:, 0]):
            x = coord[np.where(coord[:, 0] == int(i))[0], :]

            if len(x) > 4:
                splines.append(smooth_spline(x))
            else:
                splines.append(x)

        return np.concatenate(splines)

    def patch_to_segment(self,
                         graph: list,
                         coord: np.ndarray,
                         idx: list,
                         prune: int,
                         sort=True,
                         visualize: Optional[str] = None) -> np.ndarray:
        """
        Point cloud instance segmentation from graph representation

        From each point cloud (patch) first build adjacency matrix.
        Matrix is then used to iteratively search for new segments.
        For each initial node algorithm search for 2 edges with the highest prop.
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
            graph (list): Graph patch output from Dist.
            coord (np.ndarray): Coordinates for each unsorted point idx.
            idx (list): idx of point included in the segment.
            prune (int): Remove splines composed of number of node given by prune.
            sort (bool): If True sort output.
            visualize (str, None): If not None, visualize output with open3D.
        """
        """Check data"""
        if isinstance(graph, np.ndarray):
            graph = [graph]
        elif isinstance(graph, torch.Tensor):
            graph = [graph.cpu().detach().numpy()]

        if isinstance(idx, np.ndarray):
            idx = [idx.astype(int)]
        elif isinstance(graph, torch.Tensor):
            idx = [idx.cpu().detach().numpy().astype(int)]
        else:
            idx = [i.astype(int) for i in idx]

        if not isinstance(coord, np.ndarray):
            try:
                coord = self._stitch_coord(coord, idx)
            except:
                TardisError('114',
                            'tardis/dist/utils/segment_point_cloud.py',
                            'Coord must be an array of all nodes! '
                            f'Expected list of ndarrays but got {type(coord)}')

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
                    coord_segment.append(np.stack((np.repeat(segment_id,
                                                             segment.shape[0]),
                                                   segment[:, 0],
                                                   segment[:, 1],
                                                   segment[:, 2])).T)
                elif segment.shape[1] == 2:
                    coord_segment.append(np.stack((np.repeat(segment_id,
                                                             segment.shape[0]),
                                                   segment[:, 0],
                                                   segment[:, 1],
                                                   np.zeros((segment.shape[0], )))).T)
                segment_id += 1

            # Mask point assigned to the instance
            for id in idx:
                adjacency_matrix[id][1],\
                    adjacency_matrix[id][2], \
                    adjacency_matrix[id][3] = [], [], []

            if sum([sum(i[2]) for i in adjacency_matrix]) == 0:
                stop = True

        segments = np.vstack(coord_segment)
        if self.smooth:
            segments = self._smooth_segments(segments)

        if visualize is not None:
            assert visualize in ['f', 'p'], \
                TardisError('124',
                            'tardis/dist/utils/segment_point_cloud.py',
                            'To visualize output use "f" for filament '
                            'or "p" for point cloud!')

            if visualize == 'p':
                VisualizePointCloud(segments, True)
            elif visualize == 'f':
                VisualizeFilaments(segments)

        return segments
