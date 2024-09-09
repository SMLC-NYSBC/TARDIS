#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Union

import numpy as np
import torch

from tardis_em.dist_pytorch.utils.utils import VoxelDownSampling
from tardis_em.utils.errors import TardisError
from tardis_em.analysis.filament_utils import smooth_spline, sort_segment


class PropGreedyGraphCut:
    """
    PROBABILITY DRIVEN GREEDY GRAPH CUT

    Perform graph cut on predicted point cloud graph representation using in-coming
    and out-coming edges probability.

    Args:
        threshold (float): Edge connection threshold.
        connection (int): Max allowed number of connections per node.
        smooth (bool): If True, smooth splines.
    """

    def __init__(self, threshold=float, connection=2, smooth=False):
        self.threshold = threshold
        if isinstance(connection, int):
            self.connection = connection
        else:
            self.connection = 10000000
        self.smooth = smooth

    @staticmethod
    def _stitch_graph(graph_pred: list, idx: list) -> np.ndarray:
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
        graph = np.zeros((graph, graph), dtype=np.float32)

        for idx_patch, graph_patch in zip(idx, graph_pred):
            for k, _ in enumerate(idx_patch):
                row = graph_patch[k, :]
                row_v = [
                    (
                        row[id_]
                        if graph[i, idx_patch[k]] == 0
                        else np.mean((graph[i, idx_patch[k]], row[id_]))
                    )
                    for id_, i in enumerate(idx_patch)
                ]

                column = graph_patch[:, k]
                column_v = [
                    (
                        row[id_]
                        if graph[i, idx_patch[k]] == 0
                        else np.mean((graph[i, idx_patch[k]], column[id_]))
                    )
                    for id_, i in enumerate(idx_patch)
                ]

                graph[list(idx_patch), idx_patch[k]] = row_v
                graph[idx_patch[k], list(idx_patch)] = column_v

        return graph

    @staticmethod
    def _stitch_coord(coord: list, idx: list) -> np.ndarray:
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
        coord_df = np.zeros((coord_df, dim), dtype=np.float32)

        for coord_patch, idx_patch in zip(coord, idx):
            for value, id_ in zip(coord_patch, idx_patch):
                coord_df[id_, :] = value

        return coord_df

    @staticmethod
    def _stitch_cls(cls: list, idx: list) -> np.ndarray:
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
        cls_df = np.zeros(cls_df, dtype=np.float32)

        for cls_patch, idx_patch in zip(cls, idx):
            # cls_patch = [np.where(i == 1)[0][0] for i in cls_patch]
            for value, id_ in zip(cls_patch, idx_patch):
                cls_df[id_] = value

        return cls_df

    def _adjacency_list(
        self,
        graphs: list,
        coord: np.ndarray,
        output_idx: Optional[list] = None,
        threshold=0.5,
    ) -> Optional[list]:
        """
        Builder of adjacency matrix from stitched coord and graph voxels
        The output of the adjacency matrix is list containing:
        [id][coord][interactions][interaction probability]

        Args:
            graphs (list[np.ndarray]): Graph patch output from DIST.
            coord (np.ndarray): Stitched coord output from DIST.
            output_idx (list[list], None): Index lists from DIST.

        Returns:
            list: Adjacency list of all bind graph connections.
        """
        if output_idx is None:
            if len(graphs) == 1:
                output_idx = list(range(len(graphs[0])))
            else:
                return None

        all_prop = [
            [idx, coord_i.tolist(), [], []] for idx, coord_i in enumerate(coord)
        ]
        for g, o in zip(graphs, output_idx):
            top_k_indices = np.argsort(g, axis=1)
            top_k_probs = np.take_along_axis(g, top_k_indices, axis=1)
            o = np.array(o)

            for i in range(top_k_indices.shape[0]):
                indices = o[top_k_indices[i]].tolist()
                probs = top_k_probs[i].tolist()

                if threshold:
                    filtered_indices_probs = [
                        (idx, prob)
                        for idx, prob in zip(indices, probs)
                        if prob >= self.threshold
                    ]
                else:
                    filtered_indices_probs = [
                        (idx, prob) for idx, prob in zip(indices, probs) if prob != 0
                    ]

                indices, probs = (
                    zip(*filtered_indices_probs) if filtered_indices_probs else ([], [])
                )

                # Remove self-connection
                if o[i] in indices:
                    idx = indices.index(o[i])
                    indices = list(indices)
                    probs = list(probs)
                    del indices[idx]
                    del probs[idx]

                all_prop[o[i]][2].extend(indices)
                all_prop[o[i]][3].extend(probs)

        # Merge duplicates
        for p in all_prop:
            if p[2]:
                unique_indices, inv = np.unique(p[2], return_inverse=True)
                # Hot-fix for numpy 2.0.0
                if inv.ndim == 2:
                    inverse_index = inv[:, 0]

                p[2] = unique_indices.tolist()
                p[3] = [
                    np.median(np.array(p[3])[inv == i])
                    for i in range(len(unique_indices))
                ]

                # Sort by probability in descending order and apply connection limit
                sorted_indices = np.argsort(p[3])[::-1]
                p[2] = np.array(p[2])[sorted_indices].tolist()
                p[3] = np.array(p[3])[sorted_indices].tolist()

        return all_prop

    def preprocess_connections(self, adj_matrix):
        """
        Preprocess the adjacency matrix to first ensure mutual connections and then
        limit each node to its top 8 connections based on connection probability.
        """
        # Step 1: Identify potential top connections without immediately limiting to top 8
        potential_top_connections = {}
        for idx, (_, _, connections, probabilities) in enumerate(adj_matrix):
            sorted_indices = sorted(
                range(len(probabilities)), key=lambda k: probabilities[k], reverse=True
            )
            potential_top_connections[idx] = {
                connections[i]: probabilities[i] for i in sorted_indices
            }

        # Step 2: Ensure mutual connections
        mutual_connections = {}
        for idx, connections in potential_top_connections.items():
            for conn_idx, prob in connections.items():
                if idx in potential_top_connections.get(conn_idx, {}):
                    mutual_connections.setdefault(idx, {})[conn_idx] = prob
                    mutual_connections.setdefault(conn_idx, {})[idx] = (
                        potential_top_connections[conn_idx][idx]
                    )

        # Step 3: Limit to top 8 mutual connections based on probability
        for idx, connections in mutual_connections.items():
            top_indices = sorted(connections, key=connections.get, reverse=True)[
                : self.connection
            ]
            adj_matrix[idx][2] = top_indices
            adj_matrix[idx][3] = [connections[i] for i in top_indices]

        return adj_matrix

    def _find_segment_matrix_fast(self, adj_matrix: list) -> list:
        """
        Find a segment of connected nodes considering the connection strength and
        limiting each node to a maximum of 8 connections.
        """
        visited = set()
        to_visit = set()

        # Start with the first node that has connections
        for i, (_, _, connections, _) in enumerate(adj_matrix):
            if connections:
                to_visit.add(i)
                break

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue

            visited.add(current)
            _, _, connections, _ = adj_matrix[current]

            for neighbor in connections:
                if neighbor not in visited:
                    neighbor_connections = adj_matrix[neighbor][2]
                    # Check for mutual strong connection
                    if (
                        current in neighbor_connections
                        and len(neighbor_connections) <= self.connection
                    ):
                        to_visit.add(neighbor)

        non_empty_nodes = [i for i in visited if adj_matrix[i][2]]
        return non_empty_nodes

    def _find_segment_matrix(self, adj_matrix: list) -> list:
        """
        !Deprecated!
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
        while len(idx_df) == 1 and x < len(adj_matrix):
            idx_df = adj_matrix[x][2][: self.connection]
            idx_df.append(x)
            x += 1

        # Skip if no suitable initial point found
        if x > len(adj_matrix):
            return []

        x -= 1
        new = new_df = adj_matrix[x][2][: self.connection]
        visited = set(idx_df)

        # Pick all point associated with the initial point
        # Check 1: Check if node is not already on the list
        # Check 2: Check if new interaction show up secondary interaction
        # Check 3: Check if found idx's were not associated to previous instance
        while len(new_df) != 0:
            # check only new elements
            new_df = []
            for i in new:
                # Pick secondary interaction for i
                reverse_int = adj_matrix[i][2][: self.connection]

                for j in reverse_int:
                    if j not in visited and i in adj_matrix[j][2][: self.connection]:
                        new_df.append(j)
                        visited.add(j)

            new = new_df

        idx_df = list(visited)

        idx_df = [i for i in idx_df if adj_matrix[i][2] != []]
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
            x = coord[coord[:, 0] == int(i), :]

            if len(x) > 4:
                splines.append(smooth_spline(x, 2))
            else:
                splines.append(x)

        return np.concatenate(splines)

    def patch_to_segment(
        self,
        graph: list,
        coord: Union[np.ndarray, list],
        idx: list,
        prune: int,
        sort=True,
        visualize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Point cloud instance segmentation from graph representation

        From each point cloud (patch) first build adjacency matrix.
        Matrix is then used to iteratively search for new segments.
        For each initial node algorithm search for 2 edges with the highest prop.
        with given threshold. For each found edges, algorithm check if found
        node creates an edge with previous node within 2 highest probability.
        If not alg. search for new edge that fulfill the statement.

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
        elif isinstance(graph, list):
            if isinstance(graph[0], torch.Tensor):
                graph = [g.cpu().detach().numpy() for g in graph]

        if isinstance(idx, np.ndarray):
            idx = [idx.astype(int)]
        elif isinstance(idx, torch.Tensor):
            idx = [idx.cpu().detach().numpy().astype(int)]
        elif isinstance(idx, list):
            if isinstance(idx[0], torch.Tensor):
                idx = [i.cpu().detach().numpy().astype(int) for i in idx]

        if not isinstance(coord, np.ndarray):
            try:
                coord = self._stitch_coord(coord, idx)
            except:
                TardisError(
                    "114",
                    "tardis_em/dist/utils/segment_point_cloud.py",
                    "Coord must be an array of all nodes! "
                    f"Expected list of ndarrays but got {type(coord)}",
                )

        """Build Adjacency list from graph representation"""
        adjacency_matrix = self._adjacency_list(
            graphs=graph, coord=coord, output_idx=idx
        )
        adjacency_matrix = self.preprocess_connections(adjacency_matrix)

        coord_segment = []
        stop = False
        segment_id = 0

        while not stop:
            idx = self._find_segment_matrix_fast(adjacency_matrix)

            """Select segment longer then 3 points"""
            if len(idx) >= prune:
                # Sort points in segment
                if sort:
                    segment = sort_segment(coord=coord[idx])
                else:
                    segment = coord[idx]

                if segment.shape[1] == 3:
                    coord_segment.append(
                        np.stack(
                            (
                                np.repeat(segment_id, segment.shape[0]),
                                segment[:, 0],
                                segment[:, 1],
                                segment[:, 2],
                            )
                        ).T
                    )
                elif segment.shape[1] == 2:
                    coord_segment.append(
                        np.stack(
                            (
                                np.repeat(segment_id, segment.shape[0]),
                                segment[:, 0],
                                segment[:, 1],
                                np.zeros((segment.shape[0],)),
                            )
                        ).T
                    )
                segment_id += 1

            # Mask point assigned to the instance
            for id_ in idx:
                (
                    adjacency_matrix[id_][1],
                    adjacency_matrix[id_][2],
                    adjacency_matrix[id_][3],
                ) = (
                    [],
                    [],
                    [],
                )

            if sum([1 for i in adjacency_matrix if sum(i[2]) > 0]) == 0:
                stop = True

        segments = np.vstack(coord_segment)

        if self.smooth:
            voxel_ds = VoxelDownSampling(voxel=5, labels=True, KNN=True)
            segments = voxel_ds(segments)
            segments = segments[segments[:, 0].argsort()]
            df_coord = []

            for i in np.unique(segments[:, 0]):
                id_ = i
                idx = np.where(segments[:, 0] == id_)[0]

                if len(idx) > 3:
                    df_coord.append(
                        np.hstack(
                            (
                                np.repeat(id_, len(idx)).reshape(-1, 1),
                                sort_segment(segments[idx, 1:]),
                            )
                        )
                    )
            segments = np.concatenate(df_coord)
            segments = self._smooth_segments(segments)

        if visualize is not None:
            try:
                from tardis_em.utils.visualize_pc import (
                    VisualizeFilaments,
                    VisualizePointCloud,
                )

                if visualize not in ["f", "p"]:
                    TardisError(
                        "124",
                        "tardis_em/dist/utils/segment_point_cloud.py",
                        'To visualize output use "f" for filament '
                        'or "p" for point cloud!',
                    )

                if visualize == "p":
                    VisualizePointCloud(segments, True, None, False)
                elif visualize == "f":
                    VisualizeFilaments(segments, False, False)
            except:
                TardisError(
                    "124",
                    "tardis_em/dist/utils/segment_point_cloud.py",
                    'Please install Open3D library extras with pip install "tardis_em[open3d"',
                )

        return segments
