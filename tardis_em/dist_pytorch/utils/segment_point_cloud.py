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
    Manages the process of stitching graph patches and their associated
    attributes, performing adjacency operations to create a cohesive graph
    network, and preprocessing connections to ensure consistency and limit
    redundant associations.

    This class is designed for scenarios where graph predictions and corresponding
    node attributes (e.g., coordinates and class probabilities) are generated in patches
    and need to be combined into a unified representation. The provided methods allow
    for detailed control over graph stitching, class and coordinate merging, adjacency
    list construction, and preconditioning of connection data.
    """

    def __init__(self, threshold=0.5, connection=2, smooth=False):
        """
        A class that initializes configuration settings for threshold, connection,
        and smooth properties. The `threshold` represents a critical point for
        operations, `connection` determines the connectivity parameter, and
        `smooth` works as a flag to enable or disable additional processing.

        :param threshold: The threshold value for operations (default: 0.5).
        :type threshold: float
        :param connection: Connection metric; defines allowable connectivity or
            fallback default if the input is not an integer (default: 2).
        :type connection: int
        :param smooth: Enables or disables smooth processing (default: False).
        :type smooth: bool
        """
        self.threshold = threshold
        if isinstance(connection, int):
            self.connection = connection
        else:
            self.connection = 10000000
        self.smooth = smooth

    @staticmethod
    def _stitch_graph(graph_pred: list, idx: list) -> np.ndarray:
        """
        Builds a complete adjacency graph by stitching together smaller subgraphs.

        This method takes predictions for graph structure (`graph_pred`) and their corresponding
        indices (`idx`) and combines them into a complete adjacency matrix. The process involves merging
        graph segments by averaging edge weights when overlapping subgraph edges exist.

        :param graph_pred: List of adjacency matrices representing predicted subgraphs.
        :param idx: List of lists, where each sublist contains indices corresponding to
            the nodes in a subgraph.
        :return: A complete adjacency matrix as a NumPy ndarray.
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
        Stitches a list of coordinate arrays and their corresponding indices into a
        single numpy array, maintaining their positional relationship.

        :param coord: A list where each element is a 2D array-like or PyTorch tensor
            containing coordinates with shape `(n, m)`. If the elements are PyTorch
            tensors, they are converted to numpy arrays.
        :param idx: A list of 1D arrays where each array provides indices corresponding
            to the rows of each coordinate patch in `coord`.
        :return: A numpy array of shape `(max_idx, dim)` where `max_idx` is the maximum
            index value across all elements of `idx` plus one, and `dim` is the second
            dimension of the coordinate arrays.
        :rtype: np.ndarray
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
        Stitches together class prediction arrays based on provided index mappings.

        This method combines multiple predicted class arrays into a single
        comprehensive array using the given index mappings. Input arrays may
        be either PyTorch tensors or NumPy arrays. If the input is in Tensor
        form, it is converted to a NumPy array. The method creates the resulting
        array by initializing an empty array and iteratively populating it based
        on the index mappings and corresponding class values.

        :param cls: A list of class prediction arrays, each array corresponding
            to a partition of the overall data. The arrays may be PyTorch tensors
            or NumPy arrays.
        :param idx: A list of index arrays, where each index array specifies
            the indices in the resulting array corresponding to the elements
            in the matching `cls` array.
        :return: A NumPy array containing combined class data. The resultant
            array has the same size as the range of indices provided via `idx`.
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
        Generates an adjacency list representation for the input graphs based on
        threshold filtering and additional conditions. This method computes
        connections for nodes within the provided graphs, applying filters for
        probabilities and removing self-connections. The function also merges
        duplicate connections, aggregates probabilities, and sorts them in
        descending order where applicable.

        :param graphs: A list of adjacency matrices, where each entry `g[i][j]`
            represents the probability or strength of a connection between node
            `i` and node `j`.
        :type graphs: list

        :param coord: Numpy array containing coordinates or properties of each
            node. The shape of the array must correspond to the number of nodes.
        :type coord: np.ndarray

        :param output_idx: Optional list specifying the indexing for output
            nodes. If `None`, it is inferred based on the provided graphs. The
            length must align with the graph dimensions.
        :type output_idx: Optional[list]

        :param threshold: A float value specifying a filtering threshold for
            connections based on probabilities. Connections with probabilities
            below this value are ignored. Default is 0.5.
        :type threshold: float

        :return: A list of adjacency properties for each node in the input.
            Each item contains a list where the first element is the node index,
            the second is the original coordinate values, the third is the list
            of connected node indices, and the fourth is the list of corresponding
            probabilities. Returns None if the input configuration is invalid.
        :rtype: Optional[list]
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
        Preprocess the adjacency matrix to ensure mutual connections and limit the number of top
        connections for each node based on their probabilities.

        This method processes the given adjacency matrix by first identifying potential top
        connections for each node, ensuring mutual connections exist between nodes, and finally
        limiting the connections to the top N based on their probability values. The updated
        adjacency matrix is returned with these modifications.

        :param adj_matrix: A list of rows, where each row contains information about a node.
                           Each row is represented as a tuple of four elements:
                           - (node index, other metadata, list of connections, list of probabilities).
                           Connections are represented as indices of connected nodes, and probabilities indicate
                           the strength of these connections.
        :return: Modified adjacency matrix with mutual connections and top connections limited
                 to a specific number.
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
        Finds a segment matrix quickly using a depth-first-like approach for traversing
        a graph provided as an adjacency matrix. The method starts with the first node
        that has connections and continues to explore its neighbors, ensuring mutual
        strong connections are met based on a predefined connection limit.

        This algorithm identifies and returns the subset of nodes that are interconnected
        based on the connection rules specified.

        :param adj_matrix: The adjacency matrix represented as a list of tuples, where
            each tuple contains four elements: metadata, metadata, connections (list of
            neighbor indices), and metadata.
        :return: A list of indices of nodes that are non-empty and interconnected.
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
        Find a segment matrix based on connectivity rules in the adjacency matrix.

        The function starts by identifying an initial point from the adjacency matrix
        and iteratively collects associated points following specific connection
        rules. The iteration continues until no new connections are found. The final
        list of associated points respects the constraints defined by the
        `self.connection` attribute.

        :param adj_matrix: The adjacency matrix representing interactions between nodes
            where each element contains a list of connected nodes and their metadata.
        :type adj_matrix: list
        :return: A list of indices representing the nodes that are part of the segment
            matrix and satisfy the given connection rules. Returns an empty list if no
            suitable initial point is found.
        :rtype: list
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
        Smooths segments of coordinates using a spline fitting method.

        The function processes a numpy array of coordinates and applies a smoothing
        spline to segments of the array where adequate points are available. It first
        groups the coordinates by the first column's values. For each unique group, if
        the group contains more than four points, the data is smoothed using a spline
        fitting method; otherwise, the original points are kept intact. The result is
        a concatenation of all smoothed and/or original segments.

        :param coord: A numpy array containing the input coordinates to be processed.
        :return: A numpy array representing the smoothed coordinates after the spline
            fitting process or original values when smoothing criteria are not met.
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
        Converts a patch of graph data into segments based on geometric and adjacency
        relationships within the input data. This function processes spatial graph
        representations and outputs segmented components based on provided criteria
        like pruning and optional visualization.

        :param graph: Input graph representation(s), which can be a numpy array, a Torch
                      tensor, or a list of these types. Represents the connectivity
                      information of the graph data.
        :type graph: list

        :param coord: Coordinate data of the graph nodes in either numpy array or list
                      format. Contains spatial locations of nodes in the graph.
        :type coord: Union[np.ndarray, list]

        :param idx: Index information about the selected components of the graph
                    (e.g., subclusters of a larger graph) to be processed.
        :type idx: list

        :param prune: Minimum threshold for the number of points required in a segment to
                      be considered valid. Segments smaller than this threshold
                      will be ignored.
        :type prune: int

        :param sort: A flag that determines whether the output points in each segment
                     should be sorted geometrically. Defaults to True.
        :type sort: bool

        :param visualize: An optional flag indicating whether the resulting segmented
                          data should be visualized and in what mode (e.g.,
                          point cloud view or filament view). Accepts values
                          like "f" or "p".
        :type visualize: Optional[str]

        :return: Segmented graph components, represented as a numpy array, where each
                 row corresponds to a node and includes attributes like
                 segment ID and spatial coordinates.
        :rtype: np.ndarray
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
