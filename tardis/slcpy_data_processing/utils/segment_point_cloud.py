import numpy as np
from pandas import array
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira
from typing import Optional


class GraphInstanceV2:
    def __init__(self,
                 threshold=float,
                 connection=2,
                 prune=3):
        self.threshold = threshold
        self.am_build = NumpyToAmira()

        self.prune = prune
        self.connection = connection

    @staticmethod
    def _stitch_graph(graph_pred: list,
                      idx: list):
        # Build empty graph
        graph = max([max(f) for f in idx]) + 1
        graph = np.zeros((graph, graph),
                         dtype=np.float32)

        for idx_voxal, graph_voxal in zip(idx, graph_pred):
            for k, _ in enumerate(idx_voxal):
                row = graph_voxal[k, :]
                row_v = [row[id] if graph[i, idx_voxal[k]] == 0
                         else np.mean((graph[i, idx_voxal[k]], row[id]))
                         for id, i in enumerate(idx_voxal)]

                column = graph_voxal[:, k]
                column_v = [row[id] if graph[i, idx_voxal[k]] == 0
                            else np.mean((graph[i, idx_voxal[k]], column[id]))
                            for id, i in enumerate(idx_voxal)]

                graph[list(idx_voxal), idx_voxal[k]] = row_v
                graph[idx_voxal[k], list(idx_voxal)] = column_v

        return graph

    @staticmethod
    def _stitch_coord(coord: list,
                      idx: list):
        # Build empty coord array
        dim = coord[0].shape[1]
        coord_df = max([max(f) for f in idx]) + 1
        coord_df = np.zeros((coord_df, dim),
                            dtype=np.float32)

        for coord_voxal, idx_voxal in zip(coord, idx):
            for value, id in zip(coord_voxal, idx_voxal):
                coord_df[id, :] = value

        return coord_df

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

        # Current ID edge for which interacion is search
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
            graphs: graph voxal output from GraphFormer
            coord: stitched coord output from GraphFormer or input given to Graphformer
        """
        all_prop = [[id, list(i), [], []] for id, i in enumerate(coord)]

        if output_idx is None:
            output_idx = np.arange(graphs[0].shape[0])

        for g, o in zip(graphs, output_idx):
            df = np.where(g > self.threshold)

            col = list(df[0])
            row = list(df[1])
            int = [[c, r] for c, r in zip(col, row) if c != r]
            prop = [g[i[0], i[1]] for i in int]
            int = [[o[i[0]], o[i[1]]] for i in int]

            for i, p in zip(int, prop):
                if i in all_prop[i[0]][2]:
                    id = [id for id, j in enumerate(
                        all_prop[i[0]][2]) if i == j][0]

                    if all_prop[i[0]][3][id] < p:
                        all_prop[i[0]][2] = [j for idx, j in enumerate(
                            all_prop[i[0]][2]) if id != idx]
                        all_prop[i[0]][2].append(i)

                        all_prop[i[0]][3] = [j for idx, j in enumerate(
                            all_prop[i[0]][3]) if id != idx]
                        all_prop[i[0]][3].append(p)
                else:
                    all_prop[i[0]][2].append(i)
                    all_prop[i[0]][3].append(p)

        for id, a in enumerate(all_prop):
            if len(a[2]) > 0:
                prop, inter = zip(*sorted(zip(a[3], a[2]), reverse=True))

                all_prop[id][2] = [[j for j in i if j != id][0] for i in inter]
                all_prop[id][3] = prop

        return all_prop

    # @staticmethod
    # def _find_segment_list(adj_ids: list):
    #     """
    #     DEPRECIATED

    #     Iterative search mechanism that search for connected points in the
    #     adjacency list.

    #     Args:
    #         adj_ids: adjacency list of graph connections
    #     """
    #     idx_df = []
    #     x = 0

    #     while len(idx_df) == 0:
    #         idx_df = list(adj_ids[x])
    #         x += 1

    #         if x > len(adj_ids):
    #             break

    #     past = list(np.unique(idx_df))
    #     stop_iter = False

    #     new = idx_df
    #
        # while not stop_iter:
        #     [idx_df.extend(p) for p in adj_ids if p[0] in new or p[1] in new]
        #     idx_df = list(np.unique(idx_df))
        #     new = [n for n in idx_df if n not in past]

        #     if len(past) == len(idx_df):
        #         stop_iter = True

        #     past = list(np.unique(idx_df))

        # mask = []
        # for p in idx_df:
        #     mask.extend([id for id, i in enumerate(adj_ids) if p in i])
        # mask = list(np.unique(mask))

        # return idx_df, mask

    @staticmethod
    def _find_segment_matrix(adj_matrix: list):
        idx_df = [0]
        x = 0

        while len(idx_df) == 1:
            idx_df = adj_matrix[x][2][:2]
            idx_df.append(x)
            x += 1

            if x > len(adj_matrix):
                break

        x -= 1
        new = new_df = adj_matrix[x][2]

        while len(new_df) != 0:
            for i in new:
                # check new elements
                reverse_int = adj_matrix[i][2][:2]

                new_df = [j for j in reverse_int if j not in idx_df]

            new = new_df
            idx_df = idx_df + new
        return idx_df

    @ staticmethod
    def _sort_segment(coord: np.ndarray):
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
            points = kd.query(np.expand_dims(
                new_c[len(new_c) - 1], 0), 1)[1][0][0]

            new_c.append(coord[points])
            coord = np.delete(coord, points, 0)
        return np.stack(new_c)

    # def graph_to_segments(self,
    #                       graph: list,
    #                       coord: list,
    #                       idx: list):
    #     """
    #     DEPRECIATED
    #     """
    #     """Stitch voxals"""
    #     if isinstance(graph, list):
    #         graph = self._stitch_graph(graph_pred=graph,
    #                                    idx=idx)
    #     if isinstance(coord, list):
    #         coord = self._stitch_coord(coord=coord,
    #                                    idx=idx)

    #     """Build Adjacency list from graph representation"""
    #     adjacency_id, _ = self.adjacency_list(graph=graph)

    #     """Find Segments"""
    #     coord_segment = []
    #     stop = False
    #     segment_id = 0

    #     while not stop:
    #         idx, mask = self._find_segment_list(adj_ids=adjacency_id)

    #         """Select segment longer then 3 points"""
    #         if len(idx) >= self.prune:
    #             # Sort points in segment
    #             segment = self._sort_segment(coord=coord[idx],
    #                                          idx=[i for id, i in enumerate(adjacency_id) if id in idx])

    #             if segment.shape[1] == 3:
    #                 coord_segment.append(np.stack((np.repeat(segment_id, segment.shape[0]),
    #                                                segment[:, 0],
    #                                                segment[:, 1],
    #                                                segment[:, 2])).T)
    #             elif segment.shape[1] == 2:
    #                 coord_segment.append(np.stack((np.repeat(segment_id, segment.shape[0]),
    #                                                segment[:, 0],
    #                                                segment[:, 1],
    #                                                np.zeros((segment.shape[0], )))).T)
    #             segment_id += 1

    #         # Update adjacency list
    #         adjacency_id = [i for id, i in enumerate(
    #             adjacency_id) if id not in mask]

    #         if sum([sum(i) for i in adjacency_id]) == 0:
    #             stop = True

    #     return np.vstack(coord_segment)

    def voxal_to_segment(self,
                         graph: list,
                         coord: np.ndarray,
                         idx: list):
        """
        SEGMENTER FOR VOXALS

        From each point cloud (voaxl) segmenter first build adjacency matrix. 
        Matrix is then used to iteratively search for new segments.
        For each initial node algorithm search for 2 edges with highest prop. with
        given threshold. For each found edges, algorithm check if found node creates
        an edge with previous node within 2 highest probability. If not alg.
        search for new edge that fulfill the statement.

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
            graph: Graph voxal output from Dist
            coord: Coordinates for each unsorted point idx
            idx: idx of point included in the segment
        """
        """Check data"""
        if not isinstance(graph, list) and isinstance(graph, np.ndarray):
            graph = [graph]

        if not isinstance(idx, list) and isinstance(idx, np.ndarray):
            idx = [idx]

        assert isinstance(coord, np.ndarray), 'Coord must be an array of all nodes!'

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
            if len(idx) >= 2:
                # Sort points in segment
                segment = self._sort_segment(coord=coord[idx])

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

            # Update adjacency list
            for id in idx:
                adjacency_matrix[id][2] = []

            if sum([sum(i[2]) for i in adjacency_matrix]) == 0:
                stop = True

        return np.vstack(coord_segment)
