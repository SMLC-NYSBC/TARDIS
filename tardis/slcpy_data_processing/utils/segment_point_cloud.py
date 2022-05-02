import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira


class GraphInstanceV2:
    def __init__(self,
                 threshold=float):
        self.threshold = threshold
        self.am_build = NumpyToAmira()

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
                row_v = [np.max((graph[i, idx_voxal[k]], row[id]))
                         for id, i in enumerate(idx_voxal)]

                column = graph_voxal[:, k]
                column_v = [np.max((graph[i, idx_voxal[k]], column[id]))
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

            while edges != 2 and prop_id != 5:
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

    @staticmethod
    def _find_segment(adj_ids: list):
        """
        Iterative search mechanism that search for connected points in the
        adjacency list.

        Args:
            adj_ids: adjacency list of graph connections
        """
        idx_df = []
        x = 0

        while len(idx_df) == 0:
            idx_df = list(adj_ids[x])
            x += 1

            if x > len(adj_ids):
                break

        past = list(np.unique(idx_df))
        stop_iter = False

        new = idx_df

        while not stop_iter:
            [idx_df.extend(p) for p in adj_ids if p[0] in new or p[1] in new]
            idx_df = list(np.unique(idx_df))
            new = [n for n in idx_df if n not in past]

            if len(past) == len(idx_df):
                stop_iter = True

            past = list(np.unique(idx_df))

        mask = []
        for p in idx_df:
            mask.extend([id for id, i in enumerate(adj_ids) if p in i])
        mask = list(np.unique(mask))

        return idx_df, mask

    @ staticmethod
    def _sort_segment(coord: np.ndarray,
                      idx: list):
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
                id = [id for id, e in enumerate(idx) if len(e) == 1]

                if len(id) == 0:
                    id = np.where([sum(i) for i in cdist(coord, coord)] == max([sum(i) for i in cdist(coord, coord)]))[0]

                new_c.append(coord[id[0]])
                coord = np.delete(coord, id[0], 0)

            kd = KDTree(coord)
            points = kd.query(np.expand_dims(new_c[len(new_c) - 1], 0), 1)[1][0][0]

            new_c.append(coord[points])
            coord = np.delete(coord, points, 0)

        return np.stack(new_c)

    def graph_to_segments(self,
                          graph: list,
                          coord: list,
                          idx: list):
        """Stitch voxals"""
        if isinstance(graph, list):
            graph = self._stitch_graph(graph_pred=graph,
                                       idx=idx)
        if isinstance(coord, list):
            coord = self._stitch_coord(coord=coord,
                                       idx=idx)

        """Build Adjacency list from graph representation"""
        adjacency_id, _ = self.adjacency_list(graph=graph)

        """Find Segments"""
        coord_segment = []
        stop = False
        segment_id = 0

        while not stop:
            idx, mask = self._find_segment(adj_ids=adjacency_id)

            """Select segment longer then 3 points"""
            if len(idx) > 3:
                # Sort points in segment
                segment = self._sort_segment(coord=coord[idx],
                                             idx=[i for id, i in enumerate(adjacency_id) if id in idx])

                coord_segment.append(np.stack((np.repeat(segment_id, segment.shape[0]),
                                               segment[:, 0],
                                               segment[:, 1],
                                               segment[:, 2])).T)
                segment_id += 1

            # Update adjacency list
            adjacency_id = [i for id, i in enumerate(adjacency_id) if id not in mask]

            if sum([sum(i) for i in adjacency_id]) == 0:
                stop = True

        return np.vstack(coord_segment)
