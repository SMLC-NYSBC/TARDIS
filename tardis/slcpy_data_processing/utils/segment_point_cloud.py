from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from typing import Optional
import numpy as np


class GraphInstance:
    """
    INSTANCE GRAPH TO SEGMENTED POINT CLOUD

    Function taking predicted graph and output segmented point cloud. If needed
    output is represented as point cloud in order based on position in 3D.

    Args:
        coord: Coordinates of each point in the same order as graph [Length x Dimension]
        graph: Graph with float(portability) or binary data of point interactions
            of shape [Length x Length]
        order: If True points are ordered in Z
    """

    def __init__(self,
                 max_interactions=2,
                 threshold=0.5):
        self.max_interaction = max_interactions + 1
        self.threshold = threshold

    @staticmethod
    def _stitch_graph(graph_pred: list,
                      idx: list):
        # Build empty graph
        graph = max([max(f) for f in idx]) + 1
        graph = np.zeros((graph, graph),
                         dtype=np.float32)
        graph[:, :] = -1

        for idx_voxal, graph_voxal in zip(idx, graph_pred):
            for k, _ in enumerate(idx_voxal):
                column = graph_voxal[:, k]
                row = graph_voxal[k, :]

                # Column input
                for id, j in enumerate(idx_voxal):
                    if graph[j, idx_voxal[k]] == -1:
                        graph[j, idx_voxal[k]] = row[id]
                    else:
                        graph[j, idx_voxal[k]] = np.mean((row[id],
                                                          graph[j, idx_voxal[k]]))

                # Row input
                for id, j in enumerate(idx_voxal):
                    if graph[idx_voxal[k], j] == -1:
                        graph[idx_voxal[k], j] = column[id]
                    else:
                        graph[idx_voxal[k], j] = np.mean((column[id],
                                                          graph[idx_voxal[k], j]))

        # Remove -1
        graph[np.where(graph == -1)[0], np.where(graph == -1)[1]] = 0

        return graph

    @staticmethod
    def stitch_coord(coord: list,
                     idx: list):
        # Build empty coord array
        dim = coord[0].shape[1]
        coord_df = max([max(f) for f in idx]) + 1
        coord_df = np.zeros((coord_df, dim),
                            dtype=np.float32)
        coord_df[:, :] = -1

        for coord_voxal, idx_voxal in zip(coord, idx):
            for value, id in zip(coord_voxal, idx_voxal):
                coord_df[id, :] = value

        coord_df[np.where(coord_df == -1)[0],
                 np.where(coord_df == -1)[1]] = 0

        return coord_df

    @staticmethod
    def assign_coordinates(segments: np.ndarray,
                           coord: np.ndarray):
        # Add output for decoding point id to X x Y x Z coordinates
        ids = segments[:, 0]
        xyz = []

        for i in segments[:, 1]:
            xyz.append(coord[i, :])

        xyz = np.vstack(xyz)
        for i in range(len(xyz[1])):
            if i == 0:
                segments = np.vstack((ids, xyz[:, 0]))
            else:
                segments = np.vstack((segments, xyz[:, i]))

        return segments.T

    def segment_graph(self,
                      graph: Optional[list] = np.ndarray,
                      coord: Optional[list] = np.ndarray,
                      out_idx: Optional[list] = None):
        if isinstance(graph, list) or isinstance(coord, list):
            graph, coord = self.clean_graph(graph=graph,
                                            coord=coord,
                                            out_idx=out_idx)

        idx = 0  # Set initial number of segment
        point_cloud = []
        ids = []

        while not np.all(graph == -1):
            # Find initial connected point
            segments = []
            id_new_segment = 0
            next_init = True

            # Start of new segment by node != 0
            while next_init:
                next_init = graph[id_new_segment, id_new_segment] == -1
                id_new_segment += 1
            id_new_segment -= 1

            # Get all edges for initial node
            self.max_interaction -= 1
            connection = self.find_point_interaction(graph=graph,
                                                     search_for=id_new_segment)

            for i, id in enumerate(connection[0]):
                if connection[1][i] > self.threshold:
                    segments.append(id)

            segments = list(np.unique(segments))
            segments_init = len(segments)

            # Find all nodes and edges with continuos connection
            end_of_contour = True
            self.max_interaction += 1
            while end_of_contour:
                # Find edges for nodes in the countour list
                for i in segments:
                    connection = self.find_point_interaction(graph=graph,
                                                             search_for=i)

                    for i, id in enumerate(connection[0]):
                        if connection[1][i] > self.threshold:
                            segments.append(id)
                    segments = list(np.unique(segments))

                if len(segments) == segments_init:
                    end_of_contour = False
                segments_init = len(segments)

            for i in segments:
                point_cloud.append(i)
                ids.append(idx)

            self.mask_graph(graph=graph,
                            mask_list=segments)
            idx += 1

        segments = np.vstack((ids, point_cloud)).T

        return self.assign_coordinates(segments=segments, coord=coord)

    def segment_voxals(self,
                       graph_voxal: list,
                       coord_voxal: list):
        """
        Args:
            graph_voxal: List of predicted graph propability for each voxal [Length x Length]
            coord_voxal: List of coord for each voxal [Length x Dim]
        """
        idx = []
        points = []

        # Pick segments from first voxal
        graph_df_p = graph_voxal[0]
        coord_pred = coord_voxal[0]

        segments_p = self.segment_graph(graph=np.array(graph_df_p),
                                        coord=coord_pred)
        for s in segments_p:
            points.append(s[1:])
            idx.append(s[0])

        # Pick segments from first voxal and marge
        last_id = np.max(np.unique(segments_p[:, 0]))
        if len((graph_voxal)) > 1:
            for i in range(1, len(graph_voxal)):
                graph_df_p = graph_voxal[i]
                coord_pred = coord_voxal[i]

                # Pick predicted segments from voxal
                if len(coord_pred) > 1:
                    segments_p = self.segment_graph(graph=np.array(graph_df_p),
                                                    coord=coord_pred)
                else:
                    segments_p = np.hstack(([0], coord_pred[0]))[None, :]

                # Check if any point in segment is found in previous voxals
                for j in np.unique(segments_p[:, 0]):
                    point = segments_p[np.where(segments_p[:, 0] == j)[0], :]

                    match = None
                    for s in point:
                        if np.any((points == s[1:]).all(axis=1)):
                            match = idx[np.where(
                                (points == s[1:]).all(axis=1))[0][0]]
                            break

                    if match is not None:
                        for s in point:
                            points.append(s[1:])
                            idx.append(match)
                    else:
                        last_id += 1
                        for s in point:
                            points.append(s[1:])
                            idx.append(last_id)

        points = np.vstack(points)
        idx = np.reshape(np.asarray(idx), (len(idx), 1))
        segments_p = np.hstack((points, idx))

        segments_p = np.unique(segments_p, axis=0)
        segments_p = segments_p[segments_p[:, 2].argsort()]

        return segments_p


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
        graph[:, :] = -1

        for idx_voxal, graph_voxal in zip(idx, graph_pred):
            for k, _ in enumerate(idx_voxal):
                column = graph_voxal[:, k]
                row = graph_voxal[k, :]

                # Column input
                for id, j in enumerate(idx_voxal):
                    if graph[j, idx_voxal[k]] == -1:
                        graph[j, idx_voxal[k]] = row[id]
                    else:
                        graph[j, idx_voxal[k]] = np.mean((row[id],
                                                          graph[j, idx_voxal[k]]))

                # Row input
                for id, j in enumerate(idx_voxal):
                    if graph[idx_voxal[k], j] == -1:
                        graph[idx_voxal[k], j] = column[id]
                    else:
                        graph[idx_voxal[k], j] = np.mean((column[id],
                                                          graph[idx_voxal[k], j]))

        # Remove -1
        graph[np.where(graph == -1)[0], np.where(graph == -1)[1]] = 0

        return graph

    @staticmethod
    def _stitch_coord(coord: list,
                      idx: list):
        # Build empty coord array
        dim = coord[0].shape[1]
        coord_df = max([max(f) for f in idx]) + 1
        coord_df = np.zeros((coord_df, dim),
                            dtype=np.float32)
        coord_df[:, :] = -1

        for coord_voxal, idx_voxal in zip(coord, idx):
            for value, id in zip(coord_voxal, idx_voxal):
                coord_df[id, :] = value

        coord_df[np.where(coord_df == -1)[0],
                 np.where(coord_df == -1)[1]] = 0

        return coord_df

    def adjacency_list(self,
                       graph: np.ndarray):
        adjacency_list_id = []
        adjacency_list_prop = []
        for column in range(graph.shape[0]):
            """Find all interaction bellow the threshold"""
            prop = [p for p in graph[:, column]
                    if p > self.threshold]
            id = [i for i, p in enumerate(graph[:, column])
                  if p > self.threshold]

            """Check which selected interaction do not have higher prop with other points"""
            idx = [id for id, (i, p) in enumerate(zip(id, prop))
                   if len([x for x in [k for k in graph[:, i]
                                       if k > 0.25]
                           if x > p]) < 3]

            adjacency_list_id.append(list(np.array(id)[idx]))
            adjacency_list_prop.append(list(np.array(prop)[idx]))

        """Remove self connections"""
        for i in range(len(adjacency_list_id)):
            idx = [id for id, v in enumerate(adjacency_list_id[i]) if v != i]

            adjacency_list_id[i] = [j for id, j in enumerate(adjacency_list_id[i])
                                    if id in idx]
            adjacency_list_prop[i] = [j for id, j in enumerate(adjacency_list_prop[i])
                                      if id in idx]

        return adjacency_list_id, adjacency_list_prop

    @staticmethod
    def _find_segment(adj_list: list,
                      adj_prop: list):
        """
        Iterative search mechanism that search for connected points in the
        adjacency list. If the list is longer then 2 (max number of connections),
        iterator sort and pick 2 connections with highest popability.

        Args:
            adj_list: adjacency list of graph connections
            adj_prop: adjacency list of propability of the graph connections
        """
        idx_df = []
        stop_iter = False

        while not stop_iter:
            """Initiate segment search - Find edge point"""
            if len(idx_df) == 0:
                x = 0

                while len(idx_df) == 0:
                    idx_df = adj_list[x]
                    x += 1

                    if x > len(adj_list):
                        break

            past = idx_df
            """Search for all connected points"""
            for i in past:
                """Pick only 2 nodes per edge with highest propability"""
                if len(adj_list[i]) > 2:
                    sort = sorted(zip(adj_prop[i], adj_list[i]),
                                  reverse=True)

                    for i in sort[:2]:
                        idx_df = idx_df + [i[1]]
                else:
                    idx_df = idx_df + adj_list[i]

            idx_df = list(np.unique(idx_df))

            if len(past) == len(idx_df):
                stop_iter = True

        return idx_df

    @staticmethod
    def _sort_segment(coord: np.ndarray,
                      idx: list):
        new_c = []
        new_i = []

        for i in range(len(coord) - 1):
            if i == 0:
                id = [id for id, e in enumerate(idx) if len(e) == 1]
                if len(id) == 0:
                    id = np.where([sum(i) for i in cdist(coord, coord)] ==
                                  max([sum(i) for i in cdist(coord, coord)]))[0]
                new_c.append(coord[id[0]])
                new_i.append(idx[id[0]])
                coord = np.delete(coord, id[0], 0)
                del idx[id[0]]

            kd = KDTree(coord)
            points = kd.query(np.expand_dims(
                new_c[len(new_c) - 1], 0), 1)[1][0][0]

            new_c.append(coord[points])
            new_i.append(idx[points])
            coord = np.delete(coord, points, 0)
            del idx[points]

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
        adjacency_id, adjacency_prop = self.adjacency_list(graph=graph)

        """Find Segments"""
        coord_segment = []
        stop = False
        segment_id = 0
        while not stop:
            idx = self._find_segment(adj_list=adjacency_id,
                                     adj_prop=adjacency_prop)

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
            for i in idx:
                adjacency_id[i] = []

            if sum([sum(i) for i in adjacency_id]) == 0:
                stop = True

        return np.vstack(coord_segment)
