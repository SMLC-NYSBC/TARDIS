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
    def stitch_graph(graph_pred: list,
                     idx: list):
        graph = 0
        for i, _ in enumerate(idx):
            if int(np.max(idx[i])) > graph:
                graph = np.max(idx[i])

        graph = graph + 1
        graph = np.zeros((graph, graph), dtype=np.float32)

        for i, graph_voxal in enumerate(graph_pred):
            graph_voxal = graph_voxal.cpu().detach().numpy()
            idx_voxal = idx[i]

            for k, _ in enumerate(idx_voxal):
                column = graph_voxal[:, k]
                row = graph_voxal[k, :]

                for id, j in enumerate(idx_voxal):
                    if graph[idx_voxal[k], j] < column[id]:
                        graph[idx_voxal[k], j] = column[id]

                for id, j in enumerate(idx_voxal):
                    if graph[j, idx_voxal[k]] < row[id]:
                        graph[j, idx_voxal[k]] = row[id]

        return graph

    @staticmethod
    def stitch_coord(coord: list,
                     idx: list):
        dim = coord[0].shape[coord[0].dim() - 1]
        coord_df = 0

        for i, _ in enumerate(idx):
            if int(np.max(idx[i])) > coord_df:
                coord_df = np.max(idx[i])

        coord_df = coord_df + 1
        coord_df = np.zeros((coord_df, dim), dtype=np.float32)

        for i, _ in enumerate(coord):
            coord_voxal = coord[i].cpu().detach().numpy()
            idx_voxal = idx[i]

            for id, j in enumerate(idx_voxal):
                coord_df[j, :] = coord_voxal[:, id]

        return coord_df

    def clean_graph(self,
                    graph: np.ndarray,
                    coord: np.ndarray,
                    out_idx: list):
        graph = self.stitch_graph(graph_pred=graph,
                                  idx=out_idx)
        coord = self.stitch_coord(coord=coord,
                                  idx=out_idx)

        empty_idx = []
        for id, _ in enumerate(graph):
            if np.all(graph[:, id] == 0):
                empty_idx.append(id)

        while len(empty_idx) > 0:
            graph = np.delete(graph, empty_idx[0], axis=0)
            coord = np.delete(coord, empty_idx[0], axis=0)
            graph = np.delete(graph, empty_idx[0], axis=1)

            empty_idx = empty_idx[1:]
            empty_idx = [x - 1 for x in empty_idx]

        return graph, coord

    @staticmethod
    def mask_graph(graph: np.ndarray,
                   mask_list: list):
        for i in mask_list:
            graph[i, :] = -1
            graph[:, i] = -1

    def find_point_interaction(self,
                               graph: np.ndarray,
                               search_for: int):
        point = graph[search_for, :]
        connection = [point.argsort()[-self.max_interaction:][::-1],
                      point[point.argsort()[-self.max_interaction:][::-1]]]

        return connection

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
            graph_voxal: List of predicted graph propability for each voxal [Lenght x Length]
            coord_voxal: List of coord for each voxal [Batch x Length x Dim]
        """
        idx = []
        points = []

        # Pick segments from first voxal
        graph_df_p = graph_voxal[0].cpu().detach().numpy()
        coord_pred = coord_voxal[0].cpu().detach().numpy()[0, :]

        segments_p = self.segment_graph(graph=np.array(graph_df_p),
                                        coord=coord_pred)
        for s in segments_p:
            points.append(s[1:])
            idx.append(s[0])

        # Pick segments from first voxal and marge
        last_id = np.max(np.unique(segments_p[:, 0]))
        if len((graph_voxal)) > 1:
            for i in range(1, len(graph_voxal)):
                graph_df_p = graph_voxal[i].cpu().detach().numpy()
                coord_pred = coord_voxal[i].cpu().detach().numpy()[0, :]

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
