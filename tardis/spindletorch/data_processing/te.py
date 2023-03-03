#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
from typing import Union

import numpy as np


class ImportDataFromAmira:
    """
    LOADER FOR AMIRA SPATIAL GRAPH FILES

    This class read any .am file and if the spatial graph is recognized it is converted
    into a numpy array as (N, 4) with class ids and coordinates for XYZ.
    Also, due to Amira's design, file properties are encoded only in the image file
    therefore in order to properly ready spatial graph, class optionally requires
    amira binary or ASCII image file which contains transformation properties and
    pixel size. If the image file is not included, the spatial graph is returned without
    corrections.

    Args:
        src_am (str): Amira spatial graph directory.
    """

    def __init__(self,
                 src_am: str):
        self.src_am = src_am

        # Read spatial graph
        am = open(src_am, 'r', encoding="iso-8859-1").read(500)
        if 'AmiraMesh 3D ASCII' not in am and '# ASCII Spatial Graph' not in am:
            self.spatial_graph = None
        else:
            self.spatial_graph = open(src_am, "r", encoding="iso-8859-1").read().split(
                "\n")
            self.spatial_graph = [x for x in self.spatial_graph if x != '']

    def __get_segments(self) -> Union[np.ndarray, None]:
        """
        Helper class function to read segment data from amira file.

        Returns:
            np.ndarray: Array (N, 1) indicating a number of points per segment.
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with EDGE { int NumEdgePoints }
        segments = str([word for word in self.spatial_graph if
                        word.startswith('EDGE { int NumEdgePoints }')])

        segment_start = "".join((ch if ch in "0123456789" else " ") for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        try:
            segment_start = int(self.spatial_graph.index("@" + str(segment_start[0]))) + 1
        except ValueError:
            segment_start = int(self.spatial_graph.index("@" + str(
                segment_start[0]) + " ")) + 1

        # Find line define EDGE ... <- number indicate number of segments
        segments = str([word for word in self.spatial_graph if
                        word.startswith('define EDGE')])

        segment_finish = "".join((ch if ch in "0123456789" else " ") for ch in segments)
        segment_finish = [int(i) for i in segment_finish.split()]
        segment_no = int(segment_finish[0])
        segment_finish = segment_start + int(segment_finish[0])

        # Select all lines between @.. (+1) and number of segments
        segments = self.spatial_graph[segment_start:segment_finish]
        segments = [i.split(' ')[0] for i in segments]

        # return an array of number of points belonged to each segment
        segment_list = np.zeros((segment_no, 1), dtype="int")
        segment_list[0:segment_no, 0] = [int(i) for i in segments]

        return segment_list

    def __find_points(self) -> Union[np.ndarray, None]:
        """
        Helper class function to search for points in Amira file.

        Returns:
            np.ndarray: Set of all points.
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with POINT { float[3] EdgePointCoordinates }
        points = str([word for word in self.spatial_graph if
                      word.startswith('POINT { float[3] EdgePointCoordinates }')])

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @... and select last one
        try:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]))) + 1
        except ValueError:
            points_start = int(self.spatial_graph.index("@" + str(
                points_start[1]) + " ")) + 1

        # Find line define POINT ... <- number indicate number of points
        points = str([word for word in self.spatial_graph if
                      word.startswith('define POINT')])

        points_finish = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_finish = [int(i) for i in points_finish.split()][0]
        points_no = points_finish
        points_finish = points_start + points_finish

        # Select all lines between @.. (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        point_list = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(' ')[j] for i in points]
            point_list[0:points_no, j] = [float(i) for i in coord]

        return point_list

    def get_segmented_points(self) -> Union[np.ndarray, None]:
        """
        General class function to retrieve segmented point cloud.

        Returns:
            np.ndarray:  Point cloud as [ID, X, Y, Z].
        """
        if self.spatial_graph is None:
            return None

        points = self.__find_points()
        segments = self.__get_segments()

        segmentation = np.zeros((points.shape[0],))
        id = 0
        idx = 0
        for i in segments:
            segmentation[id:(id + int(i))] = idx

            idx += 1
            id += int(i)

        return np.stack((segmentation, points[:, 0], points[:, 1], points[:, 2])).T

    def get_labels(self) -> Union[dict, None]:
        """
        General class function to read all labels from amira file.

        Returns:
            dict: Dictionary with label IDs
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with EDGE { int NumEdgePoints } associated with all labels
        labels = [word for word in self.spatial_graph if
                  word.startswith('EDGE { int ') and not word.startswith('EDGE { int NumEdgePoints }')]

        # Find line define EDGE ... <- number indicate number of segments
        segment_no = str([word for word in self.spatial_graph if word.startswith('define EDGE')])

        labels_dict = {}
        for i in labels:
            # Find line starting with EDGE { int label }
            label_start = "".join((ch if ch in "0123456789" else " ") for ch in i)
            label_start = [int(i) for i in label_start.split()][-1:]

            # Find in the line directory that starts with @..
            try:
                label_start = int(self.spatial_graph.index("@" + str(label_start[0]))) + 1
            except ValueError:
                label_start = int(self.spatial_graph.index("@" + str(
                    label_start[0]) + " ")) + 1

            label_finish = "".join(
                (ch if ch in "0123456789" else " ") for ch in segment_no)
            label_finish = [int(i) for i in label_finish.split()]

            label_no = int(label_finish[0])
            label_finish = label_start + int(label_finish[0])

            # Select all lines between @.. (+1) and number of segments
            label = self.spatial_graph[label_start:label_finish]
            label = [i.split(' ')[0] for i in label]

            # return an array of number of points belonged to each segment
            label_list = np.zeros((label_no, 1), dtype="int")
            label_list[0:label_no, 0] = [int(i) for i in label]
            label_list = np.where(label_list != 0)[0]

            labels_dict.update({i[11:-5].replace(" ", "").replace("}", ""): label_list})

        return labels_dict
