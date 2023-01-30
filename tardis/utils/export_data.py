#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from datetime import datetime
from typing import List, Optional

import mrcfile
import numpy as np

from tardis.utils.errors import TardisError
from tardis.utils.spline_metric import reorder_segments_id
from tardis.version import version


class NumpyToAmira:
    """
    Builder of Amira file from numpy array.
    Support for 3D only! If 2D data, Z dim build with Z=0
    """

    @staticmethod
    def check_3d(coord: Optional[np.ndarray] = List) -> List[np.ndarray]:
        """
        Check and correct if needed to 3D

        Args:
            coord (np.ndarray, list): Coordinate file to check for 3D.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The same or converted to 3D coordinates.
        """
        if isinstance(coord, np.ndarray):
            assert coord.ndim == 2, \
                TardisError('132',
                            'tardis/utils/export_data.py',
                            'Numpy array may not have IDs for each point.')

            # Add dummy Z dimension
            if coord.shape[1] == 3:
                coord = np.hstack((coord, np.zeros((coord.shape[0], 1))))
        else:
            if not isinstance(coord, list) or not isinstance(coord, tuple):
                TardisError('130',
                            'tardis/utils/export_data.py',
                            'Expected list of np.ndarrays!')

            # Add dummy Z dimension
            coord = [np.hstack((c, np.zeros((c.shape[0], 1))))
                     if c.shape[1] == 3 else c for c in coord]

            # Fixed ordering
            ordered_coord = []
            last_id = 0
            for c in coord:
                ordered_c = reorder_segments_id(c)  # ID starts from 0
                max_id = len(np.unique(ordered_c[:, 0])) + last_id

                ordered_coord.append(reorder_segments_id(c, [last_id, max_id]))
                last_id = max_id

            return ordered_coord
        return [reorder_segments_id(coord)]

    @staticmethod
    def _build_labels(labels: Optional[tuple] = None) -> list:
        """
        Build label list

        Args:
            labels (tuple, None): List of labels.

        Returns:
            list: Set of labels.
        """
        label = ['LabelGroup']

        if labels is None:
            return label
        elif isinstance(labels[0], np.ndarray):
            for i in range(len(labels) - 1):
                label.append(f'LabelGroup{i+2}')
        elif isinstance(labels[0], str):
            label = labels

        return label

    @staticmethod
    def _build_header(coord: np.ndarray,
                      file_dir: str,
                      label: list):
        """
        Standard Amira header builder

        Args:
            coord (np.ndarray): 3D coordinate file.
            file_dir (str): Directory where the file should be saved.
            label (int): If not 0, indicate number of labels.
        """
        # Store common data for the header
        vertex = int(np.max(coord[:, 0]) + 1) * 2
        edge = int(vertex / 2)
        point = int(coord.shape[0])

        # Save header
        with open(file_dir, 'w') as f:
            f.write('# ASCII Spatial Graph \n')
            f.write('# TARDIS - Transformer And Rapid Dimensionless '
                    'Instance Segmentation (R) \n')
            f.write(f'# tardis-pytorch v{version} \r\n')
            f.write(f'# MIT License * 2021-{datetime.now().year} * '
                    'Robert Kiewisz & Tristan Bepler \n')
            f.write('\n')
            f.write(f'define VERTEX {vertex} \n'
                    f'define EDGE {edge} \n'
                    f'define POINT {point} \n')
            f.write('\n')
            f.write('Parameters { \n'
                    '    SpatialGraphUnitsVertex { \n')
            for i in label:
                f.write(f'        {i}' + ' { \n'
                        '            Unit -1, \n'
                        '            Dimension -1 \n'
                        '        } \n')
            f.write('    } \n')
            f.write('    SpatialGraphUnitsEdge { \n')
            for i in label:
                f.write(f'        {i}' + ' { \n'
                        '            Unit -1, \n'
                        '            Dimension -1 \n'
                        '        } \n')
            f.write('    } \n')
            f.write('    Units { \n'
                    '        Coordinates "Å" \n'
                    '    } \n')
            for id, i in enumerate(label):
                f.write(f'    {i}' + ' { \n'
                        '		Label0' + ' { \n'
                        '			Color 1 0.5 0.5, \n'
                        f'          Id {id + 1} \n'
                        '     } \n'
                        '        Id 0,'
                        '        Color 1 0 0'
                        '    } \n')
            f.write('	ContentType "HxSpatialGraph" \n'
                    '} \n')
            f.write('\n')
            f.write('VERTEX { float[3] VertexCoordinates } @1 \n'
                    'EDGE { int[2] EdgeConnectivity } @2 \n'
                    'EDGE { int NumEdgePoints } @3 \n'
                    'POINT { float[3] EdgePointCoordinates } @4 \n')

            label_id = 5
            for i in label:
                f.write('VERTEX { int ' + f'{i}' + '} ' + f'@{label_id} \n')
                f.write('EDGE { int ' + f'{i}' + '} ' + f'@{label_id + 1} \n')
                label_id += 2

            f.write('\n')
            f.write('# Data section follows')

    @staticmethod
    def _write_to_amira(data: list,
                        file_dir: str):
        """
        Recursively write all coordinates point

        Args:
            data (list): List of item's to save recursively.
            file_dir (str): Directory where the file should be saved.
        """
        assert file_dir.endswith('.am'), \
            TardisError('133',
                        'tardis/utils/export_data.py',
                        f'{file_dir} must be and .am file!')

        with open(file_dir, 'a+') as f:
            f.write('\n')

            for i in data:
                f.write(f'{i} \n')

    def export_amira(self,
                     file_dir: str,
                     coords: Optional[list] = np.ndarray,
                     labels: Optional[list] = None):
        """
        Save Amira file with all filaments without any labels

        Args:
            file_dir (str): Directory where the file should be saved.
            coords (np.ndarray, tuple): 3D coordinate file.
            labels: Labels names.
        """
        coord_list = self.check_3d(coord=coords)
        coords = np.concatenate(coord_list)

        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]

            if len(labels) != len(coord_list):
                TardisError(id='117',
                            py='tardis/utils/export_data.py',
                            desc='Number of labels do not mach number of Arrays!')

        # Build Amira header
        self._build_header(coord=coords,
                           file_dir=file_dir,
                           label=self._build_labels(labels))

        segments_idx = len(np.unique(coords[:, 0]))
        vertex_id_1 = -2
        vertex_id_2 = -1

        vertex_coord = ['@1']
        vertex_id = ['@2']
        point_no = ['@3']
        point_coord = ['@4']
        for i in range(segments_idx):
            # Collect segment coord and idx
            segment = coords[np.where(coords[:, 0] == i)[0]][:, 1:]

            # Get Coord for vertex #1 and vertex #2
            vertex = np.array((segment[0], segment[-1:][0]), dtype=object)

            # Append vertex #1 (aka Node #1)
            vertex_coord.append(f'{vertex[0][0]:.15e} '
                                f'{vertex[0][1]:.15e} '
                                f'{vertex[0][2]:.15e}')
            # Append vertex #2 (aka Node #2)
            vertex_coord.append(f'{vertex[1][0]:.15e} '
                                f'{vertex[1][1]:.15e} '
                                f'{vertex[1][2]:.15e}')

            # Get Update id number of vertex #1 and #2
            vertex_id_1 += 2
            vertex_id_2 += 2
            vertex_id.append(f'{vertex_id_1} {vertex_id_2}')

            # Get no. of point in edge
            point_no.append(f'{len(segment)}')

            # Get coord of points in edge
            for j in segment:
                # Append 3D XYZ coord for point
                point_coord.append(f'{j[0]:.15e} {j[1]:.15e} {j[2]:.15e}')

        self._write_to_amira(data=vertex_coord, file_dir=file_dir)
        self._write_to_amira(data=vertex_id, file_dir=file_dir)
        self._write_to_amira(data=point_no, file_dir=file_dir)
        self._write_to_amira(data=point_coord, file_dir=file_dir)

        # Write down all labels
        label_id = 5

        vertex_id = 1
        edge_id = 1

        start = 0
        total_vertex = len(np.unique(coords[:, 0])) * 2
        total_edge = len(np.unique(coords[:, 0]))
        for i in coord_list:
            vertex_label = [f'@{label_id}']
            edge_label = [f'@{label_id + 1}']

            edge = len(np.unique(i[:, 0]))
            vertex = edge * 2
            if start == 0:  # 1 1 1 1 0 0 0 0 0
                vertex_label.extend(list(np.repeat(vertex_id, vertex)))
                vertex_label.extend(list(np.repeat(0, total_vertex - vertex)))

                edge_label.extend(list(np.repeat(edge_id, edge)))
                edge_label.extend(list(np.repeat(0, total_edge - edge)))
            else:  # 0 0 0 0 1 1 1 1 1 1 0 0 0
                vertex_label.extend(list(np.repeat(0, start * 2)))
                vertex_label.extend(list(np.repeat(vertex_id, vertex)))
                fill_up = total_vertex - start * 2 - vertex
                if fill_up > 0:
                    vertex_label.extend(list(np.repeat(0, fill_up)))

                edge_label.extend(list(np.repeat(0, start)))
                edge_label.extend(list(np.repeat(edge_id, edge)))
                fill_up = total_edge - start - edge
                if fill_up > 0:
                    edge_label.extend(list(np.repeat(0, fill_up)))

            label_id += 2
            vertex_id += 1
            edge_id += 1
            start = edge

            self._write_to_amira(data=vertex_label, file_dir=file_dir)
            self._write_to_amira(data=edge_label, file_dir=file_dir)


def to_mrc(data: np.ndarray,
           file_dir: str):
    """
    Save MRC image file

    Args:
        data (np.ndarray): Image file.
        file_dir (str): Directory where the file should be saved.
    """
    try:
        with mrcfile.new(file_dir) as mrc:
            mrc.set_data(data)
    except ValueError:
        with mrcfile.new(file_dir, overwrite=True) as mrc:
            mrc.set_data(data)
