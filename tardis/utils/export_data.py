"""
TARDIS - Transformer And Rapid Dimensionless Instance Segmentation

<module> Utils - Export_Data

New York Structural Biology Center
Simons Machine Learning Center

Robert Kiewisz, Tristan Bepler
MIT License 2021 - 2023
"""
from datetime import datetime

import mrcfile
import numpy as np

from tardis.utils.errors import TardisError
from tardis.version import version


class NumpyToAmira:
    """
    Builder of Amira file from numpy array.
    Support for 3D only! If 2D data, Z dim build with Z=0
    """

    @staticmethod
    def check_3d(coord: np.ndarray):
        """
        Check and correct if needed to 3D

        Args:
            coord (np.ndarray): Coordinate file to check for 3D.

        Returns:
            np.ndarray: The same or converted to 3D coordinates.
        """
        assert coord.ndim == 2, \
            TardisError('132',
                        'tardis/utils/export_data.py',
                        'Numpy array should be of the shape [Length x Dim]')

        # Add dummy Z dimension
        if coord.shape[1] == 3:
            coord = np.hstack((coord, np.zeros((coord.shape[0], 1))))

        return coord

    @staticmethod
    def _build_header(coord: np.ndarray,
                      file_dir: str):
        """
        Standard Amira header builder

        Args:
            coord (np.ndarray): 3D coordinate file.
            file_dir (str): Directory where the file should be saved.
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
            f.write(f'define VERTEX {vertex} \n')
            f.write(f'define EDGE {edge} \n')
            f.write(f'define POINT {point} \n')
            f.write('\n')
            f.write('Parameters { ContentType "HxSpatialGraph" } \n')
            f.write('\n')
            f.write('VERTEX { float[3] VertexCoordinates } @1 \n'
                    'EDGE { int[2] EdgeConnectivity } @2 \n'
                    'EDGE { int NumEdgePoints } @3 \n'
                    'POINT { float[3] EdgePointCoordinates } @4 \n'
                    '\n')

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
                     coord: np.ndarray,
                     file_dir: str):
        """
        Save Amira file

        Args:
            coord (np.ndarray): 3D coordinate file.
            file_dir (str): Directory where the file should be saved.
        """
        coord = self.check_3d(coord=coord)

        self._build_header(coord=coord,
                           file_dir=file_dir)

        segments_idx = int(np.max(coord[:, 0]) + 1)
        vertex_id_1 = -2
        vertex_id_2 = -1

        vertex_coord = ['@1']
        vertex_id = ['@2']
        point_no = ['@3']
        point_coord = ['@4']
        for i in range(segments_idx):
            # Collect segment coord and idx
            segment = coord[np.where(coord[:, 0] == i)[0]][:, 1:]

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

        self._write_to_amira(data=vertex_coord,
                             file_dir=file_dir)
        self._write_to_amira(data=vertex_id,
                             file_dir=file_dir)
        self._write_to_amira(data=point_no,
                             file_dir=file_dir)
        self._write_to_amira(data=point_coord,
                             file_dir=file_dir)


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
