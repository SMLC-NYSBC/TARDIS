from datetime import datetime

import numpy as np
from tardis.version import version


class NumpyToAmira:
    """
    Builder of Amira file from numpy array

        Support for 3D only! If 2D data, 3D dim build with Z=0

    coord: Numpy array [Label x X x Y x Z] or [Label x X x Y] of
        shape [Length x Dim] / [x, (3,4)]
    file_dir: Full directory for saving .am file
    """

    @staticmethod
    def check_3D(coord: np.ndarray):
        assert coord.ndim == 2, 'Numpy array should be of the shape [Length x Dim]'

        # Add dummy Z dimension
        if coord.shape[1] == 3:
            coord = np.hstack((coord, np.zeros((coord.shape[0], 1))))

        return coord

    def _build_header(self,
                      coord: np.ndarray,
                      file_dir: str):
        vertex = int(np.max(coord[:, 0]) + 1) * 2
        edge = int(vertex / 2)
        point = int(coord.shape[0])

        with open(file_dir, 'w') as f:
            f.write('# ASCII Spatial Graph \r\n')
            f.write(
                '# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R) \r\n')
            f.write(f'# tardis-pytorch v{version} \r\n')
            f.write(f'# MIT License * 2021-{datetime.now().year} * Robert Kiewisz & Tristan Bepler \r\n')
            f.write('\r\n')
            f.write(f'define VERTEX {vertex} \r\n')
            f.write(f'define EDGE {edge} \r\n')
            f.write(f'define POINT {point} \r\n')
            f.write('\r\n')
            f.write('Parameters { ContentType "HxSpatialGraph" } \r\n')
            f.write('\r\n')
            f.write('VERTEX { float[3] VertexCoordinates } @1 \r\n'
                    'EDGE { int[2] EdgeConnectivity } @2 \r\n'
                    'EDGE { int NumEdgePoints } @3 \r\n'
                    'POINT { float[3] EdgePointCoordinates } @4 \r\n'
                    '\r\n')

    def _write_to_amira(self,
                        data: list,
                        file_dir: str):
        assert file_dir.endswith('.am'), f'{file_dir} must be and .am file!'

        with open(file_dir, 'a+') as f:
            f.write('\r\n')

            for i in data:
                f.write(f'{i} \r\n')

    def export_amira(self,
                     coord: np.ndarray,
                     file_dir: str):
        coord = self.check_3D(coord=coord)

        self._build_header(coord=coord,
                           file_dir=file_dir)

        segments_idx = int(np.max(coord[:, 0]) + 1)
        vertex_id_1 = -2
        vertex_id_2 = -1

        vertex_coord = []
        vertex_coord.append('@1')
        vertex_id = []
        vertex_id.append('@2')
        point_no = []
        point_no.append('@3')
        point_coord = []
        point_coord.append('@4')
        for i in range(segments_idx):
            # Collect segment coord and idx
            segment = coord[np.where(coord[:, 0] == i)[0]][:, 1:]

            # Get Coord for vertex #1 and vertex #2
            vertex = np.array((segment[0], segment[-1:][0]), dtype=object)

            vertex_coord.append(f'{vertex[0][0]:.15e} {vertex[0][1]:.15e} {vertex[0][2]:.15e}')
            vertex_coord.append(f'{vertex[1][0]:.15e} {vertex[1][1]:.15e} {vertex[1][2]:.15e}')

            # Get Update id number of vertex #1 and #2
            vertex_id_1 += 2
            vertex_id_2 += 2
            vertex_id.append(f'{vertex_id_1} {vertex_id_2}')

            # Get no. of point in edge
            point_no.append(f'{len(segment)}')

            # Get coord of points in edge
            for j in segment:
                point_coord.append(f'{j[0]:.15e} {j[1]:.15e} {j[2]:.15e}')

        self._write_to_amira(data=vertex_coord,
                             file_dir=file_dir)
        self._write_to_amira(data=vertex_id,
                             file_dir=file_dir)
        self._write_to_amira(data=point_no,
                             file_dir=file_dir)
        self._write_to_amira(data=point_coord,
                             file_dir=file_dir)
