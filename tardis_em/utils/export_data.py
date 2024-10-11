#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import codecs
import time
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Union

import numpy as np

from tardis_em.utils.errors import TardisError
from tardis_em.utils.load_data import mrc_mode, mrc_write_header, MRCHeader
from tardis_em.analysis.filament_utils import reorder_segments_id
from tardis_em._version import version
import shutil
from io import StringIO
import os


class NumpyToAmira:
    """
    Builder of the Amira file from the numpy array.
    Support for 3D only! If 2D data, Z dim build with Z=0
    """

    def __init__(self, as_point_cloud=False, header: list = None):
        self.as_point_cloud = as_point_cloud

        self.tardis_header = [
            "# ASCII Spatial Graph \n",
            "# TARDIS - Transformer And Rapid Dimensionless "
            "Instance Segmentation (R) \n",
            f"# tardis_em v{version} \r\n",
            f"# MIT License * 2021-{datetime.now().year} * "
            "Robert Kiewisz & Tristan Bepler \n",
            "",
        ]

        if header is not None:
            header = [h + " \n" if not h.endswith("\n") else h for h in header]
            header = ["# " + h if not h.startswith("#") else h for h in header]

            self.tardis_header = self.tardis_header + header

    def check_3d(
        self, coord: Optional[np.ndarray] = List
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Check and correct if needed to 3D

        Args:
            coord (np.ndarray, list): Coordinate file to check for 3D.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The same or converted to 3D coordinates.
        """
        if isinstance(coord, np.ndarray):
            if coord.shape[1] == 3 and not self.as_point_cloud:
                TardisError(
                    "132",
                    "tardis_em/utils/export_data.py",
                    "Numpy 3D array may not have IDs for each point.",
                )

            if coord.shape[1] == 2:
                if self.as_point_cloud:
                    z = np.repeat(0, len(coord)).reshape(-1, 1)
                    coord = np.hstack((coord, z))
                else:
                    TardisError(
                        "132",
                        "tardis_em/utils/export_data.py",
                        "Numpy 2D array may not have IDs for each point.",
                    )

            if coord.shape[1] == 4 and self.as_point_cloud:
                coord = coord[:, 1:]
        else:
            if not isinstance(coord, list) and not isinstance(coord, tuple):
                TardisError(
                    "130",
                    "tardis_em/utils/export_data.py",
                    "Expected list of np.ndarrays!",
                )

            # Add dummy Z dimension
            coord = [
                np.hstack((c, np.zeros((c.shape[0], 1)))) if c.shape[1] == 3 else c
                for c in coord
            ]

            # Fixed ordering
            ordered_coord = []
            last_id = 0
            for c in coord:
                ordered_c = reorder_segments_id(c)  # ID starts from 0
                max_id = len(np.unique(ordered_c[:, 0])) + last_id

                ordered_coord.append(reorder_segments_id(c, [last_id, max_id]))
                last_id = max_id

            return ordered_coord

        if self.as_point_cloud:
            return coord
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
        label = ["LabelGroup"]

        if labels is None:
            return label
        elif isinstance(labels[0], np.ndarray):
            for i in range(len(labels) - 1):
                label.append(f"LabelGroup{i + 2}")
        elif isinstance(labels[0], str):
            label = labels

        return label

    def _build_header(
        self,
        coord: np.ndarray,
        file_dir: str,
        label: Optional[list] = None,
        score: Optional[list[int, list]] = None,
    ):
        """
        Standard Amira header builder

        Args:
            coord (np.ndarray): 3D coordinate file.
            file_dir (str): Directory where the file should be saved.
            label (int): If not 0, indicate the number of labels.
        """
        # Store common data for the header
        if self.as_point_cloud:
            vertex = int(coord.shape[0])
            edge = 0
            point = 0
        else:
            vertex = int(np.max(coord[:, 0]) + 1) * 2  # Number of spline ends
            edge = int(vertex / 2)  # Number of splines
            point = int(coord.shape[0])  # Total number of points

        # Save header
        with codecs.open(file_dir, mode="w", encoding="utf-8") as f:
            for i in self.tardis_header:
                f.write(i)
            f.write("\n")
            f.write(
                f"define VERTEX {vertex} \n"
                f"define EDGE {edge} \n"
                f"define POINT {point} \n"
            )
            f.write("\n")
            f.write("Parameters { \n")
            if not self.as_point_cloud:
                f.write("    SpatialGraphUnitsVertex { \n")
                for i in label:
                    f.write(
                        f"        {i}" + " { \n"
                        "            Unit -1, \n"
                        "            Dimension -1 \n"
                        "        } \n"
                    )
                f.write("    } \n")
                f.write("    SpatialGraphUnitsEdge { \n")
                for i in label:
                    f.write(
                        f"        {i}" + " { \n"
                        "            Unit -1, \n"
                        "            Dimension -1 \n"
                        "        } \n"
                    )
                f.write("    } \n")
                f.write("    SpatialGraphUnitsPoint { \n")
                f.write("    } \n")
                f.write("    Units { \n" '        Coordinates "Å" \n' "    } \n")
                for id_, i in enumerate(label):
                    f.write(
                        f"    {i}" + " { \n"
                        "		Label0" + " { \n"
                        "			Color 1 0.5 0.5, \n"
                        f"          Id {id_ + 1} \n"
                        "     } \n"
                        "        Id 0, \n"
                        "        Color 1 0 0 \n"
                        "    } \n"
                    )
            f.write('	ContentType "HxSpatialGraph" \n' "} \n")
            f.write("\n")
            f.write(
                "VERTEX { float[3] VertexCoordinates } @1 \n"
                "EDGE { int[2] EdgeConnectivity } @2 \n"
                "EDGE { int NumEdgePoints } @3 \n"
                "POINT { float[3] EdgePointCoordinates } @4 \n"
            )

            label_id = 5
            if not self.as_point_cloud:
                for i in label:
                    f.write("VERTEX { int " + f"{i}" + "} " + f"@{label_id} \n")
                    f.write("EDGE { int " + f"{i}" + "} " + f"@{label_id + 1} \n")
                    label_id += 2

            if score is not None:
                for i in range(score[0]):
                    name_ = score[1][i]
                    f.write("EDGE { float " + f"{name_}" + " } " + f"@{label_id} \n")
                    label_id = label_id + 1

            f.write("\n")
            f.write("# Data section follows")

    @staticmethod
    def _write_to_amira(data: list, file_dir: str):
        """
        Recursively write all coordinates point

        Args:
            data (list): List of item's to save recursively.
            file_dir (str): Directory where the file should be saved.
        """
        if not file_dir.endswith(".am"):
            TardisError(
                "133",
                "tardis_em/utils/export_data.py",
                f"{file_dir} must be and .am file!",
            )

        with codecs.open(file_dir, mode="a+", encoding="utf-8") as f:
            f.write("\n")

            for i in data:
                f.write(f"{i} \n")

    def export_amira(
        self,
        file_dir: str,
        coords: Union[tuple, list, np.ndarray] = np.ndarray,
        labels: Union[tuple, list, None] = None,
        scores: Optional[list] = None,
        header: list = None,
    ):
        """
        Save Amira file with all filaments without any labels

        Args:
            file_dir (str): Directory where the file should be saved.
            coords (np.ndarray, tuple): 3D coordinate file.
            labels (tuple, list, None): Labels names.
            scores (list, None): List of confidence scores for each instance.
            header(list): Optional header information.
        """
        coord_list = self.check_3d(coord=coords)

        if not self.as_point_cloud:
            coords = np.concatenate(coord_list)

        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]

            if len(labels) != len(coord_list):
                TardisError(
                    id_="117",
                    py="tardis_em/utils/export_data.py",
                    desc="Number of labels do not mach number of Arrays!"
                    f"Labels: {len(labels)} != Coord: {len(coord_list)}",
                )

        # Build Amira header
        if scores is not None:
            score = len(scores[0])
            self._build_header(
                coord=coords,
                file_dir=file_dir,
                label=self._build_labels(labels),
                score=[score, scores[0]],
            )
        else:
            score = False
            self._build_header(
                coord=coords,
                file_dir=file_dir,
                label=self._build_labels(labels),
                score=None,
            )

        # Save only as a point cloud
        if self.as_point_cloud:
            vertex_coord = ["@1"]
            for c in coords[:, 1:]:
                vertex_coord.append(f"{c[0]:.15e} " f"{c[1]:.15e} " f"{c[2]:.15e}")
            self._write_to_amira(data=vertex_coord, file_dir=file_dir)
        else:
            segments_idx = len(np.unique(coords[:, 0]))
            vertex_id_1 = -2
            vertex_id_2 = -1

            vertex_coord = ["@1"]
            vertex_id = ["@2"]
            point_no = ["@3"]
            point_coord = ["@4"]
            for i in range(segments_idx):
                # Collect segment coord and idx
                segment = coords[np.where(coords[:, 0] == i)[0]][:, 1:]

                # Get Coord for vertex #1 and vertex #2
                vertex = np.array((segment[0], segment[-1:][0]), dtype=object)

                # Append vertex #1 (aka Node #1)
                vertex_coord.append(
                    f"{vertex[0][0]:.15e} "
                    f"{vertex[0][1]:.15e} "
                    f"{vertex[0][2]:.15e}"
                )
                # Append vertex #2 (aka Node #2)
                vertex_coord.append(
                    f"{vertex[1][0]:.15e} "
                    f"{vertex[1][1]:.15e} "
                    f"{vertex[1][2]:.15e}"
                )

                # Get Update id number of vertex #1 and #2
                vertex_id_1 += 2
                vertex_id_2 += 2
                vertex_id.append(f"{vertex_id_1} {vertex_id_2}")

                # Get no. of point in edge
                point_no.append(f"{len(segment)}")

                # Get coord of points in edge
                for j in segment:
                    # Append 3D XYZ coord for point
                    point_coord.append(f"{j[0]:.15e} {j[1]:.15e} {j[2]:.15e}")

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
                vertex_label = [f"@{label_id}"]
                edge_label = [f"@{label_id + 1}"]

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
                start += edge

                self._write_to_amira(data=vertex_label, file_dir=file_dir)
                self._write_to_amira(data=edge_label, file_dir=file_dir)

            if isinstance(score, int) and score:
                for i, s in enumerate(scores[1]):
                    label_id = label_id + i
                    edge_score = [f"@{label_id}"]

                    for j in range(segments_idx):
                        edge_score.append(f"{s[j]:.15e}")

                    self._write_to_amira(data=edge_score, file_dir=file_dir)


def to_mrc(
    data: np.ndarray,
    pixel_size: float,
    file_dir: str,
    org_header: MRCHeader = None,
    label: List = None,
):
    """
    Save MRC image file

    Args:
        data (np.ndarray): Image file.
        pixel_size (float): Image original pixel size.
        file_dir (str): Directory where the file should be saved.
        org_header(MRCHeader): Optional original header
        label(list): Optional costume label for header
    """
    mode = mrc_mode(mode=data.dtype, amin=data.min())
    time_ = time.asctime()

    text_ = f"Saved with TARDIS {version}                        {time_}"
    if label is not None:
        label = "\n ".join(label)
        if len(label) > 800:
            label = label[:800]

        label = text_ + "\n " + label
    else:
        label = text_

    if org_header is not None:
        labels = org_header.labels.split(b"\x00")[0]

        if len(labels) > 0:
            nlabl = org_header.nlabl + 1
            labels = labels + b"\x00" + bytes(label, "utf-8")
        else:
            nlabl = 1
            labels = bytes(label, "utf-8")
    else:
        nlabl = 1
        labels = bytes(label, "utf-8")

    len_ = 800 - len(labels)
    labels = labels + b"\x00" * len_

    if data.ndim == 3:
        dim_ = 3
        zlen, ylen, xlen = np.multiply(data.shape, pixel_size)
        mz, my, mx = data.shape
    else:
        dim_ = 2
        ylen, xlen = np.multiply(data.shape, pixel_size)
        zlen = 1
        my, mx = data.shape
        mz = 1

    header = mrc_write_header(
        data.shape[2] if dim_ == 3 else data.shape[1],  # nx
        data.shape[1] if dim_ == 3 else data.shape[0],  # ny
        data.shape[0] if dim_ == 3 else zlen,  # nz
        mode,  # mrc dtype mode
        0,  # nxstart
        0,  # nystart
        0,  # nzstart
        mx,  # mx
        my,  # my
        mz,  # mz
        xlen,  # xlen
        ylen,  # ylen
        zlen,  # zlen
        90.000,  # alpha
        90.000,  # beta
        90.000,  # gamma
        1,  # mapc
        2,  # mapr
        3,  # maps
        data.min(),  # amin
        data.max(),  # amax
        data.mean(),  # amean
        0,  # ispg, space group 0 means images or stack of images
        0,  # next
        0,  # creatid
        0,  # nint
        0,  # nreal
        0,  # imodStamp
        0,  # imodFlags
        0,  # idtype
        0,  # lens
        0,  # nd1
        0,  # nd2
        0,  # vd1
        0,  # vd2
        0,  # tilt_ox
        0,  # tilt_oy
        0,  # tilt_oz
        0,  # tilt_cx
        0,  # tilt_cy
        0,  # tilt_cz
        0,  # xorg
        0,  # yorg
        0,  # zorg
        b"MAP\x00",  # cmap
        b"DD\x00\x00",  # stamp
        data.std(),  # rms
        nlabl,  # nlabels
        labels,  # labels
    )

    with open(file_dir, "wb") as f:
        # write the header
        f.write(header)

        # write data
        f.write(data.tobytes())


def to_am(data: np.ndarray, pixel_size: float, file_dir: str, header: list = None):
    """
    Save image to binary Amira image file.

    Args:
        data (np.ndarray): Image file.
        pixel_size (float): Image original pixel size.
        file_dir (str): Directory where the file should be saved.
        header(list): Optional header in to form of list(str)
    """
    nz, ny, nx = data.shape
    xLen, yLen, zLen = nx * pixel_size, ny * pixel_size, nz * pixel_size

    if header is not None:
        header = ["# " + h if not h.startswith("#") else h for h in header]

    am = [
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0",
        "# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R)",
        f"# tardis_em-pytorch v{version}",
        f"# MIT License * 2021-{datetime.now().year} * Robert Kiewisz & Tristan Bepler",
    ]
    if header is not None:
        am = am + header
    am = am + [
        header,
        "",
        "",
        f"define Lattice {nx} {ny} {nz}",
        "",
        "Parameters {",
        "    Units {",
        '       Coordinates "Å"',
        "    }",
        '    DataWindow "0.000000 255.000000",',
        f'    Content "{nx}x{ny}x{nz} byte, uniform coordinates",',
        f"    BoundingBox 0 {xLen} 0 {yLen} 0 {zLen},",
        '    CoordType "uniform"',
        "}",
        "",
        "Lattice { byte Data } @1",
        "",
        "# Data section follows",
        "@1",
    ]

    with codecs.open(file_dir, mode="w", encoding="utf-8") as f:
        for i in am:
            f.write(f"{i} \n")
    with codecs.open(file_dir, mode="ab+") as f:
        bytes_data = data.flatten().tobytes()
        f.write(BytesIO(bytes_data).getbuffer())


def to_stl(data: np.ndarray, file_dir: str):
    """
    Save a point cloud as a PLY file.

    Parameters:
        data (np.ndarray): The name of the PLY file to create.
        file_dir (str): Output file location.
    """
    # STL save exception for Arm64 machines
    try:
        import pyvista as pv
    except ImportError:
        return

    def change_first_line_of_file(filename, new_first_line):
        fr = open(filename, "r")
        first_line = fr.readline()
        fr.close()
        first_line_len = len(first_line)

        new_first_line_len = len(new_first_line)
        spaces_num = first_line_len - new_first_line_len
        new_first_line = new_first_line + " " * (spaces_num - 1) + "\n"
        fw = StringIO(new_first_line)
        fr = open(filename, "r+")
        shutil.copyfileobj(fw, fr)
        fr.close()
        fw.close()
        return

    def save_multiblock_stl(multiblock, filename):
        names = multiblock.keys()
        oname, ext = os.path.splitext(filename)
        assert ext == ".stl"

        # each stl file saved (output_filenames)
        ofiles = [f"{oname}_{ii}" + ".stl" for ii in range(len(names))]

        for ii, subpart in enumerate(multiblock):
            subpart.save(ofiles[ii], binary=False)
            change_first_line_of_file(
                ofiles[ii], f"solid {names[ii]}"
            )  # basically changes "solid" to "solid <solid_name>"

        # merge files together
        total_stl = ""
        for fn in ofiles:
            f = open(fn)
            total_stl += f.read()
            f.close()

        # writes total stl file
        with open(oname + ".stl", "w") as f:
            f.write(total_stl)

        # deletes previously written stl files
        for fn in ofiles:
            os.remove(fn)

        return

    cloud = pv.MultiBlock()
    for i in np.unique(data[:, 0]):
        cloud.append(
            pv.PolyData(data[np.where(data[:, 0] == i)[0], 1:]).delaunay_2d(alpha=25),
            f"{i}",
        )

    save_multiblock_stl(cloud, file_dir)
