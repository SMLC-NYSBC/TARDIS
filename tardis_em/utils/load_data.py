#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import re
from os import listdir
from os.path import isfile, join
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import tifffile.tifffile as tif
from tardis_em.utils import MRCHeader, header_struct

try:
    import nd2
except ImportError:
    pass

from numpy import ndarray
from plyfile import PlyData
from sklearn.neighbors import KDTree

from tardis_em.dist_pytorch.utils.utils import (
    RandomDownSampling,
    VoxelDownSampling,
)
from tardis_em.utils.errors import TardisError
from tardis_em.utils.normalization import RescaleNormalize
from tardis_em.utils import SCANNET_COLOR_MAP_20

import importlib.util as lib_utils


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
        src_img (str, optional): Amira binary or ASCII image file directory.
    """

    def __init__(
        self, src_am: str, src_img: Optional[str] = None, src_surf: Optional[str] = None
    ):
        self.src_img = src_img
        self.src_am = src_am
        self.src_surf = src_surf

        self.image = None
        self.pixel_size = None
        self.physical_size, self.transformation = None, None
        self.surface = None

        # Read image and its property if existing
        if self.src_img is not None:
            if not self.src_img[-3:] == ".am":
                TardisError(
                    "130",
                    "tardis_em/utils/load_data.py",
                    f"{self.src_img} Not a .am file...",
                )

            if self.src_img.split("/")[-1:][:-3] != self.src_am.split("/")[-1:][:-20]:
                TardisError(
                    "131",
                    "tardis_em/utils/load_data.py",
                    f"Image file {self.src_img} has wrong extension for {self.src_img}!",
                )

            try:
                # Image file [Z x Y x X]
                (
                    self.image,
                    self.pixel_size,
                    self.physical_size,
                    self.transformation,
                ) = load_am(self.src_img)
            except RuntimeWarning:
                TardisError(
                    "130",
                    "tardis_em/utils/load_data.py",
                    "Directory or input .am image file is not correct..."
                    f"for given dir: {self.src_img}",
                )
        else:
            self.pixel_size = 1

        # Read surface and its property if existing
        if self.src_surf is not None:
            if not self.src_surf[-5:] == ".surf":
                TardisError(
                    "130",
                    "tardis_em/utils/load_data.py",
                    f"{self.src_img} Not an Amira .surf file...",
                )

            try:
                # Surface
                self.surface = load_am_surf(self.src_surf)
            except RuntimeWarning:
                TardisError(
                    "130",
                    "tardis_em/utils/load_data.py",
                    "Directory or input .am image file is not correct..."
                    f"for given dir: {self.src_surf}",
                )
        else:
            self.pixel_size = 1

        # Read spatial graph
        am = ""
        frame = 500
        while "# Data section follows" not in am:
            am = open(src_am, "r", encoding="iso-8859-1").read(frame)
            frame += 100

            if frame == 10000:
                break

        binary = False
        spatial_graph = ""
        if not any(
            [
                True
                for i in ["AmiraMesh 3D ASCII", "# ASCII Spatial Graph"]
                if i not in am
            ]
        ):
            if "AmiraMesh BINARY-LITTLE-ENDIAN 3.0" not in am:
                spatial_graph = None
            else:
                binary = True
        if spatial_graph is not None:
            self.spatial_graph = (
                open(src_am, "r", encoding="iso-8859-1").read().split("\n")
            )
            self.spatial_graph = [x for x in self.spatial_graph if x != ""]

        if binary:
            self.spatial_graph = None
            # self.spatial_graph = self.am_decode(self.spatial_graph)

    def __am_decode(self, am: str) -> str:
        pass

    def __get_segments(self) -> Union[np.ndarray, None]:
        """
        Helper class function to read segment data from amira file.

        Returns:
            np.ndarray: Array (N, 1) indicating a number of points per segment.
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with EDGE { int NumEdgePoints }
        segments = str(
            [
                word
                for word in self.spatial_graph
                if word.startswith("EDGE { int NumEdgePoints }")
            ]
        )

        segment_start = "".join((ch if ch in "0123456789" else " ") for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        try:
            segment_start = (
                int(self.spatial_graph.index("@" + str(segment_start[0]))) + 1
            )
        except ValueError:
            segment_start = (
                int(self.spatial_graph.index("@" + str(segment_start[0]) + " ")) + 1
            )

        # Find line define EDGE ... <- number indicate number of segments
        segments = str(
            [word for word in self.spatial_graph if word.startswith("define EDGE")]
        )

        segment_finish = "".join((ch if ch in "0123456789" else " ") for ch in segments)
        segment_finish = [int(i) for i in segment_finish.split()]
        segment_no = int(segment_finish[0])
        segment_finish = segment_start + int(segment_finish[0])

        # Select all lines between @.. (+1) and number of segments
        segments = self.spatial_graph[segment_start:segment_finish]
        segments = [i.split(" ")[0] for i in segments]

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
        points = str(
            [
                word
                for word in self.spatial_graph
                if word.startswith("POINT { float[3] EdgePointCoordinates }")
            ]
        )

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @... and select last one
        try:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]))) + 1
        except ValueError:
            points_start = (
                int(self.spatial_graph.index("@" + str(points_start[1]) + " ")) + 1
            )

        # Find line define POINT ... <- number indicate number of points
        points = str(
            [word for word in self.spatial_graph if word.startswith("define POINT")]
        )

        points_finish = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_finish = [int(i) for i in points_finish.split()][0]
        points_no = points_finish
        points_finish = points_start + points_finish

        # Select all lines between @.. (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        point_list = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(" ")[j] for i in points]
            point_list[0:points_no, j] = [float(i) for i in coord]

        return point_list

    def __find_vertex(self) -> Union[np.ndarray, None]:
        """
        Helper class function to search for VERTEX in Amira file.

        Returns:
            np.ndarray: Set of all points.
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with POINT { float[3] EdgePointCoordinates }
        points = str(
            [
                word
                for word in self.spatial_graph
                if word.startswith("VERTEX { float[3] VertexCoordinates }")
            ]
        )

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_start = [int(i) for i in points_start.split()]

        # Find line that start with the directory @... and select last one
        try:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]))) + 1
        except ValueError:
            points_start = (
                int(self.spatial_graph.index("@" + str(points_start[1]) + " ")) + 1
            )

        # Find line define POINT ... <- number indicate number of points
        points = str(
            [word for word in self.spatial_graph if word.startswith("define VERTEX")]
        )

        points_finish = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_finish = [int(i) for i in points_finish.split()][0]
        points_no = points_finish
        points_finish = points_start + points_finish

        # Select all lines between @.. (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        point_list = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(" ")[j] for i in points]
            point_list[0:points_no, j] = [float(i) for i in coord]

        return point_list

    def get_points(self) -> Union[np.ndarray, None]:
        """
        General class function to retrieve point cloud.

        Returns:
            np.ndarray: Point cloud as [X, Y, Z] after transformation and
                pixel size correction.
        """
        if self.spatial_graph is None:
            return None

        if self.src_img is None:
            self.transformation = [0, 0, 0]
        points_coord = self.__find_points()

        points_coord[:, 0] = points_coord[:, 0] - self.transformation[0]
        points_coord[:, 1] = points_coord[:, 1] - self.transformation[1]
        points_coord[:, 2] = points_coord[:, 2] - self.transformation[2]

        try:
            coordinate = str(
                [
                    word
                    for word in self.spatial_graph
                    if word.startswith("        Coordinates")
                ]
            ).split(" ")[9][1:-3]

            if coordinate == "nm":
                return points_coord / (self.pixel_size / 10)
        except IndexError:
            pass

        return points_coord / self.pixel_size

    def get_vertex(self) -> Union[np.ndarray, None]:
        """
        General class function to retrieve point cloud.

        Returns:
            np.ndarray: Point cloud as [X, Y, Z] after transformation and
                pixel size correction.
        """
        if self.spatial_graph is None:
            return None

        if self.src_img is None:
            self.transformation = [0, 0, 0]
        points_coord = self.__find_vertex()

        points_coord[:, 0] = points_coord[:, 0] - self.transformation[0]
        points_coord[:, 1] = points_coord[:, 1] - self.transformation[1]
        points_coord[:, 2] = points_coord[:, 2] - self.transformation[2]

        try:
            coordinate = str(
                [
                    word
                    for word in self.spatial_graph
                    if word.startswith("        Coordinates")
                ]
            ).split(" ")[9][1:-3]

            if coordinate == "nm":
                return points_coord / (self.pixel_size / 10)
        except IndexError:
            pass

        return points_coord / self.pixel_size

    def get_segmented_points(self) -> Union[np.ndarray, None]:
        """
        General class function to retrieve segmented point cloud.

        Returns:
            np.ndarray:  Point cloud as [ID, X, Y, Z].
        """
        if self.spatial_graph is None:
            return None

        points = self.get_points()
        segments = self.__get_segments()

        segmentation = np.zeros((points.shape[0],))
        id_ = 0
        idx = 0
        for i in segments:
            segmentation[id_ : (id_ + int(i))] = idx

            idx += 1
            id_ += int(i)

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
        labels = [
            word
            for word in self.spatial_graph
            if word.startswith("EDGE { int ")
            and not word.startswith("EDGE { int NumEdgePoints }")
        ]

        # Find line define EDGE ... <- number indicate number of segments
        segment_no = str(
            [word for word in self.spatial_graph if word.startswith("define EDGE")]
        )

        labels_dict = {}
        for i in labels:
            # Find line starting with EDGE { int label }
            label_start = "".join((ch if ch in "0123456789" else " ") for ch in i)
            label_start = [int(i) for i in label_start.split()][-1:]

            # Find in the line directory that starts with @..
            try:
                label_start = (
                    int(self.spatial_graph.index("@" + str(label_start[0]))) + 1
                )
            except ValueError:
                label_start = (
                    int(self.spatial_graph.index("@" + str(label_start[0]) + " ")) + 1
                )

            label_finish = "".join(
                (ch if ch in "0123456789" else " ") for ch in segment_no
            )
            label_finish = [int(i) for i in label_finish.split()]

            label_no = int(label_finish[0])
            label_finish = label_start + int(label_finish[0])

            # Select all lines between @.. (+1) and number of segments
            label = self.spatial_graph[label_start:label_finish]
            label = [i.split(" ")[0] for i in label]

            # return an array of number of points belonged to each segment
            label_list = np.zeros((label_no, 1), dtype="int")
            label_list[0:label_no, 0] = [int(i) for i in label]
            label_list = np.where(label_list != 0)[0]

            labels_dict.update({i[11:-5].replace(" ", "").replace("}", ""): label_list})

        return labels_dict

    def get_image(self):
        """
        General class function to return image file.

        Returns:
            np.ndarray, float: Image and if available pixel size data.
        """
        return self.image, self.pixel_size

    def get_pixel_size(self) -> float:
        """
        Catch pixel size value from image.

        Returns:
            float: Pixel size.
        """
        return self.pixel_size

    def get_surface(self) -> Union[Tuple, None]:
        return self.surface


def load_tiff(tiff: str) -> Tuple[np.ndarray, float]:
    """
    Function to load any tiff file.

    Args:
        tiff (str): Tiff file directory.

    Returns:
        np.ndarray, float: Image data and unified pixel size.
    """
    if not isfile(tiff):
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f"Indicated .tif  {tiff} file does not exist...",
        )

    return np.array(tif.imread(tiff)), 1.0


def mrc_read_header(mrc: Union[str, bytes, None] = None):
    """
    Helper function to read MRC header.

    Args:
        mrc (str): MRC file directory.

    Returns:
        class: MRC header.
    """
    if isinstance(mrc, str):
        with open(mrc, "rb") as f:
            header = f.read(1024)
    else:
        header = mrc

    return MRCHeader._make(header_struct.unpack(header))


def mrc_write_header(*args) -> bytes:
    header = MRCHeader(*args)
    return header_struct.pack(*list(header))


def mrc_mode(mode: int, amin: int):
    """
    Helper function to decode MRC mode type.

    mode int: MRC mode from mrc header.
    amin int: MRC minimum pixel value.

    Returns:
        np.dtype: Mode as np.dtype.
    """
    dtype_ = {
        0: np.uint8,
        1: np.int16,  # Signed 16-bit integer
        2: np.float32,  # Signed 32-bit real
        3: "2h",  # Complex 16-bit integers
        4: np.complex64,  # Complex 32-bit reals
        6: np.uint16,  # Unassigned int16
        12: np.float16,  # Signed 16-bit half-precision real
        16: "3B",  # RGB values
    }

    if mode == 101:
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            "4 bit .mrc file are not supported. Ask Dev if you need it!",
        )
    if mode == 1024:
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            "Are your trying to load tiff file as mrc?",
        )

    if isinstance(mode, int):
        if mode == 0 and amin >= 0:
            return dtype_[mode]
        elif mode == 0 and amin < 0:
            return np.int8

        if mode in dtype_:
            return dtype_[mode]
        else:
            TardisError(
                "130",
                "tardis_em/utils/load_data.py",
                f"Unknown dtype mode: {str(mode)} and {str(amin)}",
            )
    else:
        if mode in [np.int8, np.uint8]:
            return 0
        for name in dtype_:
            if mode == dtype_[name]:
                return name


def load_am(am_file: str):
    """
    Function to load Amira binary image data.

    Args:
        am_file (str): Amira binary image .am file.

    Returns:
        np.ndarray, float, float, list: Image file as well images parameters.
    """
    if not isfile(am_file):
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f"Indicated .am {am_file} file does not exist...",
        )

    am = open(am_file, "r", encoding="iso-8859-1").read(8000)

    asci = False
    if "AmiraMesh 3D ASCII" in am:
        if "define Lattice" not in am:
            TardisError(
                "130",
                "tardis_em/utils/load_data.py",
                f".am {am_file} file is coordinate file not image!",
            )
        asci = True

    size = [word for word in am.split("\n") if word.startswith("define Lattice ")][0][
        15:
    ].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    # Fix for ET that were trimmed
    #  ET boundary box has wrong size
    bb = str(
        [word for word in am.split("\n") if word.startswith("    BoundingBox")]
    ).split(" ")

    if len(bb) == 0:
        physical_size = np.array((float(bb[6]), float(bb[8]), float(bb[10][:-3])))
        transformation = np.array((0.0, 0.0, 0.0))
    else:
        am = open(am_file, "r", encoding="iso-8859-1").read(20000)
        bb = str(
            [word for word in am.split("\n") if word.startswith("    BoundingBox")]
        ).split(" ")

        physical_size = np.array((float(bb[6]), float(bb[8]), float(bb[10][:-3])))

        transformation = np.array((float(bb[5]), float(bb[7]), float(bb[9])))

    try:
        coordinate = str(
            [word for word in am.split("\n") if word.startswith("        Coordinates")]
        ).split(" ")[9][1:-3]
    except IndexError:
        coordinate = None

    if coordinate == "m":  # Bring meter to angstrom
        pixel_size = ((physical_size[0] - transformation[0]) / (nx - 1)) * 10000000000
    elif coordinate == "nm":  # Bring nm to angstrom
        pixel_size = ((physical_size[0] - transformation[0]) / (nx - 1)) * 10
    else:
        pixel_size = (physical_size[0] - transformation[0]) / (nx - 1)

    physical_size = (physical_size[0] - transformation[0]) / (nx - 1)
    pixel_size = np.round(pixel_size, 3)

    if "Lattice { byte Data }" in am or "Lattice { float Data }" in am:
        if asci:
            img = (
                open("../../rand_sample/T216_grid3b.am", "r", encoding="iso-8859-1")
                .read()
                .split("\n")
            )
            img = [x for x in img if x != ""]
            img = np.asarray(img)
            return img
        else:
            if "Lattice { byte Data }" in am:
                dtype_ = np.uint8
                img = np.fromfile(am_file, dtype=dtype_)
            else:
                dtype_ = np.float32
                offset = str.find(am, "\n@1\n") + 4
                img = np.fromfile(am_file, dtype=np.float32, offset=offset)
    elif "Lattice { sbyte Data }" in am:
        dtype_ = np.int8
        img = np.fromfile(am_file, dtype=dtype_)
        img = img + 128
    else:
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f".am {am_file} not supported .am format!",
        )
        return None

    if dtype_ != np.float32:
        binary_start = str.find(am, "\n@1\n") + 4
        img = img[binary_start:-1]
    if nz == 1:
        if len(img) == ny * nx:
            img = img.reshape((ny, nx))
        else:
            df_img = np.zeros((ny * nx), dtype=dtype_)
            df_img[: len(img)] = img
            img = df_img.reshape((ny, nx))
    else:
        if len(img) == nz * ny * nx:
            img = img.reshape((nz, ny, nx))
            # if dtype_ == np.float32:
            # img = np.flip(img, 0)
        else:
            df_img = np.zeros((nz * ny * nx), dtype=dtype_)
            df_img[: len(img)] = img
            img = df_img.reshape((nz, ny, nx))

    return img, pixel_size, physical_size, transformation


def load_am_surf(surf_file: str, simplify_=None) -> Tuple:
    """

    Args:
        surf_file (str): File directory

    Returns:
        Tuple: List of Materials, bounding box, all vertices, list of triangles
    """
    surf_file = open(surf_file, "r", encoding="iso-8859-1").read()
    assert surf_file.startswith("# HyperSurface"), "Not Amira surface file!"

    # Regular expression to find material names
    material_names = re.findall(
        r"\b(\w+)\s*{",
        re.search(r"Materials\s*{([\s\S]*?)\n\s*Units", surf_file, re.DOTALL).group(1),
    )
    material_names = [i for i in material_names if i not in ["Exterior", "Inside"]]

    # Regular expression to extract GridBox values (numbers after 'GridBox')
    gridbox_search = re.search(r"GridBox\s+([-\d\.eE\s]+)", surf_file)
    gridbox = np.array(gridbox_search.group(1).strip().split(), dtype=float)

    # Regular expression to extract GridSize values (numbers after 'GridSize')
    gridsize_search = re.search(r"GridSize\s+([\d\s]+)", surf_file)
    gridsize = np.array(gridsize_search.group(1).strip().split(), dtype=float)

    vertices_search = (
        re.search(
            r"Vertices\s+\d+\n((?:\s*[-\d\.]+\s+[-\d\.]+\s+[-\d\.]+\n)+)",
            surf_file,
            re.DOTALL,
        )
        .group(1)
        .strip()
    )
    vertices = np.array(
        re.findall(r"([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", vertices_search),
        dtype=float,
    )

    triangles_list = []
    for i in material_names:
        pattern = rf"""\{{\s*InnerRegion\s+{re.escape(i)}.*?Triangles\s+\d+\n((?:\s*\d+\s+\d+\s+\d+\n)+)\}}"""

        triangles_search = re.search(pattern, surf_file, re.DOTALL).group(1).strip()
        triangles_list.append(
            np.array(re.findall(r"(\d+)\s+(\d+)\s+(\d+)", triangles_search), dtype=int)
        )

    vertices_list = []
    for i in range(len(triangles_list)):
        vertices_list.append(
            vertices[np.sort(np.unique(triangles_list[i].flatten())) - 1]
        )

        t_shape = triangles_list[i].shape

        _, triangles_list[i] = np.unique(triangles_list[i], return_inverse=True)
        triangles_list[i] = triangles_list[i].reshape(t_shape)

    if simplify_ is not None:
        try:
            import open3d as o3d

            for id_, (v, t) in enumerate(zip(vertices_list, triangles_list)):
                mesh = o3d.geometry.TriangleMesh()

                mesh.vertices = o3d.utility.Vector3dVector(v)
                mesh.triangles = o3d.utility.Vector3iVector(t)

                voxel_size = (
                    max(mesh.get_max_bound() - mesh.get_min_bound()) / simplify_
                )
                mesh = mesh.simplify_vertex_clustering(
                    voxel_size=voxel_size,
                    contraction=o3d.geometry.SimplificationContraction.Average,
                )
                vertices_list[id_] = np.array(mesh.vertices)
                triangles_list[id_] = np.array(mesh.triangles)

        except ModuleNotFoundError:
            pass
    return material_names, [gridbox, gridsize], vertices_list, triangles_list


def load_mrc_file(mrc: str) -> Union[Tuple[np.ndarray, float], Tuple[None, float]]:
    """
    Function to load MRC 2014 file format.

    Args:
        mrc (str): MRC file directory.

    Returns:
        np.ndarray, float: Image data and pixel size.
    """
    if not isfile(mrc):
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f"Indicated .mrc {mrc} file does not exist...",
        )

    header = mrc_read_header(mrc)
    extended_header = header.next

    pixel_size = round(header.xlen / header.nx, 3)
    dtype = mrc_mode(header.mode, header.amin)
    nz, ny, nx = header.nz, header.ny, header.nx
    bit_len = nz * ny * nx

    # Check for corrupted files
    try:
        if nz == 1:
            image = np.fromfile(mrc, dtype=dtype)[-bit_len:].reshape((ny, nx))
        else:
            image = np.fromfile(mrc, dtype=dtype)[-bit_len:].reshape((nz, ny, nx))
    except ValueError:  # File is corrupted try to load as much as possible
        if nz > 1:
            if mrc.endswith(".rec"):
                header_len = 512
            else:
                header_len = 1024 + extended_header
            image = np.fromfile(mrc, dtype=dtype)[header_len:]

            while bit_len >= len(image):
                nz = nz - 1
                bit_len = nz * ny * nx

            image = image[:bit_len]
            image = image.reshape((nz, ny, nx))
        else:
            image = None

    if image is None:
        return None, 1.0

    # if image.min() < 0 and image.dtype == np.int8:
    #     image = image + 128
    #     image = image.astype(np.uint8)
    #
    # if image.min() < 0 and image.dtype == np.int16:
    #     image = image + 32768
    #     image = image.astype(np.uint16)

    # Detect wrongly saved cryo-EM mrc files
    if nz > int(ny * 2):
        image = image.transpose((1, 0, 2))  # YZX to ZYX
    elif nz > int(nx * 2):
        image = image.transpose((2, 1, 0))  # XYZ to ZYX

    return image, pixel_size


def load_nd2_file(
    nd2_dir: str, channels=True
) -> Union[Tuple[np.ndarray, float], Tuple[None, float]]:
    try:
        nd2.version("nd2")
    except NameError:
        TardisError(
            id_="2",
            py="tardis_em/utils/load_data.py",
            desc="You are trying to load ND2 file. Please install nd2 library or "
            'install tardis with nd2 by: pip install "tardis_em[nd2]"',
        )
        return None, 1.0

    img = nd2.imread(nd2_dir)

    if channels:
        # Search for the smallest dim, which will indicate channels
        dim0 = img[0, 0, ...]
        dim0 = int(np.round(np.mean(dim0) / np.std(dim0), 0))

        dim1 = img[0, 1, ...]
        dim1 = int(np.round(np.mean(dim1) / np.std(dim1), 0))

        dim2 = img[1, 0, ...]
        dim2 = int(np.round(np.mean(dim2) / np.std(dim2), 0))

        dim3 = img[1, 1, ...]
        dim3 = int(np.round(np.mean(dim3) / np.std(dim3), 0))

        if dim0 >= dim2 and dim0 > dim3 and dim1 > dim2 and dim1 >= dim3:
            img = img[0 if dim0 > dim2 else 1, ...]
        else:
            img = img[:, 0 if dim0 > dim1 else 1, ...]

    return img, 1.0


def load_ply_scannet(
    ply: str, downscaling=0, color: Optional[str] = None
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    Function to read .ply files.
    Args:
        ply (str): File directory.
        downscaling (float): Downscaling point cloud by fixing voxel size defaults to 0.1.
        color (str, optional): Optional color feature defaults to None.
    Returns:
        np.ndarray: Label point cloud coordinates and optionally RGB value for
            each point.
    """
    # Load .ply scannet file
    ply = PlyData.read(ply)["vertex"]

    pcd = np.stack(
        (ply["x"], ply["y"], ply["z"], ply["red"], ply["green"], ply["blue"]), axis=-1
    ).astype(np.float32)

    coord = pcd[:, :3]
    label = pcd[:, 3:]

    # Retrieve ScanNet v2 labels after down scaling
    cls_id = np.zeros((len(label), 1))
    get_key_from_value = lambda value: next(
        (key for key, val in SCANNET_COLOR_MAP_20.items() if val == value), None
    )

    for id_, i in enumerate(label):
        cls_id[id_, 0] = get_key_from_value(tuple(i))
    coord = np.hstack((cls_id, coord))

    # Retrieve Node RGB features
    if color is not None:
        ply = PlyData.read(color)["vertex"]

        rgb = np.stack((ply["red"], ply["green"], ply["blue"]), axis=-1).astype(
            np.float32
        )

        if downscaling > 0:
            down_scale = VoxelDownSampling(voxel=downscaling, labels=True)
            coord, rgb = down_scale(coord=coord, rgb=rgb)

        if coord[:, 1:].shape != rgb.shape:
            TardisError(
                "131",
                "tardis_em/utils/load_data.py",
                "RGB shape must be the same as coord! "
                f"But {coord[:, 1:].shape} != {rgb.shape}",
            )
        return coord, rgb
    else:
        # Down scaling point cloud with labels
        if downscaling > 0:
            down_scale = VoxelDownSampling(voxel=downscaling, labels=True)
            coord = down_scale(coord)
        return coord


def load_ply_partnet(ply, downscaling=0) -> np.ndarray:
    """
    Function to read .ply files.
    Args:
        ply (str): File directory.
        downscaling (float): Downscaling point cloud by fixing voxel size.
    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    # Load .ply scannet file
    ply = PlyData.read(ply)["vertex"]

    pcd = np.stack(
        (ply["x"], ply["y"], ply["z"], ply["red"], ply["green"], ply["blue"]), axis=-1
    ).astype(np.float32)

    coord_org = pcd[:, :3]
    label_org = pcd[:, 3:]
    label_uniq = np.unique(label_org, axis=0)

    # Downscaling point cloud with labels
    down_scale = VoxelDownSampling(voxel=downscaling, labels=False)
    if downscaling > 0:
        coord = down_scale(coord_org)
    else:
        coord = coord_org

    label_id = []
    tree = KDTree(coord_org, leaf_size=coord_org.shape[0])
    for i in coord:
        _, match_coord = tree.query(i.reshape(1, -1), k=1)
        match_coord = match_coord[0][0]

        label_id.append(np.where(np.all(label_org[match_coord] == label_uniq, 1))[0][0])

    return np.hstack((np.asarray(label_id)[:, None], coord))


def load_txt_s3dis(
    txt: str, rgb=False, downscaling=0
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to read .txt Stanford 3D instance scene file.

    Args:
        txt (str): File directory.
        rgb (bool): If True return RGB value.
        downscaling (float): Downscaling point cloud by fixing voxel size.

    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    coord = pd.read_csv(txt, sep=" ", on_bad_lines="skip").to_numpy()

    rgb_values = coord[:, 3:]
    coord = coord[:, :3]

    # Downscaling point cloud with labels
    down_scale = VoxelDownSampling(voxel=downscaling, labels=False)
    if downscaling > 0:
        coord = down_scale(coord=coord)

        if rgb:
            coord, rgb_values = down_scale(coord=coord, rgb=rgb_values)

    if rgb:
        return coord, rgb_values
    return coord


def load_s3dis_scene(
    dir_: str, downscaling=0, random_ds=None, rgb=False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to read .txt Stanford 3D instance scene files.

    Args:
        dir_ (str): Folder directory with all instances.
        downscaling (float): Downscaling point cloud by fixing voxel size.
        random_ds (None, float): If not None, indicate ration of point to keep.
        rgb (bool): If True, load rgb value.

    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    dir_list = [x for x in listdir(dir_) if x not in [".DS_Store", "Icon"]]

    # Build S3DIS scene with IDs [ID, X, Y, Z] [R, G, B]
    coord_scene = []
    rgb_scene = []
    id_ = 0
    for i in dir_list:
        if rgb:
            coord_inst, rgb_v = load_txt_s3dis(join(dir_, i), rgb=rgb)
            rgb_scene.append(rgb_v)
        else:
            coord_inst = load_txt_s3dis(join(dir_, i))

        coord_scene.append(
            np.hstack((np.repeat(id_, len(coord_inst)).reshape(-1, 1), coord_inst))
        )

        id_ += 1
    coord = np.concatenate(coord_scene)
    if rgb:
        rgb_v = np.concatenate(rgb_scene) / 255
    else:
        rgb_v = None

    # Down scale scene
    if downscaling > 0 or random_ds is not None:
        if random_ds is not None:
            down_scale = RandomDownSampling(threshold=random_ds, labels=True)
        else:
            down_scale = VoxelDownSampling(voxel=downscaling, labels=True)

        if rgb:
            return down_scale(coord=coord, rgb=rgb_v)
        else:
            return down_scale(coord=coord)


def load_image(
    image: str,
    normalize=False,
    px_=True,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Quick wrapper for loading image data based on the detected file format.

    Args:
        image (str): Image file directory.
        normalize (bool): Rescale histogram to 1% - 99% percentile.
        px_ (bool): Return px if True

    Returns:
        np.ndarray, float: Image array and associated pixel size.
    """
    px = 0.0

    if image.endswith((".tif", ".tiff")):
        image, _ = load_tiff(image)
        px = 1.0
    elif image.endswith((".mrc", ".rec", ".map")):
        image, px = load_mrc_file(image)
    elif image.endswith(".am"):
        image, px, _, _ = load_am(image)
    elif image.endswith(".nd2"):
        library_names = ["nd2"]
        library_installed = False
        for lib in library_names:
            if lib_utils.find_spec(lib) is not None:
                library_installed = True

        assert library_installed, (
            f"ND2 library is not installed." f'Run "pip install np2" to install it!'
        )

        image, px = load_nd2_file(image, channels=True)

    if normalize:
        norm = RescaleNormalize(clip_range=(1, 99))
        image = norm(image)

    if isinstance(image, str):
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f"Indicated .mrc {image} file does not exist...",
        )

    if px_:
        return image, px
    return image
