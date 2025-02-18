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
from tardis_em.utils.normalization import RescaleNormalize, MeanStdNormalize
from tardis_em.utils import SCANNET_COLOR_MAP_20

import importlib.util as lib_utils


class ImportDataFromAmira:
    """
    Class for importing and handling data from Amira files.

    Provides functionality to extract spatial graphs, segment data, points, and
    vertices from Amira files. This class also handles image and surface data if
    provided, performing validation for Amira-specific file formats. It is designed
    to assist in loading Amira Mesh 3D models and extracting relevant graphical
    and spatial data for further processing.
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
        This function processes a spatial graph to extract segments and their respective
        number of points. It begins by locating specific markers in the spatial graph
        to determine the range of relevant segments. Using these markers, it isolates
        the lines in the spatial graph that define each segment and calculates the
        number of points corresponding to each segment. The result is returned as a
        NumPy array, where each entry represents the number of points in a segment.
        If no spatial graph is available, it returns None.

        :param self: The current instance of the class containing the spatial graph data.
        :type self: object

        :return: A NumPy array representing the number of points in each segment, or
            None if no spatial graph is provided.
        :rtype: Union[np.ndarray, None]
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

        # Find in the line directory that starts with @...
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

        # Select all lines between @... (+1) and number of segments
        segments = self.spatial_graph[segment_start:segment_finish]
        segments = [i.split(" ")[0] for i in segments]

        # return an array of number of points belonged to each segment
        segment_list = np.zeros((segment_no, 1), dtype="int")
        segment_list[0:segment_no, 0] = [int(i) for i in segments]

        return segment_list

    def __find_points(self) -> Union[np.ndarray, None]:
        """
        Extracts and returns coordinates of points defined in the spatial graph.

        This method processes a `spatial_graph` object (list of strings) to
        identify and extract 3D coordinate points defined within it. The points
        are extracted from specific lines and parsed into a numerical array for
        further usage. If no spatial graph is set, the function returns `None`.

        :return: A 2D NumPy array where each row represents the 3D coordinates
            (x, y, z) of a point in the spatial graph. If `spatial_graph` is
            not provided or points cannot be found, returns `None`.
        :rtype: Union[np.ndarray, None]

        :raises ValueError: When the index for extracting points fails due to
            invalid input format in the `spatial_graph`.
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

        # Find in the line directory that starts with @...
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

        # Select all lines between @... (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        point_list = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(" ")[j] for i in points]
            point_list[0:points_no, j] = [float(i) for i in coord]

        return point_list

    def __find_vertex(self) -> Union[np.ndarray, None]:
        """
        Finds and retrieves the vertex coordinates from the `spatial_graph` attribute. The method
        parses a specific textual structure to locate and extract vertex data. It then processes
        this data into a NumPy array containing coordinate points.

        :raises ValueError: If the expected vertex structure is not properly indexed in the
            spatial_graph or there is an inconsistency in parsing the vertex data.

        :return: NumPy array containing vertex coordinates as floats in pixel space, or None
            if no `spatial_graph` is provided.
        :rtype: Union[np.ndarray, None]
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

        # Find in the line directory that starts with @...
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

        # Select all lines between @... (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        point_list = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(" ")[j] for i in points]
            point_list[0:points_no, j] = [float(i) for i in coord]

        return point_list

    def get_points(self) -> Union[np.ndarray, None]:
        """
        Computes and returns the transformed points' coordinates based on the
        spatial graph and image properties.

        If the `spatial_graph` is None, it returns None. The method calculates
        the transformed coordinates of points, adjusting for spatial shifts and
        applying scaling based on the pixel size or other specified factors.

        :raises IndexError: Raised if the extraction of 'Coordinate' information
            from the `spatial_graph` fails due to lack of expected content or
            format.
        :rtype: Union[numpy.ndarray, None]
        :return: A numpy array of transformed point coordinates or None if the
            `spatial_graph` attribute is unset.
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
        Computes and returns the coordinates of a vertex in the spatial graph with optional
        coordinate transformations applied. This function handles specific conditions related
        to the presence of `spatial_graph` and `src_img` attributes and applies transformations and
        scaling to the identified vertex coordinates.

        :return:
            `numpy.ndarray` containing the transformed and scaled vertex coordinates,
            or `None` in case the `spatial_graph` attribute is None.

            If the coordinate unit retrieved from the spatial graph is 'nm', then the
            vertex coordinates are scaled differently based on the `pixel_size` value.
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
        Generates segmented points based on the spatial graph and the computed
        segments. The segmentation assigns an index to each point in the graph,
        indicating which segment it belongs to.

        :return: A numpy array of shape (N, 4) where each row represents a
            point with its corresponding segment index as the first element.
            The columns correspond to:
            [segment_index, x_coordinate, y_coordinate, z_coordinate].
            Returns None if the spatial_graph is None.
        :rtype: Union[numpy.ndarray, None]
        """
        if self.spatial_graph is None:
            return None

        points = self.get_points()
        segments = self.__get_segments()

        segmentation = np.zeros((points.shape[0],))
        id_ = int(0)
        idx = int(0)

        for i in segments:
            i = int(i) if not isinstance(i, np.ndarray) else int(i.item())
            segmentation[id_ : (id_ + int(i))] = idx

            idx += 1
            id_ += int(i)

        return np.stack((segmentation, points[:, 0], points[:, 1], points[:, 2])).T

    def get_labels(self) -> Union[dict, None]:
        """
        Determines and returns a dictionary representing labels and their corresponding
        segment points from the spatial graph, if available. The method first identifies
        label lines and segment definitions within the `spatial_graph`. It calculates
        ranges for each label and extracts associated data points. If no `spatial_graph`
        exists, it will return None.

        :raises ValueError: If a label identifier is not found in the `spatial_graph`.

        :return: A dictionary mapping label names to points belonging to each segment
            or None if `spatial_graph` is not defined.
        :rtype: dict or None
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

            # Find in the line directory that starts with @...
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

            # Select all lines between @... (+1) and number of segments
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
        Retrieves the image and its corresponding pixel size.

        :return: A tuple where the first element is the image and the second
            element is the pixel size.
        :rtype: tuple
        """
        return self.image, self.pixel_size

    def get_pixel_size(self) -> float:
        """
        Retrieves the size of a single pixel.

        :return: The size of the pixel.
        :rtype: float
        """
        return self.pixel_size

    def get_surface(self) -> Union[Tuple, None]:
        """
        Provides functionality to return the surface attribute of an object.

        :return: Either the surface object or None if unavailable
        :rtype: Union[Tuple, None]
        """
        return self.surface


def load_tiff(tiff: str) -> Tuple[np.ndarray, float]:
    """
    Load a TIFF file and return its data as a NumPy array along with an intensity scale
    factor.

    :param tiff: The path to the TIFF file to be loaded.
    :type tiff: str

    :return: A tuple containing the NumPy array representation of the TIFF file and a
        float representing the intensity scale factor.
    :rtype: Tuple[np.ndarray, float]

    :raises TardisError: If the specified TIFF file does not exist.
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
    Reads the header of an MRC file and returns it as a named tuple. This function
    supports input as a file path or raw bytes. If a string is provided, the
    function will open the file in binary mode and read the first 1024 bytes as
    the header. If raw bytes are passed directly, it processes them as the
    header.

    :param mrc: The MRC file path as a string, raw header bytes, or None.

    :return: Named tuple representing the parsed header of the MRC file.
    """
    if isinstance(mrc, str):
        with open(mrc, "rb") as f:
            header = f.read(1024)
    else:
        header = mrc

    return MRCHeader._make(header_struct.unpack(header))


def mrc_write_header(*args) -> bytes:
    """
    Constructs an MRC file header.

    :param args: The arguments required to initialize an `MRCHeader` object.

    :return: Packed binary data representing the MRC header.
    """
    header = MRCHeader(*args)
    return header_struct.pack(*list(header))


def mrc_mode(mode: int, amin: int):
    """
    Determines and returns the appropriate data type or mode based on the
    provided mode and amin values. The function maps image modes to their respective
    data types, handling specific cases for input mode and amin values and also validates
    if the mode corresponds to known types.

    :param mode: The mode to evaluate. It can be either an integer or a specific
        type matching one of the predefined dtype values.
    :type mode: int
    :param amin: Minimum amplitude or intensity value used to refine the data type
        determination for the provided mode. Applicable when the mode is 0.
    :type amin: int

    :return: Returns the corresponding data type or mode index based on the
        provided parameters. For integer mode inputs, it will return the dtype
        associated with the mode or an error will be raised for unsupported modes.
        For non-integer type matching, it will return the corresponding mode index.
    :rtype: Union[numpy.dtype, str, int]
    """
    dtype_m = {
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
            return dtype_m[mode]
        elif mode == 0 and amin < 0:
            return np.int8

        if mode in dtype_m:
            return dtype_m[mode]
        else:
            TardisError(
                "130",
                "tardis_em/utils/load_data.py",
                f"Unknown dtype mode: {str(mode)} and {str(amin)}",
            )
    else:
        if mode in [np.int8, np.uint8]:
            return 0
        for name in dtype_m:
            if mode == dtype_m[name]:
                return name


def load_am(am_file: str):
    """
    Loads data from an AmiraMesh (.am) 3D image file and extracts image,
    pixel size, physical dimensions, and transformation details.

    :param am_file: File path of the .am file to be loaded.
    :type am_file: str
    :return: A tuple containing:
        - Numpy array of the image.
        - Pixel size in angstrom units.
        - Physical size of the image.
        - Transformation (offsets) in the bounding box.
    :rtype: tuple[np.ndarray, float, float, np.ndarray]

    :raises TardisError: If the file does not exist, is of unsupported format,
        or contains missing or invalid metadata.
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
        img = img.astype(np.uint8)

        img += 128
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


def load_am_surf(surf_file: str, simplify_f=None) -> Tuple:
    """
    Parses an Amira surface file and extracts material names, grid properties, vertex data,
    and triangle data. It also optionally simplifies the geometry using Open3D.

    This function reads the content of an Amira surface file, retrieves Material names,
    GridBox, GridSize, Vertices, and Triangle data. It organizes these data into structured
    formats and optionally simplifies vertex and triangle meshes if the `simplify_` argument
    is supplied. It is designed to assist in reading and processing surface geometry data
    from Amira format.

    :param surf_file: Path to the Amira surface file.
    :type surf_file: str
    :param simplify_f: Level of simplification to apply to the mesh geometry. If None, no
                      simplification is performed.
    :type simplify_f: Optional[int]

    :return: A 4-tuple containing:
             - A list of material names extracted from the file.
             - A list comprising GridBox and GridSize arrays.
             - A list of vertex arrays for each material's geometry.
             - A list of triangle arrays for each material's geometry.
    :rtype: Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
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

    if simplify_f is not None:
        try:
            import open3d as o3d

            for id_, (v, t) in enumerate(zip(vertices_list, triangles_list)):
                mesh = o3d.geometry.TriangleMesh()

                mesh.vertices = o3d.utility.Vector3dVector(v)
                mesh.triangles = o3d.utility.Vector3iVector(t)

                voxel_size = (
                    max(mesh.get_max_bound() - mesh.get_min_bound()) / simplify_f
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
    Loads and processes an .mrc file to extract image data and pixel size. The function checks for
    the existence of the .mrc file, reads its header, and computes the appropriate pixel size based
    on dimensions in the header. It attempts to load the image data, ensures file integrity by
    mitigating corrupted file instances, and performs dimensional reshaping where necessary.

    :param mrc: The file path to the .mrc file to be loaded.
    :type mrc: str

    :return: A tuple containing the loaded image data and the pixel size (in Angstroms).
             If the file is corrupted and no valid image data can be retrieved, returns
             a tuple where the image data is None and pixel size is set to 1.0.
    :rtype: Tuple[np.ndarray | None, float]
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

    # Detect wrongly saved cryo-EM mrc files
    if nz > int(ny * 2):
        image = image.transpose((1, 0, 2))  # YZX to ZYX
    elif nz > int(nx * 2):
        image = image.transpose((2, 1, 0))  # XYZ to ZYX

    return image, pixel_size


def load_nd2_file(nd2_dir: str) -> Tuple[np.ndarray, float]:
    """
    Loads an ND2 file and processes the image into a specific format. This function
    is designed to read images from ND2 files, which may include movies or still
    images. It checks the dimensionality of the image and applies transformations
    to standardize the format before calculating certain statistical criteria to
    sort them. The result is the sorted image array and a scaling factor.

    :param nd2_dir: Path to the ND2 file to load.
    :type nd2_dir: str
    :return: A tuple consisting of the processed image data as a numpy array and
        the scaling factor (float).
    :rtype: Tuple[np.ndarray, float]
    """
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
    if img.ndim == 5:
        movie_b = True
    else:
        movie_b = False

    if movie_b:
        img = np.transpose(img, (2, 1, 0, 3, 4))
    else:
        img = np.transpose(img, (1, 0, 2, 3))
        img = img[:, :, np.newaxis, ...]

    criteria = [np.mean(i) / np.std(i) for i in img[:, 0, 0, ...]]
    sorted_indices = np.argsort(criteria)[::-1]
    img = img[sorted_indices, ...]

    return img, 1.0


def load_ply_scannet(
    ply: str, downscaling=0, color: Optional[str] = None
) -> Union[Tuple[ndarray, ndarray], ndarray]:
    """
    Load and process a .ply file of the ScanNet dataset. This function reads the input
    .ply file, extracts point cloud data, and optionally loads RGB features or applies
    downscaling. It also maps ScanNet v2 labels to corresponding classes, if applicable.

    :param ply: Path to the .ply file containing point cloud data.
    :type ply: str
    :param downscaling: Voxel size for downscaling the point cloud. If set to 0, no
        downscaling is applied.
    :type downscaling: int, optional
    :param color:  path to a secondary .ply file containing RGB features
        for the point cloud.
    :type color: str, optional
    :return: Either a tuple containing the downsampled point cloud coordinates and RGB
        features, or the downsampled point cloud coordinates only, depending on whether
        a color file was provided.
    :rtype: Union[Tuple[ndarray, ndarray], ndarray]
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
    Loads a .ply file in the PartNet format and processes its point cloud by extracting coordinates and color information.
    Optionally performs downscaling of the point cloud and assigns unique labels to points based on their colors.

    :param ply: The path or file-like object referring to the .ply file to be loaded.
    :type ply: str or file-like
    :param downscaling: The voxel size used for downscaling the point cloud. If 0, no downscaling is applied.
    :type downscaling: int
    :return: A NumPy array containing the processed point cloud. The array consists of assigned label IDs and the
        downscaled or original coordinates.
    :rtype: np.ndarray
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
    Loads a point cloud dataset in `.txt` format and optionally applies
    downscaling and extracts RGB color information if present.

    The function interprets the `.txt` file specified by the ``txt`` parameter
    as a space-separated file containing point cloud data. The first three
    columns are assumed to be the X, Y, and Z coordinates. Any additional
    columns are interpreted as RGB values. Users can optionally apply voxel-based
    downscaling to reduce the resolution of the point cloud.

    The function returns either the coordinates of the point cloud or a tuple
    containing the coordinates and their corresponding RGB values, depending on
    the value of the ``rgb`` argument.

    :param txt: Path to the `.txt` file containing the point cloud data.
    :type txt: str
    :param rgb: A flag indicating whether the RGB values should be extracted.
                 If set to True, the RGB values, if available, are returned along
                 with the coordinates. Defaults to False.
    :type rgb: bool
    :param downscaling: Voxel size for downscaling the point cloud. If set to a
                        value greater than 0, the point cloud will be downsampled
                        to reduce resolution using a voxel-based approach.
                        Defaults to 0 (no downscaling).
    :type downscaling: int
    :return: If ``rgb`` is True, returns a tuple of two numpy arrays:
             the first array containing the downscaled coordinates and the
             second array containing RGB values. If ``rgb`` is False, returns
             a single numpy array containing only the downscaled coordinates.
    :rtype: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
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
    dir_s: str, downscaling=0, random_ds=None, rgb=False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Loads and processes a Stanford Large-Scale 3D Indoor Spaces (S3DIS) scene from a specified
    directory. This function creates a scene containing 3D spatial coordinates and optionally
    RGB color data. It also allows for downscaling of the scene, either with a fixed downscaling
    factor or a random downsampling threshold. The output format depends on whether RGB data
    is included and whether any downscaling is applied.

    :param dir_s: Directory containing the S3DIS scene files.
    :type dir_s: str
    :param downscaling: Downscaling factor to reduce the scene resolution by voxelizing.
        Default is 0, meaning no downscaling is applied.
    :type downscaling: int, optional
    :param random_ds: Random downsampling threshold. Overrides ``downscaling`` if provided.
        Default is None, meaning no random downsampling is applied.
    :type random_ds: float, optional
    :param rgb: Boolean indicating whether to include RGB color data in the output.
        If True, extracts [R, G, B] values along with spatial coordinates. Defaults to False.
    :type rgb: bool, optional
    :return: If ``rgb`` is True, returns a tuple containing the processed coordinate array
        and normalized RGB data as numpy arrays. If ``rgb`` is False, returns only the
        processed coordinate array. If downscaling or random downsampling is applied, the
        data is downscaled accordingly.
    :rtype: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
    """
    dir_list = [x for x in listdir(dir_s) if x not in [".DS_Store", "Icon"]]

    # Build S3DIS scene with IDs [ID, X, Y, Z] [R, G, B]
    coord_scene = []
    rgb_scene = []
    id_ = 0
    for i in dir_list:
        if rgb:
            coord_inst, rgb_v = load_txt_s3dis(join(dir_s, i), rgb=rgb)
            rgb_scene.append(rgb_v)
        else:
            coord_inst = load_txt_s3dis(join(dir_s, i))

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
    px=True,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Loads an image file and processes it based on its type. The function supports
    several image formats including TIFF, MRC, AM, and ND2. Depending on the file
    type, the image is loaded using the respective file-loader function, and
    optional normalization can be applied. Pixel size information may be returned
    for some file types.

    :param image: Path to the image file to be loaded.
    :type image: str
    :param normalize: Flag indicating whether to apply normalization to the
        loaded image. Defaults to False.
    :type normalize: bool
    :param px: Flag indicating whether to return pixel size information
        along with the image. Defaults to True.
    :type px: bool
    :return: The loaded image and optionally the pixel size information as a
        float if `px` is True. If `px` is False, only the image is returned.
    :rtype: Union[np.ndarray, Tuple[np.ndarray, float]]
    """
    px_value = 0.0

    if image.endswith((".tif", ".tiff")):
        image, _ = load_tiff(image)
        px_value = 1.0
    elif image.endswith((".mrc", ".rec", ".map")):
        image, px_value = load_mrc_file(image)
    elif image.endswith(".am"):
        image, px_value, _, _ = load_am(image)
    elif image.endswith(".nd2"):
        library_names = ["nd2"]
        library_installed = False
        for lib in library_names:
            if lib_utils.find_spec(lib) is not None:
                library_installed = True

        assert library_installed, "ND2 library is not installed."

        image, px_value = load_nd2_file(image)

    if normalize:
        mean_std = MeanStdNormalize()
        normalize = RescaleNormalize(clip_range=(1, 99))
        image = normalize(mean_std(image)).astype(np.float32)

    if isinstance(image, str):
        TardisError(
            "130",
            "tardis_em/utils/load_data.py",
            f"Indicated {image} file does not exist...",
        )

    if px:
        return image, px_value
    return image
