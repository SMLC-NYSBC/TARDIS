#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import random
import struct
from collections import namedtuple
from os import listdir
from os.path import isfile, join
from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d
import tifffile.tifffile as tif
from numpy import ndarray
from sklearn.neighbors import KDTree, NearestNeighbors

from tardis.utils.errors import TardisError
from tardis.utils.normalization import RescaleNormalize
from tardis.utils.visualize_pc import _rgb


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

    def __init__(self, src_am: str, src_img: Optional[str] = None):
        self.src_img = src_img
        self.src_am = src_am

        # Read image and its property if existing
        if self.src_img is not None:
            if not self.src_img[-3:] == ".am":
                TardisError("130", "tardis/utils/load_data.py", f"{self.src_img} Not a .am file...")

            if src_img.split("/")[-1:][:-3] != src_am.split("/")[-1:][:-20]:
                TardisError(
                    "131", "tardis/utils/load_data.py", f"Image file {src_img} has wrong extension for {src_am}!"
                )

            try:
                # Image file [Z x Y x X]
                self.image, self.pixel_size, _, self.transformation = import_am(src_img)
            except RuntimeWarning:
                TardisError(
                    "130",
                    "tardis/utils/load_data.py",
                    "Directory or input .am image file is not correct..." f"for given dir: {src_img}",
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

        if not any([True for i in ["AmiraMesh 3D ASCII", "# ASCII Spatial Graph"] if i not in am]):
            self.spatial_graph = None
        else:
            self.spatial_graph = open(src_am, "r", encoding="iso-8859-1").read().split("\n")
            self.spatial_graph = [x for x in self.spatial_graph if x != ""]

    def __get_segments(self) -> Union[np.ndarray, None]:
        """
        Helper class function to read segment data from amira file.

        Returns:
            np.ndarray: Array (N, 1) indicating a number of points per segment.
        """
        if self.spatial_graph is None:
            return None

        # Find line starting with EDGE { int NumEdgePoints }
        segments = str([word for word in self.spatial_graph if word.startswith("EDGE { int NumEdgePoints }")])

        segment_start = "".join((ch if ch in "0123456789" else " ") for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        try:
            segment_start = int(self.spatial_graph.index("@" + str(segment_start[0]))) + 1
        except ValueError:
            segment_start = int(self.spatial_graph.index("@" + str(segment_start[0]) + " ")) + 1

        # Find line define EDGE ... <- number indicate number of segments
        segments = str([word for word in self.spatial_graph if word.startswith("define EDGE")])

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
            [word for word in self.spatial_graph if word.startswith("POINT { float[3] EdgePointCoordinates }")]
        )

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ") for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @... and select last one
        try:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]))) + 1
        except ValueError:
            points_start = int(self.spatial_graph.index("@" + str(points_start[1]) + " ")) + 1

        # Find line define POINT ... <- number indicate number of points
        points = str([word for word in self.spatial_graph if word.startswith("define POINT")])

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
        id = 0
        idx = 0
        for i in segments:
            segmentation[id : (id + int(i))] = idx

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
        labels = [
            word
            for word in self.spatial_graph
            if word.startswith("EDGE { int ") and not word.startswith("EDGE { int NumEdgePoints }")
        ]

        # Find line define EDGE ... <- number indicate number of segments
        segment_no = str([word for word in self.spatial_graph if word.startswith("define EDGE")])

        labels_dict = {}
        for i in labels:
            # Find line starting with EDGE { int label }
            label_start = "".join((ch if ch in "0123456789" else " ") for ch in i)
            label_start = [int(i) for i in label_start.split()][-1:]

            # Find in the line directory that starts with @..
            try:
                label_start = int(self.spatial_graph.index("@" + str(label_start[0]))) + 1
            except ValueError:
                label_start = int(self.spatial_graph.index("@" + str(label_start[0]) + " ")) + 1

            label_finish = "".join((ch if ch in "0123456789" else " ") for ch in segment_no)
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


def import_tiff(tiff: str):
    """
    Function to load any tiff file.

    Args:
        tiff (str): Tiff file directory.

    Returns:
        np.ndarray, float: Image data and unified pixel size.
    """
    if not isfile(tiff):
        TardisError("130", "tardis/utils/load_data.py", f"Indicated .tif  {tiff} file does not exist...")

    return np.array(tif.imread(tiff)), 1.0


# int nx
# int ny
# int nz
fstr = "3i"
names = "nx ny nz"

# int mode
fstr += "i"
names += " mode"

# int nxstart
# int nystart
# int nzstart
fstr += "3i"
names += " nxstart nystart nzstart"

# int mx
# int my
# int mz
fstr += "3i"
names += " mx my mz"

# float xlen
# float ylen
# float zlen
fstr += "3f"
names += " xlen ylen zlen"

# float alpha
# float beta
# float gamma
fstr += "3f"
names += " alpha beta gamma"

# int mapc
# int mapr
# int maps
fstr += "3i"
names += " mapc mapr maps"

# float amin
# float amax
# float amean
fstr += "3f"
names += " amin amax amean"

# int ispg
# int next
# short creatid
fstr += "2ih"
names += " ispg next creatid"

# pad 30 (extra data)
# [98:128]
fstr += "30x"

# short nint
# short nreal
fstr += "2h"
names += " nint nreal"

# pad 20 (extra data)
# [132:152]
fstr += "20x"

# int imodStamp
# int imodFlags
fstr += "2i"
names += " imodStamp imodFlags"

# short idtype
# short lens
# short nd1
# short nd2
# short vd1
# short vd2
fstr += "6h"
names += " idtype lens nd1 nd2 vd1 vd2"

# float[6] tiltangles
fstr += "6f"
names += " tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz"

# NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
# float xorg
# float yorg
# float zorg
# char[4] cmap
# char[4] stamp
# float rms
fstr += "3f4s4sf"
names += " xorg yorg zorg cmap stamp rms"

# int nlabl
# char[10][80] labels
fstr += "i800s"
names += " nlabl labels"

header_struct = struct.Struct(fstr)
MRCHeader = namedtuple("MRCHeader", names)


def mrc_read_header(mrc: Optional[Union[str, bytes]] = None):
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
        TardisError("130", "tardis/utils/load_data.py", "4 bit .mrc file are not supported. Ask Dev if you need it!")
    if mode == 1024:
        TardisError("130", "tardis/utils/load_data.py", "Are your trying to load tiff file as mrc?")

    if isinstance(mode, int):
        if mode == 0 and amin >= 0:
            return dtype_[mode]
        elif mode == 0 and amin < 0:
            return np.int8

        if mode in dtype_:
            return dtype_[mode]
        else:
            TardisError("130", "tardis/utils/load_data.py", f"Unknown dtype mode: {str(mode)} and {str(amin)}")
    else:
        if mode in [np.int8, np.uint8]:
            return 0
        for name in dtype_:
            if mode == dtype_[name]:
                return name


def import_am(am_file: str):
    """
    Function to load Amira binary image data.

    Args:
        am_file (str): Amira binary image .am file.

    Returns:
        np.ndarray, float, float, list: Image file as well images parameters.
    """
    if not isfile(am_file):
        TardisError("130", "tardis/utils/load_data.py", f"Indicated .am {am_file} file does not exist...")

    am = open(am_file, "r", encoding="iso-8859-1").read(8000)

    asci = False
    if "AmiraMesh 3D ASCII" in am:
        if "define Lattice" not in am:
            TardisError("130", "tardis/utils/load_data.py", f".am {am_file} file is coordinate file not image!")
        asci = True

    size = [word for word in am.split("\n") if word.startswith("define Lattice ")][0][15:].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    # Fix for ET that were trimmed
    #  ET boundary box has wrong size
    bb = str([word for word in am.split("\n") if word.startswith("    BoundingBox")]).split(" ")

    if len(bb) == 0:
        physical_size = np.array((float(bb[6]), float(bb[8]), float(bb[10][:-3])))
        transformation = np.array((0.0, 0.0, 0.0))
    else:
        am = open(am_file, "r", encoding="iso-8859-1").read(20000)
        bb = str([word for word in am.split("\n") if word.startswith("    BoundingBox")]).split(" ")

        physical_size = np.array((float(bb[6]), float(bb[8]), float(bb[10][:-3])))

        transformation = np.array((float(bb[5]), float(bb[7]), float(bb[9])))

    try:
        coordinate = str([word for word in am.split("\n") if word.startswith("        Coordinates")]).split(" ")[9][1:2]
    except IndexError:
        coordinate = None

    if coordinate == "m":  # Bring meter to angstrom
        pixel_size = ((physical_size[0] - transformation[0]) / (nx - 1)) * 10000000000
    else:
        pixel_size = (physical_size[0] - transformation[0]) / (nx - 1)
    pixel_size = round(pixel_size, 3)

    if "Lattice { byte Data }" in am:
        if asci:
            img = open("../../rand_sample/T216_grid3b.am", "r", encoding="iso-8859-1").read().split("\n")
            img = [x for x in img if x != ""]
            img = np.asarray(img)
            return img
        else:
            img = np.fromfile(am_file, dtype=np.uint8)

    elif "Lattice { sbyte Data }" in am:
        img = np.fromfile(am_file, dtype=np.int8)
        img = img + 128

    # elif 'Lattice { byte Labels } @1(HxByteRLE' in am:
    #     img = np.fromfile(am_file, dtype=np.uint8)

    #     if nz == 1:
    #         binary_start = nx * ny
    #         img = img[-binary_start:-1].reshape((ny, nx))
    #     else:
    #         binary_start = nx * ny * nz
    #         img = img[-binary_start:].reshape((nz, ny, nx))
    # elif 'Lattice { byte Labels } @1 ' in am:
    #     img = open(am_file, 'r', encoding="iso-8859-1").readlines()

    #     if nz == 1:
    #         binary_start = nx * ny
    #         img = np.asarray(img[-binary_start:]).astype(np.uint8).reshape((ny, nx))
    #     else:
    #         binary_start = nx * ny * nz
    #         img = np.asarray(img[-binary_start:]).astype(np.uint8).reshape((nz, ny, nx))

    binary_start = str.find(am, "\n@1\n") + 4
    img = img[binary_start:-1]
    if nz == 1:
        if len(img) == ny * nx:
            img = img.reshape((ny, nx))
        else:
            df_img = np.zeros((ny * nx), dtype=np.uint8)
            df_img[: len(img)] = img
            img = df_img.reshape((ny, nx))
    else:
        if len(img) == nz * ny * nx:
            img = img.reshape((nz, ny, nx))
        else:
            df_img = np.zeros((nz * ny * nx), dtype=np.uint8)
            df_img[: len(img)] = img
            img = df_img.reshape((nz * ny * nx))

    return img, pixel_size, physical_size, transformation


def load_mrc_file(mrc: str) -> Union[Tuple[np.ndarray, float], Tuple[None, float]]:
    """
    Function to load MRC 2014 file format.

    Args:
        mrc (str): MRC file directory.

    Returns:
        np.ndarray, float: Image data and pixel size.
    """
    if not isfile(mrc):
        TardisError("130", "tardis/utils/load_data.py", f"Indicated .mrc {mrc} file does not exist...")

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
    except ValueError:  # File is corrupted
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

    if image.min() < 0 and image.dtype == np.int8:
        image = image + 128
        image = image.astype(np.uint8)

    if image.min() < 0 and image.dtype == np.int16:
        image = image + 32768
        image = image.astype(np.uint16)

    if nz > ny:
        image = image.transpose((1, 0, 2))  # YZX to ZYX
    elif nz > nx:
        image = image.transpose((2, 1, 0))  # XYZ to ZYX

    return image, pixel_size


def load_ply_scannet(ply: str, downscaling=0, color: Optional[str] = None) -> Union[Tuple[ndarray, ndarray], ndarray]:
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
    pcd = o3d.io.read_point_cloud(ply)
    coord_org = np.asarray(pcd.points)
    label_org = np.asarray(pcd.colors)

    SCANNET_COLOR_MAP_20 = {
        0: (0.0, 0.0, 0.0),
        1: (174.0, 199.0, 232.0),
        2: (152.0, 223.0, 138.0),
        3: (31.0, 119.0, 180.0),
        4: (255.0, 187.0, 120.0),
        5: (188.0, 189.0, 34.0),
        6: (140.0, 86.0, 75.0),
        7: (255.0, 152.0, 150.0),
        8: (214.0, 39.0, 40.0),
        9: (197.0, 176.0, 213.0),
        10: (148.0, 103.0, 189.0),
        11: (196.0, 156.0, 148.0),
        12: (23.0, 190.0, 207.0),
        14: (247.0, 182.0, 210.0),
        15: (66.0, 188.0, 102.0),
        16: (219.0, 219.0, 141.0),
        17: (140.0, 57.0, 197.0),
        18: (202.0, 185.0, 52.0),
        19: (51.0, 176.0, 203.0),
        20: (200.0, 54.0, 131.0),
        21: (92.0, 193.0, 61.0),
        22: (78.0, 71.0, 183.0),
        23: (172.0, 114.0, 82.0),
        24: (255.0, 127.0, 14.0),
        25: (91.0, 163.0, 138.0),
        26: (153.0, 98.0, 156.0),
        27: (140.0, 153.0, 101.0),
        28: (158.0, 218.0, 229.0),
        29: (100.0, 125.0, 154.0),
        30: (178.0, 127.0, 135.0),
        32: (146.0, 111.0, 194.0),
        33: (44.0, 160.0, 44.0),
        34: (112.0, 128.0, 144.0),
        35: (96.0, 207.0, 209.0),
        36: (227.0, 119.0, 194.0),
        37: (213.0, 92.0, 176.0),
        38: (94.0, 106.0, 211.0),
        39: (82.0, 84.0, 163.0),
        40: (100.0, 85.0, 144.0),
    }

    # Downscaling point cloud with labels
    if downscaling != 0 and downscaling > 0:
        pcd = pcd.voxel_down_sample(voxel_size=downscaling)
        coord = np.asarray(pcd.points)
    else:
        coord = coord_org

    # Retrieve Node RGB features
    if color is not None:
        rgb = o3d.io.read_point_cloud(color)
        if downscaling > 0:
            rgb = rgb.voxel_down_sample(voxel_size=downscaling)
        rgb = np.asarray(rgb.colors)
        if coord.shape != rgb.shape:
            TardisError(
                "131",
                "tardis/utils/load_data.py",
                "RGB shape must be the same as coord!" f"But {coord.shape} != {rgb.shape}",
            )

    # Retrieve ScanNet v2 labels after downscaling
    cls_id = []
    tree = KDTree(coord_org, leaf_size=coord_org.shape[0])
    for i in coord:
        _, match_coord = tree.query(i.reshape(1, -1))
        match_coord = match_coord[0][0]

        color_df = label_org[match_coord] * 255
        color_id = [key for key in SCANNET_COLOR_MAP_20 if np.all(SCANNET_COLOR_MAP_20[key] == color_df)]

        if len(color_id) > 0:
            cls_id.append(color_id[0])
        else:
            cls_id.append(0)

    cls_id = np.asarray(cls_id)[:, None]

    # Remove 0 labels
    coord = coord[np.where(cls_id != 0)[0]]

    if color is not None:
        rgb = rgb[np.where(cls_id != 0)[0]]  # Remove 0 labels
        cls_id = cls_id[np.where(cls_id != 0)[0]]

        return np.hstack((cls_id, coord)), rgb
    else:
        return np.hstack((cls_id[np.where(cls_id != 0)[0]], coord))


def load_ply_partnet(ply, downscaling=0) -> np.ndarray:
    """
    Function to read .ply files.
    Args:
        ply (str): File directory.
        downscaling (float): Downscaling point cloud by fixing voxel size.
    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    pcd = o3d.io.read_point_cloud(ply)
    label_uniq = np.unique(np.asarray(pcd.colors), axis=0)

    coord_org = np.asarray(pcd.points)
    label_org = np.asarray(pcd.colors)

    if downscaling != 0 and downscaling > 0:
        pcd = pcd.voxel_down_sample(voxel_size=downscaling)
    coord = np.asarray(pcd.points)

    label_id = []
    tree = KDTree(coord_org, leaf_size=coord_org.shape[0])
    for i in coord:
        _, match_coord = tree.query(i.reshape(1, -1), k=1)
        match_coord = match_coord[0][0]

        label_id.append(np.where(np.all(label_org[match_coord] == label_uniq, 1))[0][0])

    return np.hstack((np.asarray(label_id)[:, None], coord))


def load_txt_s3dis(txt: str, rgb=False, downscaling=0) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to read .txt Stanford 3D instance scene file.

    Args:
        txt (str): File directory.
        rgb (bool):
        downscaling (float): Downscaling point cloud by fixing voxel size.

    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    coord = np.genfromtxt(txt, invalid_raise=False)

    rgb_values = coord[:, 3:]
    coord = coord[:, :3]

    if downscaling > 0:
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(coord)

        if rgb:
            pcd.colors = o3d.utility.Vector3dVector(rgb_values)

        pcd = pcd.voxel_down_sample(voxel_size=downscaling)

        coord = np.asarray(pcd.points)

        if rgb:
            rgb_values = np.asarray(pcd.colors)
            return coord, rgb_values

    if rgb:
        return coord, rgb_values
    return coord


def load_s3dis_scene(
    dir: str, downscaling=0, random_ds=None, rgb=False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Function to read .txt Stanford 3D instance scene files.

    Args:
        dir (str): Folder directory with all instances.
        downscaling (float): Downscaling point cloud by fixing voxel size.
        random_ds (None, float): If not None, indicate ration of point to keep.

    Returns:
        np.ndarray: Labeled point cloud coordinates.
    """
    dir_list = [x for x in listdir(dir) if x not in [".DS_Store", "Icon"]]

    coord_scene = []
    rgb_scene = []
    id = 0
    for i in dir_list:
        if rgb:
            coord_inst, rgb_v = load_txt_s3dis(join(dir, i), rgb=rgb)
            rgb_scene.append(rgb_v)
        else:
            coord_inst = load_txt_s3dis(join(dir, i))

        coord_scene.append(np.hstack((np.expand_dims(np.repeat(id, len(coord_inst)), 1), coord_inst)))

        id += 1
    coord = np.concatenate(coord_scene)

    if downscaling > 0:
        if random_ds is not None:
            pick = int(len(coord) * random_ds)
            pick = random.sample(range(len(coord)), pick)
            coord = coord[pick, :]

            if rgb:
                rgb_v = np.concatenate(rgb_scene) / 255
                rgb_v = rgb_v[pick, :]
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord[:, 1:])

            if rgb:
                rgb_v = np.concatenate(rgb_scene) / 255

                pcd.colors = o3d.utility.Vector3dVector(rgb_v)
            else:
                pcd.colors = o3d.utility.Vector3dVector(_rgb(coord, True))

            pcd = pcd.voxel_down_sample(voxel_size=downscaling)
            coord_ds = np.asarray(pcd.points)

            if rgb:
                rgb_v = np.asarray(pcd.colors)

            # Associate labels
            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(coord[:, 1:])

            # Query the nearest neighbor for all points in coord_ds
            _, indices = knn.kneighbors(coord_ds)
            indices = np.concatenate(indices)
            cls_id = np.expand_dims(coord[indices, 0], 1)
            coord = np.hstack((cls_id, coord_ds))

    if rgb:
        return coord, rgb_v
    return coord


def load_image(image: str, normalize=False) -> Tuple[np.ndarray, float]:
    """
    Quick wrapper for loading image data based on detected file format.

    Args:
        image (str): Image file directory.
        normalize (bool): Rescale histogram to 1% - 99% percentile.

    Returns:
        np.ndarray, float: Image array and associated pixel size.
    """
    px = 1.0

    if image.endswith((".tif", ".tiff")):
        image, px = import_tiff(image)
    elif image.endswith((".mrc", ".rec", ".map")):
        image, px = load_mrc_file(image)
    elif image.endswith(".am"):
        image, px, _, _ = import_am(image)

    if normalize:
        norm = RescaleNormalize(clip_range=(1, 99))
        image = norm(image)

    return image, px
