import struct
from collections import namedtuple
from os.path import isfile
from typing import Optional

import numpy as np
import tifffile.tifffile as tif


class ImportDataFromAmira:
    """
    HANDLEER for loading 3D .AM DATA

    Loading of Amira 3D spatial graph and/or image data (.tif/.mrc).
    In case of including image data, coordinates from spatial graph are corrected for Amira transformation.
    Ensuring matching of the coordinates with image data

    Args:
        src_am: Source of the spatial graph in ASCII format
        src_img: Source of the 3D .tif file
    """

    def __init__(self,
                 src_am: str,
                 src_img: Optional[str] = None):
        self.src_img = src_img
        self.src_am = src_am

        if self.src_img is not None:
            if not self.src_img[-3:] == '.am':
                raise Warning("Not a .am file...")

            if src_img[:-3] != src_am[:-20]:
                raise Warning(
                    f'Image file {src_img} has wrong extension for {src_am}!')

            try:
                # Image file [Z x Y x X]
                self.image, self.pixel_size = import_am(src_img)
            except RuntimeWarning:
                raise Warning(
                    "Directory or input .am image file is not correct...")
        else:
            self.pixel_size = 1
        self.spatial_graph = open(src_am,
                                  "r",
                                  encoding="iso-8859-1").read().split("\n")

    def empty_semantic_label(self):
        return np.zeros(self.image.shape, 'int8')

    def get_segments(self):
        # Find line starting with EDGE { int NumEdgePoints }
        segments = str([word for word in self.spatial_graph if
                        word.startswith('EDGE { int NumEdgePoints }')])

        segment_start = "".join((ch if ch in "0123456789" else " ")
                                for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        segment_start = int(self.spatial_graph.index(
            "@" + str(segment_start[0]))) + 1

        # Find line define EDGE ... <- number indicate number of segments
        segments = str(
            [word for word in self.spatial_graph if word.startswith('define EDGE')])

        segment_finish = "".join(
            (ch if ch in "0123456789" else " ") for ch in segments)
        segment_finish = [int(i) for i in segment_finish.split()]
        segment_no = int(segment_finish[0])
        segment_finish = segment_start + int(segment_finish[0])

        # Select all lines between @.. (+1) and number of segments
        segments = self.spatial_graph[segment_start:segment_finish]
        segments = [i.split(' ')[0] for i in segments]

        # return an array of number of points belonged to each segment
        df = np.zeros((segment_no, 1), dtype="int")
        df[0:segment_no, 0] = [int(i) for i in segments]

        return df

    def __find_points(self):
        # Find line starting with POINT { float[3] EdgePointCoordinates }
        points = str([word for word in self.spatial_graph
                      if word.startswith('POINT { float[3] EdgePointCoordinates }')])

        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ")
                               for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @.. and select last one
        points_start = int(self.spatial_graph.index(
            "@" + str(points_start[1]))) + 1

        # Find line define POINT ... <- number indicate number of points
        points = str([word for word in self.spatial_graph
                      if word.startswith('define POINT')])

        points_finish = "".join(
            (ch if ch in "0123456789" else " ") for ch in points)
        points_finish = [int(i) for i in points_finish.split()][0]
        points_no = points_finish
        points_finish = points_start + points_finish

        # Select all lines between @.. (-1) and number of points
        points = self.spatial_graph[points_start:points_finish]

        # return an array of all points coordinates in pixel
        df = np.zeros((points_no, 3), dtype="float")
        for j in range(3):
            coord = [i.split(' ')[j] for i in points]
            df[0:points_no, j] = [float(i) for i in coord]

        return df

    def __read_am_transformation(self):
        """
        This method read the header of ET (.am) file and determines global
        transformation for all coordinates
        """

        with open(self.src_img, "r", encoding="iso-8859-1") as et:
            lines_in_et = et.read(50000).split("\n")

        transformation_list = str([word for word in lines_in_et
                                   if word.startswith('    BoundingBox')]).split(" ")

        trans_x, trans_y, trans_z = (float(transformation_list[5]),
                                     float(transformation_list[7]),
                                     float(transformation_list[9]))
        return trans_x, trans_y, trans_z

    def get_points(self):
        """Generate table of all points with coordinates in pixel"""
        if self.src_img is not None:
            self.transformation = self.__read_am_transformation()
        else:
            self.transformation = [0, 0, 0]
        points_coord = self.__find_points()

        points_coord[0:len(points_coord), 0] = points_coord[0:len(
            points_coord), 0] - self.transformation[0]
        points_coord[0:len(points_coord), 1] = points_coord[0:len(
            points_coord), 1] - self.transformation[1]
        points_coord[0:len(points_coord), 2] = points_coord[0:len(
            points_coord), 2] - self.transformation[2]

        return points_coord / self.pixel_size


def import_tiff(img: str,
                dtype=np.uint8):
    """
    Default import for tif files

    Args:
        img: x
        dtype: Type of output data
    Return:
        image: Image array of [Z, Y, X] shape
        pixel_size: None
    """
    if not isfile(img):
        raise Warning("Indicated .tif file does not exist...")

    return np.array(tif.imread(img), dtype=dtype), None


def import_mrc(img: str):
    """
    DEFAULT IMPORT FOR .mrc/.rec files

    Read out for MRC2014 files with

    Args:
        img: Source of image file

    Returns:
        image: Image array of [Z, Y, X] shape
        pixel_size: float value of the pixel size
    """
    if not isfile(img):
        raise Warning("Indicated .mrc file does not exist...")

    header = mrc_header(img)

    pixel_size = round(header.xlen / header.nx, 3)
    dtype = get_mode(header.mode)
    nz, ny, nx = header.nz, header.ny, header.nx

    if nz == 1:
        image = np.fromfile(img, dtype=dtype)[1024:].reshape((ny, nx))
    else:
        image = np.fromfile(img, dtype=dtype)[1024:].reshape((nz, ny, nx))

    return image, pixel_size


def mrc_header(img: str):
    # int nx
    # int ny
    # int nz
    fstr = '3i'
    names = 'nx ny nz'

    # int mode
    fstr += 'i'
    names += ' mode'

    # int nxstart
    # int nystart
    # int nzstart
    fstr += '3i'
    names += ' nxstart nystart nzstart'

    # int mx
    # int my
    # int mz
    fstr += '3i'
    names += ' mx my mz'

    # float xlen
    # float ylen
    # float zlen
    fstr += '3f'
    names += ' xlen ylen zlen'

    # float alpha
    # float beta
    # float gamma
    fstr += '3f'
    names += ' alpha beta gamma'

    # int mapc
    # int mapr
    # int maps
    fstr += '3i'
    names += ' mapc mapr maps'

    # float amin
    # float amax
    # float amean
    fstr += '3f'
    names += ' amin amax amean'

    # int ispg
    # int next
    # short creatid
    fstr += '2ih'
    names += ' ispg next creatid'

    # pad 30 (extra data)
    # [98:128]
    fstr += '30x'

    # short nint
    # short nreal
    fstr += '2h'
    names += ' nint nreal'

    # pad 20 (extra data)
    # [132:152]
    fstr += '20x'

    # int imodStamp
    # int imodFlags
    fstr += '2i'
    names += ' imodStamp imodFlags'

    # short idtype
    # short lens
    # short nd1
    # short nd2
    # short vd1
    # short vd2
    fstr += '6h'
    names += ' idtype lens nd1 nd2 vd1 vd2'

    # float[6] tiltangles
    fstr += '6f'
    names += ' tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz'

    # NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
    # float xorg
    # float yorg
    # float zorg
    # char[4] cmap
    # char[4] stamp
    # float rms
    fstr += '3f4s4sf'
    names += ' xorg yorg zorg cmap stamp rms'

    # int nlabl
    # char[10][80] labels
    fstr += 'i800s'
    names += ' nlabl labels'

    header_struct = struct.Struct(fstr)
    MRCHeader = namedtuple('MRCHeader', names)

    with open(img, 'rb') as f:
        header = f.read(1024)

    return MRCHeader._make(header_struct.unpack(header))


def get_mode(mode):
    if mode == 0:
        dtype = np.int8
    elif mode == 1:
        dtype = np.int16
    elif mode == 2:
        dtype = np.float32
    elif mode == 3:
        dtype = '2h'  # complex number from 2 shorts
    elif mode == 4:
        dtype = np.complex64
    elif mode == 6:
        dtype = np.uint16
    elif mode == 16:
        dtype = '3B'  # RGB values
    else:
        raise Exception('Unknown dtype mode:' + str(mode))

    return dtype


def import_am(img: str):
    """
    Default import for .am binary files

    Args:
        img: Source of image file

    Returns:
        image: Image array of [Z, Y, X] shape
        pixel_size: float value of the pixel size
    """
    if not isfile(img):
        raise Warning("Indicated .am file does not exist...")

    am = open(img, 'r', encoding="iso-8859-1").read(8000)
    size = [word for word in am.split('\n') if word.startswith(
        'define Lattice ')][0][15:].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    physical_size = str([word for word in am.split('\n') if
                        word.startswith('    BoundingBox')]).split(" ")
    if len(physical_size) == 0:
        physical_size = np.array((float(physical_size[6]),
                                  float(physical_size[8]),
                                  float(physical_size[10][:-3])))
        binary_start = str.find(am, "\n@1\n") + 4
    else:
        am = open(img, 'r', encoding="iso-8859-1").read(20000)
        physical_size = str([word for word in am.split('\n') if
                             word.startswith('    BoundingBox')]).split(" ")

        physical_size = np.array((float(physical_size[6]),
                                  float(physical_size[8]),
                                  float(physical_size[10][:-3])))
        binary_start = str.find(am, "\n@1\n") + 4

    pixel_size = round(physical_size[0] / (nx - 1), 3)

    img = np.fromfile(img, dtype=np.uint8)

    if nz == 1:
        return img[binary_start:-1].reshape((ny, nx)), pixel_size
    else:
        return img[binary_start:-1].reshape((nz, ny, nx)), pixel_size