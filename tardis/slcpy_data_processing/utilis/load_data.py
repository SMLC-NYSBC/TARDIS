from os.path import isfile
from typing import Optional

import mrcfile
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
            if not isfile(self.src_img[:-3] + "am"):
                raise Warning("Missing corresponding .am file...")

            try:
                # Image file [Z x Y x X]
                self.image, self.pixel_size = import_am(src_img)
            except RuntimeWarning:
                raise Warning(
                    "Directory or input .am image file is not correct...")

        self.spatial_graph = open(
            src_am,
            "r",
            encoding="iso-8859-1"
        ).read().split("\n")

    def empty_semantic_label(self):
        return np.zeros(self.image.shape, 'int8')

    def image_data(self):
        return self.image

    def get_segments(self):
        # Find line starting with EDGE { int NumEdgePoints }
        segments = str([
            word for word in self.spatial_graph if
            word.startswith('EDGE { int NumEdgePoints }')
        ])
        segment_start = "".join((ch if ch in "0123456789" else " ")
                                for ch in segments)
        segment_start = [int(i) for i in segment_start.split()]

        # Find in the line directory that starts with @..
        segment_start = int(self.spatial_graph.index(
            "@" + str(segment_start[0]))) + 1

        # Find line define EDGE ... <- number indicate number of segments
        segments = str([
            word for word in self.spatial_graph if word.startswith('define EDGE')
        ])
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
        points = str([
            word for word in self.spatial_graph if word.startswith('POINT { float[3] EdgePointCoordinates }')
        ])
        # Find in the line directory that starts with @..
        points_start = "".join((ch if ch in "0123456789" else " ")
                               for ch in points)
        points_start = [int(i) for i in points_start.split()]
        # Find line that start with the directory @.. and select last one
        points_start = int(self.spatial_graph.index(
            "@" + str(points_start[1]))) + 1

        # Find line define POINT ... <- number indicate number of points
        points = str([
            word for word in self.spatial_graph if word.startswith('define POINT')
        ])
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

    def __read_tiff_transformation(self):
        """
        This method read the header of ET (.am) file and determines global
        transformation for all coordinates
        """

        with open(self.src_img, "r", encoding="iso-8859-1") as et:
            lines_in_et = et.read(50000).split("\n")

        transformation_list = str([
            word for word in lines_in_et if word.startswith('    BoundingBox')
        ]).split(" ")

        trans_x, trans_y, trans_z = (
            float(transformation_list[5]),
            float(transformation_list[7]),
            float(transformation_list[9])
        )
        return trans_x, trans_y, trans_z

    def pixel_size_in_et(self):
        """
        If not specified by user, pixel size is searched in .am file

        Estimation is done by an assumption that points can be found on the top
        and the bottom surface
        pixel_size = tomogram physical size[A] / pixel_number in X[px]
        """

        if self.pixel_size is None:
            with open(self.src_img, "r", encoding="iso-8859-1") as et:
                lines_in_et = et.read(50000).split("\n")

            physical_size = str([word for word in lines_in_et if
                                 word.startswith('        XLen') or word.startswith(
                                     '        xLen')]).split(" ")

            if 'XLen' in physical_size or 'xLen' in physical_size:
                pixel_size = str([word for word in lines_in_et if
                                  word.startswith('        Nx') or word.startswith(
                                      '        nx')]).split(" ")

                physical_size = float(physical_size[9][:-3])
                pixel_size = float(pixel_size[9][:-3])

                return round(physical_size / pixel_size, 2)
            else:
                transformation_list = str([
                    word for word in lines_in_et if word.startswith('    BoundingBox')
                ]).split(" ")

                physical_size = float(transformation_list[6])
                pixel_size = float(self.image.shape[2])

                size = round(physical_size / pixel_size, 2)
                dim = np.array((23.2, 25.72))
                idx_size = (dim - size).argmin()

                return dim[idx_size]
        else:
            return self.pixel_size

    def get_points(self):
        """Generate table of all points with coordinates in pixel"""
        if self.pixel_size is not None:
            pixel_size = self.pixel_size_in_et()
            transformation = self.__read_tiff_transformation()
        else:
            self.pixel_size = 1
            transformation = [0, 0, 0]
        points_coord = self.__find_points()

        points_coord[0:len(points_coord), 0] = points_coord[0:len(
            points_coord), 0] - transformation[0]
        points_coord[0:len(points_coord), 1] = points_coord[0:len(
            points_coord), 1] - transformation[1]
        points_coord[0:len(points_coord), 2] = points_coord[0:len(
            points_coord), 2] - transformation[2]

        return points_coord / pixel_size


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
    Default import for .mrc/.rec files

    Args:
        img: Source of image file

    Returns:
        image: Image array of [Z, Y, X] shape
        pixel_size: float value of the pixel size
    """
    if not isfile(img):
        raise Warning("Indicated .mrc file does not exist...")

    mrc = mrcfile.open(img, mode='r+')

    return mrc.data, mrc.voxel_size.x


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

    am = open(img, 'r', encoding="iso-8859-1").read(5000)
    binary_start = str.find(am, "\n@1\n") + 4
    size = [word for word in am.split('\n') if word.startswith(
        'define Lattice ')][0][15:].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    physical_size = str([word for word in am.split('\n') if
                        word.startswith('    BoundingBox')]).split(" ")
    physical_size = np.array((float(physical_size[6][:-3]),
                              float(physical_size[8][:-3]),
                              float(physical_size[10][:-3])))

    pixel_size = round(physical_size[0] / (nx-1),  3)

    img = np.fromfile(img, dtype=np.uint8)

    return img[binary_start:-1].reshape((nz, ny, nx)), pixel_size
