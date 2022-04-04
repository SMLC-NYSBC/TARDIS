from os import path
from typing import Optional

import numpy as np
import tifffile.tifffile as tiff
from tardis.slcpy_data_processing.utils.load_data import ImportDataFromAmira


def preprocess_data(coord: str,
                    image: Optional[str] = None,
                    size: Optional[tuple] = None,
                    include_label=True,
                    pixel_size=None,
                    normalization: Optional[str] = 'simple',
                    memory_save: Optional[bool] = False):
    """
    PRE-PROCESSING MODULE FOR COORDINATES AND IMAGES

    The module is taking as an input a coordinate file as well as corresponding
    image file, and compose trainable input.

    Args:
        coord: Directory for coordinate file [ID x Coord x Dim].
        coord_downsample:
        image: Directory for image file.
        size: Size of the image patches.
        normalization: ['simple', 'minmax'] Type of normalization for image data.
        memory_save: If True image patches are build slower but with memory
            preservation.

    Return:
        coord: [Length x Dimension]
        img:   [Channels x Length]
        graph: [Length x Length]
    """
    if memory_save:
        import zarr

    if size is not None:
        assert (size[0] % 2) == 0, 'Size of a patch must be divided by 2!'
        assert len(size) in [2, 3], 'Size has to be 2D or 3D only!'

    """ Collect Coordinates [Length x Dimension] """
    if coord[-3:] == "csv":
        coord_label = np.genfromtxt(coord, delimiter=',')
        if str(coord_label[0, 0]) == 'nan':
            coord_label = coord_label[1:, :]

    if coord[-3:] == "npy":
        coord_label = np.load(coord)

    if coord[-3:] == ".am":
        if image is None:
            loader = LoadAmira(coord_am=coord,
                               img_am=None,
                               pixel_size=pixel_size)
            coord_label = loader.build_coord_with_label()
        else:
            loader = LoadAmira(coord_am=coord,
                               img_am=image,
                               pixel_size=pixel_size)
            coord_label = loader.build_coord_with_label()

    coords = coord_label[:, 1:]

    """ Collect Image Patches [Channels x Length] """
    if image is not None and coord[-3:] != ".am":
        if normalization == "simple":
            normalization = SimpleNormalize()
        else:
            normalization = MinMaxNormalize(0, 255)

        if image is not None and coord[-3:] != ".am":
            if memory_save:
                img_df = tiff.imread(image, aszarr=True)
                img_stack = zarr.open(img_df, mode='r')
            else:
                img_stack = tiff.imread(image)

        crop_tiff = Crop2D3D(image=img_stack,
                             size=size,
                             normalization=normalization,
                             memory_save=memory_save)  # Z x Y x X

        if len(size) == 2:
            img = np.zeros((len(coords), size[0] * size[1]))

            for i in range(img.shape[0]):
                point = coords[i]  # X x Y
                img[i, :] = np.array(crop_tiff(center_point=point)).flatten()

        elif len(size) == 3:
            img = np.zeros((len(coords), size[0] * size[1] * size[2]))

            for i in range(img.shape[0]):
                point = coords[i]  # X x Y x Z
                img[i, :] = np.array(crop_tiff(center_point=point)).flatten()
        if memory_save:
            img_df.close()
    else:
        img = np.zeros(size)

    if include_label:
        return coord_label, img
    else:
        """ Collect Graph [Length x Length] """
        build = BuildGraph(coord=coord_label,
                           pixel_size=pixel_size)
        graph = build()

        return coords, img, graph


class BuildGraph:
    """
    MODULE FOR BUILDING GRAPH REPRESENTATION OF A POINT CLOUD

    Args:
        coord: Coordinate with label from which graph is build
        pixel_size: Pixel size of image used for calculating distance
    """

    def __init__(self,
                 coord: np.ndarray,
                 pixel_size: Optional[int] = None):
        self.coord = coord
        self.pixel_size = pixel_size
        self.graph = np.zeros((len(coord), len(coord)))
        self.all_idx = np.unique(coord[:, 0])

    def __call__(self):
        for i in self.all_idx:
            points_in_contour = np.where(self.coord[:, 0] == i)
            points_in_contour = points_in_contour[0].tolist()

            for j in points_in_contour:
                self.graph[j, j] = 1
                # First point in contour
                if j == points_in_contour[0]:  # First point
                    if (j + 1) <= (len(self.coord) - 1):
                        self.graph[j, j + 1] = 1
                        self.graph[j + 1, j] = 1
                # Last point
                elif j == points_in_contour[len(points_in_contour) - 1]:
                    self.graph[j, j - 1] = 1
                    self.graph[j - 1, j] = 1
                else:  # Point in the middle
                    self.graph[j, j + 1] = 1
                    self.graph[j + 1, j] = 1
                    self.graph[j, j - 1] = 1
                    self.graph[j - 1, j] = 1

            # Check euclidian distance between fist and last point. if shorter then
            #  10 nm then connect
            ends_distance = np.linalg.norm(self.coord[points_in_contour[0]][1:] - self.coord[points_in_contour[-1]][1:])

            if self.pixel_size is not None:
                if ends_distance < round(10 / self.pixel_size):
                    self.graph[points_in_contour[0], points_in_contour[-1]] = 1
                    self.graph[points_in_contour[-1], points_in_contour[0]] = 1
            else:
                if ends_distance < 5:  # Assuming around 2 nm pixel size
                    self.graph[points_in_contour[0], points_in_contour[-1]] = 1
                    self.graph[points_in_contour[-1], points_in_contour[0]] = 1

        return self.graph


class LoadAmira:
    """
    OBJECT TO LOAD AND TRANSFORM 3D POINT CLOUD FROM .AM FILES

    Args:
        am_dir: Directory for the .am file with 3D point cloud
    !Important! Object need additional raw .am file for calculating transformation

    Modified from slcpy package
    """

    def __init__(self,
                 coord_am: str,
                 img_am: Optional[str] = None,
                 pixel_size=None):
        self.coord_am = coord_am
        self.img_am = img_am
        self.pixel_size = pixel_size

        assert path.isfile(coord_am), \
            'Indicated .am file does not exist!'

        if img_am is not None:
            assert path.isfile(img_am), \
                'Indicated Image file is not .tif!'
            assert path.isfile(img_am[:-3] + ".am"), \
                'No corresponding raw .am files was found!'

    def build_coord_with_label(self):
        importer = ImportDataFromAmira(src_am=self.coord_am,
                                       src_tiff=self.img_am,
                                       pixel_size=self.pixel_size,
                                       mask=True)
        if self.pixel_size is None:
            points = np.array(importer.get_raw_point().round(), dtype=np.int32)
        else:
            # Correct point coordinates for Amira tranformation
            points = np.array(importer.get_points().round(), dtype=np.int32)

        segments = importer.get_segments()

        seg_idx = np.zeros((len(points), 1))
        start = 0
        for id, i in enumerate(segments):
            stop = start + i[0]
            seg_idx[start:stop, 0] = id
            start = stop

        coord_label = np.hstack((seg_idx, points))  # ID x X x Y x Z

        return coord_label


class SimpleNormalize:
    """
    NORMALIZE IMAGE VALUE USING ASSUMED VALUES

    Inputs:
        x: image or target 3D or 4D arrays
    """

    def __call__(self,
                 x: np.ndarray):
        if x.min() >= 0 and x.max() <= 255:
            norm = x / 255
        elif x.min() >= 0 and x.max() <= 65535:
            norm = x / 65535
        elif x.min() < 0:
            x = x + abs(x.min())  # Move px values from negative numbers
            norm = x / x.max()  # Convert 32 to 16 bit and normalize

        return norm


class MinMaxNormalize:
    """
    NORMALIZE IMAGE VALUE USING MIN/MAX APPROACH

    Input:
        x: image or target 3D or 4D arrays

    Args:
        min: Minimal value for initialize normalization e.g. 0
        max: Maximal value for initialize normalization e.g. 255
    """

    def __init__(self,
                 min: int,
                 max: int):
        assert max > min
        self.min = min
        self.range = max - min

    def __call__(self, x):
        return (x - self.min) / self.range


class Crop2D3D:
    """
    CROPPING MODULE FOR 2D AND 3D IMAGES

    This module is highly conservative for memory used, which is especially
    useful while loading several big tiff files. The module using zarr object
    with pre-loaded image object from tifffile library.

    Input:
        center_point: 2D or 3D coordinates around which image should be cropped
            expect coordinate in shape [X x Y x Z]

    Args:
        image: Image object with pre-loaded image file
        size: Size of cropping frame for 2D or 3D images
        normalization: Normalization object, to normalize image value between 0,1
        memory_save: If True image object is used except of image loaded into memory
    """

    def __init__(self,
                 image,
                 size: tuple,
                 normalization,
                 memory_save: bool):
        self.image = image
        self.size = size
        self.normalization = normalization
        self.memory_save = memory_save

        if len(size) == 2:
            self.width, self.height = image.shape
            self.depth = None
        else:
            self.depth, self.width, self.height = image.shape

    @staticmethod
    def get_xyz_position(center_point: int,
                         size: int,
                         max_size: int):
        assert size <= max_size, \
            'Cropping frame must be smaller then image size!'
        x0 = center_point - (size / 2)
        x1 = center_point + (size / 2)

        """ Check if frame feet into image, if not shift it """
        if x0 < 0:
            move_by = abs(x0)
            x0 = x0 + move_by
            x1 = x1 + move_by

        if x1 > max_size:
            move_by = x1 - max_size
            x0 = x0 - move_by
            x1 = x1 - move_by

        return int(x0), int(x1)

    def __call__(self,
                 center_point: tuple):
        assert len(center_point) in [2, 3], \
            'Given position for cropping is not 2D or 3D!'

        if len(center_point) == 3:
            z0, z1 = self.get_xyz_position(center_point=center_point[-1],
                                           size=self.size[-1],
                                           max_size=self.depth)
            x0, x1 = self.get_xyz_position(center_point=center_point[0],
                                           size=self.size[0],
                                           max_size=self.height)
            y0, y1 = self.get_xyz_position(center_point=center_point[1],
                                           size=self.size[1],
                                           max_size=self.width)
            if self.memory_save:
                crop_img = self.image[z0:z1][y0:y1, x0:x1]
            else:
                crop_img = self.image[z0:z1, y0:y1, x0:x1]

        if len(center_point) == 2:
            x0, x1 = self.get_xyz_position(center_point=center_point[0],
                                           size=self.size[0],
                                           max_size=self.height)
            y0, y1 = self.get_xyz_position(center_point=center_point[1],
                                           size=self.size[1],
                                           max_size=self.width)
            if self.memory_save:
                crop_img = self.image[y0:y1, x0:x1]
            else:
                crop_img = self.image[y0:y1, x0:x1]

        crop_img = self.normalization(crop_img)
        return crop_img
