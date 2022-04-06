from typing import Optional

import numpy as np
import tifffile.tifffile as tiff
from tardis.slcpy_data_processing.utils.load_data import ImportDataFromAmira


def preprocess_data(coord: str,
                    image: Optional[str] = None,
                    size: Optional[int] = None,
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

    """ Collect Coordinates [Length x Dimension] """
    if coord[-3:] == "csv":
        coord_label = np.genfromtxt(coord, delimiter=',')
        if str(coord_label[0, 0]) == 'nan':
            coord_label = coord_label[1:, :]

    if coord[-3:] == "npy":
        coord_label = np.load(coord)

    if coord[-3:] == ".am":
        if image is None:
            amira_import = ImportDataFromAmira(src_am=coord,
                                               src_img=None)
            coord_label = amira_import.get_segmented_points()
            pixel_size = amira_import.get_pixel_size()
        else:
            amira_import = ImportDataFromAmira(src_am=coord,
                                               src_img=image)
            coord_label = amira_import.get_segmented_points()
            pixel_size = amira_import.get_pixel_size()

    coords = coord_label[:, 1:]

    if size is None:
        size = 1

    if coords.shape[1] == 2:
        size = (size, size)
    elif coords.shape[1] == 3:
        size = (size, size, size)

    """ Collect Image Patches [Channels x Length] """
    if image is not None and coord[-3:] != ".am":
        if normalization == "simple":
            normalization = SimpleNormalize()
        elif normalization == 'minmax':
            normalization = MinMaxNormalize(0, 255)

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
    elif image is not None and image.endswith('.am'):
        if normalization == "simple":
            normalization = SimpleNormalize()
        else:
            normalization = MinMaxNormalize(0, 255)

        img_stack, pixel_size = amira_import.get_image()

        crop_tiff = Crop2D3D(image=img_stack,
                             size=size,
                             normalization=normalization,
                             memory_save=False)  # Z x Y x X

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
            ends_distance = np.linalg.norm(
                self.coord[points_in_contour[0]][1:] - self.coord[points_in_contour[-1]][1:])

            if self.pixel_size is not None:
                if ends_distance < round(10 / self.pixel_size):
                    self.graph[points_in_contour[0], points_in_contour[-1]] = 1
                    self.graph[points_in_contour[-1], points_in_contour[0]] = 1
            else:
                if ends_distance < 5:  # Assuming around 2 nm pixel size
                    self.graph[points_in_contour[0], points_in_contour[-1]] = 1
                    self.graph[points_in_contour[-1], points_in_contour[0]] = 1

        return self.graph


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

# %%


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

            if crop_img.shape != (self.size[-1], self.size[0], self.size[1]):
                crop_df = np.array(crop_img)
                shape = crop_img.shape
                crop_img = np.zeros((self.size[2], self.size[0], self.size[1]))
                crop_img[0:shape[0], 0:shape[1], 0:shape[2]] = crop_df
        elif len(center_point) == 2:
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

            if crop_img.shape != (self.size[0], self.size[1]):
                crop_df = np.array(crop_img)
                shape = crop_img.shape
                crop_img = np.zeros((self.size[0], self.size[1]))
                crop_img[0:shape[0], 0:shape[1]] = crop_df

        crop_img = self.normalization(crop_img)

        return crop_img
