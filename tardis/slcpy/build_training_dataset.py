from os import listdir
from os.path import join

import numpy as np
from tardis.slcpy.utils.build_semantic_mask import draw_semantic
from tardis.slcpy.utils.load_data import (ImportDataFromAmira, import_am,
                                          import_mrc, import_tiff)
from tardis.slcpy.utils.trim import trim_with_stride


class BuildTrainDataSet:
    """
    MODULE FOR BUILDING TRAIN DATASET

    This module building a train dataset from each file in the specified dir.
    Module working on the file as .tif/.mrc/.rec/.am and mask in image format
    or .csv/.am
        If mask as .csv module expect image file in format of .tif/.mrc/.rec/.am
        If mask as .am module expect image file in format of .am

    For the given dir, module recognize file, and then for each file couple
    (e.g. image and mask) if needed it build mask from the point cloud
    and voxalize image and mask for given size in each dim.

    Files are saved in dir/train/imgs and dir/train/masks

    Args:
        dataset_dir: Directory with train test folders
        circle_size: Size of the segmented object in nm
        multi_layer: If True mask is build in RGB channel instead of gray (0-1)
        resize_pixel_size: Pixel size for image resizing
        tqdm: If True build with progressbar
    """

    def __init__(self,
                 dataset_dir: str,
                 circle_size: int,
                 multi_layer: bool,
                 resize_pixel_size: float,
                 tqdm: bool):
        self.dataset_dir = dataset_dir
        self.circle_size = circle_size
        self.multi_layer = multi_layer
        self.resize_pixel_size = resize_pixel_size
        self.tqdm = tqdm

        self.file_formats = ('.tif', '.am', '.mrc', '.rec')
        self.mask_format = ('_mask.tif', "_mask.mrc", '_mask.rec', '_mask.am')

        """Check what file are in the folder to build dataset"""
        self.idx_img = [f for f in listdir(
            dataset_dir) if f.endswith(self.file_formats)]
        is_csv = [f for f in self.idx_img if f.endswith('.csv')]
        is_mask_image = [
            f for f in self.idx_img if f.endswith(self.mask_format)]

        self.is_csv = False
        self.is_mask_image = False
        self.is_am = False
        if len(is_csv) > 0:  # Expect .csv as coordinate
            assert len(is_csv) * 2 == (len(self.idx_img) / 2), \
                'Not all image file in directory has corresponding .csv file...'
            self.is_csv = True
        else:
            if len(is_mask_image) > 0:  # Expect image semantic mask as input
                assert len(is_mask_image) == (len(self.idx_img) / 2), \
                    'Not all image file in directory has corresponding _mask.* file...'
                self.is_mask_image = True
            else:
                # Expect .am as coordinate
                assert len([f for f in self.idx_img if f.endswith('.CorrelationLines.am')]) == (len(self.idx_img) / 2), \
                    'Not all image file in directory has corresponding .CorrelationLines.am file...'
                self.is_am = True

        assert np.any([self.is_csv, self.is_mask_image, self.is_am]), \
            'Could not find any compatible dataset. Refer to documentation for, ' \
            'how to structure your dataset properly!'

        if self.is_csv:
            idx_mask = [f.endswith('.csv') for f in self.idx_img]
        elif self.is_mask_image:
            idx_mask = [f.endswith(self.mask_format) for f in self.idx_img]
        elif self.is_am:
            idx_mask = [f.endswith(('.CorrelationLines.am')) for f in self.idx_img]

        self.idx_mask = list(np.array(self.idx_img)[idx_mask])
        self.idx_img = list(np.array(self.idx_img)[np.logical_not(idx_mask)])
        assert len(self.idx_mask) == len(self.idx_img), \
            'Number of images and mask is not the same!'

    def __len__(self):
        return len(self.idx_img)

    def __builddataset__(self,
                         trim_xy: int,
                         trim_z: int):
        """
        Args:
            trim_xy: Voxal size of output image in x and y dimension
            trim_z: Voxal size of output image in z dimension
        """
        """Load data, build mask if not image and voxalize"""
        coord = None
        img_counter = 0
        if self.tqdm:
            from tqdm import tqdm

            batch_iter = tqdm(range(self.__len__()),
                              'Building Training dataset')
        else:
            batch_iter = range(self.__len__())

        for i in batch_iter:
            """Load image data"""
            img_name = self.idx_img[i]
            mask_name = self.idx_mask[i]
            mask = None

            """Load image file"""
            if img_name.endswith('.tif'):
                image, _ = import_tiff(join(self.dataset_dir, img_name),
                                       dtype=np.uint8)
                pixel_size = 0
            elif img_name.endswith(('.mrc', '.rec')):
                image, pixel_size = import_mrc(join(self.dataset_dir,
                                                    img_name))
            elif img_name.endswith('.am'):
                importer = ImportDataFromAmira(src_am=join(self.dataset_dir,
                                                           mask_name),
                                               src_img=join(self.dataset_dir,
                                                            img_name))
                image, pixel_size = importer.get_image()
                coord = importer.get_segmented_points()  # [ID x X x Y x Z]

            """Load mask/coord data"""
            if self.is_mask_image:
                if mask_name.endswith('_mask.tif'):
                    mask, _ = import_tiff(join(self.dataset_dir, img_name),
                                          dtype=np.uint8)
                elif mask_name.endswith(('_mask.mrc', '_mask.rec')):
                    mask, _ = import_mrc(join(self.dataset_dir,
                                              img_name))
                elif mask_name.endswith('_mask.am'):
                    mask, _ = import_am(join(self.dataset_dir,
                                             img_name))
            elif self.is_csv:
                coord = np.genfromtxt(mask_name,
                                      delimiter=',')  # [ID x X x Y x Z]
                if str(coord[0, 0]) == 'nan':
                    coord = coord[1:, :]
                if coord.shape[1] == 3:
                    coord = np.array((coord[:, 0],
                                      coord[:, 1],
                                      coord[:, 2],
                                      np.zeros((coord.shape[0], )))).T
            elif self.is_am:
                if img_name.endswith('.tif') and pixel_size == 0:
                    raise TypeError('Data are incompatible in this version. '
                                    '.tif image and .am coordinate file are incompatible '
                                    'pixel size was {pixel_size} which may yeald '
                                    'incorrect drawing of the mask!')

            """Draw mask"""
            if coord is not None:
                assert coord.shape[1] == 4, \
                    f'Coord file do not have shape [ID x X x Y x Z]. Given {coord.shape[1]}'
                if not self.is_mask_image:
                    mask = draw_semantic(mask_size=image.shape,
                                         coordinate=coord,
                                         pixel_size=pixel_size,
                                         circle_size=self.circle_size,
                                         multi_layer=self.multi_layer,
                                         tqdm=self.tqdm)

            if pixel_size == 0:
                scale_factor = 1
            else:
                scale_factor = pixel_size / self.resize_pixel_size

            """Voxalize Image and Mask"""
            trim_with_stride(image=image,
                             mask=mask,
                             scale=scale_factor,
                             trim_size_xy=trim_xy,
                             trim_size_z=trim_z,
                             output=join(self.dataset_dir, 'train'),
                             image_counter=img_counter,
                             clean_empty=True,
                             prefix='',
                             stride=25)
