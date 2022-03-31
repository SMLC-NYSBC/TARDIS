from os.path import join
from os import listdir

import numpy as np
from tardis.slcpy_data_processing.utils.build_semantic_mask import draw_semantic
from tardis.slcpy_data_processing.utils.load_data import ImportDataFromAmira, \
    import_am, import_mrc, import_tiff
from tardis.slcpy_data_processing.utils.trim import trim_with_stride

"""
# For each file
    # Load data
        # if .tif or .am require .am coord or _mask.tif _mask.am image exist
        # if .tif, .mrc, .rec require .csv coord or _mask.tif, _mask.mrc, _mask.rec exist
    # If coordinate file exist load coordinate file
        # Check if coordinates are compatible with image shape
        # Build semantic mask from coordinates
    # Voxalize image to fit needed image size
        # Save in train/imgs
    # Voxalize mask to fir needed mask size
        # Save in train/masks
"""


class BuildDataSet:
    def __init__(self,
                 dataset_dir: str,
                 circle_size: int,
                 multi_layer: bool,
                 tqdm: bool):
        self.dataset_dir = dataset_dir
        self.circle_size = circle_size
        self.multi_layer = multi_layer
        self.tqdm = tqdm

        self.file_formats = ('.tif', '.am', '.mrc', '.rec')
        self.mask_format = ('_mask.tif', "_mask.mrc", '_mask.rec', '_mask.am')

        self.idx_img = [f for f in listdir(
            dataset_dir) if f.endswith(self.file_formats)]
        is_csv = [f for f in self.idx_img if f.endswith('.csv')]
        is_mask_image = [
            f for f in self.idx_img if f.endswith(self.mask_format)]

        """Check what file are in the folder to build dataset"""
        self.is_csv = False
        self.is_mask_image = False
        self.is_am = False
        if np.any(is_csv):  # Expect .csv as coordinate
            assert sum(is_csv) * 2 == len(self.idx_img), \
                'Not all image file in directory has corresponding .csv file...'
            self.is_csv = True
        else:
            if np.any(is_mask_image):  # Expect image semantic mask as input
                assert sum(is_mask_image) * 2 == len(self.idx_img), \
                    'Not all image file in directory has corresponding _mask.* file...'
                self.is_mask_image = True
            else:  # Expect .am as coordinate
                assert len([f for f in self.idx_img if f.endswith('.am')]) == len(self.idx_img), \
                    'Not all image file in directory has corresponding _mask.* file...'
                self.is_am = True
        assert np.any([self.is_csv, self.is_mask_image, self.is_am]), \
            'Could not find any compatible dataset. Refer to documentation for, ' \
            'how to structure your dataset properly!'

        if self.is_csv:
            idx_mask = [f.endswith('.csv') for f in self.idx_img]
        elif self.is_mask_image:
            idx_mask = [f.endswith(self.mask_format) for f in self.idx_img]
        elif self.is_am:
            idx_mask = [f.endswith(('.CorrelationLines.am'))
                        for f in self.idx_img]

        self.idx_mask = list(np.array(self.idx_img)[idx_mask])
        self.idx_img = list(np.array(self.idx_img)[np.logical_not(idx_mask)])
        assert len(self.idx_mask) == len(self.idx_img), \
            'Number of images and mask is not the same!'

    def __len__(self):
        return len(self.idx_img)

    def __builddataset__(self,
                         trim_xy: int,
                         trim_z: int):
        """Load data, build mask if not image and voxalize"""
        coord = None
        img_counter = 0

        for i in range(self.__len__()):
            """Load image data"""
            img_name = self.idx_img[i]
            mask_name = self.idx_mask[i]
            mask = None

            if img_name.endswith('.tif'):
                image = import_tiff(join(self.dataset_dir, img_name),
                                    dtype=np.uint8)
                pixel_size = 1
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
                    mask = import_tiff(join(self.dataset_dir, img_name),
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

            """Voxalize Image"""
            trim_with_stride(image=image,
                             trim_size_xy=trim_xy,
                             trim_size_z=trim_z,
                             output=join(self.dataset_dir, 'train', 'imgs'),
                             image_counter=img_counter,
                             clean_empty=True,
                             prefix='',
                             stride=25)

            """Voxalize Mask"""
            trim_with_stride(image=mask,
                             trim_size_xy=trim_xy,
                             trim_size_z=trim_z,
                             output=join(self.dataset_dir, 'train', 'masks'),
                             image_counter=img_counter,
                             clean_empty=True,
                             prefix='_mask',
                             stride=25)
