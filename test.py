#%%
from tardis.slcpy_data_processing.voxalize_image import main
from os.path import join

main(images_dir=join('tests', 'test_data', 'tif'),
     output_dir=join('tests', 'test_data', 'output'),
     image_with_mask=False,
         mask_prefix=None,
         trim_xy=64,
         trim_z=64,
         stride=None,
         tqdm=True)
# %%
from tardis.slcpy_data_processing.stitch_image import main as stitcher
import numpy as np
from os.path import join

stitcher(image_dir=join('tests', 'test_data', 'output', 'imgs'),
         output_dir=join('tests', 'test_data', 'output', 'output'),
         mask=False,
         prefix='',
         binary=False,
         dtype=np.int8,
         tqdm=True)
