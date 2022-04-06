from os import getcwd, listdir
from os.path import join
from typing import Optional

import click
import numpy as np
import tifffile.tifffile as tif

from tardis.slcpy_data_processing.image_postprocess import ImageToPointCloud
from tardis.slcpy_data_processing.utils.export_data import NumpyToAmira
from tardis.utils.utils import check_uint8
from tardis.version import version


@click.command()
@click.option('-dir', '--postprocess_dataset',
              default=getcwd(),
              type=str,
              help='Directory with images for post-processing after CNN.',
              show_default=True)
@click.option('-edt', '--euclidean_distance_transform',
              default=False,
              type=bool,
              help='If True, euclidean distance transform is used for skeleton correction.',
              show_default=True)
@click.option('-fs', '--feature_size',
              default=25,
              type=int,
              help='Mask feature size in [nm] used for thresholding euclidean distance transform.',
              show_default=True)
@click.option('-ds', '--downsample',
              default=None,
              type=float,
              help='If not None, float number indicate value of voxal size for downsampling.',
              show_default=True)
@click.option('-s', '--save_format',
              default='csv',
              type=click.Choice(['csv', 'npy', 'am', 'all'],
                                case_sensitive=True),
              help='Output file format.',
              show_default=True)
@click.option('-t', '--tqdm',
              default=True,
              type=bool,
              help='If True, process with progress bar.',
              show_default=True)
@click.version_option(version=version)
def main(postprocess_dataset: str,
         euclidean_distance_transform: bool,
         feature_size: int,
         downsample: Optional[float] = None,
         save_format: Optional[str] = 'all',
         tqdm=True):
    """
    MAIN MODULE FOR IMAGE POST-PROCESSING

    Post-process dataset from Unet segmentation and save them as .csv, .npy or .am
    """
    """Check dir for compatible files"""
    idx_img = [f for f in listdir(postprocess_dataset) if f.endswith('.tif')]
    assert len(idx_img) > 0, \
        f'{postprocess_dataset} direcotry do not contain .tif files'

    """Setting up pos-processing"""
    post_processer = ImageToPointCloud(tqdm=True)
    feature_size = feature_size / 10  # Value for edt threshold

    if save_format in ['am', 'all']:
        am_convert = NumpyToAmira()

    if tqdm:
        from tqdm import tqdm
        batch_iter = tqdm(idx_img,
                          'Image post-processing to point clouds')
    else:
        batch_iter = idx_img

    """For each file run post-processing"""
    for idx in batch_iter:
        """Check file type and correct to uin8 (aka 01 binnary type"""
        image = check_uint8(tif.imread(join(postprocess_dataset, idx)))

        """Post-processing"""
        if downsample is None:
            point_cloud_HD = post_processer(image=image,
                                            euclidean_transform=euclidean_distance_transform,
                                            label_size=feature_size,
                                            down_sampling_voxal_size=None)
            point_cloud_LD = None
        else:
            point_cloud_HD, point_cloud_LD = post_processer(image=image,
                                                            euclidean_transform=euclidean_distance_transform,
                                                            label_size=feature_size,
                                                            down_sampling_voxal_size=downsample)

        """Save point cloud"""
        if save_format in ['csv', 'all']:
            np.savetxt(fname=join(postprocess_dataset, f'{idx[:-4]}_HD.csv'),
                       X=point_cloud_HD,
                       delimiter=',')
            if downsample is not None:
                np.savetxt(fname=join(postprocess_dataset, f'{idx[:-4]}_LD.csv'),
                           X=point_cloud_LD,
                           delimiter=',')

        if save_format in ['npy', 'all']:
            np.save(file=join(postprocess_dataset, f'{idx[:-4]}_HD.npy'),
                    arr=point_cloud_HD)
            if downsample is not None:
                np.save(file=join(postprocess_dataset, f'{idx[:-4]}_LD.npy'),
                        arr=point_cloud_LD)

        if save_format in ['am', 'all']:
            am_convert.export_amira(coord=point_cloud_HD,
                                    file_dir=join(postprocess_dataset, f'{idx[:-4]}_HD.am'))
            if downsample is not None:
                am_convert.export_amira(coord=point_cloud_HD,
                                        file_dir=join(postprocess_dataset, f'{idx[:-4]}_LD.am'))


if __name__ == '__main__':
    main()
