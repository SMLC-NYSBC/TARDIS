from typing import Optional
import numpy as np
import edt
from skimage.morphology import skeletonize
import tifffile.tifffile as tifffile


class BuildPointCloud:
    def __init__(self):
        pass

    def _calculate_edt(self,
                       image: np.ndarray,
                       dim=3):

        if dim == 3:
            return edt.edt(image)
        if dim == 2:
            df_img = np.zeros(image.shape, dtype=np.float16)

            for i in range(image.shape[0]):
                df_img[i, :] = edt.edt(image[i, :])

            return df_img

    def _correct_with_edt(self,
                          image: np.ndarray,
                          threshold: float):
        img_edt_2d = np.zeros(image.shape, dtype=np.int32)

        for i in range(image.shape[0]):
            img_edt_2d[i, :] = edt.edt(image[i, :])

        return np.array(img_edt_2d > threshold, dtype=np.int8)

    def _skeletonize(self,
                     image: np.ndarray,
                     dim=3):
        if dim == 3:
            min_z, max_z = 1, image.shape[0] - 2
            skeleton = np.array(skeletonize(image=image[min_z:max_z, :],
                                            method='lee'), dtype=np.int8)

        if dim == 2:
            skeleton = np.array(skeletonize(image=image,
                                            method='zhang'), dtype=np.int8)

        # [Z x Y x X] or [Y x X]
        return np.vstack(np.where(skeleton > 0)).T

    def build_point_cloud(self,
                          image: Optional[str] = np.ndarray,
                          edt_correction: Optional[float] = None):
        if isinstance(image, str):
            image = np.array(tifffile.imread(image), dtype=np.int8)

        if edt_correction is not None:
            image = self._calculate_edt(image=image,
                                        dim=image.ndim)
            image = self._correct_with_edt(image=image,
                                           threshold=edt_correction)

        coord = self._skeletonize(image=image,
                                  dim=image.ndim)

        # [X x Y x Z]
        if image.ndim == 3:
            return np.vstack((coord[:, 2], coord[:, 1], coord[:, 0])).T
        else:
            return np.vstack((coord[:, 2],
                              coord[:, 1],
                              np.zeros((len(coord[:, 1], 1))))).T
