#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################
import time
from datetime import datetime
from os import mkdir
from os.path import isdir, split, splitext, join
from typing import Union
import cv2

import numpy as np
import tifffile.tifffile as tiff
from scipy import optimize
from scipy.ndimage import zoom, affine_transform, shift, gaussian_filter
from scipy.signal import correlate2d
from scipy.stats import entropy

from tardis_em import version
from tardis_em.stitch_volume.utils import sort_tomogram_files
from tardis_em.utils.export_data import to_am, to_mrc, NumpyToAmira
from tardis_em.utils.load_data import load_image, ImportDataFromAmira
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.normalization import MeanStdNormalize, RescaleNormalize


class AlignTomograms:
    def __init__(self, images_paths: list, coords_paths: list, output_path: str, method='sift'):
        self.path = split(images_paths[0])

        self.images_path = images_paths
        self.coords_path = coords_paths
        self.output_path = output_path
        self.method = method.lower()

        assert method in ['sift', 'warp', 'powell'], f'Method must be one of ["sift", "powell"], but got {method}'
        assert len(self.images_path) == len(self.coords_path), \
            (f'Image and coord path must have same length! '
             f'But images {len(self.images_path)} != coors {len(self.coords_path)}')

        if not isdir(self.output_path):
            mkdir(self.output_path, mode=0o777)

        self.eta_predict = "NA"
        self.tardis_progress = None
        self.title = "Fully-automatic alignment of tomograms and spatial graphs"
        self.log_prediction = None
        self.device = "CPU"
        self.down_scale = None

        self.create_headers()
        self.tardis_progress = TardisLogo()
        self.tardis_progress(title=self.title, text_2=f"Device: {self.device}")

        self.course_aligner = VolumeRidgeRegistration(method=self.method,
                                                      down_scale=1)
        self.save_am_coord = NumpyToAmira()

        self.fixed_data, self.moving_data = None, None
        self.accum_angle, self.accum_tx, self.accum_ty = 0., 0., 0.

    def create_headers(self):
        """
        Creates ASCII headers and initializes logging information for a tomogram alignment.
        """
        mapping = {
            "r": "rotation",
            "t": "translation",
            "s": "scaling"
        }

        len_coord = len([x for x in self.coords_path if x is not None])
        loss = "MSE" if self.method=='powell' else "L2"
        self.log_prediction = [
            "###############################################################################",
            "# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R)      #",
            f"# tardis_em v{version}                                                           #",
            f"# MIT License * 2021-{datetime.now().year} | Robert Kiewisz & Tristan Bepler                   #",
            "###############################################################################",
            "",
            "---",
            "Course Alignment Setting:",
            "---",
            "",
            "---Directories---",
            f"Input: {self.path}",
            f"Output: {self.output_path}",
            f'Detected: {len(self.images_path)} tomograms and {len_coord} coordinates files.',
            "",
            "---Alignment Parameters---",
            f"Alignment Optimization algorithm: {self.method}",
            f"Alignment Optimization loss function: {loss}",
            f"Alignment with transformations: rotation & translation & scaling",
            f"Down scale: {self.down_scale}",
            "",
            f"Device: {self.device}",
            "",
            "---",
            "Course Alignment Started:",
            "---",
            "",
            "---",
        ]

    def update_progress(self, idx, metric=None):
        if metric is not None:
            a = np.round(metric['Angle'], 2) if 'Angle' in metric else 'NA'
            x = np.round(metric['Tx'], 2) if 'Tx' in metric else 'NA'
            y = np.round(metric['Ty'], 2) if 'Ty' in metric else 'NA'
            sc = np.round(metric['Scale'], 2) if 'Scale' in metric else 'NA'
            so = np.round(metric['Score'], 2) if 'Score' in metric else 'NA'
        else:
            a = x = y = sc = so = 'NA'
        loss = "MSE" if self.method == 'powell' else "L2"

        self.tardis_progress(title=self.title,
                             text_1=f"Found {len(self.images_path) - 1} images to align! [{self.eta_predict} min ETA]",
                             text_2=f"Device: {self.device}",
                             text_3=f"Image {idx + 1}/{len(self.images_path) - 1}:",
                             text_5=f'  Running alignment with: {self.method} model and {loss} loss function...',
                             text_6=f"  Aligned done with: rotation|translation|scaling; {self.down_scale}x Scaling",
                             text_7=f"  Angle: {a}; Tx: {x}; Ty: {y}; Scale: {sc}; Score: {so}",
                             text_9=f"Aligning image {idx+1} to {idx}...",
                             text_10=print_progress_bar(idx+1, len(self.images_path)-1),
                             )

    def save_log(self):
        with open(join(self.output_path, "course_alignment_log.txt"), "w") as f:
            f.write(" \n".join(self.log_prediction))

    def load_tomogram_pairs(self, dir_img_1: str, dir_coord_1: str,
                            dir_img_2: str, dir_coord_2: str):
        if dir_img_1 is not None:
            self.fixed_data = {'Images': np.ndarray, 'Pixel_Size': float,
                               'Coordinates': Union[np.ndarray, None], 'Amira_Transformation': [0, 0, 0],
                               'Ridge_Transform': dict}

        if dir_img_2 is not None:
            self.moving_data = {'Images': np.ndarray, 'Pixel_Size': float,
                                'Coordinates': Union[np.ndarray, None], 'Amira_Transformation': [0, 0, 0],
                                'Ridge_Transform': dict}

        if dir_coord_1 is None:
            self.fixed_data['Coordinates'] = None
            if dir_img_1 is not None:
                img_1, px_1 = load_image(dir_img_1)
            else:
                img_1, px_1 = None, None
        else:
            if dir_coord_1.endswith(".am"):
                am = ImportDataFromAmira(dir_coord_1, dir_img_1)

                img_1, px_1 = am.get_image()
                self.fixed_data['Coordinates'] = am.get_segmented_points()
                self.fixed_data['Amira_Transformation'] = am.transformation
            else:
                self.fixed_data['Coordinates'] = np.genfromtxt(dir_coord_1, delimiter=",", skip_header=1)
                img_1, px_1 = load_image(dir_img_1)
        self.fixed_data['Image'] = img_1
        self.fixed_data['Pixel_Size'] = px_1

        if self.down_scale is None:
            z_dim, y_dim, x_dim = self.fixed_data['Image'].shape
            max_dim = max(z_dim, y_dim, x_dim)
            self.down_scale = 1
            while (max_dim / self.down_scale) > 500:
                 self.down_scale += 1
            self.course_aligner.down_scale = self.down_scale

            self.create_headers()
            self.update_progress(0, None)

        if dir_coord_2 is None:
            self.moving_data['Coordinates'] = None
            if dir_img_2 is not None:
                img_2, px_2 = load_image(dir_img_2)
            else:
                img_2, px_2 = None, None
        else:
            if dir_coord_2.endswith(".am"):
                am = ImportDataFromAmira(dir_coord_2, dir_img_2)

                img_2, px_2 = am.get_image()
                self.moving_data['Coordinates'] = am.get_segmented_points()
                self.moving_data['Amira_Transformation'] = am.transformation
            else:
                self.moving_data['Coordinates'] = np.genfromtxt(dir_coord_1, delimiter=",", skip_header=1)
                img_2, px_2 = load_image(dir_img_1)
        self.moving_data['Image'] = img_2
        self.moving_data['Pixel_Size'] = px_2

    def save_data(self, i: int):
        img_name = splitext(split(self.images_path[i])[-1])[0]
        img_format = splitext(split(self.images_path[i])[-1])[-1]

        new_image_name = img_name + f'_aligned{img_format}'
        new_image_name = join(self.output_path, new_image_name)

        if img_format == ".am":
            to_am(self.moving_data['Image'], self.moving_data['Pixel_Size'], new_image_name, None)
            self.log_prediction.append(f"    - Saved Tomogram Image Data as [{img_format}] file In: {new_image_name}")
        elif img_format in [".mrc", '.rec']:
            to_mrc(self.moving_data['Image'], self.moving_data['Pixel_Size'], new_image_name)
            self.log_prediction.append(f"    - Saved Tomogram Image Data [{img_format}] file In: {new_image_name}")
        elif img_format in [".tif", '.tiff']:
            tiff.imwrite(new_image_name, self.moving_data['Image'])
            self.log_prediction.append(f"    - Saved Tomogram Image Data [{img_format}] file In: {new_image_name}")
        else:
            self.log_prediction.append(f"    - Not Saved Image Data with [{img_format}] unrecognised")

        if self.coords_path[i] is None:
            self.images_path[i] = new_image_name
            return

        if self.moving_data['Coordinates'] is not None:
            img_name = splitext(split(self.images_path[i])[-1])[0]
            coord_name = splitext(split(self.coords_path[i])[-1])[0]
            coord_format = splitext(split(self.coords_path[i])[-1])[-1]

            coord_new_name = coord_name[:len(img_name)] + '_aligned' + coord_name[len(img_name):] + '.am'
            coord_new_name = join(self.output_path, coord_new_name,)

            if coord_format == ".csv":
                np.savetxt(coord_new_name, self.moving_data['Coordinates'], delimiter=",")
                self.log_prediction.append(f"    - Saved Coordinate Data [{coord_format}] file In:     {coord_new_name}")
            elif coord_format == '.am':
                # self.moving_data['Coordinates'][:, 1:] = self.moving_data['Coordinates'][:, 1:] * self.moving_data['Pixel_Size']

                self.save_am_coord.export_amiraV2(coord_new_name, self.moving_data['Coordinates'])
                self.log_prediction.append(f"    - Saved Coordinate Data [{coord_format}] file In:     {coord_new_name}")
            else:
                self.log_prediction.append(f"   -  Not Saved Coordinate Data with [{coord_format}] unrecognised")

        #     self.coords_path[i] = coord_new_name
        # self.images_path[i] = new_image_name

    def align_tomograms(self, i, metric):
        dir_img_1, dir_coord_1 = self.images_path[i], self.coords_path[i]
        dir_img_2, dir_coord_2 = self.images_path[i+1], self.coords_path[i+1]

        self.load_tomogram_pairs(dir_img_1, dir_coord_1, dir_img_2, dir_coord_2)
        self.log_prediction.append(f"# Aligning tomograms {i+1} from {len(self.images_path) - 1}:")
        self.log_prediction = self.log_prediction + ["  - Loaded fixed and moving data for alignment:",
                                                     f"     - Fix Tomogram: {self.images_path[i]}",
                                                     f"     - Fix Coordinate: {self.coords_path[i]}",
                                                     f"         - Pixel Size: {self.fixed_data['Pixel_Size']}",
                                                     f"         - Dim Tomogram: {self.fixed_data['Image'].shape} with {self.fixed_data['Image'].dtype} dtype",
                                                     f"         - Dim Coordinates: {self.fixed_data['Coordinates'].shape}",
                                                     f"     - Moving Tomogram: {self.images_path[i + 1]}",
                                                     f"     - Moving Coordinate: {self.coords_path[i + 1]}",
                                                     f"         - Pixel Size: {self.moving_data['Pixel_Size']}",
                                                     f"         - Dim Tomogram: {self.moving_data['Image'].shape} with {self.moving_data['Image'].dtype} dtype",
                                                     f"         - Dim Coordinates: {self.moving_data['Coordinates'].shape}",
                                                     "",
                                                     ]
        self.save_log()

        # Align tomogram n to tomogram n+1
        metric = self.course_aligner(self.fixed_data['Image'],
                                   self.moving_data['Image'],
                                   self.moving_data['Coordinates'],
                                   return_aligned=False,
                                   transform_fixed=metric)

        self.moving_data['Ridge_Transform'] = metric
        self.log_prediction = self.log_prediction + ["  - Finished aligning moving tomogram:",
                                                     f"     - Angle: {self.moving_data['Ridge_Transform']['Angle']:.2f}",
                                                     f"     - Tx: {self.moving_data['Ridge_Transform']['Tx']:.2f}",
                                                     f"     - Ty: {self.moving_data['Ridge_Transform']['Ty']:.2f}",
                                                     f"     - Scale: {self.moving_data['Ridge_Transform']['Scale']:.2f}",
                                                     f"     - Score: {self.moving_data['Ridge_Transform']['Score']:.2f}",
                                                     f"     - Aligned Tomogram Shape: {self.moving_data['Image'].shape} with {self.moving_data['Image'].dtype} dtype",
                                                     f"     - Aligned Coordinates Shape: {self.moving_data['Coordinates'].shape}",
                                                     "",
                                                     ]
        self.save_log()
        self.update_progress(i, metric)

        moving_vol = self.moving_data['Image'].shape
        # self.moving_data['Image'] = self.course_aligner.get_ridge_transform(self.moving_data['Image'])
        # self.moving_data['Coordinates'] = self.course_aligner.get_ridge_transform_coord(moving_vol,
        #                                                                                 self.moving_data['Coordinates'],
        #                                                                                 *self.moving_data['Image'].shape[1:])

        # Save tomogram n+1 under the same file format
        self.log_prediction.append("  - Saved aligned moving data:")
        self.save_log()
        # self.save_data(i+1)

        self.log_prediction = self.log_prediction + ["", "---", "",]
        self.save_log()
        return metric

    def align_all_tomograms(self):
        global_start = time.time()

        metric = None
        self.update_progress(0, metric)
        for idx in range(len(self.images_path) - 1):
            start = time.time()
            metric = self.align_tomograms(idx, metric)
            end = time.time()

            self.eta_predict = round(((end - start) * (len(self.images_path) - idx - 1)) / 60, 1)
            self.update_progress(idx, metric)

        dir_img_2, dir_coord_2 = self.images_path[0], self.coords_path[0]

        self.load_tomogram_pairs(None, None, dir_img_2, dir_coord_2)
        self.save_data(0)
        global_end = time.time()
        self.log_prediction = self.log_prediction + [
            '',
            "---",
            f"Total time for aligning all tomograms: {(global_end - global_start) / 60:.2f} minutes",
        ]

    def stitch_align_volumes(self):
        output_path_images, output_path_coords = sort_tomogram_files(self.output_path)

        stitched_ = []
        for i in output_path_images:
            if not i.endswith(("stitched_volume.am", "stitched_volume.mrc", "stitched_volume.rec", "stitched_volume.tif", "stitched_volume.tiff")):
                vol, px = load_image(i, False, True)
                stitched_.append(vol)
        stitched_ = np.concatenate(stitched_, axis=0)
        to_am(stitched_, px, join(self.output_path, 'stitched_volume.am'))

        self.log_prediction = self.log_prediction + ["",
                                                      "---",
                                                     "Stitched final course alignment:",
                                                     "---",
                                                     f"     - Final Tomogram Shape: {stitched_.shape}",
                                                     "",
                                                     ]
        self.save_log()

        stitched_ = []
        last_max_z = 0
        last_max_id = 0
        for i in output_path_coords:
            if i is None:
                continue

            coord = ImportDataFromAmira(i).get_segmented_points()
            coord[:, 0] += last_max_id
            coord[:, -1] += last_max_z
            stitched_.append(coord)

            last_max_id = coord[:, 0].max() + 1
            last_max_z = coord[:, -1].max() + 1

        stitched_ = np.concatenate(stitched_)
        self.save_am_coord.export_amiraV2(join(self.output_path, 'stitched_coord.am'), stitched_)
        self.log_prediction = self.log_prediction + [
                                                     f"     - Final Coordinate Shape: {stitched_.shape}",
                                                     f"     - Final MT number: {np.max(stitched_[:, 0])}",
                                                     ]
        self.save_log()


class VolumeRidgeRegistration:
    def __init__(
            self,
            method='sift',
            down_scale=6,
            log_=False,
    ):
        method = method.lower()
        assert method in ['sift', 'warp', 'powell'], f'Method must be one of ["sift", "powell"], but got {method}'
        self.mean_std = MeanStdNormalize()
        self.normalize = RescaleNormalize(clip_range=(.1, 99.9))

        self.method, self.optimize_fn = method, 'mse'

        self.down_scale = down_scale
        self.ridge_operation = 'rst'

        self.Angle, self.Ty, self.Tx, self.Scale, self.Score = 0.0, 0.0, 0.0, 1.0, 0.0

        self.mask_fix, self.mask_moving = None, None

        self.log_ = log_

    def volume_to_projection(self, img1, img2, original_=False, transform_fixed=None):
        im1_pos = int(img1.shape[0] * 0.05)
        im2_pos = int(img2.shape[0] * 0.05)

        img1 = np.sum(img1[-im1_pos:, ...], axis=0)
        img1 = zoom(img1, 1 / self.down_scale)
        img2 = np.sum(img2[:im2_pos, ...], axis=0)
        img2 = zoom(img2, 1 / self.down_scale)

        self.img2_y, self.img2_x = 0, 0
        if not original_:
            img1, img2 = gaussian_filter(img1, sigma=1.5), gaussian_filter(img2, sigma=1.5)
            img1 = self.normalize((self.mean_std(img1)).astype(np.float32))
            img1 = (img1 - img1.min()) / (img1.max() - img1.min())
            img1 = np.clip(img1, 0, 1)
            img2 = self.normalize((self.mean_std(img2)).astype(np.float32))
            img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            img2 = np.clip(img2, 0, 1)

            self.mask_fix = np.ones_like(img1, dtype=np.uint8)
            self.mask_moving = np.ones_like(img2, dtype=np.uint8)
            self.img2_y, self.img2_x = img2.shape
            self.img2_y, self.img2_x = int(self.img2_y / self.down_scale), int(self.img2_x / self.img2_y / self.down_scale)

        pad = int((np.sqrt(img1.shape[0] ** 2 + img1.shape[1] ** 2)) - img1.shape[0]) // 2
        img1 = np.pad(img1, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
        img2 = np.pad(img2, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        if not original_:
            self.mask_fix = np.pad(self.mask_fix, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
            self.mask_moving = np.pad(self.mask_moving, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

        if transform_fixed is not None:
            img1 = self.apply_rigid_transform(img1,
                                              transform_fixed['Angle'],
                                              transform_fixed['Tx'] * 1 / self.down_scale,
                                              transform_fixed['Ty'] * 1 / self.down_scale,
                                              transform_fixed['Scale'])

            if not original_:
                self.mask_fix = self.apply_rigid_transform(self.mask_fix,
                                                  transform_fixed['Angle'],
                                                  transform_fixed['Tx'] * 1 / self.down_scale,
                                                  transform_fixed['Ty'] * 1 / self.down_scale,
                                                  transform_fixed['Scale'])
                self.mask_fix = np.where(self.mask_fix > 0, 1, 0)

        return img1, img2

    @staticmethod
    def compute_padding(h, w, angle, tx, ty, scale, _2d=True):
        # 1. Scale
        w_s, h_s = w * scale, h * scale

        # 2. Rotate
        theta = np.deg2rad(angle)
        w_r = abs(w_s * np.cos(theta)) + abs(h_s * np.sin(theta))
        h_r = abs(w_s * np.sin(theta)) + abs(h_s * np.cos(theta))

        # 3. Translate
        pad_left = (w_r - w) / 2 + max(0, tx)
        pad_right = (w_r - w) / 2 + max(0, -tx)
        pad_H = int(max(pad_left, pad_right))

        pad_top = (h_r - h) / 2 + max(0, ty)
        pad_bottom = (h_r - h) / 2 + max(0, -ty)
        pad_W = int(max(pad_top, pad_bottom))

        if _2d:
            return (pad_H, pad_H), (pad_W, pad_W)
        return (0, 0), (pad_H, pad_H), (pad_W, pad_W)

    @staticmethod
    def translate_point_cloud(points, angle_deg, distance, inv_=False):
        theta = np.deg2rad(angle_deg)
        dx = distance * np.cos(theta)
        dy = distance * np.sin(theta)

        points_copy = points.copy()
        if inv_:
            points_copy[:, 1] += dy  # X
            points_copy[:, 2] += dx  # Y
        else:
            points_copy[:, 1] += dx  # X
            points_copy[:, 2] += dy  # Y

        return points_copy

    def loss_function_ssim(self, img1, img2, valid_mask, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
        img1 = img1.astype(float)[valid_mask]
        img2 = img2.astype(float)[valid_mask]
        # Gaussian blur for stability
        mu1 = gaussian_filter(img1, sigma)
        mu2 = gaussian_filter(img2, sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        loss = -np.mean(ssim_map)

        if self.log_:
            print(loss)
        return loss

    def loss_function_l2(self, vol1, vol2, valid_mask):
        vol1 = vol1.astype(float)[valid_mask]
        vol2 = vol2.astype(float)[valid_mask]

        loss = np.linalg.norm(vol1 - vol2)

        if self.log_:
            print(loss)
        return loss

    def loss_function_mean_squared_error(self, img1, img2, valid_mask):
        """Compute Mean Squared Error between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MSE")
        img1 = img1.astype(float)[valid_mask]
        img2 = img2.astype(float)[valid_mask]

        loss = np.mean((img1 - img2) ** 2)

        if self.log_:
            print(loss)

        return loss

    def loss_function_mutual_information(self, img1, img2, valid_mask, bins=256):
        """Compute Mutual Information between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MI")

        # Normalize images to [0, 1] for histogra
        img1 = img1[valid_mask]
        img2 = img2[valid_mask]

        img1 = 255 * (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = 255 * (img2 - img2.min()) / (img2.max() - img2.min())

        # Compute joint histogram
        hist, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        joint_prob = hist / np.sum(hist)

        # Compute entropies with masking for zero probabilities
        p1 = np.sum(joint_prob, axis=1)
        p2 = np.sum(joint_prob, axis=0)
        h1 = entropy(p1)
        h2 = entropy(p2)
        h_joint = entropy(joint_prob.ravel())

        # Mutual Information
        loss = -(h1 + h2 - h_joint)

        if self.log_:
            print(loss)

        return loss

    def loss_function_normalized_cross_correlation(self, img1, img2, valid_mask):
        """Compute Normalized Cross-Correlation between two images."""
        # Ensure images are the same size
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for NCC")

        # Flatten images and normalize
        img1 = img1[valid_mask]
        img2 = img2[valid_mask]

        # Subtract mean and compute standard deviation
        img1 = (img1 - np.mean(img1)) / np.std(img1)
        img2 = (img2 - np.mean(img2)) / np.std(img2)
        corr = correlate2d(img1, img2, mode='valid')

        # Compute NCC
        loss = -(np.max(corr) / (img1.shape[0] * img1.shape[1]))

        if self.log_:
            print(loss)

        return loss

    @staticmethod
    def compute_transformation_matrix(cy, cx, angle, scale):
        # Convert angle to radians
        theta = np.deg2rad(angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Affine matrix for rotation + translation
        cy, cx = cy / 2, cx / 2

        # Translate image center to origin
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]])

        # Rotation matrix
        RS = np.array([[scale * cos_theta, -scale * sin_theta, 0],
                      [scale * sin_theta, scale * cos_theta, 0],
                      [0, 0, 1]])

        # Translate back and apply user translation
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]])

        # Combined matrix: move -> rotate -> scale -> move back + translate
        return T2 @ RS @ T1

    def apply_rigid_transform(self, img, angle, tx, ty, scale, cval=False, reshape=False):
        """Apply rotation and translation to an image without cropping."""
        if cval:
            min_ = img.min()
            if min_ > 0:
                min_ = 0
        else:
            min_ = 0

        if img.ndim == 2:
            if reshape:
                pad = self.compute_padding(img.shape[0], img.shape[1], angle, 0., 0., scale, _2d=True)
                img = np.pad(img, pad, 'constant', constant_values=cval)

            M = self.compute_transformation_matrix(*img.shape, angle, scale)
            M = np.linalg.inv(M)

            img = affine_transform(
                img,
                M[:2, :2],  # 2x2 rotation matrix
                offset=M[:2, 2],  # Combine offset and translation
                output_shape=img.shape,
                order=3,
                mode="constant",
                cval=min_,
            )

            if reshape:
                y_max, x_max = int(np.abs(ty)), int(np.abs(tx))
                img = np.pad(img, ((y_max, y_max), (x_max, x_max)), 'constant', constant_values=cval)
            img = shift(img,
                        (ty, tx),
                        order=3,
                        mode="constant",
                        cval=min_,
                        )
        else:
            if reshape:
                pad = self.compute_padding(img.shape[1], img.shape[2], angle, 0., 0., scale, _2d=False)
                img = np.pad(img, pad, 'constant', constant_values=cval)

            M = self.compute_transformation_matrix(*img.shape[1:], angle, scale)
            M = np.linalg.inv(M)

            for i in range(img.shape[0]):
                img[i] = affine_transform(
                    img[i],
                    M[:2, :2],  # 2x2 rotation matrix
                    offset=M[:2, 2],  # Combine offset and translation
                    output_shape=img.shape[1:],
                    order=3,
                    mode="constant",
                    cval=min_,
                )

            if reshape:
                y_max, x_max = int(np.abs(ty)), int(np.abs(tx))
                img = np.pad(img, ((0, 0), (y_max, y_max), (x_max, x_max)), 'constant', constant_values=cval)

            for i in range(img.shape[0]):
                img[i] = shift(img[i],
                               (ty, tx),
                               order=3,
                               mode="constant",
                               cval=min_,
                               )
        return img

    def find_best_ridge_transformation(self, params, fixed, moving):
        """Objective function to minimize (negative NCC)."""
        moving_T = self.apply_rigid_transform(moving, *params)

        # Pad image borders for interpolation artefacts
        moving_T[:2, :] = 0  # top row
        moving_T[-2:, :] = 0  # bottom row
        moving_T[:, :2] = 0  # left column
        moving_T[:, -2:] = 0  # right column

        valid_mask = np.where(self.apply_rigid_transform(self.mask_moving, *params) > 0, 1, 0)
        valid_mask = np.where(np.logical_and(valid_mask != 0, self.mask_fix != 0))
        mean_, std_ = np.mean(moving_T[valid_mask]), np.std(moving_T[valid_mask])
        moving_T[valid_mask] = (moving_T[valid_mask] - mean_) / std_

        if self.optimize_fn == "mi":
            return self.loss_function_mutual_information(fixed, moving_T, valid_mask)
        elif self.optimize_fn == "mse":
            return self.loss_function_mean_squared_error(fixed, moving_T, valid_mask)
        elif self.optimize_fn == "l2":
            return self.loss_function_l2(fixed, moving_T, valid_mask)
        elif self.optimize_fn == "ncc":
            return self.loss_function_normalized_cross_correlation(fixed, moving_T, valid_mask)
        elif self.optimize_fn == "ssim":
            return self.loss_function_ssim(fixed, moving_T, valid_mask)
        elif self.optimize_fn == "mse_mi":
            return (self.loss_function_mean_squared_error(fixed, moving_T, valid_mask)
                    + self.loss_function_mutual_information(fixed, moving_T, valid_mask))

    def optim_align_images(self, fixed_img, moving_img):
        """Align moving_img to fixed_img using intensity-based registration."""
        if self.method == "sift":
            fixed_img = np.clip(fixed_img * 255.0, 0, 255).astype(np.uint8)
            self.mask_fix = np.clip(self.mask_fix, 0, 1).astype(np.uint8)
            moving_img = np.clip(moving_img * 255.0, 0, 255).astype(np.uint8)
            self.mask_moving = np.clip(self.mask_moving, 0, 1).astype(np.uint8)

            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_sift(fixed_img, moving_img)
        if self.method == "warp":
            self.Angle, self.Tx, self.Ty, self.Scale = self.align_images_warp(fixed_img, moving_img)

        if (self.Angle == 0 and self.Tx == 0 and self.Ty == 0 and self.Scale == 1) or self.method == 'powell':
            # Initial guess: no rotation, no translation
            initial_params = [self.Angle, self.Tx, self.Ty, self.Scale]  # [angle, tx, ty, Scale]

            # Define bounds for optimization
            bounds = [
                (-180.0, 180.0) if "r" in self.ridge_operation else (0.0, 0.0),
                (
                    (-self.img2_x // 4, self.img2_x // 4)
                    if "t" in self.ridge_operation
                    else (0.0, 0.0)
                ),
                (
                    (-self.img2_y // 4, self.img2_y // 4)
                    if "t" in self.ridge_operation
                    else (0.0, 0.0)
                ),
                (0.9, 1.1) if "s" in self.ridge_operation else (1.0, 1.0),
            ]

            np.random.seed(42)
            result = optimize.minimize(
                self.find_best_ridge_transformation,
                initial_params,
                args=(fixed_img, moving_img),
                method='powell',
                bounds=bounds,
                tol=1e-9,
            )

            # Extract optimal parameters
            self.Angle, self.Tx, self.Ty, self.Scale = result.x
            self.Score = result.fun

        self.Ty /= 1 / self.down_scale
        self.Tx /= 1 / self.down_scale

    def align_images_sift(self, reference: np.ndarray, moving: np.ndarray, max_features=500, good_match_ratio=0.75):
        # Detect SIFT features and descriptors
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(reference, self.mask_fix)
        kp_mov, des_mov = sift.detectAndCompute(moving, self.mask_moving)

        # Match features using BFMatcher with a ratio test
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(des_mov, des_ref, k=2)  # Query: moving, Train: reference

        # Apply Lowe's ratio test for good matches
        good_matches = []
        for m, n in raw_matches:
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

        # Sort by distance and limit if needed
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

        if len(good_matches) < 4:  # Need at least 4 for similarity
            return 0, 0, 0, 1

        # Extract points
        pts_mov = np.float32([kp_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate similarity transformation (rotation, translation, scale) with RANSAC
        matrix, _ = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC)

        scale = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
        angle = np.arctan2(matrix[0, 1], matrix[0, 0]) * 180 / np.pi
        tx, ty = matrix[0, 2], matrix[1, 2]

        return angle, tx, ty, 1.0

    def align_images_warp(self, reference: np.ndarray, moving: np.ndarray, max_iter=5000, epsilon=1e-5):
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.zeros((3, 3), dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)
        cc, warp_matrix = cv2.findTransformECC(reference, moving, warp_matrix, warp_mode, criteria, self.mask_fix)

        angle = np.arctan2(warp_matrix[0, 1], warp_matrix[0, 0]) * 180 / np.pi
        tx, ty = warp_matrix[0, 2], warp_matrix[1, 2]
        scale = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[0, 1] ** 2)

        return angle, tx, ty, scale

    def get_transformation_metrics(self):
        return dict(
            zip(
                ["Angle", "Tx", "Ty", "Scale", "Score"],
                [self.Angle, self.Tx, self.Ty, self.Scale, self.Score],
            )
        )

    def get_ridge_transform(self, moving_vol, reshape=False):
        return self.apply_rigid_transform(moving_vol, self.Angle, self.Tx, self.Ty, self.Scale, cval=True, reshape=reshape)

    def get_ridge_transform_coord(self, moving_vol_shape, moving_coord, cy, cx):
        if moving_vol_shape[-2] != cy and moving_vol_shape[-1] != cx:
            adjust_padding = True
        else:
            adjust_padding = False

        if adjust_padding:
            pad_H, pad_W = self.compute_padding(*moving_vol_shape[1:], self.Angle, 0., 0., self.Scale)

            moving_coord[:, 1] = moving_coord[:, 1] + pad_H[0]
            moving_coord[:, 2] = moving_coord[:, 2] + pad_W[0]

        M = self.compute_transformation_matrix(cy, cx, self.Angle, 1.)
        M = np.linalg.inv(M)

        coords = np.ones((moving_coord.shape[0], 3))
        coords[:, 0] = np.copy(moving_coord[:, 1])
        coords[:, 1] = np.copy(moving_coord[:, 2])

        # Apply transformation
        coords = (M @ coords.T).T[:, :2]
        coords = np.column_stack((
            moving_coord[:, 0],  # label
            ((coords[:, 0] - (cx/2)) * self.Scale + (cx/2)),  # + self.tx,  # new_x
            ((coords[:, 1] - (cy/2)) * self.Scale + (cy/2)),  # + self.ty,  # new_y
            moving_coord[:, -1]  # z (unchanged)
        ))

        if 't' in self.ridge_operation and adjust_padding:
            y_max, x_max = np.abs(self.Ty) + 3, np.abs(self.Tx) + 3
            coords = self.translate_point_cloud(coords, -self.Angle, x_max)
            coords = self.translate_point_cloud(coords, self.Angle, y_max, inv_=True)

        coords[:, 1] += self.Tx
        coords[:, 2] += self.Ty

        return coords

    def update_log(self, log_: dict):
        for k, v in log_.items():
            setattr(self, k, v)

    def __call__(self, fixed_vol, moving_vol, moving_coord=None, return_aligned=True, transform_fixed=None):
        self.Angle, self.Ty, self.Tx, self.Scale, self.Score = 0.0, 0.0, 0.0, 1.0, 0.0

        self.optim_align_images(*self.volume_to_projection(fixed_vol, np.copy(moving_vol), original_=False, transform_fixed=transform_fixed))

        log_ = self.get_transformation_metrics()

        if return_aligned:
            moving_vol_shape = moving_vol.shape
            moving_vol = self.get_ridge_transform(moving_vol)

            if moving_coord is not None:
                moving_coord = self.get_ridge_transform_coord(moving_vol_shape,
                                                              moving_coord,
                                                              *moving_vol.shape[1:])

            return log_, moving_vol, moving_coord
        else:
            return log_


