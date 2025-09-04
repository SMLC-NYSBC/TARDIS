#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################

from scipy import optimize
from scipy.ndimage import rotate, zoom, affine_transform
import numpy as np


class VolumeRidgeRegistration:
    def __init__(
        self,
        method="cobyla",
        optimize_fn="mi",
        ridge_operation="trs",
        proj_method="top_bottom",
        crop=True,
        return_volume=False,
        down_scale=6,
        log_=False,
    ):
        method = method.lower()
        optimize_fn = optimize_fn.lower()
        ridge_operation = ridge_operation.lower()
        proj_method = proj_method.lower()

        assert any(
            ch in ridge_operation for ch in "trs"
        ), "At least one ridge operation must be supported."
        assert optimize_fn in [
            "mi",
            "mse",
            "ncc",
            "l2",
        ], "optimize_fn must be 'mi' or 'mse'"
        assert method in [
            "nelder-mead",
            "powell",
            "l-bfgs-b",
            "cobyla",
            "cobyqa",
            "slsqp",
            "tnc",
        ], f"{method} is not a valid method"

        assert proj_method in ["find", "top_bottom", "top_top"]

        self.method = method
        self.optimize_fn = optimize_fn
        self.ridge_operation = ridge_operation
        self.proj_method = proj_method

        self.crop = crop
        self.return_volume = return_volume

        self.down_scale = down_scale

        self.angle, self.ty, self.tx, self.scale, self.score = 0.0, 0.0, 0.0, 1.0, 0.0

        self.log_ = log_

    def volume_to_projection(self, img1, img2):
        assert self.proj_method in ["top_bottom", "top_top"]

        im1_pos = int(img1.shape[0] * 0.05)
        im2_pos = int(img2.shape[0] * 0.05)

        if self.proj_method == "top_bottom":
            img1 = np.sum(img1[-im1_pos:, ...], axis=0)  # Top 5% from img1
            img2 = np.sum(img2[:im2_pos, ...], axis=0)  # Bottom 5% from img2
        else:
            img1 = np.sum(img1[-im1_pos:, ...], axis=0)  # Top 5% from img1
            img2 = np.sum(img2[:-im2_pos, ...], axis=0)  # Top 5% from img2

        return img1.astype(np.float32), img2.astype(np.float32)

    def transformed_image_size(self, cy, cx):
        img = np.ones((cy, cx), dtype=np.uint8)
        img = rotate(img, self.angle, reshape=True, mode="constant", cval=0)
        img = img.shape

        if self.ty < 0:
            cy = (int(((img[0] - cy) // 2) + abs(self.ty)), int(((img[0] - cy) // 2)))
        else:
            cy = (int(((img[0] - cy) // 2)), int(((img[0] - cy) // 2) + abs(self.ty)))

        if self.tx < 0:
            cx = (int(((img[1] - cx) // 2) + abs(self.tx)), int(((img[1] - cx) // 2)))
        else:
            cx = (int(((img[1] - cx) // 2)), int(((img[1] - cx) // 2) + abs(self.tx)))

        return cy, cx

    def loss_function_l2(self, vol1, vol2):
        loss = -np.mean((vol1 - vol2) ** 2, axis=(0, 1)).mean()

        if self.log_:
            print(loss)

        return loss

    def loss_function_mean_squared_error(self, img1, img2):
        """Compute Mean Squared Error between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MSE")
        valid_mask = np.where(img2 != 0)
        img1 = img1.astype(float)[valid_mask]
        img2 = img2.astype(float)[valid_mask]
        loss = np.mean((img1 - img2) ** 2)

        if self.log_:
            print(loss)

        return loss

    def loss_function_mutual_information(self, img1, img2, bins=64):
        """Compute Mutual Information between two images."""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for MI")

        # Normalize images to [0, 1] for histogram
        img1 = img1.astype(float)
        img2 = img2.astype(float)
        valid_mask = np.where(img2 != 0)

        img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1) + 1e-10)
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2) + 1e-10)

        # Compute joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(
            img1[valid_mask].ravel(),
            img2[valid_mask].ravel(),
            bins=bins,
            range=[[0, 1], [0, 1]],
        )
        hist_2d = hist_2d / (np.sum(hist_2d) + 1e-10)  # Normalize to probability

        # Marginal histograms
        hist_1 = np.sum(hist_2d, axis=1)
        hist_2 = np.sum(hist_2d, axis=0)

        # Entropies
        H1 = -np.sum(hist_1 * np.log2(hist_1 + 1e-10))
        H2 = -np.sum(hist_2 * np.log2(hist_2 + 1e-10))
        H12 = -np.sum(hist_2d * np.log2(hist_2d + 1e-10))

        # Mutual Information
        loss = -(H1 + H2 - H12)

        if self.log_:
            print(loss)

        return loss

    def loss_function_normalized_cross_correlation(self, img1, img2):
        """Compute Normalized Cross-Correlation between two images."""
        # Ensure images are same size
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions for NCC")

        # Flatten images and normalize
        img1 = img1.astype(float)
        img2 = img2.astype(float)

        valid_mask = np.where(img2 != 0)

        # Subtract mean and compute standard deviation
        img1 = img1 - np.mean(img1)
        img2 = img2 - np.mean(img2)
        std1 = np.std(img1)
        std2 = np.std(img2)

        # Avoid division by zero
        if std1 == 0 or std2 == 0:
            return 0.0

        # Compute NCC
        loss = -(
            np.sum(img1[valid_mask] * img2[valid_mask])
            / (std1 * std2 * img1[valid_mask].size)
        )

        if self.log_:
            print(loss)

        return loss

    def compute_transformation_matrix(self, cy, cx):
        # Rotate with reshape=True to accommodate full rotated image

        # Convert angle to radians
        theta = np.deg2rad(self.angle)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        # Affine matrix for rotation + translation
        cy, cx = cy / 2, cx / 2

        # Translate image center to origin
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

        # Rotation matrix
        R = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

        # Scaling (zoom)
        S = np.array([[self.scale, 0, 0], [0, self.scale, 0], [0, 0, 1]])

        # Translate back and apply user translation
        T2 = np.array([[1, 0, cx + self.tx], [0, 1, cy + self.ty], [0, 0, 1]])

        # Combined matrix: move -> scale -> rotate -> move back + translate
        return T2 @ R @ S @ T1

    def apply_rigid_transform(self, img, cval=False):
        """Apply rotation and translation to an image without cropping."""
        if cval:
            min_ = img.min()
            if min_ > 0:
                min_ = 0
        else:
            min_ = 0

        if img.ndim == 2:
            M = self.compute_transformation_matrix(*img.shape)

            img = affine_transform(
                img,
                M[:2, :2],  # 2x2 rotation matrix
                offset=M[:2, 2],  # Combine offset and translation
                output_shape=img.shape,
                order=1,
                mode="constant",
                cval=min_,
            )
        else:
            M = self.compute_transformation_matrix(*img.shape[1:])

            for i in range(img.shape[0]):
                img[i] = affine_transform(
                    img[i],
                    M[:2, :2],  # 2x2 rotation matrix
                    offset=M[:2, 2],  # Combine offset and translation
                    output_shape=img.shape[1:],
                    order=1,
                    mode="constant",
                    cval=min_,
                )

        return img

    def find_best_ridge_transformation(self, params, fixed, moving):
        """Objective function to minimize (negative NCC)."""
        self.angle, self.tx, self.ty, self.scale = params
        moving = self.apply_rigid_transform(moving)

        if self.optimize_fn == "mi":
            return self.loss_function_mutual_information(fixed, moving, 64)
        elif self.optimize_fn == "mse":
            return self.loss_function_mean_squared_error(fixed, moving)
        elif self.optimize_fn == "l2":
            return self.loss_function_l2(fixed, moving)
        else:
            return self.loss_function_normalized_cross_correlation(fixed, moving)

    def find_best_proj_method(self, fixed_vol, moving_vol):
        best_scores = {"top_bottom": 0.0, "top_top": 0.0}

        best_metric = {}
        for i in best_scores.keys():
            self.proj_method = i
            self.align_images(*self.volume_to_projection(fixed_vol, moving_vol))
            best_scores[i] = np.abs(self.score)
            best_metric[i] = self.get_transformation_metrics()

        best_proj_method = min(best_scores, key=best_scores.get)
        self.angle, self.ty, self.tx, self.scale, self.score = best_metric[
            best_proj_method
        ].values()

        return min(best_scores, key=best_scores.get)

    def align_images(self, fixed_img, moving_img):
        fixed_img = zoom(fixed_img, 1 / self.down_scale)
        moving_img = zoom(moving_img, 1 / self.down_scale)

        """Align moving_img to fixed_img using intensity-based registration."""
        # Initial guess: no rotation, no translation
        initial_params = np.array([
            self.angle,
            self.ty,
            self.tx,
            self.scale,
        ])  # [angle, tx, ty, Scale]

        # Define bounds for optimization
        bounds = [
            (-180.0, 180.0) if "r" in self.ridge_operation else (0.0, 0.0),
            (
                (-moving_img.shape[1] // 4, moving_img.shape[1] // 4)
                if "t" in self.ridge_operation
                else (0.0, 0.0)
            ),
            (
                (-moving_img.shape[0] // 4, moving_img.shape[0] // 4)
                if "t" in self.ridge_operation
                else (0.0, 0.0)
            ),
            (0.9, 1.1) if "s" in self.ridge_operation else (1.0, 1.0),
        ]

        # Optimize transformation parameters
        result = optimize.minimize(
            self.find_best_ridge_transformation,
            initial_params,
            args=(fixed_img, moving_img),
            method=self.method,
            bounds=bounds,
        )

        # Extract optimal parameters
        self.angle, self.ty, self.tx, self.scale = result.x
        self.ty /= 1 / self.down_scale
        self.tx /= 1 / self.down_scale

        self.score = result.fun

    def get_transformation_metrics(self):
        return dict(
            zip(
                ["Angle", "Tx", "Ty", "Scale", "Score"],
                [self.angle, self.ty, self.tx, self.scale, self.score],
            )
        )

    def get_ridge_transform(self, moving_vol):
        if self.return_volume:
            if not self.crop:
                cy, cx = self.transformed_image_size(*moving_vol.shape[1:])
                moving_vol = np.pad(
                    moving_vol, ((0, 0), cy, cx), mode="constant", constant_values=0
                )

            return self.apply_rigid_transform(moving_vol, cval=True)
        else:
            if self.proj_method in ["bottom_top", "top_top"]:
                moving_vol = moving_vol[-1, ...]
            elif self.proj_method in ["bottom_bottom", "top_bottom"]:
                moving_vol = moving_vol[0, ...]
            else:
                moving_vol = moving_vol[moving_vol.shape[0] // 2, ...]

            if not self.crop:
                cy, cx = self.transformed_image_size(*moving_vol.shape)
                moving_vol = np.pad(
                    moving_vol, (cy, cx), mode="constant", constant_values=0
                )

            return self.apply_rigid_transform(moving_vol, cval=True)

    def __call__(self, fixed_vol, moving_vol, moving_coord=None):
        if self.proj_method == "find":
            self.proj_method = self.find_best_proj_method(fixed_vol, moving_vol)

            if self.proj_method == "top_top":
                moving_vol = moving_vol[::-1, ...]
                self.proj_method = "top_bottom"
        else:
            self.align_images(*self.volume_to_projection(fixed_vol, moving_vol))

        return self.get_ridge_transform(moving_vol)
