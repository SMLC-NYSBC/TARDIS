#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.fft import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import gaussian_filter


def scale_image(
    scale: tuple,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    device=torch.device("cpu"),
) -> Union[
    Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, int], Tuple[None, int]
]:
    """
    Scale image module using torch GPU interpolation

    Expect 2D (XY/YX), 3D (ZYX)

    Args:
        image (np.ndarray, Optional): image data
        mask (np.ndarray, Optional): Optional binary mask image data
        scale (tuple): scale value for image
        device (str):
    """
    dim = 1
    type_i = image.dtype
    type_m = None
    if mask is not None:
        type_m = mask.dtype

    if len(scale) == 3:
        scale = tuple([scale[0], scale[1], scale[2]])
    else:
        scale = tuple([scale[0], scale[1]])

    if image is not None:
        image = np.ascontiguousarray(image)

        if not np.all(scale == image.shape):
            if scale[0] < image.shape[0]:
                image = linear_scaling(
                    img=image, scale=scale, dtype=type_i, device=device
                )
            else:
                image = nn_scaling(
                    img=image,
                    scale=scale,
                    dtype=type_i,
                    device=device,
                    gauss=True,
                )

    if mask is not None:
        mask = np.ascontiguousarray(mask)

        if not np.all(scale == mask.shape):
            if scale[0] < mask.shape[0]:
                mask = nn_scaling(img=mask, scale=scale, dtype=type_m)
            else:
                mask = fourier_scaling(
                    img=mask.astype(np.int16), scale=scale, dtype=type_m
                )

    if image is not None and mask is not None:
        return image, mask, dim
    elif image is None and mask is not None:
        return mask, dim
    else:
        return image, dim


def nn_scaling(
    img: np.ndarray, scale: tuple, dtype: np.dtype, device="cpu", gauss=False
) -> np.ndarray:
    if gauss:
        filter_ = img.shape[0] / scale[0]  # Scaling factor
        img = gaussian_filter(img, filter_ / 2)

    img = torch.from_numpy(img)[None, None, ...].to(device)
    img = F.interpolate(img, size=scale, mode="nearest")[0, 0, ...]

    return img.cpu().detach().numpy().astype(dtype)


def pil_LANCZOS(img: np.ndarray, scale: tuple, dtype: np.dtype) -> np.ndarray:
    """
    Scale a 3D image by first down-sampling in the XY direction and then in the Z direction.

    Args:
        img: image array.
        scale: Scale array size.
        dtype: Output dtype for scale array.

    Returns:
        no.ndarray: Up or Down scale 3D array.
    """
    if len(scale) == 3:
        new_depth, new_height, new_width = scale
        scaled = []

        # Down-sample each slice in the XY direction
        for i in range(img.shape[0]):
            image = Image.fromarray(img[i, ...])
            scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
            scaled.append(np.asarray(scaled_image, dtype=dtype))

        # Convert list to 3D array
        img = np.array(scaled, dtype=dtype)

        # Down-sample each slice in the Z direction if 3D
        if len(scale) == 3:
            scaled = np.zeros(scale, dtype=dtype)

            for i in range(img.shape[2]):
                image = Image.fromarray(img[..., i])
                scaled[:, :, i] = np.asarray(
                    image.resize((new_height, new_depth), Image.LANCZOS), dtype=dtype
                )
            img = scaled.astype(dtype)
    else:
        new_height, new_width = scale
        img = Image.fromarray(img)
        img = np.asarray(
            img.resize((new_width, new_height), Image.LANCZOS), dtype=dtype
        )

    return img.astype(dtype)


def linear_scaling(
    img: np.ndarray, scale: tuple, dtype: np.dtype, device="cpu"
) -> np.ndarray:
    """
    Scaling of 2D/3D array using trilinear method from pytorch

    Args:
        img: image array.
        scale: Scale array size.
        dtype: Output dtype for scale array.

    Returns:
        no.ndarray: Up or Down scale 3D array.
    """
    if img.ndim == 3:
        current_depth, current_height, current_width = (
            img.shape[0],
            img.shape[1],
            img.shape[2],
        )
        final_depth, final_height, final_width = scale

        img = torch.from_numpy(img[None, None, :]).to(device).type(torch.float)

        # Calculate the number of steps required to reach the final scale by halving/doubling
        # Use logarithm base 2 since scaling is done by 2x at each step
        height_steps = int(abs(torch.log2(torch.tensor(final_height / current_height))))
        width_steps = int(abs(torch.log2(torch.tensor(final_width / current_width))))
        depth_steps = int(abs(torch.log2(torch.tensor(final_depth / current_depth))))

        # Perform scaling in steps
        for _ in range(max(height_steps, width_steps, depth_steps)):
            # Calculate intermediate scale
            new_height = int(
                current_height * (2 if current_height < final_height else 0.5)
            )
            new_width = int(current_width * (2 if current_width < final_width else 0.5))
            new_depth = int(current_depth * (2 if current_depth < final_depth else 0.5))

            # Stop if the desired scale is reached or exceeded
            if (
                new_height >= final_height
                and new_width >= final_width
                and new_depth >= final_depth
            ):
                break

            # Resize image
            img = F.interpolate(
                img,
                size=(new_height, new_width, new_depth),
                mode="trilinear",
                align_corners=False,
            )

            # Update current dimensions
            current_depth, current_height, current_width = (
                new_depth,
                new_height,
                new_width,
            )

            # Stop if the desired scale is reached or exceeded
            if (
                current_height >= final_height
                and current_width >= final_width
                and current_depth >= final_depth
            ):
                break

        img = F.interpolate(
            img,
            size=(scale[0], scale[1], scale[2]),
            mode="trilinear",
            align_corners=False,
        )
    else:
        current_height, current_width = img.shape[0], img.shape[1]
        final_height, final_width = scale

        img = torch.from_numpy(img[None, None, :]).to(device).type(torch.float)

        # Calculate the number of steps required to reach the final scale by halving/doubling
        # Use logarithm base 2 since scaling is done by 2x at each step
        height_steps = int(abs(torch.log2(torch.tensor(final_height / current_height))))
        width_steps = int(abs(torch.log2(torch.tensor(final_width / current_width))))

        # Perform scaling in steps
        if max(height_steps, width_steps) > 0:
            for _ in range(max(height_steps, width_steps)):
                # Calculate intermediate scale
                new_height = int(
                    current_height * (2 if current_height < final_height else 0.5)
                )
                new_width = int(
                    current_width * (2 if current_width < final_width else 0.5)
                )

                # Stop if the desired scale is reached or exceeded
                if new_height >= final_height and new_width >= final_width:
                    break

                # Resize image
                img = F.interpolate(
                    img,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                )
                current_height, current_width = new_height, new_width

        img = F.interpolate(img, size=scale, mode="bilinear", align_corners=False)
    return img.detach().numpy()[0, 0, :].astype(dtype)


def area_scaling(
    img: np.ndarray, scale: tuple, dtype: np.dtype, device="cpu"
) -> np.ndarray:
    """
    Scaling of 3D array using area method from pytorch

    Args:
        img: 3D array.
        scale: Scale array size.
        dtype: Output dtype for scale array.

    Returns:
        no.ndarray: Up or Down scale 3D array.
    """

    size_Z = [scale[0], img.shape[1], img.shape[2]]
    image_scale_Z = np.zeros(size_Z, dtype=dtype)

    # Scale Z axis
    for i in range(img.shape[2]):
        df_img = torch.from_numpy(img[:, :, i]).to(device).type(torch.float)

        image_scale_Z[:, :, i] = (
            F.interpolate(
                df_img[None, None, :], size=[int(s) for s in size_Z[:2]], mode="area"
            )
            .cpu()
            .detach()
            .numpy()[0, 0, :]
            .astype(dtype)
        )

    # Scale XY axis
    img = np.zeros(scale, dtype=dtype)
    for i in range(scale[0]):
        df_img = torch.from_numpy(image_scale_Z[i, :]).to(device).type(torch.float)
        img[i, :] = (
            F.interpolate(
                df_img[None, None, :], size=[int(s) for s in scale[1:]], mode="area"
            )
            .cpu()
            .detach()
            .numpy()[0, 0, :]
            .astype(dtype)
        )

    return img


def fourier_scaling(
    img: np.ndarray, scale: tuple, dtype: np.dtype, device="cpu"
) -> np.ndarray:
    """
    Resize a 2D or 3D image using Fourier cropping.

    Check in Topaz how

    Parameters:
    img (np.ndarray): 2D or 3D array representing the image.
    scale (tuple): Desired shape (for 2D: (height, width), for 3D: (depth, height, width)).
    dtype (np.dtype): Desired output data type.

    Returns:
        np.ndarray: Resized image in the desired data type.
    """
    if not isinstance(scale, tuple):
        scale = tuple(scale)

    # Get the current shape
    org_shape = img.shape

    # Calculate start indices for cropping or padding
    start_indices = [(cs - s) // 2 for cs, s in zip(org_shape, scale)]

    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img.to(device)

    # Perform Fourier Transform
    f_image = fftn(img)

    # Shift the zero frequency component to the center
    f_image_shifted = fftshift(f_image)

    # Slicing logic for cropping or padding
    slices_current = [
        slice(max(0, start), max(0, start) + min(cs, s))
        for start, cs, s in zip(start_indices, org_shape, scale)
    ]
    slices_resized = [
        slice(max(0, -start), max(0, -start) + min(cs, s))
        for start, cs, s in zip(start_indices, org_shape, scale)
    ]

    resized_f_image_shifted = torch.zeros(scale, dtype=torch.complex64, device=device)
    resized_f_image_shifted[slices_resized] = f_image_shifted[slices_current]

    # Shift the zero frequency component back to the original position
    resized_f_image = ifftshift(resized_f_image_shifted)

    # Perform Inverse Fourier Transform
    resized_image = ifftn(resized_f_image)

    # Return the real part of the resized image
    return torch.abs(resized_image).cpu().detach().numpy().astype(dtype)
