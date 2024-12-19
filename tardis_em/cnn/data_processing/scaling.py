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
    Scales the input image and/or mask to the specified dimensions. Depending on the sizes
    given in `scale` compared to the original dimensions of the image or mask, down-sampling
    or up-sampling is applied. When both image and mask are provided, both are processed
    and returned along with the dimensional value. If only one of them (image or mask) is
    provided, it is scaled and returned with the dimensional value. If neither is provided,
    a `None` result is returned with the dimensional value.

    :param scale: A tuple of integers specifying the desired dimensions for scaling. Can
        be of length 2 (height and width) or 3 (height, width, and depth).
    :param image: Optional. A NumPy array representing the image to be scaled. If
        provided, it is processed and scaled according to the provided `scale`.
    :param mask: Optional. A NumPy array representing the mask to be scaled. If
        provided, it is processed and scaled according to the provided `scale`.
    :param device: An optional argument specifying the device to perform scaling
        operations (e.g., `torch.device('cpu')` or `torch.device('cuda')`).

    :return: A tuple containing either:
        - `(image, mask, dim)` if both `image` and `mask` are provided and processed.
        - `(image, dim)` if only `image` is provided and processed.
        - `(mask, dim)` if only `mask` is provided and processed.
        - `(None, dim)` if neither `image` nor `mask` are provided.
    """
    dim = 1

    if len(scale) == 3:
        scale = tuple([scale[0], scale[1], scale[2]])
    else:
        scale = tuple([scale[0], scale[1]])

    if image is not None:
        image = image.astype(np.float32)
        image = np.ascontiguousarray(image)

        if not np.all(scale == image.shape):
            if scale[0] < image.shape[0]:  # Down-sampling
                image = linear_scaling(img=image, scale=scale, device=device)
            else:  # Up-sampling
                image = nn_scaling(
                    img=image,
                    scale=scale,
                    device=device,
                    gauss=True,
                )

    if mask is not None:
        mask = mask.astype(np.float32)
        mask = np.ascontiguousarray(mask)

        if not np.all(scale == mask.shape):
            mask = linear_scaling(img=mask, scale=scale)

    if image is not None and mask is not None:
        return image, mask, dim
    elif image is None and mask is not None:
        return mask, dim
    else:
        return image, dim


def nn_scaling(
    img: np.ndarray, scale: tuple, device=torch.device("cpu"), gauss=False
) -> np.ndarray:
    """
    Scale an image using nearest-neighbor interpolation, with an optional Gaussian
    filter applied before scaling.

    This function performs image resizing using PyTorch's interpolation method
    with mode set to "nearest". If the `gauss` parameter is set to True, a Gaussian
    filter is applied before resizing to smoothen the image.

    :param img: Input image as a NumPy array.
    :param scale: New dimensions for the image as a tuple (height, width).
    :param device: PyTorch device to perform operations on. Defaults to "cpu".
    :param gauss: Optional boolean flag to apply Gaussian filtering before scaling.
                  Defaults to False.
    :return: Rescaled image as a NumPy array.
    """
    if gauss:
        filter_ = img.shape[0] / scale[0]  # Scaling factor
        img = gaussian_filter(img, filter_ / 2)

    img = torch.from_numpy(img)[None, None, ...].to(device)
    img = F.interpolate(img, size=scale, mode="nearest")[0, 0, ...]

    return img.cpu().detach().numpy()


def linear_scaling(
    img: np.ndarray, scale: tuple, device=torch.device("cpu")
) -> np.ndarray:
    """
    Scales a 2D or 3D image to a desired resolution using trilinear or bilinear
    interpolation. The function performs the scaling in multiple steps, doubling or
    halving dimensions at each step, until the desired resolution is achieved or
    exceeded.

    :param img: Input image to be scaled. Can be 2D or 3D. Should be provided as
        a NumPy array.
    :param scale: Target resolution as a tuple of integers representing the
        desired dimensions. Must correspond to either (depth, height, width) for 3D
        images or (height, width) for 2D images.
    :param device: Device to use for the tensor operations (default is
        `torch.device("cpu")`). It accepts GPU devices (e.g., `torch.device("cuda")`).

    :return: A NumPy array representing the resized image, scaled to the target
        resolution.

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
    return img.detach().numpy()[0, 0, :]


def area_scaling(
    img: np.ndarray, scale: tuple, device=torch.device("cpu")
) -> np.ndarray:
    """
    Scales a 3D image array to a specified size along all axes using area-based
    interpolation. The function uses PyTorch's interpolation functionality to
    rescale the image accurately. This process first adjusts the Z-axis size,
    then scales along the XY plane to match the target scale. The function assumes
    input and output images to be 3D arrays. It takes a device parameter that
    leverages GPU acceleration if specified.

    :param img: Input 3D numpy array representing the image data to be scaled.
    :type img: np.ndarray
    :param scale: A tuple specifying the desired size of the scaled image
        (depth, height, width).
    :type scale: tuple
    :param device: The computation device to perform scaling. Defaults to CPU.
    :type device: torch.device

    :return: A scaled 3D numpy array.
    :rtype: np.ndarray
    """

    size_Z = [scale[0], img.shape[1], img.shape[2]]
    image_scale_Z = np.zeros(size_Z)

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
        )

    # Scale XY axis
    img = np.zeros(scale)
    for i in range(scale[0]):
        df_img = torch.from_numpy(image_scale_Z[i, :]).to(device).type(torch.float)
        img[i, :] = (
            F.interpolate(
                df_img[None, None, :], size=[int(s) for s in scale[1:]], mode="area"
            )
            .cpu()
            .detach()
            .numpy()[0, 0, :]
        )

    return img


def fourier_scaling(
    img: np.ndarray, scale: tuple, device=torch.device("cpu")
) -> np.ndarray:
    """
    Rescales the given image in the frequency domain to the specified spatial dimensions
    using Fourier Transform. The function accepts an image as a NumPy array, performs
    Fourier Transform, adjusts the frequency components to fit the target scale, and
    reconstructs the resized image via inverse Fourier Transform. The scaling is performed
    directly in the frequency domain, which helps preserve spatial details in the resized
    image.

    :param img: 2D or 3D image array to be processed.
    :type img: np.ndarray
    :param scale: Target dimensions for scaling the image. Must be a tuple.
    :type scale: tuple
    :param device: Computing device to perform the Fourier Transform operations.
    :type device: torch.device

    :return: Resized image array with dimensions matching the specified scale.
    :rtype: np.ndarray
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
    return torch.abs(resized_image).cpu().detach().numpy()
