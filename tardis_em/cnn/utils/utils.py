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


def scale_image(
    scale: tuple,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    nn=False,
    device="cpu",
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

    if nn:
        image = nn_scaling(img=image, scale=scale, dtype=type_i, device=device)
        return image, dim

    if image is not None:
        if not np.all(scale == image.shape):
            if scale[0] > image.shape[0]:
                if image.ndim == 3 and image.shape[2] != 3:  # 3D with Gray
                    image = area_scaling(
                        img=image, scale=scale, dtype=type_i, device=device
                    )
                else:
                    image = linear_scaling(
                        img=image, scale=scale, dtype=type_i, device=device
                    )
            else:
                image = pil_LANCZOS(
                    img=image.astype(np.float32), scale=scale, dtype=type_i
                )

    if mask is not None:
        if not np.all(scale == mask.shape):
            if scale[0] > mask.shape[0]:
                mask = linear_scaling(img=mask, scale=scale, dtype=type_m)
            else:
                mask = pil_LANCZOS(img=mask.astype(np.int16), scale=scale, dtype=type_m)

    if image is not None and mask is not None:
        return image, mask, dim
    elif image is None and mask is not None:
        return mask, dim
    else:
        return image, dim


def nn_scaling(
    img: np.ndarray, scale: tuple, dtype: np.dtype, device="cpu"
) -> np.ndarray:
    img = torch.from_numpy(img)[None, None, ...].to(device)
    img = F.interpolate(img, size=scale, mode="nearest")[0, 0, ...]

    return img.cpu().detach().numpy()


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

        img = F.interpolate(img, size=scale, mode="trilinear", align_corners=False)
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


def number_of_features_per_level(channel_scaler: int, num_levels: int) -> list:
    """
    Compute list of output channels for CNN.

    Features = channel_scaler * 2^k
        where:
        - k is layer number

    Args:
        channel_scaler (int): Number of initial input channels for CNN.
        num_levels (int): Number of channels from max_number_of_conv_layer().

    Returns:
        list: List of output channels.
    """
    return [channel_scaler * 2**k for k in range(num_levels)]


def max_number_of_conv_layer(
    img=None,
    input_volume=64,
    max_out=8,
    kernel_size=3,
    padding=1,
    stride=1,
    pool_size=2,
    pool_stride=2,
    first_max_pool=False,
) -> int:
    """
    Calculate maximum possible number of layers given image size.

    Based on the torch input automatically select number of convolution blocks,
    based on a standard settings.

        I = [(W - K + 2*P) / S] + 1
        Out_size = [(I - F) / S] + 1
        - W is the input volume
        - K is the Kernel size
        - P is the padding
        - S is the stride
        - F is the max pooling kernel size

    Args:
        img (np.ndarray, Optional): Tensor data from which size of is calculated.
        input_volume (int): Size of the multiplayer for the convolution.
        max_out (int): Maximal output dimension after max_pooling.
        kernel_size (int): Kernel size for the convolution.
        padding (int): Padding size for the convolution blocks.
        stride: (int) Stride for the convolution blocks.
        pool_size (int) Max Pooling kernel size.
        pool_stride (int): Max Pooling stride size.
        first_max_pool (bool): If first CNN block has max pooling.

    Returns:
        int: Maximum number of CNN layers.
    """
    if img is not None:
        # In case of anisotropic image select smallest size
        # to calculate max conv layers
        if len(img.shape) == 5:
            size = np.array(list(img.shape[2:]))
        elif len(img.shape) == 4:
            size = np.array(list(img.shape[1:]))
        else:
            size = np.array(list(img.shape))

        size = size[size.argmin()]
        input_volume = size

    max_layers = 0

    while input_volume >= max_out:
        img_window = ((input_volume - kernel_size + 2 * padding) / stride) + 1
        input_volume = ((img_window - pool_size) / pool_stride) + 1
        max_layers += 1

    if first_max_pool:
        max_layers -= 1

    return max_layers


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Simple image data normalizer between 0,1.

    Args:
        image: Image data set.

    Returns:
        np.ndarray: Normalized image between 0 and 1 values.
    """
    image_min = np.min(image)
    image_max = np.max(image)

    if image_min == 0 and image_max == 1:
        return image

    if image_min == 0:
        image = np.where(image > image_min, 1, 0)
    elif image_max == 0:
        image = np.where(image < image_max, 1, 0)

    return image


def check_model_dict(model_dict: dict) -> dict:
    """
    Check and rebuild model structure dictionary to ensure back-compatibility.

    Args:
        model_dict (dict): Model structure dictionary.

    Returns:
        dict: Standardize model structure dictionary.
    """
    new_dict = {}

    for key, value in model_dict.items():
        if key.endswith("type"):
            new_dict["cnn_type"] = value
        if key.endswith("cation"):
            new_dict["classification"] = value
        if key.endswith("_in") or key.startswith("in_"):
            new_dict["in_channel"] = value
        if key.endswith("_out") or key.startswith("out_"):
            new_dict["out_channel"] = value
        if key.endswith("size"):
            new_dict["img_size"] = value
        if key.endswith("dropout"):
            new_dict["dropout"] = value
        if key.endswith("layers"):
            new_dict["num_conv_layers"] = value
        if key.endswith("scaler") or key.endswith("multiplayer"):
            new_dict["conv_scaler"] = value
        if key.endswith("v_kernel"):
            new_dict["conv_kernel"] = value
        if key.endswith("padding"):
            new_dict["conv_padding"] = value
        if key.endswith("l_kernel"):
            new_dict["maxpool_kernel"] = value
        if key.endswith("components"):
            new_dict["layer_components"] = value
        if key.endswith("features"):
            new_dict["attn_features"] = value
        if key.endswith("group"):
            new_dict["num_group"] = value

    return new_dict
