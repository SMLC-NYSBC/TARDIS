#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################
import sys
import time
from os import listdir, getcwd
from os.path import isdir, isfile, join, dirname
from typing import Optional

import click
import numpy as np
import pandas as pd
import tifffile.tifffile as tif
import torch

from tardis_em.dist_pytorch.utils.utils import pc_median_dist
from tardis_em.dist_pytorch.datasets.patches import PatchDataSet
from tardis_em.dist_pytorch.dist import build_dist_network
from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud
from tardis_em.dist_pytorch.utils.segment_point_cloud import PropGreedyGraphCut
from tardis_em.cnn.data_processing.draw_mask import (
    draw_semantic_membrane,
    draw_instances,
)
from tardis_em.cnn.data_processing.stitch import StitchImages
from tardis_em.cnn.data_processing.trim import trim_with_stride
from tardis_em.cnn.datasets.dataloader import PredictionDataset
from tardis_em.cnn.cnn import build_cnn_network
from tardis_em.cnn.utils.utils import scale_image
from tardis_em.utils.aws import get_weights_aws
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.export_data import NumpyToAmira, to_am, to_mrc, to_stl
from tardis_em.utils.load_data import import_am, ImportDataFromAmira, load_image
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.normalization import MeanStdNormalize, RescaleNormalize
from tardis_em.utils.setup_envir import build_temp_dir, clean_up
from tardis_em.utils.spline_metric import (
    FilterSpatialGraph,
    SpatialGraphCompare,
    sort_by_length,
    ComputeConfidenceScore,
    length_list,
)
from tardis_em._version import version

try:
    from tardis_em.utils.ota_update import ota_update

    ota = ota_update(status=True)
except ImportError:
    ota = ""

# Pytorch CUDA optimization
torch.backends.cudnn.benchmark = True


class DataSetPredictor:
    """
    MAIN WRAPPER FOR PREDICTION MT/MEM WITH TARDIS-PYTORCH

    Args:
        predict (str): Dataset type name.
        dir_ (str): Dataset directory.
        feature_size (float): Optional size of the filament use to scale image.
        output_format (str): Two output format for semantic and instance prediction.
        patch_size (int): Image 3D crop size.
        cnn_threshold (float): Threshold for CNN model.
        dist_threshold (float): Threshold for DIST model.
        points_in_patch (int): Maximum number of points per patched point cloud.
        predict_with_rotation (bool): If True, CNN predict with 4 90* rotations.
        amira_prefix (str): Optional, Amira file prefix used for spatial graph comparison.
        filter_by_length (float): Optional, filter setting for filtering short splines.
        connect_splines (int): Optional, filter setting for connecting near splines.
        connect_cylinder (int): Optional, filter setting for connecting splines withing cylinder radius.
        amira_compare_distance (int): Optional, compare setting, max distance between two splines
        to consider them as the same.
        amira_inter_probability (float): Optional, compare setting, portability threshold
        to define comparison class.
        instances (bool): If True, run instance segmentation after semantic.
        device_ (str): Define computation device.
        debug (bool): If True, run in debugging mode.
    """

    def __init__(
        self,
        predict: str,
        dir_: str,
        binary_mask: bool,
        output_format: str,
        patch_size: int,
        cnn_threshold: float,
        dist_threshold: float,
        points_in_patch: int,
        predict_with_rotation: bool,
        amira_prefix: Optional[str],
        filter_by_length: Optional[float],
        connect_splines: Optional[int],
        connect_cylinder: Optional[int],
        amira_compare_distance: Optional[int],
        amira_inter_probability: Optional[float],
        instances: bool,
        device_: str,
        debug: bool,
        checkpoint: Optional[list] = None,
        feature_size: Optional[float] = None,
        correct_px: bool = False,
    ):
        if predict not in ["Filament", "Membrane2D", "Membrane", "Microtubule"]:
            TardisError(
                id_="01",
                py="tardis_em/utils/predictor.py",
                desc=f"TARDIS v.{version} supports only MT and Mem segmentation!",
            )
            sys.exit()

        # Directories and dataset info
        self.dir = dir_
        self.output_format = output_format
        self.predict = predict
        if self.predict == "Membrane2D":
            self.expect_2d = True
        else:
            self.expect_2d = False
        self.amira_prefix = amira_prefix
        self.checkpoint = checkpoint
        self.correct_px = correct_px

        # Pre-processing setting
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.points_in_patch = points_in_patch

        # Prediction setting
        self.cnn_threshold = cnn_threshold
        self.dist_threshold = dist_threshold
        self.rotate = predict_with_rotation

        # Global flags
        self.binary_mask = binary_mask
        self.predict_instance = instances
        self.device = get_device(device_)
        self.debug = debug

        """Initial Setup"""
        if debug:
            str_debug = " <Debugging Mode>"
        else:
            str_debug = ""

        # Initiate log output
        self.tardis_progress = TardisLogo()
        if self.predict_instance:
            if self.output_format.startswith("None"):
                self.title = (
                    f"Fully-automatic Instance {self.predict} segmentation module | {ota} "
                    f"{str_debug}"
                )
            else:
                self.title = (
                    f"Fully-automatic Semantic-Instance {self.predict} segmentation module | {ota} "
                    f"{str_debug}"
                )
        else:
            self.title = (
                f"Fully-automatic Semantic {self.predict} segmentation module | {ota} "
                f"{str_debug}"
            )

        # Check for spatial graph in folder from amira/tardis_em comp.
        self.amira_check = False
        if isdir(join(self.dir, "amira")):
            self.amira_check = True
            self.dir_amira = join(dir_, "amira")
            self.title = (
                f"Fully-automatic Instance {self.predict} segmentation module "
                f"with Amira comparison {str_debug}"
            )

        self.tardis_progress(title=self.title, text_2=f"Device: {self.device}")

        # Early stop if not semantic of instance was specified
        if self.output_format == "None_None":
            TardisError(
                id_="151",
                py="tardis_em/utils/predictor.py",
                desc=f"Require that at lest one output format is not None but {output_format} was given!",
            )
            sys.exit()

        if self.output_format.endswith("stl"):  # Check if not ARM64 machine
            import platform

            if platform.machine() == "aarch64":
                TardisError(
                    id_="151",
                    py="tardis_em/utils/predictor.py",
                    desc=f"STL output is not allowed on {platform.machine()} machine type!",
                )
                sys.exit()

        # Searching for available images for prediction
        available_format = (".tif", ".mrc", ".rec", ".am", ".map")
        omit_format = (
            "mask.tif",
            "mask.mrc",
            "mask.rec",
            "Correlation_Lines.am",
            "mask.am",
            "mask.map",
        )
        self.output = join(self.dir, "temp", "Predictions")
        self.am_output = join(self.dir, "Predictions")

        # Pickup files for the prediction
        if isdir(dir_):
            self.predict_list = [
                f
                for f in listdir(dir_)
                if f.endswith(available_format) and not f.endswith(omit_format)
            ]
        else:
            dir_ = dirname(self.dir)
            if dir_ == "":
                dir_ = getcwd()

            self.predict_list = [
                f
                for f in listdir(dir_)
                if f.endswith(available_format) and not f.endswith(omit_format)
            ]
            self.predict_list = [f for f in self.predict_list if self.dir.endswith(f)]

            # Update Dir paths
            self.dir = dir_
            self.output = join(self.dir, "temp", "Predictions")
            self.am_output = join(self.dir, "Predictions")

        # Tardis progress bar update
        if len(self.predict_list) == 0:
            TardisError(
                id_="12",
                py="tardis_em/utils/predictor.py",
                desc=f"Given {dir_} does not contain any recognizable file!",
            )
            sys.exit()
        else:
            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict!",
                text_2=f"Device: {self.device}",
                text_7="Current Task: Setting-up environment...",
            )

        """Build handler's"""
        self.click_px_overwrite, self.click_stored_px = None, None

        # Build handler's for reading data to correct format
        self.normalize = RescaleNormalize(clip_range=(1, 99))  # Normalize histogram
        self.mean_std = MeanStdNormalize()  # Standardize with mean and std

        # Sigmoid whole predicted image
        self.sigmoid = torch.nn.Sigmoid()

        # Build handler's for transforming data
        self.image_stitcher = StitchImages()
        self.post_processes = BuildPointCloud()

        # Build handler's for DIST input and output
        if self.predict_instance:
            self.patch_pc = PatchDataSet(
                max_number_of_points=points_in_patch, graph=False
            )

            if predict in ["Filament", "Microtubule"]:
                self.GraphToSegment = PropGreedyGraphCut(
                    threshold=dist_threshold, connection=999999999, smooth=True
                )

                self.filter_splines = FilterSpatialGraph(
                    connect_seg_if_closer_then=connect_splines,
                    cylinder_radius=connect_cylinder,
                    filter_short_segments=filter_by_length,
                )
                self.compare_spline = SpatialGraphCompare(
                    distance_threshold=amira_compare_distance,
                    interaction_threshold=amira_inter_probability,
                )
                self.score_splines = ComputeConfidenceScore()
            elif predict in ["Membrane2D", "Membrane"]:
                self.GraphToSegment = PropGreedyGraphCut(
                    threshold=dist_threshold, connection=999999999
                )

        # Build handler to output amira file
        self.amira_file = NumpyToAmira()

        """Build NN from checkpoints"""
        self.build_NN(NN=self.predict)

    def build_NN(self, NN: str):
        if NN == "Microtubule":
            self.normalize_px = 25
        else:
            self.normalize_px = 15

        if NN in ["Filament", "Microtubule"]:
            # Build CNN network with loaded pre-trained weights
            if not self.output_format.startswith("None") and not self.binary_mask:
                self.cnn = Predictor(
                    checkpoint=self.checkpoint[0],
                    network="fnet",
                    subtype="32",
                    model_type="microtubules",
                    img_size=self.patch_size,
                    sigmoid=False,
                    device=self.device,
                )

            # Build DIST network with loaded pre-trained weights
            if not self.output_format.endswith("None"):
                self.dist = Predictor(
                    checkpoint=self.checkpoint[1],
                    network="dist",
                    subtype="triang",
                    model_type="microtubules",
                    device=self.device,
                )
        elif NN in ["Membrane2D", "Membrane"]:
            # Build CNN network with loaded pre-trained weights
            if NN == "Membrane2D":
                if not self.output_format.startswith("None") and not self.binary_mask:
                    self.cnn = Predictor(
                        network="fnet",
                        subtype="32",
                        model_type="cryo_mem_2d",
                        img_size=self.patch_size,
                        sigmoid=False,
                        device=self.device,
                        _2d=True,
                    )

                # Build DIST network with loaded pre-trained weights
                if not self.output_format.endswith("None"):
                    self.dist = Predictor(
                        network="dist",
                        subtype="triang",
                        model_type="membrane_2d",
                        device=self.device,
                    )
            else:
                if not self.output_format.startswith("None") and self.binary_mask:
                    self.cnn = Predictor(
                        checkpoint=self.checkpoint[0],
                        network="fnet",
                        subtype="32",
                        model_type="cryo_mem",
                        img_size=self.patch_size,
                        sigmoid=False,
                        device=self.device,
                    )

                # Build DIST network with loaded pre-trained weights
                if not self.output_format.endswith("None"):
                    self.dist = Predictor(
                        checkpoint=self.checkpoint[1],
                        network="dist",
                        subtype="triang",
                        model_type="s3dis",
                        device=self.device,
                    )

    def handle_pixel_prompt(self, prompt_message):
        self.px = click.prompt(
            prompt_message,
            default=self.normalize_px,
            type=float,
        )
        if self.click_px_overwrite is None:
            self.click_px_overwrite = click.prompt(
                f"Is the pixel size {self.px} correct for all data: ",
                default=True,
                type=bool,
            )
            if self.click_px_overwrite:
                self.click_stored_px = self.px

    def load_data(self, id_name: str):
        # Build temp dir
        build_temp_dir(dir_=self.dir)

        # Load image file
        if id_name.endswith(".am"):
            self.image, self.px, _, self.transformation = import_am(
                am_file=join(self.dir, id_name)
            )
        else:
            self.image, self.px = load_image(join(self.dir, id_name))
            self.transformation = [0, 0, 0]

        # In case of unreadable pixel size, ask user
        if self.click_stored_px is not None and self.click_px_overwrite is not None:
            self.px = self.click_stored_px

        if self.px == 0:
            self.handle_pixel_prompt(
                f"Image file has pixel size {self.px}, that's obviously wrong... What is the correct value:",
            )
        elif self.px == 1:
            self.handle_pixel_prompt(
                f"Image file has pixel size {self.px}, that's maybe wrong... What is the correct value:",
            )
        elif self.correct_px:
            self.correct_px = False
            self.handle_pixel_prompt(
                f"Image file has pixel size {self.px}. What is the correct value: ",
            )

        # Normalize image histogram
        if not self.output_format.startswith("None") or not self.binary_mask:
            self.image = self.normalize(self.mean_std(self.image)).astype(np.float32)

            # Check image structure
            if (
                not self.image.min() >= -1 or not self.image.max() <= 1
            ):  # Image not between in -1 and 1
                if self.image.min() >= 0 and self.image.max() <= 1:
                    self.image = (self.image - 0.5) * 2  # shift to -1 - 1
                elif self.image.min() >= 0 and self.image.max() <= 255:
                    self.image = self.image / 255  # move to 0 - 1
                    self.image = (self.image - 0.5) * 2  # shift to -1 - 1

            if not self.image.dtype == np.float32:
                TardisError(
                    id_="11",
                    py="tardis_em/utils/predictor.py",
                    desc=f"Error while loading image {id_name}: "
                    f"Image loaded correctly, but output format "
                    f"{self.image.dtype} is not float32!",
                )
                sys.exit()
        else:
            # Check image structure
            self.image = np.where(self.image > 0, 1, 0).astype(np.int8)

            if not self.image.dtype == np.int8 or self.image.dtype == np.uint8:
                TardisError(
                    id_="11",
                    py="tardis_em/utils/predictor.py",
                    desc=f"Error while loading image {id_name}: "
                    f"Image loaded correctly, but output format "
                    f"{self.image.dtype} is not int8!",
                )
                sys.exit()

        # Calculate parameters for normalizing image pixel size
        """Note: Do not scale images of too big or small pixel size"""
        if self.predict == "Filament":
            self.scale_factor = self.normalize_px / self.feature_size
        else:
            if self.px > (self.normalize_px * 3) or self.px < (self.normalize_px / 4):
                self.scale_factor = 1.0
                self.px = self.normalize_px
            else:
                self.scale_factor = self.px / self.normalize_px

        self.org_shape = self.image.shape
        self.scale_shape = np.multiply(self.org_shape, self.scale_factor).astype(
            np.int16
        )
        self.scale_shape = [int(i) for i in self.scale_shape]

    def postprocess_CNN(self, id_name: str):
        # Stitch predicted image patches
        if self.expect_2d:
            self.image = self.image_stitcher(
                image_dir=self.output, mask=False, dtype=np.float32
            )[: self.scale_shape[0], : self.scale_shape[1]]
        else:
            self.image = self.image_stitcher(
                image_dir=self.output, mask=False, dtype=np.float32
            )[: self.scale_shape[0], : self.scale_shape[1], : self.scale_shape[2]]

        # Threshold whole image
        self.image = self.sigmoid(torch.Tensor(self.image)).detach().numpy()
        if self.cnn_threshold != 0:
            # Restored original image pixel size
            self.image, _ = scale_image(image=self.image, scale=self.org_shape)

            self.image = np.where(self.image >= self.cnn_threshold, 1, 0).astype(
                np.uint8
            )
        else:
            # Restored original image pixel size
            self.image, _ = scale_image(image=self.image, scale=self.org_shape)

            tif.imwrite(
                join(self.am_output, f"{id_name[:-self.in_format]}_CNN.tif"), self.image
            )
            self.image = None

        if len(pd.unique(self.image.flatten())) == 1:
            self.image = None

        """Clean-up temp dir"""
        clean_up(dir_=self.dir)

    def preprocess_DIST(self, id_name: str):
        # Post-process predicted image patches
        if self.predict in ["Filament", "Microtubule"]:
            self.pc_hd, self.pc_ld = self.post_processes.build_point_cloud(
                image=self.image, EDT=False, down_sampling=5
            )
        else:
            self.pc_hd, self.pc_ld = self.post_processes.build_point_cloud(
                image=self.image, EDT=False, down_sampling=5, as_2d=True
            )
        del self.image
        self._debug(id_name=id_name, debug_id="pc")

    def predict_cnn(self, id: int, id_name: str, dataloader):
        iter_time = 1
        if self.rotate:
            pred_title = "CNN prediction with four 90 degree rotations."
        else:
            pred_title = ""

        for j in range(len(dataloader)):
            if j % iter_time == 0:
                # Tardis progress bar update
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_3=f"Image {id + 1}/{len(self.predict_list)}: {id_name}",
                    text_4=f"Original Pixel size: {self.px} A",
                    text_5=pred_title,
                    text_7="Current Task: CNN prediction...",
                    text_8=print_progress_bar(j, len(dataloader)),
                )

            # Pick image['s]
            input_, name = dataloader.__getitem__(j)

            if j == 0:
                start = time.time()

                # Predict
                input_ = self.cnn.predict(input_[None, :], rotate=self.rotate)

                # Scale progress bar refresh to 10s
                end = time.time()
                iter_time = 10 // (end - start)
                if iter_time <= 1:
                    iter_time = 1
            else:
                # Predict
                input_ = self.cnn.predict(input_[None, :], rotate=self.rotate)

            tif.imwrite(
                join(self.output, f"{name}.tif"), np.array(input_, dtype=input_.dtype)
            )

    def predict_DIST(self, id: int, id_name: str):
        iter_time = int(round(len(self.coords_df) / 10))
        if iter_time == 0:
            iter_time = 1
        if iter_time >= len(self.coords_df):
            iter_time = 10

        pc = self.pc_ld.shape
        graphs = []
        for id_dist, coord in enumerate(self.coords_df):
            if id_dist % iter_time == 0:
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id + 1}/{len(self.predict_list)}: {id_name}",
                    text_4=f"Original pixel size: {self.px} A",
                    text_5=f"Point Cloud: {pc[0]} Nodes; NaN Segments",
                    text_7="Current Task: DIST prediction...",
                    text_8=print_progress_bar(id_dist, len(self.coords_df)),
                )

            graph = self.dist.predict(x=coord[None, :])
            graphs.append(graph)

        # Tardis progress bar update
        self.tardis_progress(
            title=self.title,
            text_1=f"Found {len(self.predict_list)} images to predict!",
            text_2=f"Device: {self.device}",
            text_3=f"Image {id + 1}/{len(self.predict_list)}: {id_name}",
            text_4=f"Original pixel size: {self.px} A",
            text_5=f"Point Cloud: {pc[0]}; NaN Segments",
            text_7=f"Current Task: {self.predict} segmentation...",
        )

        return graphs

    def _debug(self, id_name: str, debug_id: str):
        if self.debug:
            if debug_id == "cnn":
                tif.imwrite(
                    join(self.am_output, f"{id_name[:-self.in_format]}_CNN.tif"),
                    self.image,
                )
            elif debug_id == "pc":
                np.save(
                    join(self.am_output, f"{id_name[:-self.in_format]}_raw_pc_hd.npy"),
                    self.pc_hd,
                )
                np.save(
                    join(self.am_output, f"{id_name[:-self.in_format]}_raw_pc_ld.npy"),
                    self.pc_ld,
                )
            elif debug_id == "graph":
                try:
                    np.save(
                        join(
                            self.am_output,
                            f"{id_name[:-self.in_format]}_graph_voxel.npy",
                        ),
                        self.graphs,
                    )
                except ValueError:
                    pass
            elif debug_id == "segment":
                if self.device == "cpu":
                    np.save(
                        join(
                            self.am_output,
                            f"{id_name[:-self.in_format]}_coord_voxel.npy",
                        ),
                        self.coords_df,
                    )
                    np.save(
                        join(
                            self.am_output, f"{id_name[:-self.in_format]}_idx_voxel.npy"
                        ),
                        self.output_idx,
                    )
                else:
                    try:
                        np.save(
                            join(
                                self.am_output,
                                f"{id_name[:-self.in_format]}_coord_voxel.npy",
                            ),
                            self.coords_df.cpu().detach().numpy(),
                        )
                    except AttributeError:
                        pass
                    try:
                        np.save(
                            join(
                                self.am_output,
                                f"{id_name[:-self.in_format]}_idx_voxel.npy",
                            ),
                            self.output_idx.cpu().detach().numpy(),
                        )
                    except:
                        pass
                np.save(
                    join(self.am_output, f"{id_name[:-self.in_format]}_segments.npy"),
                    self.segments,
                )
            elif debug_id == "instance_mask":
                np.save(
                    join(
                        self.am_output, f"{id_name[:-self.in_format]}_instance_mask.npy"
                    ),
                    self.mask_semantic,
                )

    def __call__(self, *args, **kwargs):
        """Process each image with CNN and DIST"""
        for id_, i in enumerate(self.predict_list):
            """CNN Pre-Processing"""
            if i.endswith("CorrelationLines.am"):
                # Skip .am files which are spatial graphs not images
                continue

            # Find file format
            self.in_format = 0
            if i.endswith((".tif", ".mrc", ".rec", ".map")):
                self.in_format = 4
            elif i.endswith(".tiff"):
                self.in_format = 5
            elif i.endswith(".am"):
                self.in_format = 3

            # Tardis progress bar update
            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict!",
                text_2=f"Device: {self.device}",
                text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                text_4="Original Pixel size: Nan A",
                text_7="Current Task: Preprocessing for CNN...",
            )

            # Load data
            self.load_data(id_name=i)

            # Tardis progress bar update
            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict!",
                text_2=f"Device: {self.device}",
                text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                text_4=f"Original Pixel size: {self.px} A",
                text_7=f"Current Task: Sub-dividing images for {self.patch_size} size",
            )

            if not self.output_format.startswith("None") or not self.binary_mask:
                # Cut image for fix patch size and normalizing image pixel size
                trim_with_stride(
                    image=self.image,
                    scale=self.scale_shape,
                    trim_size_xy=self.patch_size,
                    trim_size_z=self.patch_size,
                    output=join(self.dir, "temp", "Patches"),
                    image_counter=0,
                    clean_empty=False,
                    stride=round(self.patch_size * 0.25),
                )
                self.image = None

                """CNN prediction"""
                self.predict_cnn(
                    id=id_,
                    id_name=i,
                    dataloader=PredictionDataset(
                        join(self.dir, "temp", "Patches", "imgs")
                    ),
                )

                """CNN Post-Processing"""
                # Tardis progress bar update
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                    text_4=f"Original pixel size: {self.px} A",
                    text_7="Current Task: Stitching...",
                )

                self.postprocess_CNN(id_name=i)
                if self.image is None:
                    continue

                # Debug flag
                self._debug(id_name=i, debug_id="cnn")

                # Check if predicted image
                if not self.image.shape == self.org_shape:
                    TardisError(
                        id_="116",
                        py="tardis_em/utils/predictor.py",
                        desc="Last Task: Stitching/Scaling/Make correction..."
                        f"Tardis Error: Error while converting to {self.px} A "
                        f"Org. shape {self.org_shape} is not the same as "
                        f"converted shape {self.image.shape}",
                    )
                    sys.exit()

                # If prediction fail aka no prediction was produces continue with next image
                if self.image is None:
                    continue

                """Save predicted mask"""
                if self.output_format.startswith("mrc"):
                    to_mrc(
                        data=self.image,
                        file_dir=join(
                            self.am_output, f"{i[:-self.in_format]}_semantic.mrc"
                        ),
                        pixel_size=self.px,
                    )
                elif self.output_format.startswith("tif"):
                    tif.imwrite(
                        join(self.am_output, f"{i[:-self.in_format]}_semantic.tif"),
                        np.flip(self.image, 1)
                        if i.endswith((".mrc", ".rec"))
                        else self.image,
                    )
                elif self.output_format.startswith("am"):
                    to_am(
                        data=self.image,
                        file_dir=join(
                            self.am_output, f"{i[:-self.in_format]}_semantic.am"
                        ),
                        pixel_size=self.px,
                    )

                if not self.image.min() == 0 and not self.image.max() == 1:
                    continue
                if not self.predict_instance:
                    continue

            """DIST pre-processing"""
            # Tardis progress bar update
            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict!",
                text_2=f"Device: {self.device}",
                text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                text_4=f"Original pixel size: {self.px} A",
                text_5="Point Cloud: In processing...",
                text_7="Current Task: Image Postprocessing...",
            )

            self.preprocess_DIST(id_name=i)
            if len(self.pc_hd) == 0:
                continue
            if len(self.pc_ld) < 100 and len(self.pc_hd) > 0:
                self.pc_ld = self.pc_hd

            self.segments = None
            if len(self.pc_ld) > 0:
                # Tardis progress bar update
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                    text_4=f"Original pixel size: {self.px} A",
                    text_5=f"Point Cloud: {self.pc_ld.shape[0]} Nodes; NaN Segments",
                    text_7="Current Task: Preparing for instance segmentation...",
                )

                # Build patches dataset
                if self.predict in ["Filament", "Microtubule", "Membrane2D"]:
                    try:
                        (
                            self.coords_df,
                            _,
                            self.output_idx,
                            _,
                        ) = self.patch_pc.patched_dataset(
                            coord=self.pc_ld / pc_median_dist(self.pc_ld, True)
                        )
                    except ValueError:
                        (
                            self.coords_df,
                            _,
                            self.output_idx,
                            _,
                        ) = self.patch_pc.patched_dataset(
                            coord=self.pc_ld / pc_median_dist(self.pc_ld, False)
                        )
                else:
                    (
                        self.coords_df,
                        _,
                        self.output_idx,
                        _,
                    ) = self.patch_pc.patched_dataset(coord=self.pc_ld / 5)

                # Predict point cloud
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                    text_4=f"Original pixel size: {self.px} A",
                    text_5=f"Point Cloud: {self.pc_ld.shape[0]} Nodes; NaN Segments",
                    text_7="Current Task: DIST prediction...",
                    text_8=print_progress_bar(0, len(self.coords_df)),
                )

                """DIST prediction"""
                self.graphs = self.predict_DIST(id=id_, id_name=i)
                self._debug(id_name=i, debug_id="graph")

                if self.predict in ["Filament", "Microtubule"]:
                    self.tardis_progress(
                        title=self.title,
                        text_1=f"Found {len(self.predict_list)} images to predict!",
                        text_2=f"Device: {self.device}",
                        text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                        text_4=f"Original pixel size: {self.px} A",
                        text_5=f"Point Cloud: {self.pc_ld.shape[0]} Nodes; NaN Segments",
                        text_7="Current Task: Instance Segmentation...",
                        text_8="MTs segmentation is fitted to:",
                        text_9=f"pixel size: {self.px}; transformation: {self.transformation}",
                    )

                    self.pc_ld = self.pc_ld * self.px
                    self.pc_ld[:, 0] = self.pc_ld[:, 0] + self.transformation[0]
                    self.pc_ld[:, 1] = self.pc_ld[:, 1] + self.transformation[1]
                    self.pc_ld[:, 2] = self.pc_ld[:, 2] + self.transformation[2]
                else:
                    self.tardis_progress(
                        title=self.title,
                        text_1=f"Found {len(self.predict_list)} images to predict!",
                        text_2=f"Device: {self.device}",
                        text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                        text_4=f"Original pixel size: {self.px} A",
                        text_5=f"Point Cloud: {self.pc_ld.shape[0]}; NaN Segments",
                        text_7="Current Task: Instance segmentation...",
                    )

                try:
                    if self.predict in ["Filament", "Microtubule"]:
                        sort = True
                        prune = 5
                    else:
                        sort = False
                        prune = 10
                    self.segments = self.GraphToSegment.patch_to_segment(
                        graph=self.graphs,
                        coord=self.pc_ld,
                        idx=self.output_idx,
                        sort=sort,
                        prune=prune,
                    )
                    self.segments = sort_by_length(self.segments)
                except:
                    self.segments = None

                if self.segments is None:
                    continue

                # Save debugging check point
                self._debug(id_name=i, debug_id="segment")

                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict!",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_ + 1}/{len(self.predict_list)}: {i}",
                    text_4=f"Original pixel size: {self.px} A",
                    text_5=f"Point Cloud: {self.pc_ld.shape[0]} Nodes;"
                    f" {np.max(self.segments[:, 0])} Segments",
                    text_7="Current Task: Segmentation finished!",
                )

                if self.predict in ["Filament", "Microtubule"]:
                    self.segments = sort_by_length(self.segments)

                """Save as .am"""
                if self.output_format.endswith("amSG") and self.predict in [
                    "Filament",
                    "Microtubule",
                ]:
                    self.amira_file.export_amira(
                        coords=self.segments,
                        file_dir=join(
                            self.am_output, f"{i[:-self.in_format]}_SpatialGraph.am"
                        ),
                        labels=["TardisPrediction"],
                        scores=[
                            ["EdgeLength", "EdgeConfidenceScore"],
                            [
                                length_list(self.segments),
                                self.score_splines(self.segments),
                            ],
                        ],
                    )

                    segments_filter = self.filter_splines(segments=self.segments)
                    segments_filter = sort_by_length(segments_filter)

                    self.amira_file.export_amira(
                        coords=segments_filter,
                        file_dir=join(
                            self.am_output,
                            f"{i[:-self.in_format]}_SpatialGraph_filter.am",
                        ),
                        labels=["TardisPrediction"],
                        scores=[
                            ["EdgeLength", "EdgeConfidenceScore"],
                            [
                                length_list(segments_filter),
                                self.score_splines(segments_filter),
                            ],
                        ],
                    )

                    if self.amira_check and self.predict == "Microtubule":
                        dir_amira_file = join(
                            self.dir_amira,
                            i[: -self.in_format] + self.amira_prefix + ".am",
                        )

                        if isfile(dir_amira_file):
                            amira_sg = ImportDataFromAmira(src_am=dir_amira_file)
                            amira_sg = amira_sg.get_segmented_points()

                            if amira_sg is not None:
                                compare_sg, label_sg = self.compare_spline(
                                    amira_sg=amira_sg, tardis_sg=segments_filter
                                )

                                if self.output_format.endswith("amSG"):
                                    self.amira_file.export_amira(
                                        file_dir=join(
                                            self.am_output,
                                            f"{i[:-self.in_format]}_AmiraCompare.am",
                                        ),
                                        coords=compare_sg,
                                        labels=label_sg,
                                        scores=None,
                                    )
                elif self.output_format.endswith("csv"):
                    np.savetxt(
                        join(self.am_output, f"{i[:-self.in_format]}_Segments.csv"),
                        self.segments,
                        delimiter=",",
                    )

                    if self.predict in ["Filament", "Microtubule"]:
                        np.savetxt(
                            join(
                                self.am_output,
                                f"{i[:-self.in_format]}_Segments_filter.csv",
                            ),
                            sort_by_length(self.filter_splines(segments=self.segments)),
                            delimiter=",",
                        )
                elif self.output_format.endswith(("mrc", "tif", "am")):
                    if self.predict in ["Membrane", "Membrane2D"]:
                        self.mask_semantic = draw_semantic_membrane(
                            mask_size=self.org_shape,
                            coordinate=self.segments,
                            pixel_size=self.px,
                            spline_size=60,
                        )
                    else:
                        self.segments[:, 1:] = self.segments[:, 1:] / self.px
                        self.mask_semantic = draw_instances(
                            mask_size=self.org_shape,
                            coordinate=self.segments,
                            pixel_size=self.px,
                            circle_size=125,
                        )
                    self._debug(id_name=i, debug_id="instance_mask")

                    if self.output_format.endswith("mrc"):
                        to_mrc(
                            data=self.mask_semantic,
                            file_dir=join(
                                self.am_output, f"{i[:-self.in_format]}_instance.mrc"
                            ),
                            pixel_size=self.px,
                        )
                    elif self.output_format.endswith("tif"):
                        tif.imwrite(
                            join(self.am_output, f"{i[:-self.in_format]}_instance.tif"),
                            self.mask_semantic,
                        )
                    elif self.output_format.endswith("am"):
                        to_am(
                            data=self.mask_semantic,
                            file_dir=join(
                                self.am_output, f"{i[:-self.in_format]}_instance.am"
                            ),
                            pixel_size=self.px,
                        )
                elif self.output_format.endswith("stl"):
                    if self.predict == "Membrane":
                        to_stl(
                            data=self.segments,
                            file_dir=join(self.am_output, f"{i[:-self.in_format]}.stl"),
                        )

            """Clean-up temp dir"""
            clean_up(dir_=self.dir)


class Predictor:
    """
    WRAPPER FOR PREDICTION

     Args:
         device (torch.device): Device on which to predict.
         checkpoint (str, Optional): Local weights files.
         network (str, Optional): Optional network type name.
         subtype (str, Optional): Optional model subtype name.
         model_type (str, Optional): Optional model type name.
         img_size (int, Optional): Optional image patch size.
         sigmoid (bool): Predict output with sigmoid.
    """

    def __init__(
        self,
        device: torch.device,
        network: Optional[str] = None,
        checkpoint: Optional[str] = None,
        subtype: Optional[str] = None,
        img_size: Optional[int] = None,
        model_type: Optional[str] = None,
        sigma: Optional[float] = None,
        sigmoid=True,
        _2d=False,
    ):
        self.device = device
        self.img_size = img_size
        if checkpoint is None and network is None:
            TardisError(
                "139",
                "tardis_em/utils/predictor.py",
                "Missing network weights or network name!",
            )

        if checkpoint is None:
            print(f"Searching for weight file for {network}_{subtype}...")

            weights = torch.load(
                get_weights_aws(network, subtype, model_type), map_location=device
            )

        elif isinstance(checkpoint, dict):
            weights = checkpoint
        else:
            print("Loading weight file...")
            weights = torch.load(checkpoint, map_location=device)

        # Allow overwriting sigma
        if sigma is not None:
            weights["model_struct_dict"]["coord_embed_sigma"] = sigma

        if "dist_type" in weights["model_struct_dict"]:
            from tardis_em.dist_pytorch.utils.utils import check_model_dict
        else:
            from tardis_em.cnn.utils.utils import check_model_dict
        model_structure = check_model_dict(weights["model_struct_dict"])

        if network is not None:
            self.network = network
        else:
            if "dist_type" in model_structure:
                self.network = "dist"
            else:
                self.network = "cnn"

        self._2d = _2d
        self.model = self._build_model_from_checkpoint(
            structure=model_structure, sigmoid=sigmoid
        )

        self.model.load_state_dict(weights["model_state_dict"])

        del weights  # Cleanup weight file from memory

    def _build_model_from_checkpoint(self, structure: dict, sigmoid=True):
        """
        Use checkpoint metadata to build compatible network

        Args:
            structure (dict): Metadata dictionary with network setting.
            sigmoid (bool): Predict output with sigmoid.

        Returns:
            pytorch model: NN pytorch model.
        """
        if "dist_type" in structure:
            model = build_dist_network(
                network_type=structure["dist_type"],
                structure=structure,
                prediction=sigmoid,
            )
        elif "cnn_type" in structure:
            model = build_cnn_network(
                network_type=structure["cnn_type"],
                structure=structure,
                img_size=self.img_size,
                prediction=sigmoid,
            )
        else:
            model = None

        return model.to(self.device)

    def predict(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, rotate=False
    ) -> np.ndarray:
        """
        General predictor.

        Args:
            x (torch.Tensor): Main feature used for prediction.
            y (torch.Tensor, None): Optional feature used for prediction.
            rotate (bool): Optional flag for CNN to output avg. from 4x 90* rotation

        Returns:
            np.ndarray: Predicted features.
        """
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.Tensor(y)

        with torch.no_grad():
            self.model.eval()

            dim_ = x.shape[-1]
            x = x.to(self.device)
            if self.network == "dist":
                if y is None:
                    out = self.model(coords=x, node_features=None)
                else:
                    out = self.model(coords=x, node_features=y.to(self.device))

                out = out.cpu().detach().numpy()[0, 0, :]
                g_len = out.shape[0]
                g_range = range(g_len)

                # Overwrite diagonal with 1
                out[g_range, g_range] = np.eye(g_len, g_len)[g_range, g_range]
                return out
            else:
                if rotate:
                    if self._2d:
                        out = np.zeros((dim_, dim_), dtype=np.float32)
                        for k in range(4):
                            x_ = torch.rot90(x, k=k, dims=(2, 3))
                            x_ = self.model(x_) / 4
                            x_ = x_.cpu().detach().numpy()[0, 0, :]

                            out += np.rot90(x_, k=-k, axes=(0, 1))
                    else:
                        out = np.zeros((dim_, dim_, dim_), dtype=np.float32)
                        for k in range(4):
                            x_ = torch.rot90(x, k=k, dims=(3, 4))
                            x_ = self.model(x_) / 4
                            x_ = x_.cpu().detach().numpy()[0, 0, :]

                            out += np.rot90(x_, k=-k, axes=(1, 2))
                else:
                    out = self.model(x).cpu().detach().numpy()[0, 0, :]

                return out


class BasicPredictor:
    """
    BASIC MODEL PREDICTOR FOR DIST AND CNN

    Args:
        model (nn.Module): ML model build with nn.Module or nn.sequential.
        structure (dict): Model structure as dictionary.
        device (str): Device for prediction.
        predicting_DataLoader (torch.DataLoader): DataLoader with prediction dataset.
        print_setting (tuple): Model property to display in TARDIS progress bar.
    """

    def __init__(
        self,
        model,
        structure: dict,
        device: str,
        print_setting: tuple,
        predicting_DataLoader,
        classification=False,
    ):
        super(BasicPredictor, self).__init__()

        self.model = model.to(device)
        self.device = device
        self.structure = structure

        if "cnn_type" in self.structure:
            self.classification = classification
            self.nn_name = self.structure["cnn_type"]
        elif "dist_type" in self.structure:
            self.nn_name = self.structure["dist_type"]

            if "node_input" in structure:
                self.node_input = structure["node_input"]

        self.predicting_DataLoader = predicting_DataLoader

        # Set-up progress bar
        self.progress_predict = TardisLogo()
        self.print_setting = print_setting

        self.id = 0
        self.predicting_idx = len(self.predicting_DataLoader)

    def _update_progress_bar(self):
        """
        Update entire Tardis progress bar.
        """
        if self.id % 50 == 0:
            self.progress_predict(
                title=f"{self.nn_name} Predicting module",
                text_1=self.print_setting[0],
                text_2=self.print_setting[1],
                text_3=self.print_setting[2],
                text_4=self.print_setting[3],
                text_8=print_progress_bar(self.id, self.predicting_idx),
            )

    def run_predictor(self):
        """
        Main prediction loop.
        """
        # Initialize progress bar.
        self.progress_predict(
            title=f"{self.nn_name} prediction module.",
            text_2="Predicted image: 0",
            text_3=print_progress_bar(0, self.predicting_idx),
        )

        self._update_progress_bar()

        """Training block"""
        self.model.eval()
        self._predict()

    def _predict(self):
        pass
