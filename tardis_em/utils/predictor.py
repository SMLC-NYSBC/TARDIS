#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
import json
import sys
import time
from datetime import datetime
from os import listdir, getcwd, mkdir
from os.path import isdir, isfile, join, expanduser
from typing import Optional, Union
import platform

import numpy as np
import pandas as pd
import tifffile.tifffile as tif
import torch

from tardis_em.dist_pytorch.utils.utils import VoxelDownSampling

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
from tardis_em.cnn.data_processing.scaling import scale_image
from tardis_em.utils.aws import get_weights_aws, get_all_version_aws
from tardis_em.utils.device import get_device
from tardis_em.utils.errors import TardisError
from tardis_em.utils.export_data import NumpyToAmira, to_am, to_mrc, to_stl
from tardis_em.utils.load_data import load_am, ImportDataFromAmira, load_image
from tardis_em.utils.logo import print_progress_bar, TardisLogo
from tardis_em.utils.normalization import (
    MeanStdNormalize,
    RescaleNormalize,
    adaptive_threshold,
)
from tardis_em.utils.setup_envir import build_temp_dir, clean_up
from tardis_em.analysis.spatial_graph_utils import (
    FilterSpatialGraph,
    FilterConnectedNearSegments,
    SpatialGraphCompare,
    ComputeConfidenceScore,
)
from tardis_em.analysis.filament_utils import sort_by_length, resample_filament
from tardis_em.analysis.geometry_metrics import length_list
from tardis_em._version import version

# try:
#     from tardis_em.utils.ota_update import ota_update
#
#     ota = ota_update(status=True)
# except ImportError:
#     ota = ""
ota = ""

# Pytorch CUDA optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class GeneralPredictor:
    """
    Summary of what the class does.

    The GeneralPredictor class is designed for handling image-based predictions
    using various neural network architectures. It provides functionalities
    to set up the environment, preprocess data, run predictions with or
    without rotation, and post-process the results for further usage or
    analysis. This class also includes methods to handle different file formats,
    apply various normalization techniques, and output results in specified formats.
    The class is modular enough to work with instances, semantic predictions, and
    custom settings based on the input parameters and configurations.
    """

    def __init__(
        self,
        predict: str,
        dir_s: Union[str, tuple[np.ndarray], np.ndarray],
        binary_mask: bool,
        output_format: str,
        patch_size: int,
        convolution_nn: str,
        cnn_threshold: str,
        dist_threshold: float,
        points_in_patch: int,
        predict_with_rotation: bool,
        instances: bool,
        device_s: str,
        debug: bool,
        checkpoint: Optional[list] = None,
        model_version: Optional[int] = None,
        correct_px: float = None,
        normalize_px: float = None,
        amira_prefix: str = None,
        filter_by_length: int = None,
        connect_splines: int = None,
        connect_cylinder: int = None,
        amira_compare_distance: int = None,
        amira_inter_probability: float = None,
        tardis_logo: bool = True,
        continue_b: bool = False,
    ):
        """
        This class initializes the configuration and parameters required for predictive models
        based on convolutional neural networks and builds the necessary handlers to process,
        transform, and analyze the provided data and predictions. It handles pre-processing,
        prediction, and post-processing steps, including normalization, stitching, spatial graph
        comparison, and various filters.

        :param predict: The predictive model type to be used.
        :param dir_s: The directory path or dataset array for input data processing.
        :param binary_mask: Flag indicating if a binary mask will be used.
        :param output_format: Specifies the output format for results.
        :param patch_size: Size of the patches in the image dataset.
        :param convolution_nn: Type of convolution neural network to use.
        :param cnn_threshold: Threshold value for CNN predictions.
        :param dist_threshold: Distance threshold used in graph segmentation.
        :param points_in_patch: Maximum number of points to include in a patch.
        :param predict_with_rotation: Flag to determine if prediction should consider rotations.
        :param instances: Flag to enable instance-based predictions.
        :param device_s: The computation device (CPU/GPU) to be used.
        :param debug: Flag to enable debugging mode.
        :param checkpoint: Optional, list of checkpoints for loading model weights.
        :param model_version: Optional, model version identifier.
        :param correct_px: Optional, value to correct pixel dimensions if needed.
        :param normalize_px: Optional, value to normalize pixel dimensions if required.
        :param amira_prefix: Optional, prefix for Amira file formats.
        :param filter_by_length: Optional, specifies filter to exclude segments of short lengths.
        :param connect_splines: Optional, parameter for connecting splines at close distances.
        :param connect_cylinder: Optional, radius for connecting cylindrical regions.
        :param amira_compare_distance: Optional, distance threshold for spatial graph comparison.
        :param amira_inter_probability: Optional, interaction threshold for Amira spatial graph probabilities.
        :param tardis_logo: Flag to handle tardis logo processing.
        :param continue_b: Flag to indicate if processing should resume from previous state.

        :raises: Raises errors for various invalid configurations or unsupported operation parameters.
        """
        self.continue_b = continue_b
        self.transformation, self.px, self.image = None, None, None
        self.tardis_logo = tardis_logo
        self.tardis_progress = None
        self.title = None

        # Directories and dataset info
        self.dir = dir_s
        self.output_format = output_format
        self.predict = predict
        if self.predict in ["Membrane2D", "Microtubule_tirf"]:
            self.expect_2d = True
        else:
            self.expect_2d = False
        self.amira_prefix = amira_prefix
        self.checkpoint = checkpoint
        self.model_version = model_version
        self.correct_px = correct_px
        self.normalize_px = normalize_px

        # Pre-processing setting
        self.cnn, self.dist = None, None
        self.patch_size = patch_size
        self.points_in_patch = points_in_patch
        self.pc_hd, self.pc_ld = np.zeros((0, 3)), np.zeros((0, 3))
        self.coords_df = np.zeros((1, 3))
        self.transformation = [0, 0, 0]
        self.segments, self.segments_filter = None, None

        # Prediction setting
        self.convolution_nn = convolution_nn
        self.cnn_threshold = cnn_threshold
        self.dist_threshold = dist_threshold
        self.rotate = predict_with_rotation

        # Global flags
        self.binary_mask = binary_mask
        self.predict_instance = instances
        self.device = get_device(device_s)
        self.debug = debug
        self.semantic_header, self.instance_header, self.log_prediction = [], [], []
        self.eta_predict = "NA"

        """Initial Setup"""
        if debug:
            self.str_debug = " <Debugging Mode>"
        else:
            self.str_debug = ""

        # Check for the spatial graph in the folder from amira/tardis_em comp.
        self.amira_check = False
        if isinstance(self.dir, str):
            if isdir(join(self.dir, "amira")):
                self.amira_check = True
                self.dir_amira = join(self.dir, "amira")

        # Searching for available images for prediction
        self.available_format = (
            ".tif",
            ".tiff",
            ".mrc",
            ".rec",
            ".am",
            ".map",
            ".npy",
        )
        self.omit_format = (
            ".nd2",
            "mask.tif",
            "mask.tiff",
            "mask.mrc",
            "mask.rec",
            "Correlation_Lines.am",
            "mask.am",
            "mask.map",
        )

        """Build handler's"""
        # Build handler's for reading data to correct format
        self.normalize = RescaleNormalize(clip_range=(1, 99))  # Normalize histogram
        self.mean_std = MeanStdNormalize()  # Standardize with mean and std

        # Sigmoid whole predicted image
        self.sigmoid = torch.nn.Sigmoid()

        # Build handler's for transforming data
        self.image_stitcher = StitchImages()
        self.post_processes = BuildPointCloud()

        # self.get_file_list()

        # Build handler for DIST input and output
        if self.predict_instance:
            self.patch_pc = PatchDataSet(
                max_number_of_points=points_in_patch, graph=False
            )

            if predict in [
                "Actin",
                "Microtubule",
                "Membrane2D",
                "General_filament",
                "Microtubule_tirf",
            ]:
                self.GraphToSegment = PropGreedyGraphCut(
                    threshold=dist_threshold, connection=2, smooth=True
                )

                if predict == "Membrane2D":
                    self.filter_splines = FilterConnectedNearSegments(
                        distance_th=connect_splines,
                        cylinder_radius=connect_cylinder,
                    )
                else:
                    self.filter_splines = FilterSpatialGraph(
                        connect_seg_if_closer_then=connect_splines,
                        cylinder_radius=connect_cylinder,
                        filter_short_segments=(
                            filter_by_length if filter_by_length is not None else 0
                        ),
                    )
                self.compare_spline = SpatialGraphCompare(
                    distance_threshold=amira_compare_distance,
                    interaction_threshold=amira_inter_probability,
                )
                self.score_splines = ComputeConfidenceScore()
            elif predict in ["Membrane", "General_object"]:
                self.GraphToSegment = PropGreedyGraphCut(
                    threshold=dist_threshold, connection=8
                )

        # Build handler to output amira file
        self.create_headers()
        self.amira_file = NumpyToAmira()
        self.amira_file_points = NumpyToAmira(as_point_cloud=True)

        # Sanity error checks
        self.init_check()

        # Build NN from checkpoints
        self.build_NN(NN=self.predict)

    def create_headers(self):
        """
        Creates ASCII headers and initializes logging information for a spatial graph
        prediction process. The headers include project details, directory setup,
        neural network configurations for semantic and instance segmentation, and
        other metadata. It verifies paths, logs prediction states appropriately,
        and dynamically determines the model version depending on the predictive
        needs and available checkpoints.

        :raises: None
        :parameter self.dir: Directory path for file output or processing.
            Default is retrieved current working directory if unset.
        :parameter self.output_format: Format of the output data produced during
           prediction, e.g. as,send LabelSetsivet nn
        """
        dir_ = self.dir if isinstance(self.dir, str) else "np.ndarray"
        if dir_ == ".":
            dir_ = getcwd()

        main_ = [
            "# ASCII Spatial Graph",
            "# TARDIS - Transformer And Rapid Dimensionless Instance Segmentation (R)",
            f"# tardis_em-pytorch v{version} \r",
            f"# MIT License * 2021-{datetime.now().year} * "
            "Robert Kiewisz & Tristan Bepler",
            "Robert Kiewisz & Tristan Bepler",
            "",
            f"Directory: {dir_}",
            f"Output Format: {self.output_format}",
            f"Predict: {self.predict}",
            f"Device: {self.device}",
            "",
        ]

        if self.model_version is None:
            model = self.predict

            if model == "Actin":
                model = "actin_3d"
            elif model == "Microtubule":
                model = "microtubules_3d"
            elif model == "Microtubule_tirf":
                model = "microtubules_tirf"
            elif model == "Membrane":
                model = "membrane_3d"
            elif model == "Membrane2D":
                model = "membrane_2d"
            else:
                model = "None"

            if model != "None":
                if self.model_version is None:
                    try:
                        model_version = get_all_version_aws(
                            self.convolution_nn, "32", model
                        )
                        model_version = model_version[-1]
                    except:
                        if isfile(join(expanduser("~"), ".tardis_em", "fnet_attn_32", model, "model_version.json")):
                            model_version = json.load(open(
                                join(expanduser("~"), ".tardis_em", "fnet_attn_32", model, "model_version.json")
                            ))
                            model_version = model_version["version"]
                        else:
                            model_version = "Local"
                else:
                    model_version = self.model_version
            else:
                model_version = "None"
        else:
            model_version = self.model_version

        self.semantic_header = [
            (
                f"CNN type: {self.convolution_nn} {model_version} "
                if self.checkpoint[0] is None
                else f"CNN model loaded from checkpoint: {self.checkpoint[0]}"
                f"Image patch size used for CNN: {self.patch_size}"
            ),
            f"CNN threshold: {self.cnn_threshold}",
            f"CNN predicted with 4x 90 degrees rotations: {self.rotate}",
        ]

        self.instance_header = [
            (
                "DIST 2D model"
                if self.predict
                in ["Actin", "Microtubule", "Membrane2D", "Microtubule_tirf"]
                else "DIST 3D model"
            ),
            (
                ""
                if self.checkpoint[1] is None
                else f"DIST model loaded from checkpoint: {self.checkpoint[1]}"
                f"DIST threshold: {self.dist_threshold}"
            ),
            f"Maximum number of points used in single DIST run: {self.points_in_patch}",
        ]
        init_log = (
            main_
            + ["----Semantic Segmentation----"]
            + self.semantic_header
            + ["", "----Instance Segmentation----"]
            + self.instance_header
            + [""]
        )

        if self.continue_b and isfile(
            join(self.dir, "Predictions", "prediction_log.txt")
        ):
            self.log_prediction = np.genfromtxt(
                join(self.dir, "Predictions", "prediction_log.txt"),
                delimiter=",",
                dtype=str,
            )
            self.log_prediction = [str(s) for s in self.log_prediction]
            self.log_prediction = (
                self.log_prediction + ["", "---Continue prediction---", ""] + init_log
            )
        else:
            self.log_prediction = init_log

    def init_check(self):
        """
        Perform initialization and validation checks for a segmentation prediction task.

        This method performs several checks and initializations before executing a segmentation
        task. It first validates whether the requested prediction type is supported. If support is
        enabled for a TARDIS logo, additional checks are performed for user configurations, and error
        messages are displayed if any invalid settings are detected. Subsequently, the log output
        initialization begins with a defined title according to the segmentation type and configuration.

        This function also validates the chosen output format to ensure its compatibility with the
        system (e.g., checking for unsupported machine types like ARM64 or ensuring that at least one
        valid output format is selected). Invalid configurations or unsupported settings result in
        terminating the execution with proper error messaging.

        :param self: Instance of the class to which the initialization belongs.

        :raises AssertionError: If the requested segmentation type is unsupported.
        :raises TardisError: If any invalid configuration is detected based on
                              TARDIS-specific rules (e.g., invalid output format, machine-type
                              dependencies).
        """
        msg = f"TARDIS v.{version} supports only MT and Mem segmentation!"
        assert_b = self.predict in [
            "Actin",
            "Membrane2D",
            "Membrane",
            "Microtubule",
            "Microtubule_tirf",
            "General_filament",
            "General_object",
        ]
        if self.tardis_logo:
            # Check if users ask to predict the correct structure
            if not assert_b:
                TardisError(
                    id_="01",
                    py="tardis_em/utils/predictor.py",
                    desc=msg,
                )
                sys.exit()
        else:
            assert assert_b, msg

        # Initiate log output
        if self.tardis_logo:
            self.tardis_progress = TardisLogo()

            if isdir(join(self.dir, "amira")):
                self.title = (
                    f"Fully-automatic Instance {self.predict} segmentation module "
                    f"with Amira comparison {self.str_debug}"
                )
            elif self.predict_instance:
                if self.output_format.startswith("None"):
                    self.title = (
                        f"Fully-automatic Instance {self.predict} segmentation module | {ota} "
                        f"{self.str_debug}"
                    )
                else:
                    self.title = (
                        f"Fully-automatic Semantic-Instance {self.predict} segmentation module | {ota} "
                        f"{self.str_debug}"
                    )
            else:
                self.title = (
                    f"Fully-automatic Semantic {self.predict} segmentation module | {ota} "
                    f"{self.str_debug}"
                )

            self.tardis_progress(title=self.title, text_2=f"Device: {self.device}")

        # Check for any other errors
        # Early stop if not semantic of instance was specified
        msg = f"Require that at lest one output format is not None but {self.output_format} was given!"
        assert_b = self.output_format == "None_None"
        if assert_b:
            if self.tardis_logo:
                TardisError(
                    id_="151",
                    py="tardis_em/utils/predictor.py",
                    desc=msg,
                )
                sys.exit()
            else:
                assert assert_b, msg

        # Check for know error if using ARM64 machine
        msg = f"STL output is not allowed on {platform.machine()} machine type!"
        assert_b = platform.machine() == "aarch64"
        if self.output_format.endswith("stl"):
            if self.tardis_logo:
                TardisError(
                    id_="151",
                    py="tardis_em/utils/predictor.py",
                    desc=msg,
                )
                sys.exit()
            else:
                assert not assert_b, msg

    def build_NN(self, NN: str):
        """
        Builds the neural network and distance prediction modules based on the specified
        neural network (NN) type. Depending on the NN type, the appropriate configurations
        and pre-trained weights are loaded for CNN and DIST networks. This method supports
        multiple NN types, including Actin, Microtubule, Membrane, and General models.
        Additionally, configurations for 2D, 3D, and other specialized models are supported.

        :param NN: A string denoting the neural network type. Supported types include
            "Actin", "Microtubule", "Microtubule_tirf", "Membrane2D", "Membrane",
            or any type starting with "General".
        :type NN: str
        """
        if NN in ["Actin", "Microtubule", "Microtubule_tirf"]:
            # None - default value
            # 0.0 - Do not normalize
            # > 0 - normalzie to specific A resolution
            if self.normalize_px is None:
                self.normalize_px = 25
            elif self.normalize_px == 0.0:
                self.normalize_px = None

            if NN == "Actin" and self.normalize_px is not None:
                if self.normalize_px == 0.0:
                    self.normalize_px = None
                else:
                    self.normalize_px = 15

            # Build CNN network with loaded pre-trained weights
            if not self.binary_mask:
                if NN in ["Actin", "Microtubule"]:
                    self.cnn = Predictor(
                        checkpoint=self.checkpoint[0],
                        network=self.convolution_nn,
                        subtype="32",
                        model_type=(
                            "microtubules_3d" if NN == "Microtubule" else "actin_3d"
                        ),
                        model_version=self.model_version,
                        img_size=self.patch_size,
                        sigmoid=False,
                        device=self.device,
                    )
                else:
                    self.cnn = Predictor(
                        checkpoint=self.checkpoint[0],
                        network=self.convolution_nn,
                        subtype="32",
                        model_type="microtubules_tirf",
                        model_version=self.model_version,
                        img_size=self.patch_size,
                        sigmoid=False,
                        _2d=True,
                        device=self.device,
                    )
            # Build DIST network with loaded pre-trained weights
            if not self.output_format.endswith("None"):
                self.dist = Predictor(
                    checkpoint=self.checkpoint[1],
                    network="dist",
                    subtype="triang",
                    model_type="2d",
                    model_version=None,
                    device=self.device,
                )
        elif NN in ["Membrane2D", "Membrane"]:
            # None - default value
            # 0.0 - Do not normalize
            # > 0 - normalzie to specific A resolution
            if self.normalize_px is None:
                self.normalize_px = 15
            elif self.normalize_px == 0.0:
                self.normalize_px = None

            # Build CNN network with loaded pre-trained weights
            if NN == "Membrane2D":
                if not self.binary_mask:
                    self.cnn = Predictor(
                        checkpoint=self.checkpoint[0],
                        network=self.convolution_nn,
                        subtype="32",
                        model_type="membrane_2d",
                        model_version=self.model_version,
                        img_size=self.patch_size,
                        sigmoid=False,
                        device=self.device,
                        _2d=True,
                    )

                # Build DIST network with loaded pre-trained weights
                if not self.output_format.endswith("None"):
                    self.dist = Predictor(
                        checkpoint=self.checkpoint[1],
                        network="dist",
                        subtype="triang",
                        model_type="2d",
                        model_version=None,
                        device=self.device,
                    )
            else:
                if not self.binary_mask:
                    self.cnn = Predictor(
                        checkpoint=self.checkpoint[0],
                        network=self.convolution_nn,
                        subtype="32",
                        model_type="membrane_3d",
                        model_version=self.model_version,
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
                        model_type="3d",
                        model_version=None,
                        device=self.device,
                    )
        elif NN.startswith("General"):
            self.cnn = Predictor(
                checkpoint=self.checkpoint[0],
                model_version=self.model_version,
                network=self.convolution_nn,
                img_size=self.patch_size,
                sigmoid=False,
                device=self.device,
                _2d=self.expect_2d,
            )
            if not self.output_format.endswith("None"):
                if NN.endswith("filament"):
                    self.dist = Predictor(
                        checkpoint=self.checkpoint[1],
                        network="dist",
                        subtype="triang",
                        model_type="2d",
                        model_version=None,
                        device=self.device,
                    )
                else:
                    self.dist = Predictor(
                        checkpoint=self.checkpoint[1],
                        network="dist",
                        subtype="triang",
                        model_type="3d",
                        model_version=None,
                        device=self.device,
                    )

    def load_data(self, id_name: Union[str, np.ndarray]):
        """
        Loads and processes image data or point cloud data from a specified file or array. Depending on the
        input type, the function determines if the data is an AmiraMesh 3D ASCII file, a general image file,
        or a preloaded array. It performs normalization, sanity checks, and prepares the data for further
        processing.

        :param id_name:
            Specifies either the path to the file to be loaded or an already loaded numpy array.
            Can be a string filename or a numpy array.

        :raises AssertionError:
            - If the Amira Spatial Graph has dimensions other than 4.
            - If the loaded image's dtype is not `float32` after normalization.
            - If the processed binary mask dtype is not `int8` or `uint8`.

        :raises TardisError:
            - If `tardis_logo` is True and any of the aforementioned conditions fail.

        :raises SystemExit:
            - If certain errors occur during processing and `tardis_logo` is True.

        :return: None
        """
        self.log_prediction.append("\n")
        self.log_prediction.append(f"Loaded image: {id_name}")

        # Build temp dir
        build_temp_dir(dir_s=self.dir)

        if isinstance(id_name, str):
            # Load image file
            if id_name.endswith(".am"):
                am = open(join(self.dir, id_name), "r", encoding="iso-8859-1").read(500)

                if "AmiraMesh 3D ASCII" in am:
                    self.amira_image = False
                    self.pc_hd = ImportDataFromAmira(
                        join(self.dir, id_name)
                    ).get_segmented_points()

                    self.image = None
                    self.px = self.correct_px
                    self.transformation = [0, 0, 0]

                    assert_b = self.pc_hd.shape[1] == 4
                    msg = f"Amira Spatial Graph has wrong dimension. Given {self.pc_hd.shape[1]}, but expected 4."
                    if self.tardis_logo:
                        if not assert_b:
                            TardisError(
                                id_="11",
                                py="tardis_em/utils/predictor.py",
                                desc=msg,
                            )
                            sys.exit()
                    else:
                        assert assert_b, msg
                else:
                    self.amira_image = True
                    self.image, self.px, _, self.transformation = load_am(
                        am_file=join(self.dir, id_name)
                    )
            else:
                self.amira_image = True
                self.image, self.px = load_image(join(self.dir, id_name))
                self.transformation = [0, 0, 0]
        else:
            self.image = id_name
            self.px = self.correct_px
            self.transformation = [0, 0, 0]

        self.log_prediction.append(
            f"Image pixel size: {self.px}A"
            if self.correct_px is None
            else f"Image pixel size: {self.correct_px}A"
        )
        # Normalize image histogram
        msg = f"Error while loading image {id_name}. Image loaded correctly, but output format "
        if self.amira_image and not self.binary_mask:
            self.image = self.normalize(self.mean_std(self.image)).astype(np.float32)

            # Sanity check image histogram
            if (
                not self.image.min() >= -1 or not self.image.max() <= 1
            ):  # Image not between in -1 and 1
                if self.image.min() >= 0 and self.image.max() <= 1:
                    self.image = (self.image - 0.5) * 2  # shift to -1 - 1
                elif self.image.min() >= 0 and self.image.max() <= 255:
                    self.image = self.image / 255  # move to 0 - 1
                    self.image = (self.image - 0.5) * 2  # shift to -1 - 1

            assert_b = self.image.dtype == np.float32
            if self.tardis_logo:
                if not assert_b:
                    TardisError(
                        id_="11",
                        py="tardis_em/utils/predictor.py",
                        desc=msg + f" {self.image.dtype} is not float32!",
                    )
                    sys.exit()
            else:
                assert assert_b, msg

            # Calculate parameters for image pixel size with optional scaling
            if self.correct_px is not None:
                self.px = self.correct_px

            if self.normalize_px is not None:
                self.scale_factor = self.px / self.normalize_px
            else:
                self.scale_factor = 1.0

            self.org_shape = self.image.shape
            self.scale_shape = np.multiply(self.org_shape, self.scale_factor).astype(
                np.int16
            )
            self.scale_shape = [int(i) for i in self.scale_shape]
        elif self.amira_image and self.binary_mask:
            # Check image structure
            self.image = np.where(self.image > 0, 1, 0).astype(np.int8)

            assert_b = self.image.dtype == np.int8 or self.image.dtype == np.uint8
            if self.tardis_logo:
                if not assert_b:
                    TardisError(
                        id_="11",
                        py="tardis_em/utils/predictor.py",
                        desc=msg + f" {self.image.dtype} is not int8!",
                    )
                    sys.exit()
            else:
                assert assert_b, msg

        if self.image.ndim == 2 or self.predict == "Microtubule_tirf":
            self.expect_2d = True
        else:
            self.expect_2d = False

        self.log_prediction.append(
            f"Image scaled to: {self.normalize_px}A and shape {self.scale_shape}",
        )

    def predict_cnn(self, id_i: int, id_name: str, dataloader):
        """
        Predict images using a Convolutional Neural Network (CNN) with options for image rotation
        and progress tracking integrated with the Tardis progress bar interface.

        This method iterates over a dataloader to retrieve images, predicts their output using
        a CNN model (optionally with four 90° rotations), and writes the output to `.tif` files.
        The method supports progress tracking with Tardis interface updates
        and dynamically optimizes the progress bar refresh rate based on initial iteration timing.

        :param id_i: Integer representing the ID of the image being processed.
        :type id_i: int
        :param id_name: The name of the image being processed.
        :type id_name: str
        :param dataloader: An iterable object that provides access to image data and corresponding names.
        :return: None
        """
        iter_time = 1
        if self.rotate:
            pred_title = f"CNN prediction with four 90 degree rotations with {self.convolution_nn}"
        else:
            pred_title = f"CNN prediction with {self.convolution_nn}"
        start, end = 0, 0

        for j in range(len(dataloader)):
            if j % iter_time == 0 and self.tardis_logo:
                eta_time = (
                    str(round(((end - start) * (len(dataloader) - j - 1)) / 60, 1))
                    + "min"
                )

                # Tardis progress bar update
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_i + 1}/{len(self.predict_list)} [{eta_time} ETA]: {id_name}",
                    text_4=f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
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
            tif.imwrite(join(self.output, f"{name}.tif"), input_)

    def predict_cnn_napari(self, input_t: torch.Tensor, name: str):
        """
        Predicts an output using the CNN model on the provided input tensor, saves the
        result in TIFF format, and returns the output tensor.

        This function performs a prediction using the Convolutional Neural Network
        (CNN) model on the given input tensor. The result is saved as a TIFF file
        using the provided file name in the specified output directory.

        :param input_t:
            Input tensor on which prediction needs to be performed, should follow
            the required input format for the CNN model.
        :param name:
            Name of the output file to save the predicted result.

        :return:
            The output tensor resulting from the CNN prediction, after processing
            with the input tensor.
        """
        input_t = self.cnn.predict(input_t[None, :], rotate=self.rotate)
        tif.imwrite(join(self.output, f"{name}.tif"), input_t)

        return input_t

    def postprocess_CNN(self, id_name: str):
        """
        Post-processes the CNN prediction by stitching predicted image patches,
        restoring the original pixel size, applying a threshold, and optionally
        saving the results in the specified format. This function also performs
        clean-up of temporary directories after processing.

        :param id_name: Identifier of the input data used to track and log the
            processed output.
        :type id_name: str
        :return: None
        """
        # Stitch predicted image patches
        if self.expect_2d:
            self.image = self.image_stitcher(
                image_dir=self.output, mask=False, dtype=np.float32
            )
            if self.image.ndim == 3:
                self.image = self.image[:, : self.scale_shape[0], : self.scale_shape[1]]
            else:
                self.image = self.image[: self.scale_shape[0], : self.scale_shape[1]]
        else:
            self.image = self.image_stitcher(
                image_dir=self.output, mask=False, dtype=np.float32
            )[: self.scale_shape[0], : self.scale_shape[1], : self.scale_shape[2]]

        # Restored original image pixel size
        self.image = torch.sigmoid(torch.from_numpy(self.image)).cpu().detach().numpy()
        self.image, _ = scale_image(image=self.image, scale=self.org_shape)

        # Threshold CNN prediction
        if self.cnn_threshold == "auto":
            self.image = adaptive_threshold(self.image)
        else:
            self.cnn_threshold = float(self.cnn_threshold)

            if self.cnn_threshold != 0:
                self.image = np.where(self.image >= self.cnn_threshold, 1, 0).astype(
                    np.uint8
                )
            else:
                if not self.output_format.startswith("return"):
                    self.log_prediction.append(
                        f"Semantic Prediction: {id_name[:-self.in_format]}"
                        f" | Number of pixels: {np.sum(self.image)}"
                    )
                    with open(join(self.am_output, "prediction_log.txt"), "w") as f:
                        f.write(" \n".join(self.log_prediction))

                    # ToDo output format specified by user aka MRC/REC/AM/TIFF
                    tif.imwrite(
                        join(self.am_output, f"{id_name[:-self.in_format]}_CNN.tif"),
                        self.image,
                    )
                    self.image = None

        """Clean-up temp dir"""
        clean_up(dir_s=self.dir)

    def preprocess_DIST(self, id_name: str):
        """
        Preprocesses a given dataset identifier (id_name) to produce and manipulate
        high-density and low-density point clouds, typically used for structural or
        image data analysis. Depending on the prediction type and the presence of
        an Amira image, this function either post-processes predicted image patches
        to construct point clouds using provided processing utilities or applies
        optimization methods like voxel down-sampling for refining existing point
        clouds.

        :param id_name: The unique dataset identifier used in debugging and processing.
        :type id_name: str
        :return: None
        """
        if self.amira_image:
            # Post-process predicted image patches
            if self.predict in ["Actin", "Microtubule", "General_filament"]:
                self.pc_hd, self.pc_ld = self.post_processes.build_point_cloud(
                    image=self.image, down_sampling=5
                )
            elif self.predict in ["Membrane2D", "Microtubule_tirf"]:
                self.pc_hd, self.pc_ld = self.post_processes.build_point_cloud(
                    image=self.image, down_sampling=5, as_2d=True
                )
            else:
                self.pc_hd, self.pc_ld = self.post_processes.build_point_cloud(
                    image=self.image, down_sampling=5, as_2d=True
                )

            self.image = None
            self._debug(id_name=id_name, debug_id="pc")
        else:
            self.pc_hd = resample_filament(self.pc_hd, self.px)
            down_sample = VoxelDownSampling(voxel=5, labels=False, KNN=True)
            self.pc_ld = down_sample(coord=self.pc_hd[:, 1:])

    def predict_DIST(self, id_i: int, id_name: str):
        """
        Predicts DIST graphs for the given coordinates using the provided DIST prediction
        model. The method processes coordinate data in chunks and updates the progress bar
        if visual feedback is enabled. The progress bar reflects the current task,
        the percentage of completion, and relevant details of the segmentation process.
        The function ensures predictive modeling for the total images with a
        controlled iteration mechanism.

        :param id_i: An integer representing the identifier of the image to be processed.
        :param id_name: A string denoting the name of the image corresponding to the ID.
        :return: A list of predicted graph representations for each coordinate dataset.
        """
        iter_time = int(round(len(self.coords_df) / 10))

        if iter_time == 0:
            iter_time = 1
        if iter_time >= len(self.coords_df):
            iter_time = 10

        pc = self.pc_ld.shape
        graphs = []
        for id_dist, coord in enumerate(self.coords_df):
            if id_dist % iter_time == 0 and self.tardis_logo:
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
                    text_2=f"Device: {self.device}",
                    text_3=f"Image {id_i + 1}/{len(self.predict_list)}: {id_name}",
                    text_4=f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                    text_5=f"Point Cloud: {pc[0]} Nodes; NaN Segments",
                    text_7="Current Task: DIST prediction...",
                    text_8=print_progress_bar(id_dist, len(self.coords_df)),
                )

            graph = self.dist.predict(x=coord[None, :])
            graphs.append(graph)

        # Tardis progress bar update
        if self.tardis_logo:
            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
                text_2=f"Device: {self.device}",
                text_3=f"Image {id_i + 1}/{len(self.predict_list)}: {id_name}",
                text_4=f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                text_5=f"Point Cloud: {self.pc_ld.shape[0]}; NaN Segments",
                text_7=f"Current Task: {self.predict} segmentation...",
            )

        return graphs

    def postprocess_DIST(self, id_i, i):
        """
        Processes and postprocesses data based on given inputs.

        This function adjusts the pixel data, logs information based
        on specific prediction types, and handles the transformation
        of graphs to segments. Additionally, updates the Tardis
        progress bar to provide task-specific updates.

        :param id_i: Identification number for the current image
            being processed.
        :type id_i: int

        :param i: Index of the current image.
        :type i: int

        :return: None, modifies instance attributes based
            on the processing steps.
        :rtype: None
        """
        self.pc_ld = (
            self.pc_ld * self.px
            if self.correct_px is None
            else self.pc_ld * self.correct_px
        )
        self.pc_ld = self.pc_ld + self.transformation

        if self.predict in [
            "Actin",
            "Microtubule",
            "General_filament",
            "Microtubule_tirf",
        ]:
            self.log_tardis(id_i, i, log_id=6.1)
        else:
            self.log_tardis(id_i, i, log_id=6.2)

        if self.predict in [
            "Actin",
            "Microtubule",
            "Membrane2D",
            "General_filament",
            "Microtubule_tirf",
        ]:
            sort = True
            prune = 5
        else:
            sort = False
            prune = 15

        try:
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

        # Tardis progress bar update
        if self.tardis_logo:
            no_segments = ""
            if self.segments is None:
                no_segments = f"Point Cloud: {self.pc_ld.shape[0]}; None Segments"
            else:
                no_ = np.max(np.unique(self.segments[:, 0])).item(0)
                no_segments = f"Point Cloud: {self.pc_ld.shape[0]}; {no_} Segments"

            self.tardis_progress(
                title=self.title,
                text_1=f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
                text_2=f"Device: {self.device}",
                text_3=f"Image {id_i + 1}/{len(self.predict_list)}: {i}",
                text_4=f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                text_5=no_segments,
                text_7=f"Current Task: {self.predict} segmented...",
            )

    def get_file_list(self):
        """
        Retrieves and processes a list of files to be used for prediction based on the directory
        or input provided. Filters files according to specified formats, handles input as either
        single directories or lists/tuples, and logs the processed files. Additionally, performs
        setup tasks for the prediction workflow, including generating paths for output directories
        and checking prediction readiness.

        :param self: The instance of the object that contains attributes such as directory paths,
                     filtering formats, continuation settings, and progress handlers for processing.
        :type self: Object containing attributes: `dir`, `available_format`, `omit_format`,
                    `continue_`, `tardis_logo`, `tardis_progress`, `output`, `am_output`,
                    `predict_list`, `device`, `title`.

        :raises AssertionError: Raised when no recognizable files exist in the provided directory
                                structure, based on specified formats, and appropriate progress
                                handling or error logging is not enabled.
        :raises Exception: Additional exceptions may occur if environmental setup or file reading
                           operations fail, depending on external utilities used and provided paths.

        :return: None
        """
        # Pickup files for the prediction
        if not isinstance(self.dir, str):
            if isinstance(self.dir, tuple) or isinstance(self.dir, list):
                self.predict_list = self.dir
            else:
                self.predict_list = [self.dir]
        else:
            if isdir(self.dir):
                self.predict_list = [
                    f
                    for f in listdir(self.dir)
                    if f.endswith(self.available_format)
                    and not f.endswith(self.omit_format)
                ]
            else:
                self.predict_list = [
                    f
                    for f in [self.dir]
                    if f.endswith(self.available_format)
                    and not f.endswith(self.omit_format)
                ]
        out_ = [
            i
            for i in self.dir.split("/")
            if not i.endswith((".mrc", ".rec", ".map", ".tif", ".tiff", ".am"))
        ]
        self.dir = join("/".join(out_))

        # Update Dir paths
        self.output = join(self.dir, "temp", "Predictions")
        self.am_output = join(self.dir, "Predictions")

        if self.continue_b:
            if isfile(join(self.am_output, "prediction_log.txt")):
                logs = np.genfromtxt(
                    join(self.am_output, "prediction_log.txt"), delimiter=",", dtype=str
                )
                logs = logs[-1].split(" ")[2]
                logs = [
                    i for i, x in enumerate(self.predict_list) if x.startswith(logs)
                ]

            if len(logs) > 0:
                logs = logs[0]
                self.predict_list = self.predict_list[logs:]

        # Check if there is anything to predict in the user-indicated folder
        msg = f"Given {self.dir} does not contain any recognizable file!"
        assert_b = len(self.predict_list) == 0

        if self.tardis_logo and self.tardis_progress is not None:
            if assert_b:
                TardisError(
                    id_="12",
                    py="tardis_em/utils/predictor.py",
                    desc=msg,
                )
                sys.exit()
            else:
                self.tardis_progress(
                    title=self.title,
                    text_1=f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
                    text_2=f"Device: {self.device}",
                    text_7="Current Task: Setting-up environment...",
                )
        else:
            assert not assert_b, msg

    def log_tardis(self, id_i: int, i: Union[str, int, np.ndarray], log_id: float):
        """
        Logs various states and processing stages of the TARDIS application based on the
        provided `log_id` and input data. Depending on the log ID and input type, it generates
        log messages showcasing the progress of various computational tasks and updates a
        progress bar accordingly.

        :param id_i: Identifier for the current image being processed in the list
            of input images.
        :type id_i: int
        :param i: Input data for logging, representing either a string description
            or a numpy array. If a numpy array is passed, it is converted into a
            string representation.
        :type i: Union[str, numpy.ndarray]
        :param log_id: Numeric identifier specifying the current task or processing
            stage. Determines the type of logging information generated, and can
            optionally include different subtasks.
        :type log_id: float
        :return: None
        """
        if not isdir(self.am_output):
            mkdir(self.am_output)

        with open(join(self.am_output, "prediction_log.txt"), "w") as f:
            f.write(" \n".join(self.log_prediction))

        if isinstance(i, np.ndarray):
            i = "Numpy array"

        if not self.tardis_logo:
            return

        # Common text for all configurations
        common_text = {
            "text_1": f"Found {len(self.predict_list)} images to predict! [{self.eta_predict} ETA]",
            "text_2": f"Device: {self.device}",
            "text_3": f"Image {id_i + 1}/{len(self.predict_list)}: {i}",
        }

        if log_id == 7:
            no_segments = np.max(self.segments[:, 0]) if len(self.segments) > 0 else 0
        else:
            no_segments = "None"

        # Define text configurations for each log_id
        pc_ld_shape = self.pc_ld.shape[0]

        text_configurations = {
            0: {
                **common_text,
                "text_4": "Org. Pixel size: Nan A",
                "text_7": "Current Task: Preprocessing for CNN...",
            },
            1: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_7": f"Current Task: Sub-dividing images for {self.patch_size} size",
            },
            2: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_7": "Current Task: Stitching...",
            },
            3: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": "Point Cloud: In processing...",
                "text_7": "Current Task: Image Postprocessing...",
            },
            4: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": f"Point Cloud: {pc_ld_shape} Nodes; NaN Segments",
                "text_7": "Current Task: Preparing for instance segmentation...",
            },
            5: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": f"Point Cloud: {pc_ld_shape} Nodes; NaN Segments",
                "text_7": "Current Task: DIST prediction...",
                "text_8": (
                    print_progress_bar(0, len(self.coords_df))
                    if self.tardis_logo
                    else ""
                ),
            },
            6.1: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": f"Point Cloud: {pc_ld_shape} Nodes; NaN Segments",
                "text_7": "Current Task: Instance Segmentation...",
                "text_8": "Filament segmentation is fitted to:",
                "text_9": f"pixel size: {self.px}; transformation: {self.transformation}",
            },
            6.2: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": f"Point Cloud: {pc_ld_shape}; NaN Segments",
                "text_7": "Current Task: Instance segmentation...",
            },
            7: {
                **common_text,
                "text_4": f"Org. Pixel size: {self.px} A | Norm. Pixel size: {self.normalize_px}",
                "text_5": f"Point Cloud: {pc_ld_shape} Nodes; {no_segments} Segments",
                "text_7": "Current Task: Segmentation finished!",
            },
        }

        # Retrieve text configuration based on log_id
        config = text_configurations.get(log_id)

        # Tardis progress bar update
        self.tardis_progress(title=self.title, **config)

    def save_semantic_mask(self, i):
        """
        Saves a semantic mask prediction in a specified format and logs the prediction
        details. Supported formats include MRC, TIF, AM, and NPY. The function also
        updates a log file with prediction details and writes the semantic mask output
        to the appropriate directory in the chosen format.

        :param i: The input file name used to derive the output file name.
        :type i: str

        :raises IOError: If there are issues writing the output files or logs.
        :raises ValueError: If the specified output format is unsupported.
        """
        self.log_prediction.append(
            f"Semantic Prediction: {i[:-self.in_format]}"
            f" | Number of pixels: {np.sum(self.image)}"
        )

        with open(join(self.am_output, "prediction_log.txt"), "w") as f:
            f.write(" \n".join(self.log_prediction))

        if self.output_format.startswith("mrc"):
            to_mrc(
                data=self.image,
                file_dir=join(self.am_output, f"{i[:-self.in_format]}_semantic.mrc"),
                pixel_size=(self.px if self.correct_px is None else self.correct_px),
                label=self.semantic_header,
            )
        elif self.output_format.startswith("tif"):
            tif.imwrite(
                join(self.am_output, f"{i[:-self.in_format]}_semantic.tif"),
                (self.image if i.endswith((".mrc", ".rec")) else self.image),
            )
        elif self.output_format.startswith("am"):
            to_am(
                data=self.image,
                file_dir=join(self.am_output, f"{i[:-self.in_format]}_semantic.am"),
                pixel_size=(self.px if self.correct_px is None else self.correct_px),
                header=self.semantic_header,
            )
        elif self.output_format.startswith("npy"):
            np.save(
                join(self.am_output, f"{i[:-self.in_format]}_semantic.npy"), self.image
            )

    def save_instance_PC(self, i, overwrite_save=False):
        """
        Save processed prediction instance data to disk in various formats. This method handles
        logging, filtering, and outputting of prediction data based on the specified output format
        or other input parameters. It supports multiple output types like CSV, MRC, TIF, AM, STL,
        and NPY for different prediction types such as "Actin", "Microtubule", "Membrane",
        and general filaments or objects. Depending on the output format, it can further refine
        data through filtering, save semantic masks, or interface with Amira for spatial graph
        comparison and exportation.

        :param i: The identifier for the instance being saved.
        :type i: str
        :param overwrite_save: A flag to denote whether an existing file should be overwritten.
                               Defaults to False.
        :type overwrite_save: bool
        :return: None
        """
        self.log_prediction.append(
            f"Instance Prediction: {i[:-self.in_format]}; Number of segments: {np.max(self.segments[:, 0])+1}"
        )

        if self.predict in [
            "Actin",
            "Microtubule",
            "General_filament",
            "Microtubule_tirf",
        ]:
            try:
                self.segments_filter = self.filter_splines(segments=self.segments,
                                                           px=None if self.predict == "Microtubule_tirf" else self.px)
                self.segments_filter = sort_by_length(self.segments_filter)

                self.log_prediction.append(
                    f"Instance Prediction: {i[:-self.in_format]};"
                    f" Number of segments: {np.max(self.segments_filter[:, 0])+1}"
                )
            except ValueError:
                self.segments_filter = None

        with open(join(self.am_output, "prediction_log.txt"), "w") as f:
            f.write(" \n".join(self.log_prediction))

        if self.output_format.endswith("amSG"):
            if self.predict in [
                "Actin",
                "Microtubule",
                "General_filament",
                "Microtubule_tirf",
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

                if self.segments_filter is not None:
                    self.amira_file.export_amira(
                        coords=self.segments_filter,
                        file_dir=join(
                            self.am_output,
                            f"{i[:-self.in_format]}_SpatialGraph_filter.am",
                        ),
                        labels=["TardisPrediction"],
                        scores=[
                            ["EdgeLength", "EdgeConfidenceScore"],
                            [
                                length_list(self.segments_filter),
                                self.score_splines(self.segments_filter),
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
                            if self.segments_filter is not None:
                                compare_sg, label_sg = self.compare_spline(
                                    amira_sg=amira_sg, tardis_sg=self.segments_filter
                                )
                            else:
                                compare_sg, label_sg = self.compare_spline(
                                    amira_sg=amira_sg, tardis_sg=self.segments
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
            else:
                self.amira_file_points.export_amira(
                    coords=self.segments,
                    file_dir=join(
                        self.am_output, f"{i[:-self.in_format]}_SpatialGraph.am"
                    ),
                    labels=["TardisPrediction"],
                )
                if self.segments_filter is not None:
                    self.amira_file_points.export_amira(
                        coords=self.segments_filter,
                        file_dir=join(
                            self.am_output,
                            f"{i[:-self.in_format]}_SpatialGraph_filter.am",
                        ),
                        labels=["TardisPrediction"],
                    )
        elif self.output_format.endswith("csv") or overwrite_save:
            segments = pd.DataFrame(self.segments)
            segments.to_csv(
                join(self.am_output, f"{i[:-self.in_format]}_instances.csv"),
                header=["IDs", "X [A]", "Y [A]", "Z [A]"],
                index=False,
                sep=",",
            )

            if self.predict in [
                "Actin",
                "Microtubule",
                "Microtubule_tirf",
                "Membrane2D",
                "General_filament",
            ]:
                if self.segments_filter is not None:
                    if len(self.segments_filter) > 0:
                        self.segments_filter = pd.DataFrame(self.segments_filter)
                        self.segments_filter.to_csv(
                            join(
                                self.am_output,
                                f"{i[:-self.in_format]}_instances_filter.csv",
                            ),
                            header=["IDs", "X [A]", "Y [A]", "Z [A]"],
                            index=False,
                            sep=",",
                        )
                        self.segments_filter = self.segments_filter.to_numpy()
        elif self.output_format.endswith(("mrc", "tif", "am")):
            if self.predict in [
                "Membrane",
                "Membrane2D",
                "General_object",
                "Microtubule_tirf",
            ]:
                self.mask_semantic = draw_semantic_membrane(
                    mask_size=self.org_shape,
                    coordinate=self.segments,
                    pixel_size=(
                        self.px if self.correct_px is None else self.correct_px
                    ),
                    spline_size=(
                        60
                        if self.predict
                        not in [
                            "Microtubule_tirf",
                            "Microtubule",
                            "General_filament",
                            "Actin",
                        ]
                        else 25
                    ),
                )
            else:
                self.segments[:, 1:] = (
                    self.segments[:, 1:] / self.px
                    if self.correct_px is None
                    else self.correct_px
                )

                self.mask_semantic = draw_instances(
                    mask_size=self.org_shape,
                    coordinate=self.segments,
                    pixel_size=(
                        self.px if self.correct_px is None else self.correct_px
                    ),
                    circle_size=125,
                )
            self._debug(id_name=i, debug_id="instance_mask")

            if self.output_format.endswith("mrc"):
                to_mrc(
                    data=self.mask_semantic,
                    file_dir=join(
                        self.am_output, f"{i[:-self.in_format]}_instance.mrc"
                    ),
                    pixel_size=(
                        self.px if self.correct_px is None else self.correct_px
                    ),
                    label=self.instance_header,
                )
            elif self.output_format.endswith("tif"):
                tif.imwrite(
                    join(self.am_output, f"{i[:-self.in_format]}_instance.tif"),
                    self.mask_semantic,
                )
            elif self.output_format.endswith("am"):
                to_am(
                    data=self.mask_semantic,
                    file_dir=join(self.am_output, f"{i[:-self.in_format]}_instance.am"),
                    pixel_size=(
                        self.px if self.correct_px is None else self.correct_px
                    ),
                    header=self.instance_header,
                )
        elif self.output_format.endswith("stl"):
            if self.predict == "Membrane":
                to_stl(
                    data=self.segments,
                    file_dir=join(self.am_output, f"{i[:-self.in_format]}.stl"),
                )
        elif self.output_format.endswith("npy"):
            np.save(
                join(self.am_output, f"{i[:-self.in_format]}_instance.npy"),
                self.segments,
            )

    def _debug(self, id_name: str, debug_id: str):
        """
        Executes specific debug procedures based on the provided debug identifier. Depending
        on the `debug_id` and other instance properties, it processes and writes files
        including TIFF images, NPY arrays, or other data outputs to the defined output
        directory.

        :param id_name: Name or identifier for the current operation or file being processed.
        :param debug_id: Debug identifier to specify the type of debug operation to perform.
                         Acceptable values include "cnn", "pc", "graph", "segment", and
                         "instance_mask".
        :return: None
        """
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

    def __call__(self, save_progres=False):
        """
        Executes the object as a callable, processing and predicting on a dataset of files
        using a combination of semantic segmentation and instance segmentation workflows.
        The function involves data loading, pre-processing, neural network (CNN) inference,
        post-processing, and saving results. Predictions can be returned optionally if
        specified in the output format.

        The function supports multiple input formats and provides options for binary masks,
        instance-level predictions, and debug checkpoints. Additionally, it handles errors,
        progress updates via a Tardis logger, and optional clean-up of temporary directories.
        Intermediate and final results, like semantic masks or instance segmentations
        from DIST predictions, are saved and optionally returned for further analysis.

        :param save_progres: Boolean flag indicating whether to overwrite and save the
            progress of processing for each file. Default is False.
        :type save_progres: bool
        :return: Depending on the output format, returns semantic segmentation outputs,
            instance segmentation outputs, and instance-filtered outputs in the form
            of lists, where each list entry corresponds to the prediction of an individual file.
        :rtype: tuple or list
        """
        self.get_file_list()

        semantic_output, instance_output, instance_filter_output = [], [], []
        for id_, i in enumerate(self.predict_list):
            start_predict = time.time()

            """CNN Pre-Processing"""
            if isinstance(i, str):
                # Find a file format
                self.in_format = 0
                if i.endswith((".tif", ".mrc", ".rec", ".map", ".npy")):
                    self.in_format = 4
                elif i.endswith(".tiff"):
                    self.in_format = 5
                elif i.endswith(".am"):
                    self.in_format = 3

            # Tardis progress bar update
            self.log_tardis(id_, i, log_id=0)

            # Load data
            self.load_data(id_name=i)

            msg = (
                f"Predicted file {id_} is numpy array without pixel size metadate {self.px}."
                "Please pass correct_px argument as a correct pixel size value."
            )
            assert_b = self.px is None and not isinstance(i, str)
            if assert_b:
                if self.tardis_logo:
                    TardisError(id_="161", py="tardis_em.utils.predictor.py", desc=msg)
                    sys.exit()
                else:
                    assert not assert_b, msg

            # Tardis progress bar update
            self.log_tardis(id_, i, log_id=1)

            """Semantic Segmentation"""
            if not self.binary_mask:
                trim_with_stride(
                    image=self.image,
                    scale=self.scale_shape,
                    trim_size_xy=self.patch_size,
                    trim_size_z=self.patch_size,
                    output=join(self.dir, "temp", "Patches"),
                    image_counter=0,
                    clean_empty=False,
                    stride=round(self.patch_size * 0.125),
                )
                self.image = None

                """CNN prediction"""
                self.predict_cnn(
                    id_i=id_,
                    id_name=i,
                    dataloader=PredictionDataset(
                        join(self.dir, "temp", "Patches", "imgs")
                    ),
                )

                """CNN Post-Processing"""
                # Tardis progress bar update
                self.log_tardis(id_, i, log_id=2)

                # Post-process
                self.postprocess_CNN(id_name=i)

                # Store optionally for return
                if self.output_format.startswith("return"):
                    semantic_output.append(self.image)

                # Check if the image is binary
                if self.image is not None and not self.output_format.startswith(
                    "return"
                ):
                    if len(pd.unique(self.image.flatten())) == 1:
                        self.image = None
                        self.log_prediction.append(
                            f"Semantic prediction: No semantic detected."
                        )
                    else:
                        end_predict = time.time()
                        half_time = round((end_predict - start_predict) / 60, 1)
                        self.log_prediction.append(
                            f"Semantic Prediction Finished in: {half_time} min"
                        )

                    with open(join(self.am_output, "prediction_log.txt"), "w") as f:
                        f.write(" \n".join(self.log_prediction))

                if self.image is None:
                    continue

                # Debug flag
                self._debug(id_name=i, debug_id="cnn")

                # Check if predicted image
                assert_b = self.image.shape == self.org_shape
                if not assert_b:
                    msg = (
                        "Last Task: Stitching/Scaling/Make correction..."
                        f"Tardis Error: Error while converting to {self.px} A "
                        f"Org. shape {self.org_shape} is not the same as "
                        f"converted shape {self.image.shape}"
                    )
                    if self.tardis_logo:
                        TardisError(
                            id_="116",
                            py="tardis_em/utils/predictor.py",
                            desc=msg,
                        )
                    else:
                        assert assert_b, msg
                    sys.exit()

                # Save predicted mask as file
                self.save_semantic_mask(i)

                # Sanity check for binary mask
                if not self.image.min() == 0 and not self.image.max() == 1:
                    continue

                # Close iter loop if not instance prediction
                if not self.predict_instance:
                    continue

            """Instance Segmentation"""
            # Tardis progress bar update
            self.log_tardis(id_, i, log_id=3)

            self.preprocess_DIST(id_name=i)
            self.segments = np.zeros((0, 4))
            self.segments_filter = None

            # Break iter loop for instances if no point cloud is found
            if len(self.pc_ld) == 0:
                self.log_prediction.append(
                    f"Instance prediction: Not enough point from semantic mask were generated."
                )

                if self.output_format.endswith("return"):
                    instance_output.append(np.zeros((0, 4)))
                    instance_filter_output.append(np.zeros((0, 4)))
                continue
            else:
                half_time = round((time.time() - end_predict) / 60, 1)
                self.log_prediction.append(
                    f"Instance Prediction Finished in: {half_time} min"
                )
            with open(join(self.am_output, "prediction_log.txt"), "w") as f:
                f.write(" \n".join(self.log_prediction))

            # Tardis progress bar update
            self.log_tardis(id_, i, log_id=4)

            # Build patches dataset
            (
                self.coords_df,
                _,
                self.output_idx,
                _,
            ) = self.patch_pc.patched_dataset(coord=self.pc_ld)

            # Predict point cloud
            self.log_tardis(id_, i, log_id=5)

            # DIST prediction
            self.graphs = self.predict_DIST(id_i=id_, id_name=i)
            self._debug(id_name=i, debug_id="graph")
            # Save debugging check point
            self._debug(id_name=i, debug_id="segment")

            # DIST Instance graph-cut
            self.postprocess_DIST(id_, i)

            if self.segments is None:
                self.log_prediction.append(
                    f"Instance prediction: Not segments could be predicted with point cloud generated from Semantic mask."
                )
            self.log_prediction.append('\n')
            with open(join(self.am_output, "prediction_log.txt"), "w") as f:
                f.write(" \n".join(self.log_prediction))

                continue

            self.log_tardis(id_, i, log_id=7)

            """Save as .am"""
            self.save_instance_PC(i, overwrite_save=save_progres)

            if self.segments is None:
                if self.output_format.endswith("return"):
                    instance_output.append(np.zeros((0, 4)))
                    instance_filter_output.append(np.zeros((0, 4)))
            else:
                instance_output.append(self.segments)
                if self.predict in [
                    "Actin",
                    "Microtubule",
                    "General_filament",
                    "Microtubule_tirf",
                ]:
                    instance_filter_output.append(self.segments_filter)

            """Clean-up temp dir"""
            clean_up(dir_s=self.dir)
            end_predict = time.time()

            self.eta_predict = (
                str(
                    round(
                        (
                            (end_predict - start_predict)
                            * (len(self.predict_list) - id_ - 1)
                        )
                        / 60,
                        1,
                    )
                )
                + " min"
            )

        """Optional return"""
        if self.output_format.startswith("return"):
            if self.output_format.endswith("return"):
                return semantic_output, instance_output, instance_filter_output
            return semantic_output


class Predictor:
    """
    Handles model prediction workflows for neural networks, including loading pretrained
    weights, configuring model architectures dynamically, and predicting data. The
    purpose of this class is to abstract away the complexities of network setup and
    enhance user focus on utilizing pretrained networks, streamlining predictions.

    The class ensures compatibility with various deep learning frameworks, supports
    CNNs and distance-based networks dynamically, and provides inference-time adjustments like
    rotations for robustness.
    """

    def __init__(
        self,
        device: torch.device,
        network: Optional[str] = None,
        checkpoint: Optional[str] = None,
        subtype: Optional[str] = None,
        model_version: Optional[int] = None,
        img_size: Optional[int] = None,
        model_type: Optional[str] = None,
        sigma: Optional[float] = None,
        sigmoid=True,
        _2d=False,
        logo=True,
    ):
        """
        This class initializer is responsible for setting up a model in the
        TARDIS framework. It includes initializing the model's structure,
        loading weights from a checkpoint or AWS, configuring model parameters,
        and setting up essential model properties like sigmoid activation,
        coordinate embeddings, and device placement.

        :param device: Specifies the torch device (e.g., CPU or GPU)
                       on which the model will be loaded.
        :type device: torch.device
        :param network: The name of the network to be used. If None, the
                        function will automatically determine it based on
                        the model structure.
        :type network: Optional[str]
        :param checkpoint: Specifies the path to the model checkpoint file or
                           directly a preloaded weights dictionary. If not
                           provided, it attempts to find weights on AWS.
        :type checkpoint: Optional[str]
        :param subtype: Defines the network subtype to search specific weights.
                        This is typically used to distinguish between variants
                        of the same base network.
        :type subtype: Optional[str]
        :param model_version: Indicates the model version to use for locating
                              specific weights on AWS.
        :type model_version: Optional[int]
        :param img_size: Determines the image size to which the model will
                         process input. This is primarily for patch-based CNNs.
        :type img_size: Optional[int]
        :param model_type: The type or category of the model (e.g., cnn or dist).
        :type model_type: Optional[str]
        :param sigma: Used for setting the sigma value for coordinate embedding
                      if provided. It can also overwrite the value in weights
                      dictionary.
        :type sigma: Optional[float]
        :param sigmoid: A boolean flag to determine whether sigmoid activation
                        should be applied to the model's output.
        :type sigmoid: bool
        :param _2d: Indicates whether the model processes 2D data (True) or
                    3D data (False).
        :type _2d: bool
        :param logo: A flag that controls whether to enforce specific checks
                     like model network and weights validation on initialization.
        :type logo: bool

        :raises TardisError: Custom TARDIS-related error when neither network
                             nor checkpoint is provided in certain scenarios.
        :raises AssertionError: Raised when logo is False and both network and
                                checkpoint parameters are not defined.
        """
        self.logo = logo

        self.device = device
        self.img_size = img_size

        msg = "Missing network weights or network name!"
        if self.logo:
            if checkpoint is None and network is None:
                TardisError(
                    "139",
                    "tardis_em/utils/predictor.py",
                    msg,
                )
        else:
            assert checkpoint is not None and network is not None, msg

        if checkpoint is None:
            print(
                f"Searching for weight file for {network}_{subtype} for {model_type}..."
            )

            weights = torch.load(
                get_weights_aws(network, subtype, model_type, model_version),
                map_location="cpu",
                weights_only=False,
            )
        elif isinstance(checkpoint, dict):
            print("Loading weight dictionary...")
            weights = checkpoint
        else:
            print("Loading weight model...")
            weights = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # Load model
        if isinstance(weights, dict):  # Deprecated support for dictionary
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
        else:  # Load onnx or another model
            self.network = network
            self._2d = _2d
            self.model = weights

            if not network == "dist":
                self.model.update_patch_size(self.img_size, sigmoid)
        self.model.to(self.device)
        self.model.eval()

        del weights  # Cleanup weight file from memory

    def _build_model_from_checkpoint(self, structure: dict, sigmoid=True):
        """
        Builds a model from the given checkpoint structure.

        This method constructs a machine learning model based on a provided
        configuration dictionary. The configuration can specify different
        types of networks such as `dist_type` or `cnn_type`. In case the
        structure does not match these options, it will return `None`.
        For `dist_type`, it creates a distribution-based network, while for
        `cnn_type`, it constructs a CNN-based network.

        :param structure: A dictionary containing the model configuration
                          blueprint, including the type of network
                          (`dist_type` or `cnn_type`) and other necessary
                          parameters.
        :type structure: dict
        :param sigmoid: A boolean indicating whether the model's final
                        output should apply a sigmoid function or not.
                        Defaults to ``True``.
        :type sigmoid: bool
        :return: A constructed model instance based on the specified
                 structure and type, or ``None`` if no valid network type
                 is provided.
        :rtype: object | None
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

        return model

    def predict(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, rotate=False
    ) -> np.ndarray:
        """
        Predicts an output based on given input data using a trained model. The method supports
        various modes of operation, including computing outputs for specific network types or
        applying rotations to the input for models with two-dimensional or three-dimensional
        spatial components. This function can handle data provided as PyTorch tensors or convert
        NumPy arrays into tensors internally. Outputs are generated either in a transformed or
        direct form according to the network type and additional parameters provided.

        :param x: Input tensor containing the primary data for prediction. Expected to have
                  shapes conforming to the model's requirements.
        :type x: torch.Tensor
        :param y: Optional secondary input tensor containing additional data or node features.
                  Defaults to None. Expected to match the compatible input feature dimensions
                  of the model if provided.
        :type y: Optional[torch.Tensor]
        :param rotate: Boolean flag indicating whether rotations should be applied to the input
                       tensor to generate averaged transformed outputs. Defaults to False.
        :type rotate: bool
        :return: The processed output from the model, structured as a NumPy array. Its dimensions
                 and content correspond to the defined task of the provided trained model. Output
                 is adjusted according to whether rotations were applied or if specific
                 computations are required by the network type.
        :rtype: np.ndarray
        """
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.Tensor(y)

        with torch.no_grad():
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
