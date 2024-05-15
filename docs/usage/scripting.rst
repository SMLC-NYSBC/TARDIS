.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: guilabel

Scripting in Python
-------------------
TARDIS-em library can be use to simply and fast script your own workflows.

More examples can be find in: tardis_em/examples/TARDIS_em_Script.ipynb

Example
-------

.. code-block::

    from tardis_em.utils.predictor import GeneralPredictor

    predictor = GeneralPredictor(
        predict: str,
        dir_: Union[str, tuple[np.ndarray], np.ndarray],
        binary_mask: bool,
        output_format: str,
        patch_size: int,
        convolution_nn: str,
        cnn_threshold: float,
        dist_threshold: float,
        points_in_patch: int,
        predict_with_rotation: bool,
        instances: bool,
        device_: str,
        debug: bool,
        checkpoint: Optional[list] = None,
        correct_px: float = None,
        amira_prefix: str = None,
        filter_by_length: int = None,
        connect_splines: int = None,
        connect_cylinder: int = None,
        amira_compare_distance: int = None,
        amira_inter_probability: float = None,
        tardis_logo: bool = True,
    )

    semantic, instance, instance_filter = predictor()


:bash:`predict`: File directory to visualize.
    - :guilabel:`Allowed options:` Microtubule, Membrane2D, Membrane

:bash:`-dir_`: Directory to a single file, folder with files or numpy array with tomogram/micrograph.
    - :guilabel:`Allowed options:` str, np.ndarray

:bash:`-binary_mask`: If True, Predictor assume, that input images are binary mask. The semantic segmentation step would be skipped and only instance segmentation results will be produce.
    - :guilabel:`Allowed options:` bool

:bash:`-output_format`: Two output format for semantic and instance prediction.
    - :guilabel:`Tips:` Define as `semantic_instance` format. For example to output semantic segmentation mask as .mrc file format, and instance segmentation as .csv file. Type `mrc_csv`
    - :guilabel:`Allowed options Semantics:` None, am, mrc, tif, npy
    - :guilabel:`Allowed options Instances:` None, am, mrc, tif, npy, amSG, csv, stl

:bash:`-patch_size`: Image crop size used during semantic segmentation.
    - :guilabel:`Allowed options:` int

:bash:`-convolution_nn`: Type of pre-train CNN model.
    - :guilabel:`Allowed options:` unet, fnet_attn

:bash:`-cnn_threshold`: Threshold for CNN model. Used during semantic segmentation.
    - :guilabel:`Allowed options:` float

:bash:`-dist_threshold`: Threshold for DIST model. Used during instance segmentation.
    - :guilabel:`Allowed options:` float

:bash:`-points_in_patch`: Maximum number of points per patched point cloud.
    - :guilabel:`Tip`: About 1000 points require ~ 12Gb of GPU or RAM (if device_ == 'cpu')
    - :guilabel:`Allowed options:` int

:bash:`-predict_with_rotation`: If True, CNN predict with 4 90* rotations.
    - :guilabel:`Allowed options:` bool

:bash:`-instances``: If True, run instance segmentation after semantic.
    - :guilabel:`Allowed options:` bool

:bash:`-device_`: Device on which prediction will take place.
    - :guilabel:`Allowed options:` cpu, gpu or number between 0-9 indicating gpu id

:bash:`-debug`: If True, enable debugging mode which save all intermediate files.
    - :guilabel:`Allowed options:` bool

:bash:`-checkpoint`: List of model checkpoints for semantic and instance segmentation. If its None, TARDIS retrieves weights from AWS.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` list[str], list[dict]

:bash:`-correct_px`: Indicate correct pixel size for image data. If its None, TARDIS retrieves pixels size from the file header.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` float, None

:bash:`-amira_prefix``: Optional, Amira file prefix name used for spatial graph comparison.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` str, None

:bash:`-filter_by_length`: Optional, filter setting for filtering short splines. Value given in Angstrom.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` int, None

:bash:`-connect_splines`: Optional, filter setting for connecting near splines. Value given in Angstrom.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` int, None

:bash:`-connect_cylinder`: Optional, filter setting for connecting splines withing cylinder radius. Value given in Angstrom.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` int, None

:bash:`-amira_compare_distance`: Optional, compare setting, max distance between two splines to consider them as the same. Value given in Angstrom.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` int, None

:bash:`-amira_inter_probability`: Optional, compare setting, portability threshold to define comparison class. Value given between 0-1 as a probability.
    - :guilabel:`Default:` None
    - :guilabel:`Allowed options:` float, None

:bash:`-tardis_logo`: If True, GeneralPredictor will display terminal or command-line logs.
    - :guilabel:`Default:` True
    - :guilabel:`Allowed options:` bool
