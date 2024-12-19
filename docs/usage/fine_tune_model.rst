.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: guilabel

Tuning new CNN module for specific dataset
-------------------------------------------

TARDIS has general training and prediction model build in. With TARDIS you can
train your UNet, ResNet, UNet3Plus and our custom build FNet CNN model.

Training TARDIS CNN models
==========================

1. Data preparation
~~~~~~~~~~~~~~~~~~~

TARDIS requires a folder containing training datasets as input. The dataset should include images
in formats such as .tif, .mrc, .rec, or .am, and corresponding binary masks/labels in .tif, .mrc, .rec, .am, or .csv formats.
The images and mask files must share the same names, with mask files having an additional _mask prefix.

E.g. For example image.mrc and image_mask.mrc or image_mask.csv

Tips:
    - For .mrc files:
        Ensure your data is not flipped. You can verify this by loading your file in Fiji and checking if the image and mask file overlap.
    - Normalization:
        No normalization is required. It is best to use raw image files, as TARDIS will handle all necessary normalizations.


2. Training
~~~~~~~~~~~

To train a CNN model with TARDIS, use the following command to get detailed information:

.. code-block::

    tardis_cnn_train --help

Basic usage:
````````````

To run training. To start training, use the following command:

.. code-block::

    tardis_cnn_train -dir <path-to-your-dataset; str> -ps <patch_size; int> -cnn <cnn_type; str> -b <batch_size> -cs 3gcl <or 2gcl for 2D>


TARDIS will create a folder named cnn_model_checkpoint, containing files named `*_checkpoint.pth` and `model_weights.pth`.

- For re-training the model, use the `*_checkpoint.pth` file.
- For predictions with TARDIS-em, use the `model_weights.pth` file.

Advance usage:
``````````````
Below you can find all available arguments you can use with :bash:`tardis_mt`,
with the explanation for their functionality:

:bash:`-dir` or :bash:`--path`: Directory path with all dataset and labels.
    - :guilabel:`default:` Current command line directory.
    - :guilabel:`type:` str

:bash:`-ps` or :bash:`--patch_size`: Image size used for prediction.
    - :guilabel:`default:` 64
    - :guilabel:`type:` int

:bash:`-ms` or :bash:`--mask_size`: Size of drawn mask in A. If you are using .csv files as labels.
    - :guilabel:`default:` 150
    - :guilabel:`type:` int

:bash:`-cnn` or :bash:`--cnn_type`: Type of NN used for training.
    - :guilabel:`default:` fnet_attn
    - :guilabel:`type:` str
    - :guilabel:`options:` unet, resunet, unet3plus, big_unet, fnet, fnet_attn

:bash:`-co` or :bash:`--cnn_out_channel`: Number of output channels for the NN.
    - :guilabel:`default:` 1
    - :guilabel:`type:` int

:bash:`-b` or :bash:`--training_batch_size`: Batch size.
    - :guilabel:`default:` 25
    - :guilabel:`type:` int

:bash:`-cl` or :bash:`--cnn_layers`: Number of convolution layer for NN.
    - :guilabel:`default:` 5
    - :guilabel:`type:` int

:bash:`-cm` or :bash:`--cnn_scaler`: Convolution multiplayer for CNN layers.
    - :guilabel:`default:` 32
    - :guilabel:`type:` int

:bash:`-cs` or :bash:`--cnn_structure`: Define structure of the convolution layer.
    - :guilabel:`default:` 3gcl
    - :guilabel:`type:` str
    - :guilabel:`options:` 2 or 3 - dimension in 2D or 3D;  c - convolution;    g - group normalization;    b - batch normalization;    r - ReLU;   l - LeakyReLU;  e - GeLu;   p - PReLu

:bash:`-ck` or :bash:`--conv_kernel`: Kernel size for 2D or 3D convolution.
    - :guilabel:`default:` 3
    - :guilabel:`type:` int

:bash:`-cp` or :bash:`--conv_padding`: Padding size for convolution.
    - :guilabel:`default:` 1
    - :guilabel:`type:` int

:bash:`-cmpk` or :bash:`--pool_kernel`: Max_pooling kernel.
    - :guilabel:`default:` 2
    - :guilabel:`type:` int

:bash:`-l` or :bash:`--cnn_loss`: Loss function use for training.
    - :guilabel:`default:` BCELoss
    - :guilabel:`type:` str
    - :guilabel:`options:` AdaptiveDiceLoss, BCELoss, WBCELoss, BCEDiceLoss, CELoss, DiceLoss, ClDiceLoss, ClBCELoss, SigmoidFocalLoss, LaplacianEigenmapsLoss, BCEMSELoss

:bash:`-lr` or :bash:`--loss_lr_rate`: Learning rate for NN.
    - :guilabel:`default:` 0.0005
    - :guilabel:`type:` float

:bash:`-lrs` or :bash:`--lr_rate_schedule`: If True learning rate scheduler is used.
    - :guilabel:`default:` False
    - :guilabel:`type:` bool

:bash:`-dv` or :bash:`--device`: Define which device use for training:
    - :guilabel:`default:` 0
    - :guilabel:`type:` str
    - :guilabel:`options:` gpu - Use ID 0 gpus;  cpu - Usa CPU; mps - Apple silicon; 0-9 - specified gpu device id to use

:bash:`-w` or :bash:`--warmup`: Number of warmup steps.
    - :guilabel:`default:` 100
    - :guilabel:`type:` int

:bash:`-e` or :bash:`--epochs`: Number of epoches.
    - :guilabel:`default:` 10000
    - :guilabel:`type:` int

:bash:`-es` or :bash:`--early_stop`: Number of epoches without improvement after which early stop is initiated. Default should is 10% of the total number of epochs.
    - :guilabel:`default:` 1000
    - :guilabel:`type:` int

:bash:`-cch` or :bash:`--cnn_checkpoint`: If indicated, dir to training checkpoint to reinitialized training.
    - :guilabel:`default:` None
    - :guilabel:`type:` str

:bash:`-dp` or :bash:`--dropout_rate`: If indicated, value of dropout for CNN.
    - :guilabel:`default:` 0.5
    - :guilabel:`type:` float

3.1 Pre-train model from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run re-training:

.. code-block::

    tardis_cnn_train -dir <path-to-your-dataset; str> -ps <patch_size; int> -cnn <cnn_type; str> -b <batch_size> -cch <checkpoint.pth_file_dir>

3.2 Fine-tune existing models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All TARDIS models are stored locally in ~/tardis_em/

For example a default model for membrane segmentation can be found in

.. code-block::

    ./tardis_em/fnet_attn_32/membrane_3d/model_weights.pth

In order to fine-tune it on your existing data:

.. code-block::

    tardis_cnn_train ... -cch ./tardis_em/fnet_attn_32/membrane_3d/model_weights.pth


4. Predict with train model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To predict images with your newly train model, you can use the following command:

.. code-block::

    tardis_predict --help
    tardis_predict -dir <dir to folder of file to predict> -ch <model_weight.pth_directory> -ps <patch_size> -out <output_format> mrc|tif|rec|am -rt True -ct <CNN_threshold> -dv 0