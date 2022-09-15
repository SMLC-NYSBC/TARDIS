================================
Transformer And Rapid Dimensionless Instance Segmentation [TARDIS]
================================
.. image:: https://img.shields.io/badge/release-0.1.0_beta1-success
        :target: https://img.shields.io/badge/release-0.1.0_beta1-success

.. image:: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python-package.yml/badge.svg?event=release
        :target: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python-package.yml

.. image:: https://readthedocs.org/projects/tardis-pytorch/badge/?version=latest
        :target: https://tardis-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
....

Python based software for generali instance segmentation of object from electron microscopy (EM) and 
cryo-EM micrographs. Software package is builded on general workflow where predicted semantic segmentation 
is used for instance segmentation of 2D/3D and 4D/5D fluorescent images in the future.

.. image:: /resources/workflow.jpg
        :target: /resources/workflow.jgg
        :alt: TARDIS workflow

Documentation: https://tardis-pytorch.readthedocs.io/en/latest/

Features
--------
* Training of Unet/ResNet/Unet3Plus for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
        * 4D and 5D point clouds segmentation in the future
* Point cloud instance segmentation by point cloud graph representation

============
Requirements
============
  $ conda install --file requirements.txt
  
or install following requirements::

        click>=8.0.4
        edt>=2.1.2
        imagecodecs>=2021.8.26
        numpy>=1.22.3
        open3d>=0.9.0
        scikit-image>=0.19.2
        scikit-learn >=1.0.2
        scipy>=1.8.0
        tifffile>=2022.5.4
        torch>=1.12.0
        tqdm>=4.63.0
        requests>=2.27.1
        zarr>=2.8.1
        pandas>=1.3.5
        opencv-python>=4.5.5.64

============
Installation
============

From sources
------------

The sources for TARDIS-pytorch can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
    $ python setup.py install
    $ pip install -r requirements.txt

.. _Github repo: https://github.com/SMLC-NYSBC/TARDIS-pytorch
.. _tarball: https://github.com/SMLC-NYSBC/TARDIS-pytorch/tarball/master

=====
Usage
=====
**!IMPORTANT!** Training expect to be in a directory which contains 2 folders: 
data/train/ and data/test both of which should have ./imgs and ./masks folders

Training modules:
        Semantic Segmentation with Unet/Unet3Plus/FNet:
.. code-block:: console

        tardis_cnn_train -dir str -ttr float -ps int -px float -cnn str -co int -b int -cl int -cm int -cs str -ck int -cp int -cmpk int -dp None/float -l str -la None/float -lr float -lrs bool -d str -e int -es int -cch None/str

        int    [-dir]   Directory for training and test dataset.
                [-default]      None
        float  [-ttr]   Percentage value of train dataset that will become test.
                [-default]      10
        int    [-ps]    Image size used for perdition.
                [-default]      64
        float  [-px]    Pixel size to which all images are resize.
                [-default]      23.2
        str    [-cnn]   Type of NN used for training.
                [-default]      'Unet'
                [-choice]       'unet', 'resunet', 'unet3plus', 'big_unet', 'fnet'
        int    [-co]    Number of output channels for the NN.
                [-default]      1
        int    [-b]     Batch size.
                [-default]      25
        int    [-cl]    Number of convolution layer for NN.
                [-default]      5
        int    [-cm]    Convolution multiplayer for CNN layers.
                [-default]      64
        str    [-cs]    Define structure of the convolution layer.
                [-default]      '3gcl'
                [-choice]       '2 or 3 - dimension in 2D or 3D'
                                'c - convolution'
                                'g - group normalization'
                                'b - batch normalization'
                                'r - ReLU'
                                'l - LeakyReLU'
        int    [-ck]    Kernel size for 2D or 3D convolution.
                [-default]      3
        int    [-cp]    Padding size for convolution.
                [-default]      1
        int    [-cmpk]    Maxpooling kernel
                [-default]      2
        float  [-dp]    If indicated, value of dropout for CNN.
                [-default]      None
        str    [-l]    Loss function use for training.
                [-default]      'bce'
                [-choice]       'bce', 'dice', 'hybrid', 'adaptive_dice'
        float  [-la]    Value of alpha used for adaptive dice loss.
                [-default]      None
        float  [-lr]    Learning rate for NN.
                [-default]      0.001
        bool   [-lrs]    If True learning rate scheduler is used.
                 [-default]      False
        str    [-d]    Define which device use for training:
                [-default]      0
                [-choice]       'gpu: Use ID 0 gpus'
                                'cpu: Usa CPU'
                                'mps: Apple silicon'
                                '0-9 - specified gpu device id to use'
        int    [-e]    Number of epoches
                [-default]      100
        int    [-es]    Number of epoches without improvement after which early stop is initiated.
                [-default]      10
        str    [-cch]    If indicated, dir to training checkpoint to reinitialized training.
                [-default]      None

        Point cloud instance segmentation
.. code-block:: console


Prediction modules:
        Semantic Segmentation with Unet/Unet3Plus/FNet:
.. code-block:: console

        tardis_cnn_predict -dir str -ps int -cnn str -co int -cl int -cm int -cs str -ck int -cp int -cmpk int -dp None/float -cch (None, None)/ (str, str) -d str -th float -tq bool

        int    [-dir]   Directory for training and test dataset.
                [-default]      None
        int    [-ps]    Image size used for perdition.
                [-default]      64
        str    [-cnn]   Type of NN used for training.
                [-default]      'Unet'
                [-choice]       'unet', 'resunet', 'unet3plus'
        int    [-co]    Number of output channels for the NN.
                [-default]      1
        int    [-b]     Batch size.
                [-default]      25
        int    [-cl]    Number of convolution layer for NN.
                [-default]      5
        int    [-cm]    Convolution multiplayer for CNN layers.
                [-default]      64
        str    [-cs]    Define structure of the convolution layer.
                [-default]      '3gcl'
                [-choice]       '2 or 3 - dimension in 2D or 3D'
                                'c - convolution'
                                'g - group normalization'
                                'b - batch normalization'
                                'r - ReLU'
                                'l - LeakyReLU'
        int    [-ck]    Kernel size for 2D or 3D convolution.
                [-default]      3
        int    [-cp]    Padding size for convolution.
                [-default]      1
        int    [-cmpk]    Maxpooling kernel
                [-default]      2
        float  [-dp]    If indicated, value of dropout for CNN.
                [-default]      None
        str    [-cch]    If indicated, dir to training checkpoint to reinitialized training.
                         None value force to download most up-to-data weights
                [-default]      (None, None)
        str    [-d]    Define which device use for training:
                [-default]      0
                [-choice]       'gpu: Use ID 0 gpus'
                                'cpu: Usa CPU'
                                'mps: Apple silicon'
                                '0-9 - specified gpu device id to use'
        float  [-th]    Threshold use for model prediction.
                [-default]      0.5
        bool  [-tq]    If True, build with progress bar.
                [-default]      True

        Point cloud instance segmentation
.. code-block:: console


        Microtubules segmentation
.. code-block:: console
