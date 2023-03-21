TARDIS
======

.. image:: https://img.shields.io/badge/release-0.1.0_RC1-success
        :target: https://img.shields.io/badge/release-0.1.0_RC1-success

.. image:: https://readthedocs.org/projects/tardis-pytorch/badge/?version=latest
        :target: https://tardis-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Python based software for generali instance segmentation of object from electron microscopy (EM) and 
cryo-EM micrographs. Software package is built on general workflow where predicted semantic segmentation
is used for instance segmentation of 2D/3D and 4D/5D fluorescent images in the future.

.. image:: resources/workflow.jpg
        :target: resources/workflow.jpg
        :alt: TARDIS workflow


Features
--------
* Training of Unet/ResNet/Unet3Plus for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
* Point cloud instance segmentation by point cloud graph representation


News
----
.. raw:: html

    <details>
    <summary><b>TARDIS v0.1.0 release candidate 2 (RC2) - (2023-03-21)</b></summary>

    <ul>
    <li> General improvement form MT prediction </li>
    <li> Added support for Cry-mem prediction </li>
    <li> Added support for node (RGB) features in DIST </li>
    <li> Full support for Pytorch 2.0 </li>
    </ul>
    </details>

    <details>
    <summary><b>TARDIS v0.1.0 release candidate 1 (RC1) - (2023-02-08)</b></summary>

    <ul>
    <li> Overall clean-up for final release </li>
    <li> Added full code documentation </li>
    <li> Added full stable suport for MT prediction </li>
    <li> Added support for ScanNetV2 dataset prediction with DIST </li>
    <li> Added costume TARDIS error and console logo outputs </li>
    <li> TARDIS error handling </li>
    </ul>
    </details>

    <details>
    <summary><b>TARDIS v0.1.0 beta 2 - (2022-09-14)</b></summary>

    <ul>
    <li> Cryo-Membrane 2D support </li>
    <li> Stable training and prediction entries for spindletorch and DIST </li>
    <li> Restructure and standardized naming iand versioning in TARDIS </li>
    <li> Combined all side-code into TARDIS </li>
    <li> Full support for Amira formats, MRC/REC, TIF </li>
    </ul>
    </details>

    <details><summary><b>TARDIS v0.1.0 beta 1</b></summary>
    <ul>
    <li> Cryo-Membrane 2D support </li>
    <li> Stable training and prediction entries for spindletorch and DIST </li>
    <li> Restructure and standardized naming iand versioning in TARDIS </li>
    </details>

Requirements
------------
.. code-block::

    python 3.7 or newer


Installation
------------
**From Source**

The sources for TARDIS-pytorch can be downloaded from the ***Available soon***.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
    $ python setup.py install
    $ pip install -r requirements.txt

Or install from pre-build python package:
Install:
    - Python 3.7


Windows x64 and Linux:

.. code-block:: console

    $ conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
    $ pip install ./tardis_pytorch-0.1.0b2-py3-none-any.whl


Known installation errors on Linux:

.. code-block:: console

    OSError: /lib64/libc.so.6: version `GLIBC_2.18' not found

Solution:

.. code-block:: console

    $ pip install open3d==0.9.0
Usage
-----
Prediction of MT from electron tomograms:

.. code-block::

    **All setting:**
    -dir   (str): Directory with electron micrographs   [*.mrc, *.rec, *.am]
    -ps    (int): Patch size used for prediction.       [default: 128].
    -cnn   (str): CNN network name.                     [default: 'fnet_t 0.2 '].
    -cch   (str): If not None, str checkpoints for CNN. [default: None]
    -ct  (float): Threshold use for model prediction.   [default: 0.3]
    -dch   (str): If not None, checkpoints for DIST.    [default: None]
    -dt  (float): Threshold use for graph segmentation. [default: 0.5]
    -pv    (int): Number of point per voxel.            [default: 1000]
    -d     (str): Define which device use for training: [default: 0]
              cpu: cpu
              gpu: 0-9 - specific GPU.
    -db   (bool): If True, save debuting output.        [default: False]
    -v     (str): If not None, output visualization of  [default: None]
              the prediction:
              - f: Output as filaments:
              - p: Output color coded point cloud
    --version     Show the version and exit.
    --help        Show this message and exit.

    **Recommended usage for electron tomograms:**
    $ tardis_mt -dir ./.. -ct 0.2 -pv 1000

    **Recommended usage for cryo-electron tomograms/micrographs:**
    $ tardis_mt -dir ./.. -ct 0.2 -pv 1000
