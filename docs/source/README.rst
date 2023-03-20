TARDIS
======

.. image:: https://img.shields.io/badge/release-0.1.0_beta1-success
        :target: https://img.shields.io/badge/release-0.1.0_beta1-success

.. image:: https://readthedocs.org/projects/tardis-pytorch/badge/?version=latest
        :target: https://tardis-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Python based software for generali instance segmentation of object from electron microscopy (EM) and 
cryo-EM micrographs. Software package is builded on general workflow where predicted semantic segmentation 
is used for instance segmentation of 2D/3D and 4D/5D fluorescent images in the future.

.. image:: ../../resources/workflow.jpg
        :target: ../../resources/workflow.jpg
        :alt: TARDIS workflow


Features
--------
* Training of Unet/ResNet/Unet3Plus for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
* Point cloud instance segmentation by point cloud graph representation

News
----
<details>
  <summary>TARDIS v0.1.0 release candidate 2 (RC2) - (2023-03-21)</summary>
    * General improvement form MT prediction
    * Added support for Cry-mem prediction
    * Added support for node (RGB) features in DIST
    * Full support for Pytorch 2.0
</details>

<details>
  <summary>TARDIS v0.1.0 release candidate 1 (RC1) - (2023-02-08)</summary>
    * Overall clean-up for final release
    * Added full code documentation
    * Added full stable suport for MT prediction
    * Added support for ScanNetV2 dataset prediction with DIST
    * Added costume TARDIS error and console logo outputs
    * TARDIS error handling
</details>

<details>
  <summary>TARDIS v0.1.0 beta 2 - (2022-09-14)</summary>
    * Cryo-Membrane 2D support
    * Stable training and prediction entries for spindletorch and DIST
    * Restructure and standardized naming iand versioning in TARDIS
    * Combined all side-code into TARDIS
    * Full support for Amira formats, MRC/REC, TIF
</details>

<details>
  <summary>TARDIS v0.1.0 beta 1</summary>
    * Cryo-Membrane 2D support
    * Stable training and prediction entries for spindletorch and DIST
    * Restructure and standardized naming iand versioning in TARDIS
</details>

Requirements
------------
.. code-block::

	conda install --file requirements.txt

or install following requirements:
	.. include:: ../../requirements.txt
		:literal:


Installation
------------
**From Source**

The sources for TARDIS-pytorch can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
    $ python setup.py install
    $ pip install -r requirements.txt

.. _Github repo: https://github.com/SMLC-NYSBC/TARDIS-pytorch
.. _tarball: https://github.com/SMLC-NYSBC/TARDIS-pytorch/tarball/master


Usage
-----
**!IMPORTANT!** 

Training expect to be in a directory which contains 2 folders: 
data/train/ and data/test both of which should have ./imgs and ./masks folders

Coming Soon!