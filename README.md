# TARDIS

![Version](https://img.shields.io/badge/release-0.1.0_RC2-success)
![Documentation](https://readthedocs.org/projects/tardis-pytorch/badge/?version=latest)

Python based software for generalized object instance segmentation from (cryo-)electron microscopy
micrographs/tomograms. Software package is built on general workflow where predicted semantic segmentation
is used for instance segmentation of 2D/3D images.

![Tardis Workflow](resources/workflow.jpg)

## Features
* Training of Unet/ResNet/Unet3Plus/FNet for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
* Point cloud instance segmentation by point cloud graph representation

## News
<details>
    <summary><b>TARDIS v0.1.0 release candidate 2 (RC2) - (2023-03-22)</b></summary>

    * General improvement form MT prediction
    * Added support for Cry-mem prediction
    * Added support for node (RGB) features in DIST
    * Full support for Pytorch 2.0
</details>

<details>
    <summary><b>TARDIS v0.1.0 release candidate 1 (RC1) - (2023-02-08)</b></summary>

    * Overall clean-up for final release 
    * Added full code documentation
    * Added full stable suport for MT prediction 
    * Added support for ScanNetV2 dataset prediction with DIST 
    * Added costume TARDIS error and console logo outputs 
    * TARDIS error handling 

</details>

<details>
    <summary><b>TARDIS v0.1.0 beta 2 - (2022-09-14)</b></summary>

    * Cryo-Membrane 2D support 
    * Stable training and prediction entries for spindletorch and DIST 
    * Restructure and standardized naming iand versioning in TARDIS 
    * Combined all side-code into TARDIS 
    * Full support for Amira formats, MRC/REC, TIF 

</details>

<details>
    <summary><b>TARDIS v0.1.0 beta 1</b></summary>

    * Cryo-Membrane 2D support 
    * Stable training and prediction entries for spindletorch and DIST 
    * Restructure and standardized naming iand versioning in TARDIS 
</details>

# Requirements

    # Python
        python 3.7, 3.8. 3.9, 3.10

    # ML library
        torch>=1.13.1
        numpy>=1.21.6
        pandas>=1.3.5
    
    # Image loading library
        tifffile>=2021.11.2
        imagecodecs>=2021.11.20
        mrcfile >= 1.4.3
    
    # Image processign library 
        scikit-learn>=1.0.2
        scikit-image>=0.19.2
        opencv-python>=4.6.0.66
        scipy>=1.7.3
        edt>=2.3.0

    # Point cloud processing library
        open3d==0.9.0
    
    # Other
        requests>=2.28.2
        ipython>=7.34.0
        click>=8.0.4
        nvidia-smi>=0.1.3


## Installation
**From Source**

The sources for TARDIS-pytorch can be downloaded from the ***Available upon stable release***.

You can either clone the public repository:

    $ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
    $ python setup.py install
    $ pip install -r requirements.txt
    $ pip install -r requirements-dev.txt

Or install from pre-build python package:

Install:
* Python 3.7, 3.8, 3.9, 3.10

Windows x64 and Linux:

    $ conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    $ pip install ./tardis_pytorch-0.1.0rc2-py3-none-any.whl

MacOS:
    
    $ conda install pytorch -c pytorch
    $ pip install ./tardis_pytorch-0.1.0rc2-py3-none-any.whl

<details><summary><b>Known issues with installation:</b></summary>

Linux:

    Error:
        OSError: /lib64/libc.so.6: version `GLIBC_2.18' not found
    
    Solution:
        $ pip install open3d==0.9.0
</details>

## Usage
<details><summary><b>Microtubule Prediction</b></summary>

<details><summary><i>Semantic microtubule prediction:</i></summary>

![Prediction example1]()

</details>

<details><summary><i>Instance microtubule prediction:</i></summary>

![Prediction example2]()
</details>

</details>


<details><summary><b>Membrane Prediction</b></summary>

<details><summary><i>Semantic membrane prediction:</i></summary>

![Prediction example3]()

</details>

<details><summary><i>Instance membrane prediction*:</i></summary>

![Prediction example4]()

*Stable support for membrane instance segmentation is expected in TARDIS-0.1.0-RC3.
TARDIS from v0.1.0-RC2 allows for instance membrane segmentation. Results may vary.

</details>

</details>
