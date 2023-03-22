# TARDIS

![Version](https://img.shields.io/badge/release-0.1.0_RC2-success)
[![Python PyTest](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_package.yml)
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
    * Pre-trained network for Cryo-mem, General-MT, S3DIS dataset
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

Or install directly from pre-build python package:

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

### Example:
![Prediction example1](resources/tardis_semantic_mt.jpg)

### Usage:

</details>

<details><summary><i>Instance microtubule prediction:</i></summary>

### Example: 
![Prediction example2](resources/tardis_instance_mt.jpg)

### Usage:
```
recomanded usage: tardis_mt [-dir path/to/folder/with/input/tomogram]
advance usage: tardis_mt [-dir str] [-out str] [-ps int] [-ct float] [-dt float]
                         [-pv int] [-ap str] ...
```
```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show current TARDIS version
  
  
  -dir, --dir [str]     
                        Directory with images for prediction with CNN model.
                        Default: getcwd()
                        
  -out, --output_format [str]
                        Type of output files. The First optional output file is the binary mask 
                        which can be of type None [no output], am [Amira], mrc or tif. 
                        Second output is instance segmentation of objects, which can be 
                        output as amSG [Amira], mrcM [mrc mask], tifM [tif mask],
                        csv coordinate file [ID, X, Y, Z] or None [no instance prediction].
                        Default: None_amSG
  
  -ps, --patch_size [int]
                        Size of image patch used for prediction. This will break 
                        the tomogram volumes into 3D patches where each patch will be
                        separately predicted and then stitched back together 
                        with 25% overlap.
                        Default: 128
                        
  -ct, --cnn_threshold [float]
                        Threshold used for CNN prediction.
                        Default: 0.5

  -dt, --dist_threshold [float]
                        Threshold used for instance prediction.
                        Default: 0.75

  -pv, --points_in_patch [int]
                        Size of the cropped point cloud, given as a max. number of points
                        per crop. This will break generated from the binary mask
                        point cloud into smaller patches with overlap. 
                        Default: 1000

  -ap, --amira_prefix [str]
                        If dir/amira foldr exist, TARDIS will serach for files with
                        given prefix (e.g. file_name.CorrelationLines.am). If the correct
                        file is found, TARDIS will use its instance segmentation with
                        ZiB Amira prediction, and output additional file called
                        file_name_AmiraCompare.am.
                        Default: .CorrelationLines                       
 
  -fl, --filter_by_length [int]
                        Filtering parameters for microtubules, defining maximum microtubule 
                        length in angstrom. All filaments shorter then this length 
                        will be deleted.
                        Default: 500
                        
  -cs, --connect_splines [int]
                        Filtering parameter for microtubules. Some microtubules may be 
                        predicted incorrectly as two separate filaments. To overcome this
                        during filtering for each spline, we determine the vector in which 
                        filament end is facing and we connect all filament that faces 
                        the same direction and are within the given connection 
                        distance in angstrom.
                        Default: 2500

  -cr, --connect_cylinder [int]
                        Filtering parameter for microtubules. To reduce false positive 
                        from connecting filaments, we reduce the searching are to cylinder 
                        radius given in angstrom. For each spline we determine vector 
                        in which filament end is facing and we search for a filament 
                        that faces the same direction and their end can be found 
                        within a cylinder.
                        Default: 250
  
  -acd, --amira_compare_distance [int]
                        If dir/amira/file_amira_prefix.am is recognized, TARDIS runs
                        a comparison between its instance segmentation and ZiB Amira prediction.
                        The comparison is done by evaluating the distance of two filaments from
                        each other. This parameter defines the maximum distance used to 
                        evaluate the similarity between two splines based on their 
                        coordinates [A].
                        Default: 175

  -aip, --amira_inter_probability [flaot]
                        If dir/amira/file_amira_prefix.am is recognized, TARDIS runs
                        a comparison between its instance segmentation and ZiB Amira prediction.
                        This parameter defines the interaction threshold used to identify splines 
                        that are similar overlaps between TARDIS and ZiB Amira.
                        Default: 0.25
                        
 -dv, --device [str]
                        Define which device to use for training:
                        * gpu: Use ID 0 GPU
                        * cpu: Usa only CPU
                        * mps: Apple silicon (experimental)
                        * 0-9 - specified GPU device id to use    

  -db, --debug [bool]
                        If True, save the output from each step for debugging.
                        Default: False                          
```

</details>

</details>

<details><summary><b>Membrane Prediction</b></summary>

<details><summary><i>Semantic membrane prediction:</i></summary>

### Example: 
![Prediction example3](resources/tardis_semantic_mem.jpg)

### Usage:
```
recomanded usage: tardis_mem [-dir path/to/folder/with/input/tomogram]
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...
```
```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show current TARDIS version
  
  
  -dir, --dir [str]     
                        Directory with images for prediction with CNN model.
                        Default: getcwd()
                        
  -out, --output_format [str]
                        Type of output files. The First optional output file is the binary mask 
                        which can be of type None [no output], am [Amira], mrc or tif. 
                        Second output is instance segmentation of objects, which can be 
                        output as amSG [Amira], mrcM [mrc mask], tifM [tif mask],
                        csv coordinate file [ID, X, Y, Z] or None [no instance prediction].
                        Default: mrc_None
  
  -ps, --patch_size [int]
                        Size of image patch used for prediction. This will break 
                        the tomogram volumes into 3D patches where each patch will be
                        separately predicted and then stitched back together 
                        with 25% overlap.
                        Default: 128
                        
  -ct, --cnn_threshold [float]
                        Threshold used for CNN prediction.
                        Default: 0.15

  -dt, --dist_threshold [float]
                        Threshold used for instance prediction.
                        Default: 0.95

  -pv, --points_in_patch [int]
                        Size of the cropped point cloud, given as a max. number of points
                        per crop. This will break generated from the binary mask
                        point cloud into smaller patches with overlap. 
                        Default: 1000
                        
 -dv, --device [str]
                        Define which device to use for training:
                        * gpu: Use ID 0 GPU
                        * cpu: Usa only CPU
                        * mps: Apple silicon (experimental)
                        * 0-9 - specified GPU device id to use    

  -db, --debug [bool]
                        If True, save the output from each step for debugging.
                        Default: False  
```

</details>

<details><summary><i>Instance membrane prediction*:</i></summary>

### Example: 
![Prediction example4](resources/tardis_instance_mem.jpg)

*Stable support for membrane instance segmentation is expected in TARDIS-0.1.0-RC3.
TARDIS from v0.1.0-RC2 allows for instance membrane segmentation. Results may vary.

### Usage:
```
recomanded usage: tardis_mem [-dir path/to/folder/with/input/tomogram] [-out mrc_mrcM]
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...
```
```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show current TARDIS version
  
  
  -dir, --dir [str]     
                        Directory with images for prediction with CNN model.
                        Default: getcwd()
                        
  -out, --output_format [str]
                        Type of output files. The First optional output file is the binary mask 
                        which can be of type None [no output], am [Amira], mrc or tif. 
                        Second output is instance segmentation of objects, which can be 
                        output as amSG [Amira], mrcM [mrc mask], tifM [tif mask],
                        csv coordinate file [ID, X, Y, Z] or None [no instance prediction].
                        Default: mrc_None
  
  -ps, --patch_size [int]
                        Size of image patch used for prediction. This will break 
                        the tomogram volumes into 3D patches where each patch will be
                        separately predicted and then stitched back together 
                        with 25% overla
                        Default: 128
                        
  -ct, --cnn_threshold [float]
                        Threshold used for CNN prediction.
                        Default: 0.5

  -dt, --dist_threshold [float]
                        Threshold used for instance prediction.
                        Default: 0.5

  -pv, --points_in_patch [int]
                        Size of the cropped point cloud, given as a max. number of points
                        per crop. This will break generated from the binary mask
                        point cloud into smaller patches with overlap. 
                        Default: 1000
                        
 -dv, --device [str]
                        Define which device to use for training:
                        * gpu: Use ID 0 GPU
                        * cpu: Usa only CPU
                        * mps: Apple silicon (experimental)
                        * 0-9 - specified GPU device id to use    

  -db, --debug [bool]
                        If True, save the output from each step for debugging.
                        Default: False  
```
</details>

</details>
