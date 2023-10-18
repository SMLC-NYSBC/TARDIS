<p align="center">
  <img src="resources/Tardis_logo_2.png" width="512"/>
</p>

![Version](https://img.shields.io/badge/release-0.1.0-success)
[![Python PyTest](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_pytest.yml/badge.svg)](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_pytest.yml)
[![Check License Lines](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/licensed.yml/badge.svg)](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/licensed.yml)
[![Documentation](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/sphinx_documentation.yml/badge.svg)](https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/sphinx_documentation.yml)

# TARDIS-EM
Python-based software for generalized object instance segmentation from (cryo-)electron microscopy
micrographs/tomograms. The software package is built on a general workflow where predicted semantic segmentation
is used for instance segmentation of 2D/3D images.

![Tardis Workflow](resources/workflow.jpg)

## Features
* Training of Unet/ResNet/Unet3Plus/FNet for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
* Point cloud instance segmentation by point cloud graph representation

## News
<details open>
    <summary><b>TARDIS v0.1.0 - RC3 - (2023-08-28)</b></summary>

    * General improvement from MT prediction
    * Added full support for OTA updates of the entire package
    * Improved accuracy for semantic and instance segmentation of MT and Membrane
    * Added support for 2D membrane segmentation and update to MT and membrane 3D models
    * Added experimental SparseDIST module
    * Support for ply export file
    * Fixed AWS access denied error on some networks
    * Added filament filtering for removing false-positive rapid 150-degree connections
    * Microtubule output is now sorted by the length
    * Each instance receives a segmentation confidence score by which the user can filter out predictions
</details>

<details>
    <summary><b>TARDIS v0.1.0 - RC2 - (2023-03-22)</b></summary>

    * General improvement from MT prediction
    * Added support for Cry-mem prediction
    * Added support for node (RGB) features in DIST
    * Pre-trained network for Cryo-mem, General-MT, S3DIS dataset
    * Full support for Pytorch 2.0
</details>

<details>
    <summary><b>TARDIS v0.1.0 - RC1 - (2023-02-08)</b></summary>

    * Overall clean-up for the final release 
    * Added full code documentation
    * Added full stable support for MT prediction 
    * Added support for ScanNetV2 dataset prediction with DIST 
    * Added costume TARDIS error and console logo outputs 
    * TARDIS error handling 

</details>

<details>
    <summary><b>TARDIS v0.1.0 - beta 2 - (2022-09-14)</b></summary>

    * Cryo-Membrane 2D support 
    * Stable training and prediction entries for spindletorch and DIST 
    * Restructure and standardize naming and versioning in TARDIS 
    * Combined all side-code into TARDIS 
    * Full support for Amira formats, MRC/REC, TIF 

</details>

<details>
    <summary><b>TARDIS v0.1.0 - beta 1</b></summary>

    * Cryo-Membrane 2D support 
    * Stable training and prediction entries for spindletorch and DIST 
    * Restructure and standardize naming and versioning in TARDIS 
</details>

# Requirements
    # Python
        python 3.7, 3.8. 3.9, 3.10, 3.11

    # ML library
        torch>1.12.0
        numpy>1.21.0
        pandas>1.3.0
    
    # Image loading library
        tifffile>2021.11.0
        imagecodecs>2021.11.00
    
    # Image processing library 
        scikit-learn>1.0.1
        scikit-image>0.19.2
        scipy>1.9.0
        edt>=2.3.0

    # External file format reader
        plyfile>=0.9

    # Other
        requests>2.28.0
        chardet>5.0.0
        ipython>7.33.0
        click>8.0.4
        nvidia-smi>=0.1.3; sys_platform != 'darwin'
        setuptools>=67.6.0


## Installation
**From Source**

The sources for TARDIS-em can be downloaded from the ***Available upon stable release***.

You can either clone the public repository:

    $ git clone git://github.com/SMLC-NYSBC/TARDIS
    $ python setup.py install
    $ pip install -r requirements.txt

    # Development only
    $ pip install -r requirements-dev.txt

Or install directly from the pre-build python package:

Install:
* Python 3.7, 3.8, 3.9, 3.10, 3.11

Windows x64 and Linux:

    $ conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    $ pip install ./tardis_em-0.1.0rc3-py3-none-any.whl

MacOS:
    
    $ conda install pytorch -c pytorch
    $ pip install ./tardis_em-0.1.0rc3-py3-none-any.whl

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
```
recommended usage: tardis_mt [-dir path/to/folder/with/input/tomogram] -out mrc_None
advance usage: tardis_mt [-dir str] [-out str] [-ps int] [-ct float] [-dt float]
                         [-pv int] [-ap str] ...
```

</details>

<details><summary><i>Instance microtubule prediction:</i></summary>

### Example: 
![Prediction example2](resources/tardis_instance_mt.jpg)

### Usage:
```
recommended usage: tardis_mt [-dir path/to/folder/with/input/tomogram]
advance usage: tardis_mt [-dir str] [-out str] [-ps int] [-ct float] [-dt float]
                         [-pv int] [-ap str] ...
```

</details>

```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         shows the current TARDIS version
  
  
Options:
  -dir, --dir TEXT                Directory with images for prediction with
                                  CNN model.
                                  [default: /local/dir/]
                                  
  -ms, --mask BOOL                Define if you input tomogram images or binary 
                                  mask with pre-segmented microtubules.
                                  [default: False]
                                  
  -ch, --checkpoint TEXT          Optional list of pre-trained weights
                                  [default: None|None]
                                  
  -out, --output_format [None_amSG|am_amSG|mrc_amSG|tif_amSG|None_mrcM|am_mrcM|
                         mrc_mrcM|tif_mrcM|None_tifM|am_tifM|mrc_tifM|tif_tifM|
                         None_mrcM|am_csv|mrc_csv|tif_csv|None_csv|am_None|mrc_None|
                         tif_None|am_ply|mrc_ply|tif_ply|None_ply]
                                  Type of output files. The First optional
                                  output file is the binary mask which can be
                                  of type None [no output], am [Amira], mrc or
                                  tif. The second output is instance segmentation
                                  of objects, which can be output as amSG
                                  [Amira], mrcM [mrc mask], tifM [tif mask],
                                  csv coordinate file [ID, X, Y, Z] or None
                                  [no instance prediction].  
                                  [default: None_amSG]
                                  
  -ps, --patch_size INTEGER       Size of image patch used for prediction.
                                  This will break the tomogram volumes into 3D
                                  patches where each patch will be separately
                                  predicted and then stitched back together
                                  with 25% overlap.  
                                  [default: 128]
                                  
  -rt, --rotate BOOLEAN           If True, during CNN prediction image is
                                  rotated 4x by 90 degrees. This will increase
                                  prediction time 4x. However, may lead to more
                                  cleaner output.  
                                  [default: True]
                                  
  -ct, --cnn_threshold FLOAT      Threshold used for CNN prediction.
                                  [default: 0.25]
                                  
  -dt, --dist_threshold FLOAT     Threshold used for instance prediction.
                                  [default: 0.5]
                                  
  -pv, --points_in_patch INTEGER  Size of the cropped point cloud, given as a
                                  max. number of points per crop. This will
                                  break generated from the binary mask point
                                  cloud into smaller patches with overlap.
                                  [default: 1000]
                                  
  -ap, --amira_prefix TEXT        If dir/amira folder exists, TARDIS will search
                                  for files with a given prefix (e.g.
                                  file_name.CorrelationLines.am). If the
                                  the correct file is found, TARDIS will use its
                                  instance segmentation with ZiB Amira
                                  prediction, and output additional file
                                  called file_name_AmiraCompare.am.  
                                  [default: .CorrelationLines]
  -fl, --filter_by_length INTEGER
                                  Filtering parameters for microtubules,
                                  defining maximum microtubule length in
                                  angstrom. All filaments shorter than this
                                  length will be deleted.
                                  [default: 500]
                                  
  -cs, --connect_splines INTEGER  Filtering parameter for microtubules. Some
                                  microtubules may be predicted incorrectly as
                                  two separate filaments. To overcome this
                                  during filtering for each spline, we
                                  determine the vector in which the filament end
                                  is facing and we connect all filaments that
                                  faces the same direction and are within the
                                  given connection distance in Angstrom.
                                  [default: 2500]
                                  
  -cr, --connect_cylinder INTEGER
                                  Filtering parameter for microtubules. To
                                  reduce false positives from connecting
                                  filaments, we reduce the search area to
                                  cylinder radius is given in Angstrom. For each
                                  spline we determine the vector in which the filament
                                  end is facing and we search for a filament
                                  that faces the same direction and their end
                                  can be found within a cylinder.
                                  [default: 250]
                                  
  -acd, --amira_compare_distance INTEGER
                                  If dir/amira/file_amira_prefix.am is
                                  recognized, TARDIS runs a comparison between
                                  its instance segmentation and ZiB Amira
                                  prediction. The comparison is done by
                                  evaluating the distance of two filaments
                                  from each other. This parameter defines the
                                  the maximum distance used to evaluate the
                                  similarity between two splines based on
                                  their coordinates [A].
                                  [default: 175]
                                  
  -aip, --amira_inter_probability FLOAT
                                  If dir/amira/file_amira_prefix.am is
                                  recognized, TARDIS runs a comparison between
                                  its instance segmentation and ZiB Amira
                                  prediction. This parameter defines the
                                  interaction threshold used to identify
                                  splines that are similar overlaps between
                                  TARDIS and ZiB Amira.
                                  [default: 0.25]
                                  
  -dv, --device TEXT              Define which device to use for training:
                                  gpu: Use ID 0 GPUcpu: Usa CPUmps: Apple
                                  silicon (experimental)0-9 - specified GPU
                                  device id to use.
                                  [default: 0]
                                  
  -db, --debug BOOLEAN            If True, save the output from each step for
                                  debugging.
                                  [default: False]
                      
```

</details>


<details><summary><b>Membrane Prediction</b></summary>

```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         shows the current TARDIS version
  
  
  -dir, --dir TEXT                Directory with images for prediction with
                                  CNN model.  
                                  [default: /local/dir/]

  -ms, --mask BOOL                Define if you input tomogram images or binary 
                                  mask with pre-segmented microtubules.
                                  [default: False]
                                                      
  -ch, --checkpoint TEXT          Optional list of pre-trained weights
                                  [default: None|None]
                                   
  -out, --output_format [None_amSG|am_amSG|mrc_amSG|tif_amSG|None_mrcM|am_mrcM|
                         mrc_mrcM|tif_mrcM|None_tifM|am_tifM|mrc_tifM|tif_tifM|
                         None_mrcM|am_csv|mrc_csv|tif_csv|None_csv|am_None|mrc_None|
                         tif_None|am_ply|mrc_ply|tif_ply|None_ply]
                                  Type of output files. The First optional
                                  output file is the binary mask which can be
                                  of type None [no output], am [Amira], mrc or
                                  tif. Second output is instance segmentation
                                  of objects, which can be output as amSG
                                  [Amira], mrcM [mrc mask], tifM [tif mask],
                                  csv coordinate file [ID, X, Y, Z] or None
                                  [no instance prediction].  
                                  [default: mrc_None]
                                  
  -ps, --patch_size INTEGER       Size of image patch used for prediction.
                                  This will break the tomogram volumes into 3D
                                  patches where each patch will be separately
                                  predicted and then stitched back together
                                  with 25% overlap.  
                                  [default: 256]
                                  
  -rt, --rotate BOOLEAN           If True, during CNN prediction image is
                                  rotate 4x by 90 degrees.This will increase
                                  prediction time 4x. However, may lead to more
                                  cleaner output.  
                                  [default: True]
                                  
  -ct, --cnn_threshold FLOAT      Threshold used for CNN prediction.
                                  [default: 0.5]
                                  
  -dt, --dist_threshold FLOAT     Threshold used for instance prediction.
                                  [default: 0.95]
                                  
  -pv, --points_in_patch INTEGER  Size of the cropped point cloud, given as a
                                  max. number of points per crop. This will
                                  break generated from the binary mask point
                                  cloud into smaller patches with overlap.
                                  [default: 1000]
                                  
  -dv, --device TEXT              Define which device to use for training:
                                  gpu: Use ID 0 GPUcpu: Usa CPUmps: Apple
                                  silicon0-9 - specified GPU device id to use
                                  [default: 0]
                                  
  -db, --debug BOOLEAN            If True, save the output from each step for
                                  debugging.  [default: False]
```

<details><summary><i>Semantic membrane prediction:</i></summary>

### Example: 
![Prediction example3](resources/tardis_semantic_mem.jpg)

### Usage:

```
2D prediction
-------------

recommended usage: tardis_mem2d [-dir path/to/folder/with/input/tomogram] -out mrc_None
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...

3D prediction
-------------
recommended usage: tardis_mem [-dir path/to/folder/with/input/tomogram] -out mrc_None
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...
```

</details>

<details><summary><i>Instance membrane prediction:</i></summary>

### Example: 
![Prediction example4](resources/tardis_instance_mem.jpg)

### Usage:

```

2D prediction
-------------

recommended usage: tardis_mem2d [-dir path/to/folder/with/input/tomogram]
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...

3D prediction
-------------
recommended usage: tardis_mem [-dir path/to/folder/with/input/tomogram]
advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...
```


</details>

</details>
