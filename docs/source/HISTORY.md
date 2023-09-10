# History

<details open>
    <summary><b>0.1.0-RC3 OTA (2023-08-10)</b></summary>

*This intends to be a release submitted with Nature Method 2023*

* **General changes:**
    * *General*:
        * Documentation update
        * Added full support for OTA updates of the entire pacakge
        * Few bug fixes

* **SpindleTorch module changes:**
    * *General*:
        * Update for FNet CNN model for membrane 3D

* **DIST module changes:**
    * *Optimize*:
        * Update DIST model for 2D and 3D membrane
    * *BugFix*:
        * Fixed few issues in the membrane segmentation pipeline

</details>

<details>
    <summary><b>0.1.0-RC3 (2023-08-25)</b></summary>

* **General changes:**
    * *General*:
        * Full support for 2D data
        * Black
        * Introduced the TARDIS Logo and rebranding to Tardis-PyTorch
        * Remove Open3D library (conflict in CentOS7)
        * Fixed MRC read-out during training that forcibly rotated .mrc files

    * *New_Feature*: 
        * Added new output format .ply
        * New general tardis call
        * Added helper functions csv_am and am_csv
        * Added instance prediction from semantic binary masks 

    * *Optimize*: 
        * Added an optional checkpoint to all Tardis calls
        * Improvements in training for CNN and DIST by users
        * Amira possible output as a raw point cloud

    * *BugFix*:
        * Fixed save for .mrc files

* **SpindleTorch module changes:**
    * *General*:
        * Retrained FNet_32 model for membrane and microtubules
        * Train FNet_32 for 2D membrane segmentation

    * *Optimize*: 
        * 2D CNN network set-up

* **DIST module changes:**
    * *General*:
        * Added simulated data for training on filament-like structures
        * Re-train model no simulated + real data
        * Fine-tuned setting for predictions and post-processing

    * *New_Feature*: 
        * Experimental SparseDist model to offer more memory-efficient performance,  
      for instance segmentation

    * *Optimize*: 
        * Improved visualization outputs
        * Mcov metric optimization
        * Rebuild Graph prediction function to be more robust
        * Reverse-engineered Open3D voxal downsampling and added random downsampling
        * Added distance embedding with a range value

</details>

<details>
    <summary><b>0.1.0-RC2-HotFix2 (2023-03-28)</b></summary>

* **General changes:**
  * Fixed saving int8 semantic output as mrc
  * Added rotation for CNN prediction

</details>

<details>
    <summary><b>0.1.0-RC2-HotFix1 (2023-03-23)</b></summary>

* **General changes:**
  * Fixed loading for corrupted mrc files
  * Fixed for loading and saving mrc/rec files (fix for reading headers size)
  * Fix for loading new Amira SG with coordinates in 'nm' not 'Angstrom'
  * Small fixed in general prediction loops
  * Fixed missing membrane instance prediction output

</details>

<details>
    <summary><b>0.1.0-RC2 (2023-03-22)</b></summary>

* **General changes:**
    * *General*: 
      * Normalized all documentation to *.md
    * *New_Feature*: 
      * Ensure support for PyTorch 2.0 
      * Added benchmark entry
      * Added ClBCE and ClDice loss functions
      * Added binary Amira image file export
      * Full membrane support (training and prediction of cryo-mem)
      * Added costume LR schedular (ISR - invert square root)
    * *Optimize*: 
      * Loss functions pytest and general cleanup
      * Formatting and missing TardisErrors
      * 20x Speed up for Tardis logo for Linux/OS X
    * *BugFix*:
      * Fixed small bugs in metrics calculation

* **SpindleTorch module changes:**
    * *New_Feature* 
      * Added and tested clDice and clBCE loss function
    * *Optimize* 
      * Support for membrane training dataset
    * *General* 
      * Globally change normalization (0-1) to image standardization (-1-1)
      with mean and standard deviation

* **DIST module changes:**
  * *New_Feature* 
        * Node embedding with furier random
        * Added calculation of mcov metric during training and save checkpoint based on it
  * *Optimize* 
    * Point cloud visualization can be now with or without animation

</details>

<details>
    <summary><b>0.1.0-RC1 (2023-02-08)</b></summary>

* **Code restructure:**
    * *Optimize*:
        * autonomization of tests for all python version

* **SpindleTorch module changes:**
    * *Optimize*:
      * Rebuild interpolation for images and mask
      * Simplified building training/testing of datasets
      * Redo mask building from coordinates
      * Build_Point_Cloud
      * New model train with optimize image normalization
    * *BugFix*:
      * image normalization for few very specific cases

* **DIST module changes:**
    * *Optimize*:
      * Change how DIST distance embedding is computed for GT data.
      * Change normalization for point cloud
        - MT normalized by pixel size
        - All other by open3d downsampling value optional random downsamling
      * F1 eval metric and BCE loss without diagonal axis
    * *New_Feature*:
      * DataLoader for stanford data
      * spline filtering include geometric filtering and margin of spline
    * *BugFix*:
      * in point cloud segmenter when feed with coord idx as float not int

* **General changes:**
    * *General*:
      * Added data competition with Amira mesh output
      * Added license footnote
    * *General* code *Optimize* for speed
    * *BugFix* and *Optimize* for post-processing of spatial-graphs
    * *BugFix* and *New_Feature* for amira export format (now build multi-label)
    * *BugFix*:
      * AWS weight import when aws don't allow read access
    * *New_Feature*:
      * TardisError for all error handling

</details>

<details>
    <summary><b>0.1.0-beta2 (2022-09-14)</b></summary>

* **Code restructure:**
    * Finished documentation with Sphinx
    * Build testes for the whole tardis-pytorch
    * Push to RC branch

* **SpindleTorch module changes:**
    * Cryo-membrane model support 
    * Build prediction module for Cryo-membrane
    * Removed scaling module (after extensive testes it shows not benefits)
    * Fixes in building train data set and small restructure (more in documentation)
    * Added more support for 2D images while building test/train dataset
    * Added support for pure probability prediction output in float32

* **DIST module changes:**
    * Last clean-up and prepare for release with ICLR2023

* **General changes:**
    * Added support for mrc and csv file outputs
    * Support for python 3.11 (awaiting pytorch and open3d)
    * requirements.txt changes and include pytroch with support for different os

</details>

<details>
    <summary><b>0.1.0-pre_beta2 (2022-09-14)</b></summary>

* **Code restructure:**
    * Clean-up
    * Restructure code organization
    * Removed slcpy and unified it with spindletorch and dist
    * Rebuild main classes and make them more general
    * Simplified overall structure
    * Full documentation with Sphinx
    * Separate dev. requirements
    * Cleaned S3 aws loading and removed old models from S3 bucket

* **SLCPY module changes:**
    * Removed and marge with SpindleTorch and DIST

* **SpindleTorch module changes:**
    * Retrained FNet_16, FNet_32 and UNet_16, UNet_32

* **DIST module changes:**
    * Introduced DIST for semantic segmentation
    * Retrained model on ScanNet v2 datasets
    * Added node feature embedding with images or RGB values
    * Retrained DIST model on ScanNet v2 + RGB

* **General changes:**
    * Load image data, marge and fixed for int8 and uint8
    * Amira binary import fixes. Amira defined import type. Previously assumption was
      that Amira load all binary as uint8. Amira loads files as uint8 or int8 and
      have different structures when loading mask data which can be binary or ascii.
    * Overall stability improvements
    * Tardis logo was integrated with all TARDIS modules
    * Build tests for the whole tardis-pytorch
    * Introduced tardis_dev and divided stable and developmental branches
    * Fixed image normalization and ensure correct normalized output for training
      and prediction
    * Added MRC export
    * Minor bugfixes from prebeta2 and new additions to beta2

</details>

<details>
    <summary><b>0.1.0-beta1 (2022-09-14)</b></summary>

* **DIST module changes:**
    * Added new classification model based on DIST
    * Simplified logic for patching big point cloud + reduction of number of patches
    * Model structure now embedded in the model weight file
    * Spline smoothing added to graph prediction
    * Small bugfixes:
        * Fixe initial_scale in model nn.Modules
        * Fixed graph builder for ScanNet and PartNet
    * Speed improved dataloader during training
    * Added support for .ply file format and meshes
    * Re-train model on different DIST structure for the paper and for searching 
      of the best approach
    * Bugfixes for segmentation of point cloud from graph probabilities
        * Speed-up boost with simplifying the building and reading adjacency matrix
        * Fix in masking adjacency matrix for points already connected
        * Moved from greedy segmentation to 1-step-back segmentation

* **SpindleTorch changes:**
    * Quick retrained model on hand-curated dataset
    * Added and trained new FNet
    * Standardized pixel size input. Now all data are reshaped to the pixel size of 2.32
    * Change up-sampling from align_corners=True to align_corners=False
    * Added new data for training from @Stefanie_Redemann and @Gunar
    * Ground-up rebuild spindletorch model
        * New Big UNet model combining both UNet and UNet3Plus
        * Unet/Unet3Plus re-trained <- rejected big_unet is better
        * Train Big UNet
    * Speed-up prediction with new Big UNet model

* **SLCPY module changes:**
    * Fix interpolation handling for up-sampled datasets
    * Post-processing improvements and speeds-up
    * MRC2014 file format expand readable formats
    * Processing image data with standardized pixel size of 25 A
    * Bugfixes for floating point precision in Amira output
        * Change floating point from 3 to 15
    * Improvements from importing data from binary Amira file format
        * Change how pixel size is calculated. Amira has weird behavior whenever ET 
          is trimmed. Include this in pixel size calculation
    * Improvements in .rec, .mrc file loader
        * .rec and .mrc file are format with uint8 (value from -128 to 128) or 
          int8 (value from 0 to 255). Fix reading of these files

* **TARDIS**
    * Cleaned log output for easier reading
    * New beautiful log progress window
    * Moved loss fun. to common directory
    * Clean-up
    * Flake8 and pyteset fixes
    * Global tunning for segmentation quality

</details>

<details>
    <summary><b>0.1.0-alpha6 (2022-07-12)</b></summary>

* Check pipeline for image embedding (normalization to enhance features)
    * Introduce new normalization ResaleNormalize that spread histogram from 
    2-98 projectile of intensity distribution
* Model retraining for MTs and membranes (generalization)
* Redone PC normalization
* Additional work on speed up training by optimizing DataLoader
* TODO: Model retraining for MTs with real image data
* Closed #7 an #9 issue
* Added removal of dist_embedding as an input
* SpindleTorch rebuild to work on 2D and 3D datasets
* DIST training progress bar update (simplified output and removed prints)
* Add Visualizer module for point clouds
* Added hotfix for output of coordinates to fit Amira coordinates transformation
* Spellings and documentation fixes
* Bumped version for DIST and slcpy
* Cleaned code and documentation

</details>

<details>
    <summary><b>0.1.0-alpha5 (2022-04-25)</b></summary>

* Rename GraphFormer to DIST (Dimensionless instance Segmentation Transformer)
* Updates for DIST
    * SetUp metric evaluation
    * Changes in handling point cloud
        * Normalization based on K-NN distance
    * Setup for easy dissection of the model
    * Dist version to 0.1.5
    * Added evaluation pipeline

</details>

<details>
    <summary><b>0.1.0a2-alpha4 (2022-04-25)</b></summary>

* Fix for better handling graph prediction
* Fix for #4-#6 issues
* Small bugfixes for GraphFormer while training
* Add point cloud normalization before training/prediction

</details>

<details>
    <summary><b>0.1.0-alpha1 (2022-04-13)</b></summary>

* Rename tardis to tardis-pytorch
* Build tests for all modules
* Integrated slcpy, spindletorch and graphformer
* Added general workflow for MT prediction
    * SLCPY:
        * Loading of data types: .tif, .am, .mrc, .rec for 2D and 3D
        * Included all slcpy modules
        * Move Amira file output of point cloud from graphformer
        * SetUp workflows for data pre- and post-processing 

* SPINDLETORCH
    * Included all spindletorch modules
    * Build standard workflows for training and prediction of 2D and 3D images

* GRAPHFORMER
    * Included all graphformer modules

</details>

<details>
    <summary><b>0.0.1 (2022-03-24)</b></summary>

* Initial commit

</details>
