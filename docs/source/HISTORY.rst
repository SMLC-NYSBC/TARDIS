History
=======

0.1.0-beta2 (2022-09-14)
----------------------------
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

0.1.0-pre_beta2 (2022-09-14)
----------------------------
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

0.1.0-beta1 (2022-09-14)
------------------------
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

0.1.0-alpha6 (2022-07-12)
-------------------------
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

0.1.0-alpha5 (2022-04-25)
-------------------------
* Rename GraphFormer to DIST (Dimensionless instance Segmentation Transformer)
* Updates for DIST
    * SetUp metric evaluation
    * Changes in handling point cloud
        * Normalization based on K-NN distance
    * Setup for easy dissection of the model
    * Dist version to 0.1.5
    * Added evaluation pipeline

0.1.0a2-alpha4 (2022-04-25)
---------------------------
* Fix for better handling graph prediction
* Fix for #4-#6 issues
* Small bugfixes for GraphFormer while training
* Add point cloud normalization before training/prediction

0.1.0-alpha1 (2022-04-13)
-------------------------
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

0.0.1 (2022-03-24)
------------------
* Initial commit
