=======
History
=======

0.1.0b (2022-07-**)
--------------------
* SpindleTorch changes:
    * Standardized pixel size input. Now all data are reshaped to the pixel 
        size of 2.32
    * Change up-sampling from align_corners=True to align_corners=False
    * Added new data for training from @Stefanie Redemann and @Gunar
    * Ground-up rebuild spindletorch model
        * New Big UNet model combining both UNet and UNet3Plus
        * Unet/Unet3Plus re-trained
        * Train Big UNet
    * Speed-up prediction with new Big UNet model

* DIST module changes:
    * Added support for .ply file format and meshes
    * Re-train model on different DIST structure for the paper and for searching 
        of the best approach

    * Bugfixes for segmentation of point cloud from graph probabilities
        * Speed-up boost with simplifying the building and reading adjacency matrix
        * Fix in masking adjacency matrix for points already connected
        * Moved from greedy segmentation to 1-step-back segmentation

* SLCPY module changes:
    * Bugfixes for floating point precision in Amira output
        * Change floating point from 3 to 15

    * Improvements from importing data from binary Amira file format
        * Change how pixel size is calculated. Amira has weird behavior whenever ET 
            is trimmed. Include this in pixel size calculation

    * Improvements in .rec, .mrc file loader
        * .rec and .mrc file are format with uint8 (value from -128 to 128) or 
            int8 (value from 0 to 255). Fix reading of these files

0.1.0a (2022-07-12)
--------------------
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

0.1.0a5 (2022-04-25)
--------------------
* Rename GraphFormer to DIST (Dimensionless instance Segmentation Transformer)
* Updates for DIST
    * SetUp metric evaluation
    * Changes in handling point cloud
        * Normalization based on K-NN distance
    * Setup for easy dissection of the model
    * Dist version to 0.1.5
    * Added evaluation pipeline

0.1.0a2-a4 (2022-04-25)
-----------------------
* Fix for better handling graph prediction
* Fix for #4-#6 issues
* Small bugfixes for GraphFormer while training
* Add point cloud normalization before training/prediction

0.1.0a1 (2022-04-13)
--------------------
* Rename tardis to tardis-pytorch
* Build tests for all modules
* Integrated slcpy, spindletorch and graphformer
* Added general workflow for MT prediction
    * SLCPY:
        * Loading of data types: *.tif, *.am, *.mrc, *.rec for 2D and 3D
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
