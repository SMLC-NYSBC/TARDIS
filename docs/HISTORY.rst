=======
History
=======

0.1.0a6 (2022-05-xx)
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
