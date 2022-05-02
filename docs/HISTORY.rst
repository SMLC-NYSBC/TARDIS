=======
History
=======

0.1.0a2-a4 (2022-04-25)
------------------
* Fix for better handling graph prediction
* Fix for #4-#6 issues
* Small bugfixes for GraphFormer while training
* Add point cloud normalization before training/prediction

0.1.0a1 (2022-04-13)
------------------
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
