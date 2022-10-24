DIST -> DataLoader
==================

General DataLoader
------------------
General class for creating Datasets. It works by detecting all specified 
file formats in the given directory and return the index list.

.. autoclass:: tardis_dev.dist_pytorch.utils.dataloader.BasicDataset


Specialized DataLoader's
------------------------
Filament structure DataSet
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tardis_dev.dist_pytorch.utils.dataloader.FilamentDataset

PartNet synthetic DataSet
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: tardis_dev.dist_pytorch.utils.dataloader.PartnetDataset

ScanNet V2 synthetic DataSet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: tardis_dev.dist_pytorch.utils.dataloader.ScannetDataset

ScanNet V2 with RGB values synthetic DataSet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: tardis_dev.dist_pytorch.utils.dataloader.ScannetColorDataset


Helper Functions
----------------
Build dataloader
^^^^^^^^^^^^^^^^
.. autofunction:: tardis_dev.dist_pytorch.utils.dataloader.build_dataset