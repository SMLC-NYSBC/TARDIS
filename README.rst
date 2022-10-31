TARDIS
======

.. image:: https://img.shields.io/badge/release-0.1.0_beta2-success
        :target: https://img.shields.io/badge/release-0.1.0_beta2-success

.. image:: https://readthedocs.org/projects/tardis-pytorch/badge/?version=latest
        :target: https://tardis-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Python based software for generali instance segmentation of object from electron microscopy (EM) and 
cryo-EM micrographs. Software package is builded on general workflow where predicted semantic segmentation 
is used for instance segmentation of 2D/3D and 4D/5D fluorescent images in the future.

.. image:: resources/workflow.jpg
        :target: resources/workflow.jpg
        :alt: TARDIS workflow


Features
--------
* Training of Unet/ResNet/Unet3Plus for 2D and 3D images [.tif, .mrc, .rec, .am]
* Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
* Training of DIST ML model for instance segmentation of 2D and 3D point clouds
        * 4D and 5D point clouds segmentation in the future
* Point cloud instance segmentation by point cloud graph representation


Requirements
------------
.. code-block::

	python 3.7 or newer


Installation
------------
**From Source**

The sources for TARDIS-pytorch can be downloaded from the ***Available soon***.

You can either clone the public repository:

.. code-block:: console

	$ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
	$ python setup.py install
	$ pip install -r requirements.txt

Or install from pre-build python package:
Install:
	- Python 3.7


Windows x64 and Linux:

.. code-block:: console

	$ conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
	$ pip install ./tardis_pytorch-0.1.0b2-py3-none-any.whl


Known installation errors on Linux:

.. code-block:: console

	OSError: /lib64/libc.so.6: version `GLIBC_2.18' not found
	
Solution:

.. code-block:: console

	$ pip install open3d==0.9.0
Usage
-----
Prediction of MT from electron tomograms:

.. code-block::
	
	**All setting:**
	-dir   (str): Directory with electron micrographs   [*.mrc, *.rec, *.am]
	-ps    (int): Patch size used for prediction.       [default: 128].
	-cnn   (str): CNN network name.                     [default: 'fnet_t 0.2 '].
	-cch   (str): If not None, str checkpoints for CNN. [default: None]
	-ct  (flaot): Threshold use for model prediction.   [default: 0.3]
	-dch   (str): If not None, checkpoints for DIST.    [default: None]
	-dt  (float): Threshold use for graph segmentation. [default: 0.5]
	-pv    (int): Number of point per voxal.            [default: 1000]
	-d     (str): Define which device use for training: [default: 0]
		      cpu: cpu
		      gpu: 0-9 - specific GPU.
	-db   (bool): If True, save debuging output.        [default: False]
	-v     (str): If not None, output visualization of  [default: None]
		      the prediction:
		      - f: Output as filamentsp: 
		      - p: Output color coded point cloud
	--version     Show the version and exit.
	--help        Show this message and exit.
	
	**Recomanded usage for electron tomograms:**
	$ tardis_mt -dir ./.. -ct 0.2 -pv 1000 
	
	**Recomanded usage for cryo-electron tomograms/micrographs:**
	$ tardis_mt -dir ./.. -ct 0.2 -pv 1000 	
