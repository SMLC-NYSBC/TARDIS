================================
Transformer And Rapid Dimensionless Instance Segmentation [TARDIS]
================================
.. image:: https://img.shields.io/github/v/release/SMLC-NYSBC/tardis
        :target: https://img.shields.io/github/v/release/SMLC-NYSBC/tardis

.. image:: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python-package.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python-package.yml

.. image:: https://readthedocs.org/projects/tardis/badge/?version=latest
        :target: https://tardis-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
....

Python based software for generali instance segmentation of object from electron microscopy (EM) and 
cryo-EM micrographs. Software package is builded on general workflow where predicted semantic segmentation 
is used for instance segmentation of 2D/3D and 4D/5D fluorescent images in the future.

.. image:: /resources/workflow.png
        :target: /resources/workflow.png
        :alt: TARDIS workflow

* Documentation: https://tardis-pytorch.readthedocs.io/en/latest/

Features
--------
        * Training of Unet/ResNet/Unet3Plus for 2D and 3D images [.tif, .mrc, .rec, .am]
        * Prediction of binary semantic segmentation of 2D and 3D images [.tif, .mrc, .rec, .am]
        * Training of DIST ML model for instance segmentation of 2D and 3D point clouds
                * 4D and 5D point clouds segmentation in the future
        * Point cloud instance segmentation by point cloud graph representation

============
Installation
============

From sources
------------

The sources for TARDIS-pytorch can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/SMLC-NYSBC/TARDIS-pytorch
    $ python setup.py install
    $ pip install -r requirements.txt

.. _Github repo: https://github.com/SMLC-NYSBC/TARDIS-pytorch
.. _tarball: https://github.com/SMLC-NYSBC/TARDIS-pytorch/tarball/master

=====
Usage
=====

