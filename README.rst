.. image:: resources/Tardis_logo_2.png
    :width: 512
    :align: center
    :target: https://smlc-nysbc.github.io/TARDIS/

========

.. image:: https://img.shields.io/badge/Release-0.1.1-success
    :target: https://shields.io

.. image:: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_pytest.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/python_pytest.yml

.. image:: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/licensed.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/licensed.yml

.. image:: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/sphinx_documentation.yml/badge.svg
        :target: https://github.com/SMLC-NYSBC/TARDIS/actions/workflows/sphinx_documentation.yml

Python-based software for generalized object instance segmentation from (cryo-)electron microscopy
micrographs/tomograms. The software package is built on a general workflow where predicted semantic segmentation
is used for instance segmentation of 2D/3D images.

.. image:: resources/workflow.jpg

Features
========

- Robust and high-throughput semantic/instance segmentation of all microtubules:
    - Supported file formats: [.tif, .mrc, .rec, .am]
    - Supported modality: [ET, Cryo-ET]
    - Supported Å resolution: [all]
    - 2D micrograph modality microtubule segmentation will come soon!

- Robust and high-throughput semantic/instance segmentation of membranes:
    - Supported file formats: [.tif, .mrc, .rec, .am]
    - Supported modality: [EM, ET, Cryo-EM, Cryo-ET]
    - Supported Å resolution: [all]

- Fully automatic segmentation solution!
- Napari plugin [Coming soon]
- Cloud computing [Coming soon]


What's new?
===========

`Full History <https://smlc-nysbc.github.io/TARDIS/HISTORY.html>`__

TARDIS-em v0.1.0 (2023-08-10):
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

Quick Start
===========

For more examples and an advance usage please fine more details in our `Documentation <https://smlc-nysbc.github.io/TARDIS/>`__

Microtubule Prediction
----------------------

2D prediction
^^^^^^^^^^^^^

TBD

3D prediction
^^^^^^^^^^^^^

Example:
""""""""

TBD

Usage:
""""""

.. code-block:: bash

    recommended usage: tardis_mt [-dir path/to/folder/with/input/tomogram]
    advance usage: tardis_mt [-dir str] [-out str] [-ps int] [-ct float] [-dt float]
                             [-pv int] [-ap str] ...


Membrane Prediction
-------------------

2D prediction
^^^^^^^^^^^^^

Example:
""""""""

TBD

Usage:
""""""

.. code-block:: bash

    recommended usage: tardis_mem2d [-dir path/to/folder/with/input/tomogram] -out mrc_csv
    advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...

3D prediction
^^^^^^^^^^^^^

Example:
""""""""

TBD

Usage:
""""""

.. code-block:: bash


    recommended usage: tardis_mem [-dir path/to/folder/with/input/tomogram] -out mrc_csv
    advance usage: tardis_mem [-dir str] [-out str] [-ps int] ...
