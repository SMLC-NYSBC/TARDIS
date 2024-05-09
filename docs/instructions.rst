Installation
============

This is a preliminary installation instruction. This may not work on all systems.
If any problems come up, do not hesitate to contact us `rkiewisz@nysbc.org <mailto:rkiewisz@nysbc.org>`__,
or contact us on our `Slack Channel <https://tardis-em.slack.com>`__.

We are working on a more intuitive installation of our software. In the meantime please use the following options.

Option 1:
---------
Install TARDIS using the newest released package on `Github <https://github.com/SMLC-NYSBC/TARDIS/releases>`__

.. code-block:: bash

    pip install package_name-py3-none-any.whl


And jump to `Validate`_.

Option 2:
---------
Build a package from the source

Step 1: Clone repository
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/SMLC-NYSBC/TARDIS.git


Step 2: Create a conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting, it is beneficial to create a virtual Python environment.
In these cases, we use Miniconda.

If you don't know if you have installed Miniconda on your machine, please run:

.. code-block:: bash

    conda -h


If you don't have Miniconda, you could install it following `official instruction <https://docs.conda.io/projects/miniconda/en/latest>`__.

Now you can create a new conda environment:

.. code-block:: bash

    conda create -n <env_name> python=3.11


And to use it, you need to active it:

.. code-block:: bash

    conda activate <env_name>


Step 3: Install TARDIS
~~~~~~~~~~~~~~~~~~~~~~

The following command will install TARDIS and all its dependencies

.. code-block:: bash

    pip install tardis-em

(Optional) Install from GitHub master branch:

.. code-block:: bash

    cd TARDIS
    pip install .


.. _Validate:

Validate installation
---------------------

To check if the installation was successful and check for any new OTA updates, you can run:

.. code-block:: bash

    tardis

This should display the TARDIS home screen, similar to the screenshot below:

    .. image:: resources/main_tardis.jpg
      :width: 512


Run automatic segmentation
--------------------------

- Advance Tutorial - Predict Microtubules in 3D [`Tutorial <usage/3d_mt.html>`__].

.. code-block:: bash

    tardis_mt -dir path/to/folder/with/your/tomograms

- Advance Tutorial - Predict Microtubules in 2D [Coming soon] [`Tutorial <usage/2d_mt.html>`__]

.. code-block:: bash

    TBD

- Advance Tutorial - Predict Membrane in 3D [`Tutorial <usage/3d_membrane.html>`__]

.. code-block:: bash

    tardis_mem -dir path/to/folder/with/your/tomograms

- Advance Tutorial - Predict Membrane in 2D [`Tutorial <usage/2d_membrane.html>`__]

.. code-block:: bash

    tardis_mem2d -dir path/to/folder/with/your/tomograms
