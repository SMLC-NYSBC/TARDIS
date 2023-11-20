.. role:: bash(code)
   :language: bash

.. role:: guilabel

Prediction of 3D Membrane semantics/instances
---------------------------------------------

This guide provides detailed instruction to perform fully automatic membrane
segmentation on all of your tomograms using our most up-to-data model.

TARDIS is able to predict fully automatic membrane as a semantic labels, or
instances [labels, point cloud].

.. image:: ../resources/3d_mem.jpg

Example of segmented tomograms with indicated predicted semantic binary segmentation
and individual instances represented as a tracks of different colours.

TARDIS Workflow
_______________

#. Prepare folder with data.
#. Predict membrane segmentation
#. (Optional) Optimize prediction setting

Preparation
___________
Simply store all your tomograms in one folder. TARDIS will recognize all
image file with the extension [*.tif, *.tiff, *.rec, *.map, *.mrc, *.am].

`Tip:` In case of REC/MAP/MRC files try to make sure that files have embedded
in the header pixel size information.

Prediction
__________

(Optional) Type following to check if TARDIS is up-to-data and is working properly.

`Tips:` If any error occur, try using our `troubleshooting chapter <troubleshooting.html>`__.

.. code-block::

    tardis

This will display the TARDIS interface and show available option or available updates.

.. image:: ../resources/main_tardis.jpg

Semantic/Instance segmentation:
```````````````````````````````
For the semantic prediction, you only need to type:

.. code-block::

    tardis_mem -dir <path-to-your-tomograms> -out <output_type>

TARDIS will save predictions in default folder :bash:`Prediction` located in
the folder with your data.

Running this will segment all tomograms in the indicated path. Predicted output
will be store in file format indicated in :bash:`-out <output_type>` [details bellow].

For example:

.. code-block::

    tardis_mem -dir <path-to-your-tomograms> -out mrc_None

Will perform only semantic segmentation and save output file as *.mrc file.


.. code-block::

    tardis_mem -dir <path-to-your-tomograms> -out None_csv

Will perform only instance segmentation and save output file as *.csv file with data
structure as [Membrane ID x X x Y]

.. code-block::

    tardis_mem -dir <path-to-your-tomograms> -out mrc_csv

Will perform semantic and instance segmentation and save output file as *.mrc and *.csv files.

Advance usage:
``````````````

:bash:`--dir`
