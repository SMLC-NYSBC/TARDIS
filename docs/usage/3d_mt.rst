.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: guilabel

Prediction of 2D MT semantics/instances
---------------------------------------

This guide provides detailed instructions to perform fully automatic microtubule
segmentation on all of your tomograms using our most up-to-date model.

TARDIS can predict fully automatic microtubules as semantic labels, or
instances [track, or labels].

.. image:: ../resources/3d_mt.jpg

Data source: Gunar Fabig and Thomas Müller-Reichert, CFCI MTZ-TU Dresden

Example of segmented tomograms with indicated predicted semantic binary segmentation
and individual instances represented as tracks of different colors.

TARDIS Workflow
_______________

#. Prepare a folder with data
#. (Optional) Prepare for comparing TARDIS results with Amira
#. Predict microtubule segmentation
#. (Optional) Advance prediction setting

Preparation
___________
Simply store all your tomograms in one folder. TARDIS will recognize all
image file with the extension [.tif, .tiff, .rec, .map, .mrc, .am].

`Tip:` In the case of REC/MAP/MRC files try to make sure that files have embedded
in the header pixel size information.

(Optional) Amira Comparison
___________________________
TARDIS at the end of the instance prediction can use already generated Amira prediction
and create a file that will contain information that predicts microtubules are a match
between both tools and which are possible False-Positive instances.

To do this create a new sub-folder, named :bash:`amira` in your directory with all tomograms.
For each predicted tomogram, TARDIS will search for a file with :bash:`.CorrelationLines.am` prefix.
And will use that file to compare both outputs.

`Tips`:

#. Amira file with :bash:`.CorrelationLines.am` prefix should have the same name as a tomogram.
#. Amira spatial graph file should be saved in ASCII format. Remember that by default Amira saves files in binary format.
#. In TARDIS you can also find a few arguments you can use to tune this comparison setting [:ref:`compare`].

Prediction
__________

(Optional) Type the following to check if TARDIS is up-to-date and is working properly.

`Tips:` If any error occurs, try using our `troubleshooting chapter <troubleshooting.html>`__.

.. code-block::

    tardis

This will display the TARDIS interface and show available options or available updates.

.. image:: ../resources/main_tardis.jpg
  :width: 512

Semantic/Instance segmentation:
```````````````````````````````
For the semantic prediction, you only need to type:

.. code-block::

    tardis_mt -dir <path-to-your-tomograms> -out <output_type>

TARDIS will save predictions in the default folder :bash:`Prediction` located in
the folder with your data.

Running this will segment all tomograms in the indicated path. Predicted output
will be store in file format indicated in :bash:`-out <output_type>` [:ref:`see all -out options <out>`].

For example:

.. code-block::

    tardis_mt -dir <path-to-your-tomograms> -out mrc_None

Will perform only semantic segmentation and save the output file as .a mrc file.


.. code-block::

    tardis_mt -dir <path-to-your-tomograms> -out None_amSG

Will perform only instance segmentation and save the output file as .am spatial graph file with data
structure as [Microtubule ID x X x Y x Z]

.. code-block::

    tardis_mt -dir <path-to-your-tomograms> -out mrc_amSG

Will perform semantic and instance segmentation and save the output file as .mrc and .am spatial graph files.

`Tips`: As a final product of TARDIS instance segmentation for microtubules, TARDIS produces two files.
The first file with :bash:`_SpatialGraph.am` extension contains all predicted microtubules. The second file
with :bash:`_SpatialGraph_filter.am` extension contains filter microtubules based on length and curvature [:ref:`filter`].

Advance usage:
``````````````
Below you can find all available arguments you can use with :bash:`tardis_mt`,
with the explanation for their functionality:

:bash:`-dir` or :bash:`--path`: Directory path with all tomograms for TARDIS prediction.
    - :guilabel:`default:` Current command line directory.

:bash:`-ms` or :bash:`--mask`: Define if your input is a binary mask with a pre-segmented microtubules.
    - :guilabel:`Example:` You can set this argument to :bash:`-ms True` if you have already segmented microtubules
      and you only want to segment instances.

    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` True, False

:bash:`-px` or :bash:`--correct_px`: Overwrite pixel value.
    - :guilabel:`Example:` You can set this argument to :bash:`-px True` if you want to overwrite
      the pixel size value that is being recognized by TARDIS.

    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` True, False

:bash:`-ch` or :bash:`--checkpoint`: Directories to pre-train models.
    - :guilabel:`Example:` If you fine-tuned TARDIS on your data you can indicate here
      file directories for semantic and instance model. To do this type your directory
      as follow: :bash:`-ch <semantic-model-directory>|<instance-model-directory>`. For example,
      if you want to pass only semantic model type: :bash:`-ch <semantic-model-directory>|None`.

    - :guilabel:`default:` None|None

:bash:`-out` or :bash:`--output_format`: Type of output files.
    - :guilabel:`Example:` Output format argument is compose of two elements :bash:`-out <format>_<format>`.
      The first output format is the semantic mask, which can be of type: None [no output], am [Amira], mrc, or tif.
      The second output is predicted instances of detected objects, which can be of type:
      output as amSG [Amira spatial graph], mrc [mrc instance mask], tif [tif instance mask],
      csv coordinate file [ID, X, Y, Z], stl [mesh grid], or None [no instance prediction].

    - :guilabel:`default:` mrc_csv
    - :guilabel:`Allowed options:` am_None, mrc_None, tif_None, None_am, am_am, mrc_am, tif_am,
      None_amSG, am_amSG, mrc_amSG, tif_amSG, None_mrc, am_mrc, mrc_mrc, tif_mrc,
      None_tif, am_tif, mrc_tif, tif_tif, None_csv, am_csv, mrc_csv, tif_csv,
      None_stl, am_stl, mrc_stl, tif_stl

:bash:`-ps` or :bash:`--patch_size`: Window size used for prediction.
    - :guilabel:`Example:` This will break tomograms into smaller patches with 25% overlap.
      Smaller values than 128 consume less GPU, but also may lead to worse segmentation results!

    - :guilabel:`default:` 128
    - :guilabel:`Allowed options:` 32, 64, 96, 128, 256, 512

:bash:`-rt` or :bash:`--rotate`: Predict the image 4 times rotating it each time by 90 degrees.
    - :guilabel:`Example:` If :bash:`-rt True`, during semantic prediction tomograms is rotate 4x by 90 degrees.
      This will increase prediction time 4 times. However, it usually will result in cleaner output.

    - :guilabel:`default:` True
    - :guilabel:`Allowed options:` True, False

:bash:`-ct` or :bash:`--cnn_threshold`: Threshold used for semantic prediction.
    - :guilabel:`Example:` Higher value then :bash:`-ct 0.25` will lead to a reduction in noise
      and microtubule prediction recall. A lower value will increase microtubules prediction
      recall but may lead to increased noise.

    - :guilabel:`default:` 0.25
    - :guilabel:`Allowed options:` Float value between 0.0 and 1.0

:bash:`-dt` or :bash:`--dist_threshold`: Threshold used for instance prediction.
    - :guilabel:`Example:` Higher value then :bash:`-dt 0.5` will lower number of the
      predicted instances, a lower value will increase the number of predicted instances.

    - :guilabel:`default:` 0.5
    - :guilabel:`Allowed options:` Float value between 0.0 and 1.0

:bash:`-pv` or :bash:`--points_in_patch`: Window size used for instance prediction.
    - :guilabel:`Example:` This value indicates the maximum number of points that could be
       found in each point cloud cropped view. Essentially, this will lead to dividing
       a point cloud into smaller overlapping areas that would be segmented individually and
       then stitched and predicted together. `Tips`: 1000 points per crop requires
       ~12 GB of GPU memory. For GPUs with smaller amounts of GPU memory, you can use
       lower numbers 500 or 800. A higher number will always lead to faster inference,
       and may slightly improve segmentation.

    - :guilabel:`default:` 1000
    - :guilabel:`Allowed options:` Int value between 250 and 5000.

.. _compare:

:bash:`-ap` or :bash:`--amira_prefix`: Amira prefix file name.
    - :guilabel:`Example:`  If amira sub-foldr exist, TARDIS will search for files with
      given prefix (e.g. <file_name><amira_prefix>.am). If the correct file is found,
      TARDIS will use its instance segmentation with Amira prediction, and output
      additional file called <file_name>_AmiraCompare.am.

    - :guilabel:`default:` .CorrelationLines

:bash:`-acd` or :bash:`--amira_compare_distance`: Overlying distance threshold between Amira and TARDIS
    - :guilabel:`Example:` The comparison with Amira prediction is done by evaluating
      filaments distance between Amira and TARDIS. This parameter defines the maximum
      distance to the similarity between two splines. Value given in Angstrom [A].

    - :guilabel:`default:` 175

:bash:`-aip` or :bash:`--amira_inter_probability`: Microtubule overlap between Amira and TARDIS
    - :guilabel:`Example:` This parameter define normalize between 0 and 1 overlap
      between filament from TARDIS na Amira sufficient to identifies microtubule as
      a match between both software.

    - :guilabel:`default:` 0.25

.. _filter:

:bash:`-fl` or :bash:`--filter_by_length`: Minimum microtubule length
    - :guilabel:`Example:` Filtering parameters for microtubules, defining maximum microtubule
      length in Angstrom. All filaments shorter then this length will be deleted.

    - :guilabel:`default:` 1000

:bash:`-cs` or :bash:`--connect_splines`: Threshold distance between two microtubules
    - :guilabel:`Example:` To address the issue where microtubules are mistakenly
      identified as two different filaments, we use a filtering technique.
      This involves identifying the direction each filament end points towards and then
      linking any filaments that are facing the same direction and are within
      a certain distance from each other, measured in angstroms. This distance threshold
      determines how far apart two microtubules can be, while still being considered
      as a single unit if they are oriented in the same direction.

    - :guilabel:`default:` 2500

:bash:`-cc` or :bash:`--connect_cylinder`: Microtubule thickens in Angstrom
    - :guilabel:`Example:` To minimize false positives when linking microtubules,
      we limit the search area to a cylindrical radius specified in angstroms.
      For each spline, we find the direction the filament end is pointing in
      and look for another filament that is oriented in the same direction.
      The ends of these filaments must be located within this cylinder
      to be considered connected.

    - :guilabel:`default:` 250

:bash:`-dv` or :bash:`--device`: Define which device to use for inference.
    - :guilabel:`Example:` You can use :bash:`-dv gpu` to use the first available gpu on your system.
      You can also specify the exact GPU device with the number  :bash:`-dv 0`, :bash:`-dv 1`, etc. where 0 is always the default GPU.
      You can also use :bash:`-dv cpu` to perform inference only on the CPU.

    - :guilabel:`default:` 0
    - :guilabel:`Allowed options:` cpu, gpu, 0, 1, 2, 3, etc.

:bash:`-db` or :bash:`--debug`: Enable debugging mode.
    - :guilabel:`Example:` Debugging mode saves all intermediate files allowing for
      debugging any errors. Use only as a developer or if specifically asked for by the developer.

    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` True, False