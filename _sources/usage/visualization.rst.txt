.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: guilabel

Visualization
-------------
TARDIS is pre-build with visualization tools that can help you quickly visualize your results.

You can visualize sematic binary masks in `[.mrc, .rec, .am, .tif]` format,
or instance segmentation results in `[.csv, .npy, .am]` formats.

Simple Usage:
`````````````

.. code-block::

    tardis-vis -dir <path/to/your/file>


Advance usage:
``````````````
Below you can find all available arguments you can use with :bash:`tardis_mt`,
with the explanation for their functionality:

:bash:`-dir` or :bash:`--path`: File directory to visualize.
    - :guilabel:`Allowed options:` str

:bash:`-2d` or :bash:`--_2d`: If True, expect 2D data.
    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` bool

:bash:`-t` or :bash:`--type_`: Visualize as filaments lines or point cloud.
    - :guilabel:`default:` p
    - :guilabel:`Allowed options:` f, p

:bash:`-a` or :bash:`--animate`: If True, show visualization with pre-build rotation animation.
    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` bool

:bash:`-wn` or :bash:`--with_node`: If visualizing filaments, you can show color-codded filaments with nodes.
    - :guilabel:`default:` False
    - :guilabel:`Allowed options:` bool