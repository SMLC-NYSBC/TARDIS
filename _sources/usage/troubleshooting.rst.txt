.. role:: bash(code)
   :language: bash
   :class: highlight

.. role:: guilabel

Troubleshooting
---------------
Here are few know issue and how to troubleshoot them.

Pytorch not compiled with CUDA
______________________________

Fix by reinstall pytorch pre-build with CUDA:


.. code-block::

    pip uninstall pytorch

or

.. code-block::

    conda uninstall torch

and follow instruction for `Pytorch website  <https://pytorch.org>`__
on how to install current pytorch version with CUDA on your system.

TARDIS produce no output and no error message
_____________________________________________
This could be because TARDIS could not find any know object (e.g. membrane or microtubules) in your
tomogram or micrograph.

You can try to reduce :bash:`-ct <float; default 0.25>` to 0.1 or 0.05. If this wont help, please
contact `Developers <rkiewisz@nysbc.org>`__.