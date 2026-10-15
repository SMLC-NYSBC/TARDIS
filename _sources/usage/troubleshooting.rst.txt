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

Error loading model: ``invalid load key, '<'``
_______________________________________________
The full message looks like:

.. code-block::

    _pickle.UnpicklingError: invalid load key, '<'.

This means the ``model_weights.pth`` file is **not** a model checkpoint - it is an
XML error page (starting with ``<``) that was saved by a failed download, for
example ``NoSuchKey`` or ``AccessDenied`` returned by S3. A genuine checkpoint is
several MB to hundreds of MB; a corrupted one is usually a few hundred bytes.

Fix:

1. Delete the cached weights and let TARDIS re-download them with internet access:

.. code-block:: bash

    rm -rf ~/.tardis_em

2. Or, if you pre-downloaded the weights manually, replace the corrupted file with
   a direct download (see `Offline / air-gapped clusters`_ below).

Recent TARDIS versions detect this automatically: a corrupted cached file is
re-downloaded, and a failed download aborts with a clear error instead of being
saved as a checkpoint.

Offline / air-gapped clusters
_____________________________
On clusters where compute nodes have no internet access, pre-download the weights
on a login node and point TARDIS at them with ``-cch`` / ``-dch`` (or via the
``$TARDIS_CNN_MODELS`` / ``$TARDIS_DIST_MODELS`` environment variables).

The weights live in the public S3 bucket ``tardis-weigths``:

- CNN models: ``tardis_em/fnet_attn_32/<model>/<version>/model_weights.pth``
- DIST models: ``tardis_em/dist_triang/<model>/<version>/model_weights.pth``

Option 1 - ``tardis_fetch`` (downloads one model at a time):

.. code-block:: bash

    tardis_fetch -dir ./weights -mc actin_3d -md 3d

Option 2 - mirror the whole bucket with ``rclone`` (no AWS credentials needed):

.. code-block:: bash

    mkdir -p weights

    rclone config create tardis_s3 s3 \
      provider AWS \
      env_auth false \
      region us-east-1

    rclone copy \
      --s3-no-check-bucket \
      --progress \
      tardis_s3:tardis-weigths/tardis_em \
      ./weights

Option 3 - download a single file directly, e.g. the 3D actin CNN model:

.. code-block:: bash

    curl -L -o model_weights.pth \
      https://tardis-weigths.s3.us-east-1.amazonaws.com/tardis_em/fnet_attn_32/actin_3d/V_3/model_weights.pth

After downloading, sanity-check the file size (a few hundred bytes means the
download failed and returned an error page):

.. code-block:: bash

    ls -lh weights/**/model_weights.pth