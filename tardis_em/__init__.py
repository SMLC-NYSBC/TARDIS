from tardis_em._version import version
import os

# Temporal fallback for mps devices
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__version__ = version

format_choices = [
    f"{prefix}_{suffix}"
    for prefix in ["None", "am", "mrc", "tif", "npy"]
    for suffix in ["None", "am", "mrc", "tif", "npy", "amSG", "csv", "stl"]
]
