from tardis_em._version import version
import os
import logging

# Import logging configuration
from tardis_em.utils.logging_config import configure_tardis_logging

# Temporal fallback for mps devices
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set up centralized logging for TARDIS
configure_tardis_logging(level=logging.INFO)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("TARDIS initialized")

__version__ = version

format_choices = [
    f"{prefix}_{suffix}"
    for prefix in ["None", "am", "mrc", "tif", "npy"]
    for suffix in ["None", "am", "mrc", "tif", "npy", "amSG", "csv", "stl"]
]
