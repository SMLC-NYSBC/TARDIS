from tardis_pytorch._version import version

__version__ = version

# Uncomment on deployment
from tardis_pytorch.utils.ota_update import ota_update
ota = ota_update()
