from tardis_em._version import version

__version__ = version

# Uncomment on deployment
from tardis_em.utils.ota_update import ota_update

ota = ota_update()
