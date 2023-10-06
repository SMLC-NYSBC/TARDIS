from tardis_em._version import version as __version__

# Uncomment on deployment
from tardis_em.utils.ota_update import ota_update

ota = ota_update()
