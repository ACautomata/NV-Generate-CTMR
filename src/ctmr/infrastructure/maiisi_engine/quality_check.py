# Bridge module (see this package's __init__): helpers still homed in scripts.quality_check.
# The frozen engine bodies' relative ``from .quality_check import ...`` resolves through
# here. Update THIS file only when the helpers move homes -- never the bodies.
from ctmr.infrastructure.dataio.quality_check import is_outlier  # noqa: F401  (bridge re-export)
