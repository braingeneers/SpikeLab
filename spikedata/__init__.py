# Import everything in __all__ in spikedata.py, which for some reason loads the module
# object itself into global scope, so delete it afterwards.
from .spikedata import *  # noqa F401
from ._version import version as __version__

del spikedata, _version  # noqa F821
