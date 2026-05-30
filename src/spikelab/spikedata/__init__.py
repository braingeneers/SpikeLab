# Import everything in __all__ in spikedata.py, which for some reason loads the module
# object itself into global scope, so delete it afterwards.
from .spikedata import *  # noqa: F401,F403
from .ratedata import *
from .rateslicestack import *
from .spikeslicestack import *
from .hippie_adapter import *  # noqa: F401,F403  — module is import-safe without [hippie]
