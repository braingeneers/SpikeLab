# Import everything in __all__ in spikedata.py, which for some reason loads the module
# object itself into global scope, so delete it afterwards.
from .spikedata import *  # noqa: F401,F403
from .neuron_attributes import NeuronAttributes  # noqa: F401

__version__ = "0.0.0"
