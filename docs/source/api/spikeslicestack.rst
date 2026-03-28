===============
SpikeSliceStack
===============

The ``SpikeSliceStack`` class stores event-aligned slices of raw spike train
data. It holds a list of ``SpikeData`` objects (one per trial or event window)
and provides methods for converting to raster arrays of shape
``(num_units, num_bins, num_slices)`` and computing trial-averaged statistics.

.. autoclass:: spikelab.spikedata.spikeslicestack.SpikeSliceStack
   :members:
   :show-inheritance:
   :special-members: __init__
