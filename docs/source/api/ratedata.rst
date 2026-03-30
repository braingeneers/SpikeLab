========
RateData
========

The ``RateData`` class represents instantaneous firing rates for one or more
units. It stores a 2-D array of shape ``(num_units, num_bins)`` along with
bin size and time axis metadata. ``RateData`` objects are typically created
from a ``SpikeData`` object via binning and optional smoothing.

.. autoclass:: spikelab.spikedata.ratedata.RateData
   :members:
   :show-inheritance:
   :special-members: __init__
