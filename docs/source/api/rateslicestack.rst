==============
RateSliceStack
==============

The ``RateSliceStack`` class stores event-aligned slices of firing rate data.
It holds a 3-D array of shape ``(num_units, num_bins, num_slices)`` where each
slice corresponds to one trial or event window. This structure supports
trial-averaging, per-unit statistics, and visualization of peri-event firing
rate profiles.

.. autoclass:: spikelab.spikedata.rateslicestack.RateSliceStack
   :members:
   :show-inheritance:
   :special-members: __init__
