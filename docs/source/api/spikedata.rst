=========
SpikeData
=========

The ``SpikeData`` class is the primary data structure in SpikeLab. It holds
spike trains for one or more units (neurons or electrodes), where each spike
train is a sequence of spike times in milliseconds. Most analysis workflows
start by creating or loading a ``SpikeData`` object.

.. autoclass:: spikelab.spikedata.SpikeData
   :members:
   :show-inheritance:
   :special-members: __init__
