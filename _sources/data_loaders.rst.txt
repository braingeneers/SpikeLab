Data Loaders
============

Overview
--------

Data loaders convert common neurophysiology formats into ``SpikeData``. Times are normalized to milliseconds internally.

Import convenience:

.. code-block:: python

   from data_loaders import (
       load_spikedata_from_hdf5,
       load_spikedata_from_hdf5_raw_thresholded,
       load_spikedata_from_nwb,
       load_spikedata_from_kilosort,
       load_spikedata_from_spikeinterface,
       load_spikedata_from_spikeinterface_recording,
   )

----

.. toctree::
   :maxdepth: 2
   :caption: Loader Functions

   data_loaders/hdf5
   data_loaders/hdf5_thresholding
   data_loaders/nwb
   data_loaders/kilosort
   data_loaders/spikeinterface

----

Notes
-----

* Times are stored in milliseconds in ``SpikeData``.
* Optional dependencies are imported lazily (e.g., ``h5py``, ``pynwb``, ``pandas``).
* See ``tests/test_dataloaders.py`` for runnable examples and edge cases.
