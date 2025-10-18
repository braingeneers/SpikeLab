Data Exporters
==============

Overview
--------

Data exporters write ``SpikeData`` to common neuroscience formats. Times are converted from internal milliseconds to requested units. You can call the standalone functions or use the convenience instance methods on ``SpikeData``.

Import convenience:

.. code-block:: python

   from data_loaders.data_exporters import (
       export_spikedata_to_hdf5,
       export_spikedata_to_nwb,
       export_spikedata_to_kilosort,
   )

----

.. toctree::
   :maxdepth: 2
   :caption: Exporter Functions

   data_exporters/hdf5
   data_exporters/nwb
   data_exporters/kilosort

----

Notes
-----

* Requires ``h5py`` for HDF5/NWB exports. Install with ``pip install h5py``.
* See ``tests/test_dataexporters.py`` for runnable examples and edge cases.
* All exporters handle time unit conversions automatically from internal milliseconds.
