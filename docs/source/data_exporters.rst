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

HDF5 (Generic)
--------------

export_spikedata_to_hdf5
^^^^^^^^^^^^^^^^^^^^^^^^^

Export spike trains to HDF5 using one of four styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can also be exported.

Parameters (Selected)
"""""""""""""""""""""

* **sd** (``SpikeData``): The spike data object to export.
* **filepath** (``str``): Path to the output HDF5 file.
* **style** (``'raster'|'ragged'|'group'|'paired'``): Output organization style.

**Raster Style:**

* ``raster_dataset`` (``str``): Dataset path for the raster matrix.
* ``raster_bin_size_ms`` (``float``): Bin size in milliseconds.

**Ragged Style:**

* ``spike_times_dataset`` (``str``): Dataset path for concatenated spike times.
* ``spike_times_index_dataset`` (``str``): Dataset path for end indices per unit.
* ``spike_times_unit`` (``'s'|'ms'|'samples'``): Time unit for spike times.
* ``fs_Hz`` (``float|None``): Sampling frequency (Hz), required when ``spike_times_unit='samples'``.

**Group Style:**

* ``group_per_unit`` (``str``): Group path where each unit gets its own dataset.
* ``group_time_unit`` (``'s'|'ms'|'samples'``): Time unit for spike times.
* ``fs_Hz`` (``float|None``): Required when ``group_time_unit='samples'``.

**Paired Style:**

* ``idces_dataset`` (``str``): Dataset path for unit indices.
* ``times_dataset`` (``str``): Dataset path for spike times.
* ``times_unit`` (``'s'|'ms'|'samples'``): Time unit for spike times.
* ``fs_Hz`` (``float|None``): Required when ``times_unit='samples'``.

**Optional Raw Data:**

* ``raw_dataset`` (``str|None``): Dataset path for raw analog data.
* ``raw_time_dataset`` (``str|None``): Dataset path for raw time vector.
* ``raw_time_unit`` (``'s'|'ms'|'samples'``): Time unit for raw time vector.

Example
"""""""

.. code-block:: python

   # Using the instance method
   sd.to_hdf5(
       "out.h5",
       style="ragged",
       spike_times_unit="s",
   )

   # Using the standalone function
   from data_loaders.data_exporters import export_spikedata_to_hdf5
   
   export_spikedata_to_hdf5(
       sd,
       "out.h5",
       style="ragged",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s",
   )

----

NWB (Units)
-----------

export_spikedata_to_nwb
^^^^^^^^^^^^^^^^^^^^^^^

Write ragged spike times to ``/units/spike_times`` and ``/units/spike_times_index`` in seconds, sufficient for round-tripping with the NWB loader (h5py-based path).

Parameters
""""""""""

* **sd** (``SpikeData``): The spike data object to export.
* **filepath** (``str``): Path to the output NWB file.
* **spike_times_dataset** (``str``, default ``'spike_times'``): Name of the spike times dataset.
* **spike_times_index_dataset** (``str``, default ``'spike_times_index'``): Name of the spike times index dataset.
* **group** (``str``, default ``'units'``): HDF5 group name.

Example
"""""""

.. code-block:: python

   # Using the instance method
   sd.to_nwb("out.nwb")

   # Using the standalone function
   from data_loaders.data_exporters import export_spikedata_to_nwb
   
   export_spikedata_to_nwb(sd, "out.nwb")

----

KiloSort / Phy
--------------

export_spikedata_to_kilosort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create ``spike_times.npy`` and ``spike_clusters.npy`` suitable for KiloSort/Phy. Times default to integer samples at ``fs_Hz``; can also write in ``'ms'`` or ``'s'``.

Parameters
""""""""""

* **sd** (``SpikeData``): The spike data object to export.
* **folder** (``str``): Path to the output directory.
* **fs_Hz** (``float``): Sampling frequency (Hz), required when ``time_unit='samples'``.
* **spike_times_file** (``str``, default ``'spike_times.npy'``): Filename for spike times.
* **spike_clusters_file** (``str``, default ``'spike_clusters.npy'``): Filename for cluster assignments.
* **time_unit** (``'samples'|'ms'|'s'``, default ``'samples'``): Time unit for spike times.
* **cluster_ids** (``list|array|None``): Custom cluster IDs mapping; if ``None``, uses 0-based indices.

Example
"""""""

.. code-block:: python

   # Using the instance method
   sd.to_kilosort("ks_out", fs_Hz=30000.0)

   # Using the standalone function
   from data_loaders.data_exporters import export_spikedata_to_kilosort
   
   export_spikedata_to_kilosort(
       sd,
       "ks_out",
       fs_Hz=30000.0,
       time_unit="samples",
   )

----

Notes
-----

* Requires ``h5py`` for HDF5/NWB exports. Install with ``pip install h5py``.
* See ``tests/test_dataexporters.py`` for runnable examples and edge cases.
* All exporters handle time unit conversions automatically from internal milliseconds.

