HDF5 (Generic)
==============

export_spikedata_to_hdf5
-------------------------

Export spike trains to HDF5 using one of four styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can also be exported.

Parameters (Selected)
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^

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

