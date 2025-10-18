NWB (Units)
===========

export_spikedata_to_nwb
-----------------------

Write ragged spike times to ``/units/spike_times`` and ``/units/spike_times_index`` in seconds, sufficient for round-tripping with the NWB loader (h5py-based path).

Parameters
^^^^^^^^^^

* **sd** (``SpikeData``): The spike data object to export.
* **filepath** (``str``): Path to the output NWB file.
* **spike_times_dataset** (``str``, default ``'spike_times'``): Name of the spike times dataset.
* **spike_times_index_dataset** (``str``, default ``'spike_times_index'``): Name of the spike times index dataset.
* **group** (``str``, default ``'units'``): HDF5 group name.

Example
^^^^^^^

.. code-block:: python

   # Using the instance method
   sd.to_nwb("out.nwb")

   # Using the standalone function
   from data_loaders.data_exporters import export_spikedata_to_nwb
   
   export_spikedata_to_nwb(sd, "out.nwb")

