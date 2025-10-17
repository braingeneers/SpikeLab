KiloSort / Phy
==============

export_spikedata_to_kilosort
-----------------------------

Create ``spike_times.npy`` and ``spike_clusters.npy`` suitable for KiloSort/Phy. Times default to integer samples at ``fs_Hz``; can also write in ``'ms'`` or ``'s'``.

Parameters
^^^^^^^^^^

* **sd** (``SpikeData``): The spike data object to export.
* **folder** (``str``): Path to the output directory.
* **fs_Hz** (``float``): Sampling frequency (Hz), required when ``time_unit='samples'``.
* **spike_times_file** (``str``, default ``'spike_times.npy'``): Filename for spike times.
* **spike_clusters_file** (``str``, default ``'spike_clusters.npy'``): Filename for cluster assignments.
* **time_unit** (``'samples'|'ms'|'s'``, default ``'samples'``): Time unit for spike times.
* **cluster_ids** (``list|array|None``): Custom cluster IDs mapping; if ``None``, uses 0-based indices.

Example
^^^^^^^

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

