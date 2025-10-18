Export Methods
==============

to_hdf5(filepath, ..., style='ragged')
---------------------------------------

Export spike data to an HDF5 file using one of four styles.

**Styles:**

* ``raster``: Binary raster or count matrix
* ``ragged``: Flat concatenated arrays with index array (NWB-like)
* ``group``: One dataset per unit in a group
* ``paired``: Separate indices and times arrays

**Key Parameters:**

* ``raster_bin_size_ms`` (for ``raster`` style)
* ``spike_times_unit`` and ``fs_Hz`` (for ``ragged`` with ``'samples'``)
* ``group_per_unit``, ``group_time_unit`` (for ``group``)
* ``idces_dataset``, ``times_dataset``, ``times_unit`` (for ``paired``)
* Optional ``raw_dataset``, ``raw_time_dataset``, ``raw_time_unit`` to write continuous data/time

Times are converted from internal ms to requested units.

----

to_nwb(filepath, spike_times_dataset='spike_times', spike_times_index_dataset='spike_times_index', group='units')
------------------------------------------------------------------------------------------------------------------

Export spike data to a minimal NWB-compatible file (HDF5 backend).

* Writes ``/units/spike_times`` and ``/units/spike_times_index`` in seconds
* Round-trippable with ``load_spikedata_from_nwb(..., prefer_pynwb=False)``

----

to_kilosort(folder, fs_Hz, spike_times_file='spike_times.npy', spike_clusters_file='spike_clusters.npy', time_unit='samples', cluster_ids=None)
---------------------------------------------------------------------------------------------------------------------------------------------------

Export spike data to KiloSort/Phy format.

* Produces ``spike_times.npy`` and ``spike_clusters.npy``
* ``time_unit``: ``'samples'`` (default, requires ``fs_Hz``), ``'ms'``, or ``'s'``
* ``cluster_ids`` maps SpikeData unit indices to arbitrary cluster IDs

