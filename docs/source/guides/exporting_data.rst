==============
Exporting Data
==============

SpikeLab provides export functions for saving
:class:`~spikelab.spikedata.spike_data.SpikeData` objects to several common
electrophysiology formats. All exporters assume spike times are in
**milliseconds** and convert to the target format's native time unit
automatically.

The export functions live in ``spikelab.data_loaders.data_exporters``.


To HDF5
-------

The HDF5 exporter supports the same four storage styles as the loader:

- **Ragged** (default) -- a flat spike times array plus an index array.
- **Raster** -- a ``(N, T)`` spike count matrix.
- **Group** -- one HDF5 group per unit.
- **Paired** -- two parallel arrays of unit indices and spike times.

.. code-block:: python

   from spikelab.data_loaders.data_exporters import export_spikedata_to_hdf5

   # Default ragged style (spike_times + spike_times_index)
   export_spikedata_to_hdf5(sd, "output.h5")

   # Raster style
   export_spikedata_to_hdf5(
       sd, "output.h5",
       style="raster",
       raster_dataset="raster",
       raster_bin_size_ms=1.0,
   )

   # Group-per-unit style
   export_spikedata_to_hdf5(
       sd, "output.h5",
       style="group",
       group_per_unit="units",
       group_time_unit="s",
   )

   # Paired style
   export_spikedata_to_hdf5(
       sd, "output.h5",
       style="paired",
       idces_dataset="idces",
       times_dataset="times",
       times_unit="ms",
   )

Ragged vs raster style
^^^^^^^^^^^^^^^^^^^^^^^

The **ragged** style is space-efficient for sparse data: it stores one
concatenated array of spike times and a separate index array that marks unit
boundaries. This is the same layout used by NWB files.

The **raster** style stores a full ``(N, T)`` matrix of spike counts. This is
convenient for downstream tools that expect a matrix, but can be large for long
recordings with fine time bins. When using raster style, you must specify
``raster_bin_size_ms``.

The ``spike_times_unit`` parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ragged style, the ``spike_times_unit`` parameter (default ``'s'``) controls
the unit of the exported spike times. Set to ``'ms'`` to write times in
milliseconds instead of seconds. Analogous parameters exist for the other
styles (``group_time_unit``, ``times_unit``).

Exporting raw data
^^^^^^^^^^^^^^^^^^

If the ``SpikeData`` object has raw voltage traces attached (e.g. from
:func:`~spikelab.spikedata.spike_data.SpikeData.from_thresholding`), you can
include them in the HDF5 file:

.. code-block:: python

   export_spikedata_to_hdf5(
       sd, "output.h5",
       raw_dataset="raw_data",
       raw_time_dataset="raw_time",
       raw_time_unit="ms",
   )


To NWB
------

Export to Neurodata Without Borders format. Spike times are saved in seconds.
Unit IDs, electrode assignments, and electrode locations are written when
available in ``neuron_attributes``.

.. code-block:: python

   from spikelab.data_loaders.data_exporters import export_spikedata_to_nwb

   export_spikedata_to_nwb(sd, "session.nwb")

   # With custom dataset names
   export_spikedata_to_nwb(
       sd, "session.nwb",
       spike_times_dataset="spike_times",
       spike_times_index_dataset="spike_times_index",
       group="units",
   )


To KiloSort
-----------

Export to the KiloSort/Phy folder format. This writes ``spike_times.npy`` and
``spike_clusters.npy`` files, and optionally ``channel_map.npy`` if electrode
information is available.

The ``fs_Hz`` parameter is required to convert spike times from milliseconds to
the target time unit (samples by default).

.. code-block:: python

   from spikelab.data_loaders.data_exporters import export_spikedata_to_kilosort

   spike_times_path, spike_clusters_path = export_spikedata_to_kilosort(
       sd,
       "output_folder/",
       fs_Hz=30000,
   )

   # With custom options
   export_spikedata_to_kilosort(
       sd,
       "output_folder/",
       fs_Hz=30000,
       time_unit="samples",
       spike_times_file="spike_times.npy",
       spike_clusters_file="spike_clusters.npy",
       cluster_ids=None,
   )


To pickle
---------

Save a ``SpikeData`` object as a pickle file. Supports optional upload to
Amazon S3.

.. code-block:: python

   from spikelab.data_loaders.data_exporters import export_spikedata_to_pickle

   # Local file
   path = export_spikedata_to_pickle(sd, "my_data.pkl")

   # Upload to S3
   s3_url = export_spikedata_to_pickle(
       sd,
       "s3://my-bucket/my_data.pkl",
       s3_upload=True,
       aws_access_key_id="...",
       aws_secret_access_key="...",
   )

The function returns the file path (local) or S3 URL as a string.
