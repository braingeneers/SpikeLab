HDF5 (Generic)
==============

load_spikedata_from_hdf5
-------------------------

Load spike trains from an HDF5 file using one of four input styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can be attached. Times are converted to milliseconds.

Parameters
^^^^^^^^^^

* **filepath** (``str``): Path to the HDF5 file.
* **raster_dataset** (``str | None``): Dataset path for a 2D raster/counts matrix (units × time). Use with ``raster_bin_size_ms``.
* **raster_bin_size_ms** (``float | None``): Bin size (ms) for ``raster_dataset``.
* **spike_times_dataset** (``str | None``): Dataset path for flat concatenated spike times (ragged array style).
* **spike_times_index_dataset** (``str | None``): Dataset path for end indices per unit (ragged array style).
* **spike_times_unit** (``'s' | 'ms' | 'samples'``): Unit for ``spike_times_dataset``.
* **fs_Hz** (``float | None``): Sampling frequency (Hz). Required when any unit is ``'samples'``.
* **group_per_unit** (``str | None``): Group path containing one dataset per unit with that unit's spike times.
* **group_time_unit** (``'s' | 'ms' | 'samples'``): Unit for datasets under ``group_per_unit``.
* **idces_dataset** (``str | None``): Dataset path for unit indices (paired arrays style).
* **times_dataset** (``str | None``): Dataset path for spike times (paired arrays style).
* **times_unit** (``'s' | 'ms' | 'samples'``): Unit for ``times_dataset``.
* **raw_dataset** (``str | None``): Dataset path for optional raw analog data (e.g., channels × time).
* **raw_time_dataset** (``str | None``): Dataset path for time vector corresponding to ``raw_dataset``.
* **raw_time_unit** (``'s' | 'ms' | 'samples'``): Unit of ``raw_time_dataset``.
* **length_ms** (``float | None``): Recording duration; inferred from last spike if not provided.
* **metadata** (``Mapping[str, object] | None``): Extra metadata to attach; ``source_file`` is added automatically.
* **cache_dir** (``str | None``): Directory for caching S3 downloads. Only used if filepath is an S3 URI. Default: None
* **s3_endpoint_url** (``str | None``): S3 endpoint URL. Only used if filepath is an S3 URI. Default: ``'https://s3-west.nrp-nautilus.io'`` (Nautilus)

Returns
^^^^^^^

* **SpikeData**: Spike trains in milliseconds; may include attached ``raw_data`` and ``raw_time``.

Raises
^^^^^^

* **ValueError**: If not exactly one input style is specified, or missing required arguments for a style, or invalid time units.
* **ImportError**: If ``h5py`` is unavailable.

Behavior and Notes
^^^^^^^^^^^^^^^^^^

Exactly one of the four styles must be provided. Raster uses ``SpikeData.from_raster``; paired arrays use ``SpikeData.from_idces_times``; ragged arrays and group-per-unit build per-unit trains by splitting and converting to ms. Optional raw arrays are attached if both ``raw_dataset`` and ``raw_time_dataset`` are provided.

Example
^^^^^^^

.. code-block:: python

   sd = load_spikedata_from_hdf5(
       "data.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s",  # 's' | 'ms' | 'samples' (requires fs_Hz)
       fs_Hz=20000.0,
   )

Supported Styles
^^^^^^^^^^^^^^^^

Specify exactly one:

* **Raster** (units × time): ``raster_dataset`` + ``raster_bin_size_ms``
* **Flat ragged arrays** (NWB-like): ``spike_times_dataset`` + ``spike_times_index_dataset``
* **Group-per-unit**: ``group_per_unit`` (each child holds spike times)
* **Paired** (indices, times): ``idces_dataset`` + ``times_dataset`` + ``times_unit``

Optional raw attachments: ``raw_dataset`` + ``raw_time_dataset`` with ``raw_time_unit`` in ``s/ms/samples`` (needs ``fs_Hz`` for samples).

S3 Support
^^^^^^^^^^

Load HDF5 files directly from S3 storage:

.. code-block:: python

   # Load from S3 URI
   sd = load_spikedata_from_hdf5(
       "s3://my-bucket/data/recording.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s"
   )
   
   # With custom cache directory
   sd = load_spikedata_from_hdf5(
       "s3://my-bucket/data/recording.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s",
       cache_dir="/path/to/cache"
   )
   
   # With custom S3 endpoint
   sd = load_spikedata_from_hdf5(
       "s3://my-bucket/data/recording.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s",
       s3_endpoint_url="https://s3.amazonaws.com"
   )

**Requirements:** S3 support requires ``boto3``:

.. code-block:: bash

   pip install boto3

**Caching:** S3 files are automatically downloaded and cached in ``cache_dir`` (or a temporary directory if not specified). Subsequent loads with the same ``cache_dir`` will use the cached file.

