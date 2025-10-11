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

HDF5 (Generic)
--------------

load_spikedata_from_hdf5
^^^^^^^^^^^^^^^^^^^^^^^^^

Load spike trains from an HDF5 file using one of four input styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can be attached. Times are converted to milliseconds.

Parameters
""""""""""

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

Returns
"""""""

* **SpikeData**: Spike trains in milliseconds; may include attached ``raw_data`` and ``raw_time``.

Raises
""""""

* **ValueError**: If not exactly one input style is specified, or missing required arguments for a style, or invalid time units.
* **ImportError**: If ``h5py`` is unavailable.

Behavior and Notes
""""""""""""""""""

Exactly one of the four styles must be provided. Raster uses ``SpikeData.from_raster``; paired arrays use ``SpikeData.from_idces_times``; ragged arrays and group-per-unit build per-unit trains by splitting and converting to ms. Optional raw arrays are attached if both ``raw_dataset`` and ``raw_time_dataset`` are provided.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_hdf5(
       "data.h5",
       spike_times_dataset="/units/spike_times",
       spike_times_index_dataset="/units/spike_times_index",
       spike_times_unit="s",  # 's' | 'ms' | 'samples' (requires fs_Hz)
       fs_Hz=20000.0,
   )

Supported Styles
""""""""""""""""

Specify exactly one:

* **Raster** (units × time): ``raster_dataset`` + ``raster_bin_size_ms``
* **Flat ragged arrays** (NWB-like): ``spike_times_dataset`` + ``spike_times_index_dataset``
* **Group-per-unit**: ``group_per_unit`` (each child holds spike times)
* **Paired** (indices, times): ``idces_dataset`` + ``times_dataset`` + ``times_unit``

Optional raw attachments: ``raw_dataset`` + ``raw_time_dataset`` with ``raw_time_unit`` in ``s/ms/samples`` (needs ``fs_Hz`` for samples).

----

HDF5 Thresholding
-----------------

load_spikedata_from_hdf5_raw_thresholded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Threshold-and-detect spikes from an HDF5 dataset of raw traces shaped (channels × time) or (time × channels). Returns a ``SpikeData`` built from detected spikes per channel.

Parameters
""""""""""

* **filepath** (``str``): Path to the HDF5 file.
* **dataset** (``str``): Dataset path containing raw traces.
* **fs_Hz** (``float``): Sampling frequency (Hz).
* **threshold_sigma** (``float``, default ``5.0``): Threshold in units of per-channel standard deviation.
* **filter** (``dict | bool``, default ``True``): If ``True``, apply default Butterworth bandpass; if a ``dict``, passed as filter configuration; if ``False``, no filtering.
* **hysteresis** (``bool``, default ``True``): Use rising-edge detection if ``True``.
* **direction** (``'both' | 'up' | 'down'``, default ``'both'``): Polarity of detection.

Returns
"""""""

* **SpikeData**: Detected spike trains per channel in milliseconds.

Raises
""""""

* **ImportError**: If ``h5py`` is unavailable.
* **ValueError**: Propagated from detection if invalid arguments are provided.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_hdf5_raw_thresholded(
       "raw.h5",
       dataset="/raw",
       fs_Hz=20000.0,
       threshold_sigma=5.0,
       filter=True,
       hysteresis=True,
       direction="both",
   )

----

NWB (Units)
-----------

load_spikedata_from_nwb
^^^^^^^^^^^^^^^^^^^^^^^

Load spike trains from an NWB file's Units table. Prefers ``pynwb``; falls back to ``h5py`` to read ``/units/spike_times`` and ``/units/spike_times_index``. Times are in seconds and converted to milliseconds.

Parameters
""""""""""

* **filepath** (``str``): Path to the NWB file.
* **prefer_pynwb** (``bool``, default ``True``): If ``True``, try ``pynwb`` first; on failure, fall back to ``h5py`` with a warning.
* **length_ms** (``float | None``): Recording duration; inferred from last spike if not provided.

Returns
"""""""

* **SpikeData**: Spike trains in milliseconds; metadata includes ``source_file`` and ``format='NWB'``.

Raises
""""""

* **ValueError**: If the file lacks a Units table or spike_times datasets.
* **ImportError**: If ``h5py`` is unavailable for fallback.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_nwb("recording.nwb", prefer_pynwb=True)

Falls back to ``h5py`` if ``pynwb`` is unavailable.

----

KiloSort / Phy
--------------

load_spikedata_from_kilosort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load clusters from KiloSort/Phy outputs: ``spike_times.npy`` and ``spike_clusters.npy``. Optionally parse a TSV to filter clusters by label. Spike times are converted from samples or seconds to milliseconds.

Parameters
""""""""""

* **folder** (``str``): Path to KiloSort/Phy output directory.
* **fs_Hz** (``float``): Sampling frequency (Hz) when ``time_unit='samples'``.
* **spike_times_file** (``str``, default ``'spike_times.npy'``): File with spike times.
* **spike_clusters_file** (``str``, default ``'spike_clusters.npy'``): File with cluster assignments.
* **cluster_info_tsv** (``str | None``): Optional TSV file with cluster metadata (``group``/``KSLabel``, ``cluster_id``/``id``).
* **time_unit** (``'samples' | 's' | 'ms'``, default ``'samples'``): Unit of ``spike_times.npy``.
* **include_noise** (``bool``, default ``False``): If ``False``, keep only clusters labeled ``good``/``mua`` when TSV provided; if ``True``, keep all clusters.
* **length_ms** (``float | None``): Recording duration; inferred from last spike if not provided.

Returns
"""""""

* **SpikeData**: Spike trains grouped by cluster; metadata contains ``cluster_ids``, ``fs_Hz``, and ``source_folder``.

Raises
""""""

* **ValueError**: If ``spike_times`` and ``spike_clusters`` lengths mismatch.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_kilosort(
       "path/to/ks/",
       fs_Hz=30000.0,
       cluster_info_tsv="cluster_info.tsv",  # optional
   )

----

SpikeInterface
--------------

load_spikedata_from_spikeinterface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert a SpikeInterface ``SortingExtractor``-like object to ``SpikeData`` by retrieving unit spike trains and converting sample indices to milliseconds using the sorting's sampling frequency.

Parameters
""""""""""

* **sorting** (object): Exposes ``get_unit_ids()``, ``get_sampling_frequency()``, and ``get_unit_spike_train(...)``.
* **sampling_frequency** (``float | None``): Override sampling frequency (Hz); if ``None``, use ``sorting.get_sampling_frequency()``.
* **unit_ids** (``Sequence[int | str] | None``): Subset of unit IDs to include; if ``None``, include all.
* **segment_index** (``int``, default ``0``): Segment index for multi-segment sortings.

Returns
"""""""

* **SpikeData**: Spike trains in milliseconds; metadata includes ``source_format='SpikeInterface'``, ``unit_ids``, and ``fs_Hz``.

Raises
""""""

* **TypeError**: If ``sorting`` lacks required methods.
* **ValueError**: If sampling frequency is not positive.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_spikeinterface(sorting)

----

load_spikedata_from_spikeinterface_recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert a SpikeInterface ``BaseRecording``-like object into ``SpikeData`` by thresholding raw traces. Automatically orients the trace matrix to (channels × time) using a robust heuristic.

Parameters
""""""""""

* **recording** (object): Provides ``get_traces(segment_index=...)`` returning a 2D array, ``get_sampling_frequency()`` or ``sampling_frequency`` attribute, and optionally ``get_num_channels()``.
* **segment_index** (``int``, default ``0``): Segment index to read traces from.
* **threshold_sigma** (``float``, default ``5.0``): Threshold in units of per-channel standard deviation.
* **filter** (``dict | bool``, default ``False``): If ``True``, apply default bandpass; if ``dict``, pass as filter config; if ``False``, no filtering.
* **hysteresis** (``bool``, default ``True``): Use rising-edge detection if ``True``.
* **direction** (``'both' | 'up' | 'down'``, default ``'both'``): Detection polarity.

Returns
"""""""

* **SpikeData**: Detected spike trains per channel in milliseconds.

Raises
""""""

* **ValueError**: If sampling frequency is not positive or traces are not 2D.

Example
"""""""

.. code-block:: python

   sd = load_spikedata_from_spikeinterface_recording(
       recording,
       threshold_sigma=5.0
   )

----

Notes
-----

* Times are stored in milliseconds in ``SpikeData``.
* Optional dependencies are imported lazily (e.g., ``h5py``, ``pynwb``, ``pandas``).
* See ``tests/test_dataloaders.py`` for runnable examples and edge cases.

