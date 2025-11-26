KiloSort / Phy
==============

load_spikedata_from_kilosort
-----------------------------

Load clusters from KiloSort/Phy outputs: ``spike_times.npy`` and ``spike_clusters.npy``. Optionally parse a TSV to filter clusters by label. Spike times are converted from samples or seconds to milliseconds.

Parameters
^^^^^^^^^^

* **folder** (``str``): Path to KiloSort/Phy output directory.
* **fs_Hz** (``float``): Sampling frequency (Hz) when ``time_unit='samples'``.
* **spike_times_file** (``str``, default ``'spike_times.npy'``): File with spike times.
* **spike_clusters_file** (``str``, default ``'spike_clusters.npy'``): File with cluster assignments.
* **time_unit** (``'samples' | 's' | 'ms'``, default ``'samples'``): Unit of ``spike_times.npy``.
* **include_noise** (``bool``, default ``False``): Reserved for future use.
* **length_ms** (``float | None``): Recording duration; inferred from last spike if not provided.

Returns
^^^^^^^

* **SpikeData**: Spike trains grouped by cluster; metadata contains ``fs_Hz`` and ``source_folder``.

Raises
^^^^^^

* **ValueError**: If ``spike_times`` and ``spike_clusters`` lengths mismatch.

Example
^^^^^^^

.. code-block:: python

   sd = load_spikedata_from_kilosort(
       "path/to/ks/",
       fs_Hz=30000.0,
   )

