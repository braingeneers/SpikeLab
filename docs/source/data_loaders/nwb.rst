NWB (Units)
===========

load_spikedata_from_nwb
-----------------------

Load spike trains from an NWB file's Units table. Prefers ``pynwb``; falls back to ``h5py`` to read ``/units/spike_times`` and ``/units/spike_times_index``. Times are in seconds and converted to milliseconds.

Parameters
^^^^^^^^^^

* **filepath** (``str``): Path to the NWB file.
* **prefer_pynwb** (``bool``, default ``True``): If ``True``, try ``pynwb`` first; on failure, fall back to ``h5py`` with a warning.
* **length_ms** (``float | None``): Recording duration; inferred from last spike if not provided.

Returns
^^^^^^^

* **SpikeData**: Spike trains in milliseconds; metadata includes ``source_file`` and ``format='NWB'``.

Raises
^^^^^^

* **ValueError**: If the file lacks a Units table or spike_times datasets.
* **ImportError**: If ``h5py`` is unavailable for fallback.

Example
^^^^^^^

.. code-block:: python

   sd = load_spikedata_from_nwb("recording.nwb", prefer_pynwb=True)

Falls back to ``h5py`` if ``pynwb`` is unavailable.

