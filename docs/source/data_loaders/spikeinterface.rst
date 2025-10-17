SpikeInterface
==============

load_spikedata_from_spikeinterface
-----------------------------------

Convert a SpikeInterface ``SortingExtractor``-like object to ``SpikeData`` by retrieving unit spike trains and converting sample indices to milliseconds using the sorting's sampling frequency.

Parameters
^^^^^^^^^^

* **sorting** (object): Exposes ``get_unit_ids()``, ``get_sampling_frequency()``, and ``get_unit_spike_train(...)``.
* **sampling_frequency** (``float | None``): Override sampling frequency (Hz); if ``None``, use ``sorting.get_sampling_frequency()``.
* **unit_ids** (``Sequence[int | str] | None``): Subset of unit IDs to include; if ``None``, include all.
* **segment_index** (``int``, default ``0``): Segment index for multi-segment sortings.

Returns
^^^^^^^

* **SpikeData**: Spike trains in milliseconds; metadata includes ``source_format='SpikeInterface'``, ``unit_ids``, and ``fs_Hz``.

Raises
^^^^^^

* **TypeError**: If ``sorting`` lacks required methods.
* **ValueError**: If sampling frequency is not positive.

Example
^^^^^^^

.. code-block:: python

   sd = load_spikedata_from_spikeinterface(sorting)

----

load_spikedata_from_spikeinterface_recording
---------------------------------------------

Convert a SpikeInterface ``BaseRecording``-like object into ``SpikeData`` by thresholding raw traces. Automatically orients the trace matrix to (channels × time) using a robust heuristic.

Parameters
^^^^^^^^^^

* **recording** (object): Provides ``get_traces(segment_index=...)`` returning a 2D array, ``get_sampling_frequency()`` or ``sampling_frequency`` attribute, and optionally ``get_num_channels()``.
* **segment_index** (``int``, default ``0``): Segment index to read traces from.
* **threshold_sigma** (``float``, default ``5.0``): Threshold in units of per-channel standard deviation.
* **filter** (``dict | bool``, default ``False``): If ``True``, apply default bandpass; if ``dict``, pass as filter config; if ``False``, no filtering.
* **hysteresis** (``bool``, default ``True``): Use rising-edge detection if ``True``.
* **direction** (``'both' | 'up' | 'down'``, default ``'both'``): Detection polarity.

Returns
^^^^^^^

* **SpikeData**: Detected spike trains per channel in milliseconds.

Raises
^^^^^^

* **ValueError**: If sampling frequency is not positive or traces are not 2D.

Example
^^^^^^^

.. code-block:: python

   sd = load_spikedata_from_spikeinterface_recording(
       recording,
       threshold_sigma=5.0
   )

