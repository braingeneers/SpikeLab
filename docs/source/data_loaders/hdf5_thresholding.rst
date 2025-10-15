HDF5 Thresholding
=================

load_spikedata_from_hdf5_raw_thresholded
-----------------------------------------

Threshold-and-detect spikes from an HDF5 dataset of raw traces shaped (channels × time) or (time × channels). Returns a ``SpikeData`` built from detected spikes per channel.

Parameters
^^^^^^^^^^

* **filepath** (``str``): Path to the HDF5 file.
* **dataset** (``str``): Dataset path containing raw traces.
* **fs_Hz** (``float``): Sampling frequency (Hz).
* **threshold_sigma** (``float``, default ``5.0``): Threshold in units of per-channel standard deviation.
* **filter** (``dict | bool``, default ``True``): If ``True``, apply default Butterworth bandpass; if a ``dict``, passed as filter configuration; if ``False``, no filtering.
* **hysteresis** (``bool``, default ``True``): Use rising-edge detection if ``True``.
* **direction** (``'both' | 'up' | 'down'``, default ``'both'``): Polarity of detection.

Returns
^^^^^^^

* **SpikeData**: Detected spike trains per channel in milliseconds.

Raises
^^^^^^

* **ImportError**: If ``h5py`` is unavailable.
* **ValueError**: Propagated from detection if invalid arguments are provided.

Example
^^^^^^^

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

