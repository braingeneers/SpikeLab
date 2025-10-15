Standalone Utilities
====================

spike_time_tiling(tA, tB, delt=20.0, length=None)
--------------------------------------------------

Compute the spike time tiling coefficient (STTC) between two spike trains.

**Parameters:**

* ``tA``, ``tB`` (array-like): Sorted spike times for each train.
* ``delt`` (float): Window size (ms).
* ``length`` (float, optional): Total duration (ms).

**Returns:**

* ``float``: STTC value.

----


randomize(ar, swap_per_spike=5)
--------------------------------

Randomize a binary spike raster while preserving row/column sums.

**Parameters:**

* ``ar`` (array-like): Binary matrix (neurons × time or time × neurons).
* ``swap_per_spike`` (int): Target number of successful degree-preserving swaps per spike.

**Returns:**

* ``np.ndarray``: Randomized binary matrix with same shape and marginals.

**Example:**

.. code-block:: python

   import numpy as np
   from spikedata import randomize

   # neurons × time raster
   raster = (np.random.rand(50, 1000) < 0.02).astype(float)
   rnd = randomize(raster, swap_per_spike=5)

   assert np.allclose(raster.sum(axis=0), rnd.sum(axis=0))  # column sums preserved
   assert np.allclose(raster.sum(axis=1), rnd.sum(axis=1))  # row sums preserved

----

get_pop_rate(t_spk_mat, SQUARE_WIDTH, GAUSS_SIGMA)
---------------------------------------------------

Compute population firing rate by smoothing summed spike counts.

**Parameters:**

* ``t_spk_mat`` (array-like): Time-major spike matrix (T × N), values 0/1 or counts.
* ``SQUARE_WIDTH`` (int): Moving-average window width (samples), 0 to disable.
* ``GAUSS_SIGMA`` (float): Gaussian sigma (samples) for additional smoothing, 0 to disable.

**Returns:**

* ``np.ndarray``: Population rate vector of length T.

**Example:**

.. code-block:: python

   import numpy as np
   from spikedata import get_pop_rate

   # Build T × N spike matrix
   T, N = 1000, 64
   t_spk_mat = (np.random.rand(T, N) < 0.01).astype(float)

   pop_rate = get_pop_rate(t_spk_mat, SQUARE_WIDTH=5, GAUSS_SIGMA=2)

----

get_bursts(pop_rate, pop_rate_acc, THR_BURST, MIN_BURST_DIFF, BURST_EDGE_MULT_THRESH)
--------------------------------------------------------------------------------------

Detect bursts from a population rate trace using peak detection and amplitude-scaled edges.

**Parameters:**

* ``pop_rate`` (array-like): Population rate vector (length T).
* ``pop_rate_acc`` (array-like): Optional accumulator with same length T for peak localization; pass an empty list to skip.
* ``THR_BURST`` (float): Multiplier on RMS(pop_rate) for peak height threshold.
* ``MIN_BURST_DIFF`` (int): Minimum distance (samples) between consecutive peaks.
* ``BURST_EDGE_MULT_THRESH`` (float): Edge threshold as a fraction of each burst's peak amplitude.

**Returns:**

* ``(tburst, edges, peak_amp)``: peak times, edge indices per burst, and amplitudes.

**Example:**

.. code-block:: python

   import numpy as np
   from spikedata import get_bursts

   # Suppose pop_rate is computed from get_pop_rate(...)
   pop_rate = np.zeros(500)
   pop_rate[95:106] = np.array([0,2,4,6,8,10,8,6,4,2,0])
   pop_rate[295:306] = np.array([0,3,6,9,12,15,12,9,6,3,0])

   tburst, edges, peak_amp = get_bursts(
       pop_rate,
       pop_rate_acc=[],
       THR_BURST=0.5,
       MIN_BURST_DIFF=10,
       BURST_EDGE_MULT_THRESH=0.2,
   )

   # tburst: indices near burst peaks; edges[i] brackets burst i

