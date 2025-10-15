Migration & Usage Tips
======================

Population Firing Rate
----------------------

Use ``SpikeData.binned(bin_size)`` and apply external smoothing:

.. code-block:: python

   bins = sd.binned(10)
   smoothed = np.convolve(bins / 10, np.ones(5), 'same') / 5

Pairwise Correlations
---------------------

Compute on the dense raster output using NumPy/SciPy:

.. code-block:: python

   r = sd.raster(1.0)
   corr = np.corrcoef(r)

Burst/Avalanche/DCC Analysis
-----------------------------

These features are no longer included; use or develop dedicated modules for these analyses.

