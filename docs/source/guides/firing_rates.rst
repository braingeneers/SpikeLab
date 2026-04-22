=============================
Firing Rates and Correlations
=============================

This guide covers how to compute instantaneous firing rates from spike trains,
build pairwise firing-rate correlation matrices, and compare correlation
structure across experimental conditions. By the end you will be able to go from
a :class:`~spikelab.SpikeData` object to publication-ready correlation heatmaps
and distribution plots.

All examples assume you already have a :class:`~spikelab.SpikeData` object
loaded. See the :doc:`../getting_started/quickstart` for how to create one.

.. code-block:: python

   from spikelab import SpikeData, RateData


Instantaneous Firing Rates
--------------------------

Mean firing rates (``sd.rates()``) give one number per unit, but many analyses
require a time-resolved estimate. SpikeLab provides two methods for computing
per-unit instantaneous firing rate traces, both returning a
:class:`~spikelab.RateData` object.

ISI interpolation
^^^^^^^^^^^^^^^^^

:meth:`~spikelab.SpikeData.resampled_isi` estimates the instantaneous rate by
interpolating the inverse inter-spike interval at a set of time points and
smoothing with a Gaussian kernel:

.. code-block:: python

   import numpy as np

   # Define the time grid (e.g. every 1 ms across the recording)
   times = np.arange(0, sd.length, 1.0)

   rd = sd.resampled_isi(
       times,
       sigma_ms=10.0,   # Gaussian smoothing kernel width (ms)
   )
   # rd.inst_Frate_data.shape == (sd.N, len(times))

Sliding window
^^^^^^^^^^^^^^

:meth:`~spikelab.SpikeData.sliding_rate` computes rates by counting spikes in
a centered sliding window and dividing by the window width. Both the
sliding-window step and an additional Gaussian smoothing step are optional and
can be used independently or in combination:

.. code-block:: python

   rd = sd.sliding_rate(
       window_size=50,       # sliding window width (ms)
       step_size=1.0,        # advance step (ms)
       gauss_sigma=10.0,     # optional Gaussian smoothing (ms); 0 to disable
       apply_square=True,    # set False to skip the sliding window step
   )
   # rd.inst_Frate_data.shape == (sd.N, T)

You can also specify ``sampling_rate`` instead of ``step_size``
(they are mutually exclusive).

Working with RateData
^^^^^^^^^^^^^^^^^^^^^

Both methods return a :class:`~spikelab.RateData` object that carries the rate
matrix together with its time axis. ``RateData`` supports unit and time
subsetting (``rd.subset()``, ``rd.subtime()``), and provides the correlation
and dimensionality-reduction methods described below.

.. code-block:: python

   from spikelab import RateData

   print(rd.N)                      # number of units
   print(rd.inst_Frate_data.shape)  # (N, T)

   # Plot raster with firing rate heatmap
   fig = sd.plot(
       show_raster=True,
       show_fr_rates=True,
       fr_rates=rd.inst_Frate_data,
   )

.. figure:: /_static/images/raster_fr_heatmap_D0.png
   :width: 100%
   :alt: Spike raster with instantaneous firing rate heatmap

   Spike raster (top) and instantaneous firing rate heatmap (bottom) for an
   MEA recording. Each row in the heatmap is one unit; colour intensity
   reflects firing rate.


Pairwise Firing-Rate Correlations
----------------------------------

A natural next step is to ask how similarly pairs of units modulate their
firing over time. :meth:`~spikelab.RateData.get_pairwise_fr_corr` computes the
cross-correlation between every pair of units' instantaneous rate traces:

.. code-block:: python

   corr_matrix, lag_matrix = rd.get_pairwise_fr_corr(
       max_lag=10,    # maximum lag offset to search (in time bins)
       n_jobs=-1,     # parallel threads (-1 = all cores)
   )

Both return values are :class:`~spikelab.spikedata.pairwise.PairwiseCompMatrix`
objects wrapping an ``(N, N)`` NumPy array:

- ``corr_matrix`` — peak cross-correlation coefficient for each pair. The
  diagonal is always 1 (self-correlation).
- ``lag_matrix`` — lag (in bins) at which the peak correlation occurs. The
  diagonal is always 0 (self-lag).

You can extract the lower triangle for summary statistics:

.. code-block:: python

   # 1-D array of all unique pairwise correlations
   corr_values = corr_matrix.extract_lower_triangle()

   print(f"Median pairwise correlation: {np.median(corr_values):.3f}")


Correlation Matrices
--------------------

Visualise the full ``(N, N)`` correlation matrix as a heatmap using the
built-in :func:`~spikelab.spikedata.plot_utils.plot_heatmap` utility:

.. code-block:: python

   from spikelab.spikedata.plot_utils import plot_heatmap

   fig, ax = plot_heatmap(
       corr_matrix.matrix,
       vmin=-1,
       vmax=1,
       cmap="RdBu_r",
       xlabel="Unit",
       ylabel="Unit",
       cbar_label="Correlation",
   )

The correlation matrix can also be converted to a NetworkX graph for
graph-theoretic analysis:

.. code-block:: python

   G = corr_matrix.to_networkx()
   print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

.. figure:: /_static/images/fr_corr_matrices.png
   :width: 100%
   :alt: Firing-rate correlation matrices

   Pairwise firing-rate correlation matrices for two conditions (left, middle)
   and their difference (right).


Cross-Condition Comparisons
---------------------------

When you have recordings from multiple conditions (e.g. baseline vs.
treatment, or successive days in culture), comparing the distributions of
pairwise correlations reveals changes in network coordination.

First, compute correlations for each condition:

.. code-block:: python

   # Assume sd_baseline and sd_treatment are SpikeData objects
   times_base = np.arange(0, sd_baseline.length, 1.0)
   times_treat = np.arange(0, sd_treatment.length, 1.0)

   rd_base = RateData(sd_baseline.resampled_isi(times_base, sigma_ms=10.0), times_base)
   rd_treat = RateData(sd_treatment.resampled_isi(times_treat, sigma_ms=10.0), times_treat)

   corr_base, _ = rd_base.get_pairwise_fr_corr(max_lag=10)
   corr_treat, _ = rd_treat.get_pairwise_fr_corr(max_lag=10)

Then visualise the distributions side by side using
:func:`~spikelab.spikedata.plot_utils.plot_distribution`:

.. code-block:: python

   from spikelab.spikedata.plot_utils import plot_distribution
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 4))
   plot_distribution(
       ax,
       metric_data={
           "Baseline": corr_base.extract_lower_triangle(),
           "Treatment": corr_treat.extract_lower_triangle(),
       },
       ylabel="FR correlation",
       show_violin=True,
       show_box=True,
   )

.. figure:: /_static/images/fr_corr_violins.png
   :width: 100%
   :alt: Violin plots of pairwise FR correlations across conditions

   Distribution of pairwise firing-rate correlations for the different
   experimental conditions. Shifts in the median or spread indicate changes
   in functional network connectivity.


Further Reading
---------------

- :doc:`spike_analysis` — population rate, burst detection, and per-unit
  spike train metrics.
- :doc:`../getting_started/quickstart` — creating and loading
  :class:`~spikelab.SpikeData` objects.
- The full :doc:`../api/ratedata` API reference documents every method on
  :class:`~spikelab.RateData`.
- The :class:`~spikelab.spikedata.pairwise.PairwiseCompMatrix` API reference
  covers thresholding, masking, and graph conversion.
