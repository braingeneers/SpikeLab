==========================
Instantaneous Firing Rates
==========================

This guide covers how to compute instantaneous firing rates from spike trains,
project them into low-dimensional manifolds, and work with the
:class:`~spikelab.RateData` class. For pairwise firing-rate correlations, see
the :doc:`pairwise_analysis` guide.

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


Dimensionality Reduction
------------------------

:meth:`~spikelab.RateData.get_manifold` projects the firing-rate matrix into a
low-dimensional space using PCA or UMAP. This is useful for visualising
population dynamics over time or comparing activity structure across
conditions.

.. code-block:: python

   # PCA — returns (embedding, explained_variance_ratio, components)
   embedding, var_ratio, components = rd.get_manifold(
       method="PCA",
       n_components=3,
   )
   # embedding.shape == (T, 3) — one point per time bin

   # UMAP — returns (embedding, trustworthiness)
   embedding, trust = rd.get_manifold(
       method="UMAP",
       n_components=2,
   )

Visualise the embedding with
:func:`~spikelab.spikedata.plot_utils.plot_manifold`:

.. code-block:: python

   from spikelab.spikedata.plot_utils import plot_manifold

   fig, ax = plt.subplots(figsize=(5, 5))
   plot_manifold(
       ax,
       embedding,
       var_explained=var_ratio,
       groups=condition_labels,       # integer array (T,)
       group_labels=["D0", "D3", "D10", "D30", "D50"],
       marker_size=1,
       alpha=0.3,
   )

.. figure:: /_static/images/rate_pca_combined.png
   :width: 100%
   :alt: PCA embedding of firing rate dynamics

   PCA embedding of instantaneous firing rate dynamics across conditions.
   Each point is one time bin; colours indicate experimental conditions.


Pairwise Firing-Rate Correlations
----------------------------------

Once you have a ``RateData`` object, you can compute pairwise correlations
between all unit pairs using
:meth:`~spikelab.RateData.get_pairwise_fr_corr`. See the
:doc:`pairwise_analysis` guide for full details, code examples, and
visualisation of correlation matrices across conditions.


Further Reading
---------------

- :doc:`spike_analysis` — population rate, burst detection, and per-unit
  spike train metrics.
- :doc:`pairwise_analysis` — pairwise firing-rate correlations, STTC,
  network analysis, and spatial network visualisation.
- :doc:`../getting_started/quickstart` — creating and loading
  :class:`~spikelab.SpikeData` objects.
- The full :doc:`../api/ratedata` API reference documents every method on
  :class:`~spikelab.RateData`.
- The :class:`~spikelab.spikedata.pairwise.PairwiseCompMatrix` API reference
  covers thresholding, masking, and graph conversion.
