==============
Spike Analysis
==============

This guide covers the core spike train analysis tools in SpikeLab: computing
the population firing rate, detecting network bursts, extracting per-unit
metrics, and running parameter sensitivity sweeps. By the end you will know how
to characterise the temporal structure of an MEA recording from raw spike data.

All examples assume you already have a :class:`~spikelab.SpikeData` object
loaded. See the :doc:`../getting_started/quickstart` for how to create one.

.. code-block:: python

   from spikelab import SpikeData

   # sd is your loaded SpikeData (e.g. from pickle, HDF5, or NWB)


Population Rate
---------------

The population rate is the smoothed aggregate firing rate across all units.
SpikeLab computes it in two stages: an optional square (boxcar) convolution
followed by an optional Gaussian convolution, both applied to the binned
population spike count.

.. code-block:: python

   pop_rate = sd.get_pop_rate(
       square_width=20,         # full width of the boxcar kernel (ms)
       gauss_sigma=100,         # standard deviation of the Gaussian kernel (ms)
       raster_bin_size_ms=1.0,  # bin size for binning spike times to the raster (ms)
   )

The returned array has one value per millisecond-wide bin and represents the
smoothed population firing rate over time.

Adjusting ``square_width`` and ``gauss_sigma`` controls the trade-off between
temporal resolution and smoothness. A smaller ``gauss_sigma`` preserves fast
transients; a larger value reveals slow trends. Setting either to ``0`` skips
that smoothing step entirely. The ``raster_bin_size_ms`` parameter sets the bin
size used for binning spike times to the internal raster.


Burst Detection
---------------

Network bursts are periods of elevated population activity that stand out above
baseline. :meth:`~spikelab.SpikeData.get_bursts` detects them by thresholding
the smoothed population rate:

.. code-block:: python

   tburst, edges, peak_amp = sd.get_bursts(
       thr_burst=2.5,                # RMS multiplier for peak detection
       min_burst_diff=1000,           # minimum distance between burst peaks (bins)
       burst_edge_mult_thresh=0.2,    # fraction of peak height for edge detection
       square_width=20,               # pop rate smoothing: boxcar full width (ms)
       gauss_sigma=100,               # pop rate smoothing: Gaussian sigma (ms)
       raster_bin_size_ms=1.0,        # raster bin size (ms)
   )

The return values are:

- ``tburst`` — 1-D array of burst peak times (in bin indices).
- ``edges`` — ``(B, 2)`` array where each row is ``[start_bin, end_bin]`` for
  one burst.
- ``peak_amp`` — 1-D array of peak amplitudes.

The three main parameters control detection sensitivity:

- ``thr_burst`` sets how many times the baseline RMS the population rate must
  exceed at a burst peak. Lower values detect weaker bursts; higher values
  are more conservative.
- ``min_burst_diff`` sets the minimum separation between consecutive burst
  peaks. This prevents a single long burst from being split into many short
  ones.
- ``burst_edge_mult_thresh`` sets the fraction of the burst peak height at
  which the burst edges are drawn. In peak-to-trough mode (default), the
  height is measured relative to the neighbouring interburst trough; in
  peak-to-zero mode, it is measured relative to zero. A lower value pushes
  edges closer to baseline.

Set ``peak_to_trough=True`` (the default) for trough-to-trough baseline
subtraction, or ``False`` for a zero baseline.

.. figure:: /_static/images/raster_poprate_D0.png
   :width: 100%
   :alt: Spike raster with population rate and burst edges

   Spike raster (top) and population rate (bottom) for an MEA recording.
   Detected burst boundaries overlay the population rate trace.


Burst Sensitivity Analysis
--------------------------

Choosing burst detection parameters requires understanding how the number of
detected bursts changes with threshold and distance settings. The
:meth:`~spikelab.SpikeData.burst_sensitivity` method sweeps a grid of
``thr_burst`` and ``min_burst_diff`` values and returns the burst count at each
combination:

.. code-block:: python

   import numpy as np

   thr_values = np.arange(1.0, 5.5, 0.5)     # RMS multipliers to test
   dist_values = np.arange(200, 2200, 200)    # minimum peak distances to test

   counts = sd.burst_sensitivity(
       thr_values=thr_values,
       dist_values=dist_values,
       burst_edge_mult_thresh=0.2,
       square_width=20,
       gauss_sigma=100,
   )
   # counts.shape == (len(thr_values), len(dist_values))

The result is a 2-D matrix that you can visualise as a heatmap using the
built-in plotting utility:

.. code-block:: python

   from spikelab.spikedata.plot_utils import plot_burst_sensitivity

   fig, ax = plot_burst_sensitivity(
       ax=None,
       thresholds=thr_values,
       burst_counts=counts,
       dist_values=dist_values,
       xlabel="Threshold (RMS mult.)",
       ylabel="Min burst distance (bins)",
   )

.. figure:: /_static/images/burst_sensitivity.png
   :width: 100%
   :alt: Burst sensitivity heatmap

   Heatmap of detected burst counts as a function of threshold multiplier
   (x-axis) and minimum burst distance (y-axis). Warm colours indicate more
   bursts detected.

Inspecting this heatmap helps you identify a plateau region where the burst
count is stable — a sign that your chosen parameters are robust.

To test a single parameter in isolation, pass a length-1 list for the other.
The output is then effectively 1-D, and ``plot_burst_sensitivity`` renders it
as a line plot instead of a heatmap. To compare multiple recordings, pass
``burst_counts`` as a dict mapping condition labels to arrays — each condition
is drawn as a separate line on the same axes:

.. code-block:: python

   # Sweep thresholds with a fixed min_burst_diff
   counts_d0 = sd_d0.burst_sensitivity(
       thr_values=thr_values,
       dist_values=[1000],
       burst_edge_mult_thresh=0.2,
   )
   counts_d50 = sd_d50.burst_sensitivity(
       thr_values=thr_values,
       dist_values=[1000],
       burst_edge_mult_thresh=0.2,
   )

   plot_burst_sensitivity(
       ax=None,
       thresholds=thr_values,
       burst_counts={"D0": counts_d0.ravel(), "D50": counts_d50.ravel()},
   )


Per-Unit Metrics
----------------

Beyond population-level summaries, SpikeLab provides several per-unit metrics
that characterise individual neurons within the network.

Firing rates
^^^^^^^^^^^^

.. code-block:: python

   # Mean firing rate per unit
   rates_hz = sd.rates(unit="Hz")     # shape (N,)
   rates_khz = sd.rates(unit="kHz")   # default unit

The returned array has one entry per unit. Use the ``unit`` parameter to choose
between Hz and kHz.

Inter-spike intervals
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # ISI arrays per unit (list of N arrays, each in ms)
   isis = sd.interspike_intervals()

Each element is the ``np.diff`` of that unit's spike train — useful for ISI
histograms, coefficient of variation, or burst index calculations.

Spike-triggered population rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The spike-triggered population rate (stPR) measures how much the overall
network rate deviates whenever a given neuron fires. It reveals functional
coupling between individual units and the population:

.. code-block:: python

   stpr, coupling_zero, coupling_max, delays, lags = sd.compute_spike_trig_pop_rate(
       window_ms=80,       # lag window: [-80, +80] ms
       cutoff_hz=20,       # low-pass filter cutoff (Hz)
       fs=1000,            # sampling rate for the low-pass filter
       bin_size=1,         # raster bin size (ms)
       cut_outer=10,       # trim edge artefacts (ms)
   )

- ``stpr`` — ``(N, 2*window_ms+1)`` matrix of filtered stPR traces.
- ``coupling_zero`` — ``(N,)`` coupling strength at zero lag.
- ``coupling_max`` — ``(N,)`` maximum coupling strength across all lags.
- ``delays`` — ``(N,)`` lag at which each unit's coupling peaks.
- ``lags`` — ``(2*window_ms+1,)`` lag axis in ms.

.. figure:: /_static/images/firing_rate_violins.png
   :width: 100%
   :alt: Firing rate distributions across conditions

   Violin plots of per-unit firing rates compared across experimental
   conditions.


Burst Widths
-------------

Once bursts are detected, you can examine their temporal extent by computing the
width of each burst from the ``edges`` array:

.. code-block:: python

   tburst, edges, peak_amp = sd.get_bursts(
       thr_burst=2.5,
       min_burst_diff=1000,
       burst_edge_mult_thresh=0.2,
   )

   # Burst widths in ms (when raster_bin_size_ms=1.0)
   burst_widths = edges[:, 1] - edges[:, 0]

.. figure:: /_static/images/burst_widths.png
   :width: 100%
   :alt: Distribution of burst widths

   Distribution of burst widths across detected bursts. Wider bursts
   correspond to sustained periods of elevated network activity.


Further Reading
---------------

- :doc:`firing_rates` — instantaneous firing rates, pairwise correlations,
  and the :class:`~spikelab.RateData` class.
- :doc:`../getting_started/quickstart` — creating and loading
  :class:`~spikelab.SpikeData` objects.
- The full :doc:`../api/spikedata` API reference documents every method on
  :class:`~spikelab.SpikeData`.
