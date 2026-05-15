===================
Parallel Computing
===================

Several SpikeLab methods support parallel computation to speed up
analyses on large recordings.

.. contents:: On this page
   :local:
   :depth: 2


Thread-Pool Parallelism
-----------------------

Methods that compute pairwise metrics across all unit pairs accept an
``n_jobs`` parameter that controls thread-pool parallelism:

- :meth:`~spikelab.SpikeData.get_pairwise_ccg` — pairwise cross-correlograms
- :meth:`~spikelab.RateData.get_pairwise_fr_corr` — pairwise firing-rate
  correlations

Pass ``n_jobs=-1`` (the default) to use all available cores, ``n_jobs=1`` for
serial execution, or a specific integer to limit the number of threads:

.. code-block:: python

   corr, lag = rd.get_pairwise_fr_corr(max_lag=10, n_jobs=-1)


Numba Acceleration
------------------

When the optional ``numba`` package is installed, SpikeLab automatically uses
JIT-compiled parallel kernels for the most computationally expensive
operations:

- **STTC** — :meth:`~spikelab.SpikeData.spike_time_tilings` uses
  ``numba.prange`` to parallelize across all unit pairs
- **Pairwise latencies** — :meth:`~spikelab.SpikeData.get_pairwise_latencies`
  uses parallel nearest-spike search
- **Spike-triggered population rate** —
  :meth:`~spikelab.SpikeData.compute_spike_trig_pop_rate` uses parallel
  per-unit computation

No code changes are needed — SpikeLab detects numba at import time and
switches to the accelerated kernels automatically. Install with:

.. code-block:: bash

   pip install spikelab[numba]
