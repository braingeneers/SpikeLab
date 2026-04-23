==============
GPLVM Analysis
==============

SpikeLab can fit a Gaussian Process Latent Variable Model (GPLVM) to spike
train data, uncovering discrete latent states that describe the network's
activity over time (Zheng et al. 2025). This is useful for identifying
recurring population dynamics such as UP/DOWN states or multi-state
transitions.

- Zheng, Z., Zutshi, I., Huszar, R. et al. From labels to latents:
  revealing state-dependent hippocampal computations with Jump Latent Variable
  Model. *bioRxiv* (2025).

.. note::

   GPLVM fitting requires the optional ``poor_man_gplvm`` and ``jax``
   packages. Install them separately before using these features.

.. contents:: On this page
   :local:
   :depth: 2


Fitting a GPLVM
----------------

:meth:`~spikelab.SpikeData.fit_gplvm` bins the spike trains and fits a GPLVM
via expectation-maximisation:

.. code-block:: python

   result = sd.fit_gplvm(
       bin_size_ms=50.0,           # temporal bin size for spike counts
       n_latent_bin=100,           # number of discrete latent states
       n_iter=20,                  # EM iterations
       movement_variance=1.0,     # state transition variance
       tuning_lengthscale=10.0,   # GP tuning curve lengthscale
       random_seed=3,
   )

The returned dict contains:

- ``decode_res`` — the full decode result from the GPLVM model, including
  posterior marginals over latent states and dynamics.
- ``reorder_indices`` — unit ordering derived from the model's tuning curves.
- ``model`` — the fitted model object.
- ``binned_spike_counts`` — the ``(T, N)`` binned spike matrix used for fitting.
- ``bin_size_ms`` — the bin size used.
- ``log_marginal_l`` — log marginal likelihood per EM iteration.

All arrays are returned as NumPy ndarrays (not JAX arrays).


Post-fit Analysis
-----------------

SpikeLab provides utility functions that extract summary statistics from the
GPLVM decode result.

State entropy
^^^^^^^^^^^^^

Compute the Shannon entropy of the latent state posterior at each time bin.
Higher entropy indicates greater uncertainty about the current state:

.. code-block:: python

   from spikelab.spikedata.utils import gplvm_state_entropy

   entropy = gplvm_state_entropy(
       result["decode_res"].posterior_latent_marg,
   )
   # entropy.shape == (T,)

Continuity probability
^^^^^^^^^^^^^^^^^^^^^^

Extract the probability that the network remains in the same state from one
time bin to the next (i.e. no state transition):

.. code-block:: python

   from spikelab.spikedata.utils import gplvm_continuity_prob

   cont_prob = gplvm_continuity_prob(result["decode_res"])
   # cont_prob.shape == (T,)

Average state probability
^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the mean probability of each latent state across all time bins. This
reveals which states dominate the recording:

.. code-block:: python

   from spikelab.spikedata.utils import gplvm_average_state_probability

   avg_prob = gplvm_average_state_probability(
       result["decode_res"].posterior_latent_marg,
   )
   # avg_prob.shape == (K,), where K = n_latent_bin

Consecutive state durations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measure how long the network stays in a given condition. For example, compute
the durations of high-continuity epochs:

.. code-block:: python

   from spikelab.spikedata.utils import consecutive_durations

   durations = consecutive_durations(
       cont_prob,
       threshold=0.8,
       mode="above",     # runs where cont_prob >= 0.8
       min_dur=1,         # minimum run length to include
   )
   # durations is a 1-D array of run lengths (in time bins)


Visualisation
-------------

The GPLVM results integrate with SpikeLab's plotting utilities. Use
:meth:`~spikelab.SpikeData.plot` with ``show_model_states=True`` to overlay
the decoded states on a raster plot:

.. code-block:: python

   fig = sd.plot(
       show_raster=True,
       show_pop_rate=True,
       show_model_states=True,
       gplvm_result=result,
   )

For dimensionality reduction on the latent posteriors, use
:func:`~spikelab.spikedata.utils.PCA_reduction` or
:func:`~spikelab.spikedata.utils.UMAP_reduction` on the posterior marginal
matrix, and visualise with :func:`~spikelab.spikedata.plot_utils.plot_manifold`.
