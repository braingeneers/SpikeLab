==============
Shuffled Data
==============

SpikeLab provides methods for generating shuffled null distributions from spike
train data using degree-preserving double-edge swaps (Okun et al. 2012). These
are useful for testing whether observed patterns (e.g. correlations, synchrony
metrics) exceed what would be expected by chance.

- Okun, M., Yger, P., Marguet, S. L. et al. Population rate dynamics and
  multineuron firing patterns in sensory cortex. *J Neurosci* 32, 17108--17119
  (2012).

.. contents:: On this page
   :local:
   :depth: 2


Single Shuffle
--------------

:meth:`~spikelab.SpikeData.spike_shuffle` produces a single shuffled copy of
a ``SpikeData`` object using degree-preserving double-edge swaps on the binned
spike matrix (Okun et al. 2012). This preserves each neuron's spike count and
each time bin's population rate while disrupting the precise spike timing:

.. code-block:: python

   sd_shuffled = sd.spike_shuffle(
       swap_per_spike=5,   # number of swap attempts per spike
       seed=42,            # random seed for reproducibility
       bin_size=1,         # raster bin size (ms)
   )

The returned ``SpikeData`` has the same number of units and the same total
spike counts per unit as the original, but the temporal correlations between
units are destroyed.


Shuffle Stacks
--------------

To build a null distribution of shuffled copies, use
:meth:`~spikelab.SpikeData.spike_shuffle_stack`. This calls ``spike_shuffle``
repeatedly and collects the results into a
:class:`~spikelab.spikedata.spikeslicestack.SpikeSliceStack`:

.. code-block:: python

   shuffle_stack = sd.spike_shuffle_stack(
       n_shuffles=100,
       seed=42,             # base seed; incremented per shuffle
       swap_per_spike=5,
   )

   # Each element is an independent shuffled SpikeData
   print(len(shuffle_stack.spike_stack))  # 100

You can then compute a metric on each shuffled copy and compare to the
observed value — for example using
:meth:`~spikelab.spikedata.spikeslicestack.SpikeSliceStack.apply`:

.. code-block:: python

   import numpy as np

   # Compute mean pairwise STTC for each shuffle
   def mean_sttc(sd):
       pcm = sd.spike_time_tilings(delt=20)
       return np.nanmean(pcm.extract_lower_triangle())

   shuffle_values = shuffle_stack.apply(mean_sttc)  # shape (100,)
   observed = mean_sttc(sd)


Unit Subset Stacks
------------------

:meth:`~spikelab.SpikeData.subset_stack` generates random subsets of units,
useful for sub-sampling analyses or bootstrap confidence intervals:

.. code-block:: python

   subset_stack = sd.subset_stack(
       n_subsets=50,
       units_per_subset=10,
       seed=42,
   )

Each element in the stack is a ``SpikeData`` with ``units_per_subset`` randomly
selected units (sampled without replacement within each subset). The same unit
may appear in multiple subsets.


Comparing to the Null Distribution
-----------------------------------

SpikeLab provides two functions for comparing an observed value against a
shuffle distribution:

.. code-block:: python

   from spikelab.spikedata.utils import shuffle_z_score, shuffle_percentile

   # Z-score: (observed - mean) / std of shuffles
   z = shuffle_z_score(observed, shuffle_values)

   # Percentile rank: fraction of shuffle values <= observed
   p = shuffle_percentile(observed, shuffle_values)

Both functions operate element-wise and accept arrays, so you can compare
entire matrices or vectors of observed values against their corresponding
shuffle distributions at once.
