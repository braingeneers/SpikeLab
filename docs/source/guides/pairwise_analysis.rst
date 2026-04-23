=================
Pairwise Analysis
=================

This guide covers computing pairwise similarity between neurons, visualizing
the relationship between firing-rate correlation and spike timing, and
building spatial network representations of functional connectivity on the
MEA layout.

After working through this guide you will know how to:

- Compute spike time tiling coefficient (STTC) matrices with
  :meth:`~spikelab.SpikeData.spike_time_tilings`.
- Compare firing-rate correlation and STTC with scatter plots using marginal
  histograms.
- Convert a :class:`~spikelab.PairwiseCompMatrix` to a NetworkX graph for
  graph analyses.
- Plot functional connectivity as a spatial network on MEA electrode
  positions.


Spike Time Tiling Coefficient (STTC)
-------------------------------------

The STTC (Cutts & Eglen 2014) quantifies the temporal co-occurrence of spikes
between two neurons. It ranges from -1 (anti-correlated) to +1 (perfectly
coincident) and is unbiased with respect to firing rate, making it preferable
to raw cross-correlation for comparing pairs with different activity levels.

- Cutts, C. S. & Eglen, S. J. Detecting pairwise correlations in spike trains:
  an objective comparison of methods and application to the study of retinal
  waves. *J Neurosci* 34, 14288--14303 (2014).

The ``delt`` parameter sets the coincidence window in milliseconds.  A spike
in neuron A is considered coincident with neuron B if any spike of B falls
within ``delt`` ms.  Typical values range from 5 to 50 ms; 20 ms is a common
default for cortical data.

.. code-block:: python

   from spikelab import SpikeData, PairwiseCompMatrix

   # sd is a SpikeData with N units
   # delt: coincidence window in ms (default 20 ms)
   sttc_matrix = sd.spike_time_tilings(delt=20.0)

   # sttc_matrix is a PairwiseCompMatrix with shape (N, N)
   print(sttc_matrix.matrix.shape)   # (N, N)
   print(sttc_matrix.metadata)       # {'delt': 20.0}

The returned :class:`~spikelab.PairwiseCompMatrix` stores the symmetric
N-by-N matrix together with optional labels and metadata.  When ``numba`` is
installed and the number of units exceeds two, the computation is
automatically parallelized across all unit pairs.

To compare multiple conditions, compute one STTC matrix per recording and
visualize them side by side:

.. code-block:: python

   import matplotlib.pyplot as plt
   from spikelab.spikedata.plot_utils import plot_heatmap

   fig, axes = plt.subplots(1, 2, figsize=(10, 4))

   plot_heatmap(
       sttc_d0.matrix,
       ax=axes[0],
       cmap="RdBu_r",
       vmin=-1, vmax=1,
       xlabel="Unit", ylabel="Unit",
       colorbar_label="STTC",
   )
   axes[0].set_title("Baseline")

   plot_heatmap(
       sttc_d50.matrix,
       ax=axes[1],
       cmap="RdBu_r",
       vmin=-1, vmax=1,
       xlabel="Unit", ylabel="Unit",
       colorbar_label="STTC",
   )
   axes[1].set_title("Treated")

.. figure:: /_static/images/sttc_matrices.png
   :width: 100%
   :alt: STTC matrices for baseline and treated conditions

   Pairwise STTC matrices for two conditions (left, middle)
   and their difference (right).


Firing-Rate Correlations
-------------------------

Firing-rate correlation measures how similarly pairs of units modulate their
activity over time. See :doc:`firing_rates` for how to compute the
instantaneous rate traces.

:meth:`~spikelab.RateData.get_pairwise_fr_corr` computes the peak
cross-correlation between every pair of units' rate traces:

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

Visualise the full ``(N, N)`` correlation matrix as a heatmap:

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

.. figure:: /_static/images/fr_corr_matrices.png
   :width: 100%
   :alt: Firing-rate correlation matrices

   Pairwise firing-rate correlation matrices for two conditions (left, middle)
   and their difference (right).

To compare distributions across conditions:

.. code-block:: python

   from spikelab.spikedata.plot_utils import plot_distribution
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 4))
   plot_distribution(
       ax,
       metric_data={
           "D0": corr_d0.extract_lower_triangle(),
           "D3": corr_d3.extract_lower_triangle(),
           "D10": corr_d10.extract_lower_triangle(),
           "D30": corr_d30.extract_lower_triangle(),
           "D50": corr_d50.extract_lower_triangle(),
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


FR Correlation vs STTC Scatter
-------------------------------

Firing-rate correlation and STTC capture different aspects of functional
connectivity. You can plot one against the other to see how they compare.

Use :func:`~spikelab.spikedata.plot_utils.plot_scatter_with_marginals` to
create a scatter plot with marginal histograms.  The function accepts a
``GridSpec`` slot and a parent figure, so you can embed multiple panels in a
single layout:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.gridspec as gridspec
   from spikelab.spikedata.plot_utils import plot_scatter_with_marginals

   # Extract upper-triangle values from each matrix
   mask = np.triu(np.ones(sttc_matrix.matrix.shape, dtype=bool), k=1)
   x = fr_corr_matrix.matrix[mask]   # firing-rate correlation per pair
   y = sttc_matrix.matrix[mask]      # STTC per pair

   # Remove NaN pairs
   valid = np.isfinite(x) & np.isfinite(y)
   x, y = x[valid], y[valid]

   fig = plt.figure(figsize=(5, 5))
   gs = gridspec.GridSpec(1, 1, figure=fig)

   ax_scatter, ax_histx, ax_histy, sc = plot_scatter_with_marginals(
       gs[0],                     # GridSpec slot for the sub-layout
       fig,
       x, y,
       xlabel="FR correlation",
       ylabel="STTC",
       color_vals="density",      # KDE-based density coloring
       cmap="viridis",
       marker_size=0.5,
       alpha=1.0,
       show_identity=True,        # draw x=y reference line
       show_colorbar=False,
   )

.. figure:: /_static/images/fr_vs_sttc_scatter.png
   :width: 100%
   :alt: Scatter plot of firing-rate correlation vs STTC with marginal histograms

   Each point is one unit pair.  Density coloring highlights the core of the
   distribution.  The identity line shows where the two metrics would agree
   perfectly.


Network Analysis
-----------------

A :class:`~spikelab.PairwiseCompMatrix` can be converted to a weighted
NetworkX graph for standard graph-theoretic analyses such as clustering
coefficient, shortest path length, and community detection.

.. code-block:: python

   import networkx as nx

   # Convert to a NetworkX graph
   # threshold: only include edges with |weight| > threshold
   # invert_weights: if True, weight = 1 - value (useful for path-length metrics)
   G = sttc_matrix.to_networkx(threshold=0.0, invert_weights=False)

   print(f"Nodes: {G.number_of_nodes()}")
   print(f"Edges: {G.number_of_edges()}")

For path-based metrics like global efficiency, use ``invert_weights=True`` so
that strongly correlated pairs correspond to short paths:

.. code-block:: python

   G_inv = sttc_matrix.to_networkx(threshold=0.0, invert_weights=True)
   efficiency = nx.global_efficiency(G_inv)
   print(f"Global efficiency: {efficiency:.3f}")

Community detection with the Louvain algorithm identifies groups of neurons
that are more densely connected to each other than to the rest of the
network.  This requires the ``python-louvain`` package (``pip install
python-louvain``):

.. code-block:: python

   from community import community_louvain

   # Build a graph with only positive edges (raw weights)
   G_raw = sttc_matrix.to_networkx(threshold=0.0, invert_weights=False)

   # Run Louvain community detection
   # resolution: higher values produce more communities
   partition = community_louvain.best_partition(G_raw, weight="weight", resolution=1.0)
   modularity = community_louvain.modularity(partition, G_raw, weight="weight")

   print(f"Modularity: {modularity:.3f}")
   print(f"Number of communities: {len(set(partition.values()))}")

   # partition is a dict mapping node index -> community label
   community_labels = [partition[i] for i in range(G_raw.number_of_nodes())]

You can also compute per-node metrics such as weighted degree (strength),
clustering coefficient, and betweenness centrality:

.. code-block:: python

   # Node strength (sum of edge weights per node)
   strengths = np.array([d for _, d in G_raw.degree(weight="weight")])

   # Weighted clustering coefficient
   clustering = nx.clustering(G_raw, weight="weight")


Spatial Network Visualization
------------------------------

When electrode positions are available in ``neuron_attributes``, you can
overlay the pairwise connectivity on the physical MEA layout.  The
:meth:`~spikelab.SpikeData.plot_spatial_network` method plots each unit at
its (x, y) position with node size proportional to its mean pairwise metric
and edges drawn between pairs that exceed a threshold.

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 6))

   # sd must have neuron_attributes with 'x' and 'y' keys (positions in um)
   sc = sd.plot_spatial_network(
       ax,
       sttc_matrix.matrix,          # (N, N) pairwise metric
       edge_threshold=0.3,          # draw edges where STTC > 0.3
       node_size_range=(2, 20),     # min/max marker size
       node_cmap="viridis",         # colormap for node color (by row-mean)
       edge_color="red",            # color of connecting lines
       edge_linewidth=0.6,
       edge_alpha_range=(0.15, 1.0),  # alpha scales with edge strength
       scale_bar_um=500,            # scale bar length in micrometers
   )

   fig.colorbar(sc, ax=ax, label="Mean STTC")

Alternatively, if you have positions as a separate array, call the
:meth:`~spikelab.PairwiseCompMatrix.plot_spatial_network` method on the
matrix directly:

.. code-block:: python

   positions = sd.unit_locations   # (N, 2) array of (x, y) in um

   sc = sttc_matrix.plot_spatial_network(
       ax,
       positions,
       edge_threshold=0.3,
       node_cmap="viridis",
       edge_color="red",
       scale_bar_um=500,
   )

You can also use the ``top_pct`` parameter instead of ``edge_threshold`` to
draw only the top percentile of edges, which is useful when the absolute
scale varies across conditions:

.. code-block:: python

   sc = sd.plot_spatial_network(
       ax,
       sttc_matrix.matrix,
       top_pct=10,    # draw only the top 10% of edges
   )

.. figure:: /_static/images/mea_network_D0.png
   :width: 100%
   :alt: Spatial network of functional connectivity on the MEA layout

   Units plotted at their MEA positions.  Node size and color reflect mean
   pairwise connectivity; red lines connect pairs exceeding the threshold,
   with line opacity proportional to edge strength.
