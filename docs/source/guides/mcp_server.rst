==========
MCP Server
==========

SpikeLab includes an MCP (Model Context Protocol) server that exposes analysis
tools for programmatic access by AI agents and other MCP clients. The server
provides a workspace-centric interface to SpikeLab's data loading, analysis,
and export capabilities over the standard MCP stdio transport.


Starting the server
-------------------

Run the MCP server as a Python module:

.. code-block:: bash

   python -m spikelab.mcp_server

This starts the server on stdio transport (the default). MCP clients (such as
Claude Desktop or other MCP-compatible tools) connect by launching this command
as a subprocess and communicating over stdin/stdout.

Example configuration for an MCP client:

.. code-block:: json

   {
     "mcpServers": {
       "spikelab": {
         "command": "python",
         "args": ["-m", "spikelab.mcp_server"]
       }
     }
   }

To use SSE (Server-Sent Events) transport instead, pass ``--transport sse``:

.. code-block:: bash

   python -m spikelab.mcp_server --transport sse --port 8080

SSE transport requires the ``sse`` optional dependency group
(``pip install spikelab[sse]``).

The server requires the ``mcp`` Python package as a dependency.


Tool categories
---------------

The server registers over 90 tools organized into several categories:

Data loading tools
^^^^^^^^^^^^^^^^^^

Load spike train data from various formats into the workspace. Each loader
stores a :class:`~spikelab.spikedata.spike_data.SpikeData` object in the
workspace and returns the workspace ID, namespace, key, and a data summary.

- ``load_from_hdf5_raster`` -- HDF5 raster style
- ``load_from_hdf5_ragged`` -- HDF5 ragged (flat spike_times + index) style
- ``load_from_hdf5_group`` -- HDF5 group-per-unit style
- ``load_from_hdf5_paired`` -- HDF5 paired (idces + times) style
- ``load_from_nwb`` -- Neurodata Without Borders format
- ``load_from_kilosort`` -- KiloSort/Phy output folder
- ``load_from_pickle`` -- Pickle files (local or S3)
- ``load_from_hdf5_thresholded`` -- Raw HDF5 traces with threshold detection
- ``load_from_ibl`` -- International Brain Laboratory public data

Analysis tools
^^^^^^^^^^^^^^

Run analysis operations on data already loaded into the workspace. Results are
stored back into the workspace for subsequent use. Examples include:

- ``compute_rates``, ``compute_binned``, ``compute_raster`` -- basic firing statistics
- ``compute_spike_time_tilings`` -- pairwise spike time tiling coefficients
- ``compute_pairwise_fr_corr``, ``compute_pairwise_ccg`` -- pairwise comparisons
- ``create_rate_slice_stack``, ``create_spike_slice_stack`` -- event-aligned slicing
- ``align_to_events`` -- align data to stimulus or behavioral events
- ``compute_rate_manifold`` -- dimensionality reduction (PCA, UMAP, t-SNE)
- ``fit_gplvm`` -- Gaussian Process Latent Variable Model fitting

Export tools
^^^^^^^^^^^^

Export data from the workspace to files:

- ``export_to_hdf5`` -- HDF5 format (multiple styles)
- ``export_to_nwb`` -- NWB format
- ``export_to_kilosort`` -- KiloSort/Phy folder format
- ``export_to_pickle`` -- Pickle format (local or S3)


Workspace addressing
--------------------

All MCP tools use a workspace-centric addressing model. Data items are
identified by three coordinates:

- **workspace_id** -- identifies the workspace instance. When left empty on a
  loader call, a new workspace is created and its ID is returned.
- **namespace** -- a logical grouping within a workspace, typically one per
  recording or experimental session. When left empty on a loader call, the
  namespace is derived from the input file or folder name. If that namespace
  already exists, a numeric suffix (``_1``, ``_2``, ...) is appended.
- **key** -- identifies a specific data item within a namespace. For example,
  ``SpikeData`` is always stored at key ``"spikedata"``. Analysis results are
  stored at user-specified keys.

A typical workflow looks like this:

1. Call a loader tool (e.g. ``load_from_hdf5_ragged``) with an empty
   ``workspace_id``. The server creates a workspace and returns its ID.
2. Call analysis tools using the returned ``workspace_id`` and ``namespace``,
   specifying output keys for each result.
3. Call further analysis tools that build on previous results by referencing
   their keys.
4. Call export tools to save results to files.

All tools return JSON-serializable dictionaries. Errors are returned as
``{"error": "...", "type": "..."}`` rather than raising exceptions, making the
server safe for automated use.
