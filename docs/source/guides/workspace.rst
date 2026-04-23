======================
Workspace Persistence
======================

This guide covers saving and loading analysis results with the
:class:`~spikelab.AnalysisWorkspace`, so that expensive computations do not
need to be repeated across sessions.

After working through this guide you will know how to:

- Create a workspace and store analysis results in named namespaces.
- Save a workspace to disk and reload it later.
- Browse the contents of a workspace with ``list_namespaces``,
  ``list_keys``, and ``describe``.
- Use :class:`~spikelab.workspace.workspace.LazyAnalysisWorkspace` for
  large datasets that do not fit in RAM.


Creating a Workspace
---------------------

An :class:`~spikelab.AnalysisWorkspace` is a two-level dictionary: items are
addressed by a ``(namespace, key)`` pair.  A typical convention is to use the
experimental condition as the namespace and the analysis result name as the
key.

.. code-block:: python

   from spikelab import AnalysisWorkspace

   ws = AnalysisWorkspace(name="my_experiment")

   # Store a SpikeData object under namespace "D0", key "spikedata"
   ws.store("D0", "spikedata", sd, note="Baseline recording, 120 units")

   # Store a pairwise matrix in the same namespace
   ws.store("D0", "sttc_matrix", sttc, note="STTC, delt=20 ms")

   # Store a numpy array under a shared namespace
   ws.store("all", "burst_times", tburst, note="Burst peak times across all conditions")

The ``note`` parameter is optional free-text metadata that helps you remember
what each item contains.

Supported types include :class:`~spikelab.SpikeData`,
:class:`~spikelab.RateData`, :class:`~spikelab.RateSliceStack`,
:class:`~spikelab.SpikeSliceStack`, :class:`~spikelab.PairwiseCompMatrix`,
:class:`~spikelab.PairwiseCompMatrixStack`, ``numpy.ndarray``, and ``dict``
(with serializable leaf values such as scalars, strings, arrays, or nested
supported types).

Use ``describe`` to get a summary of every item in the workspace:

.. code-block:: python

   summary = ws.describe()

   # summary is a nested dict: {namespace: {key: info_dict}}
   for ns, keys in summary.items():
       for key, info in keys.items():
           print(f"  {ns}/{key}: {info['type']}, {info.get('shape', '')}")


Saving and Loading
-------------------

Call ``save`` to write the workspace to disk.  This produces two files: an
HDF5 file (``*.h5``) containing all data, and a JSON sidecar with workspace
metadata.

.. code-block:: python

   # Save to disk — creates workspace.h5 and workspace.json
   ws.save("results/workspace")

To reload the workspace in a later session:

.. code-block:: python

   ws = AnalysisWorkspace.load("results/workspace")

   # All items are available immediately
   sd = ws.get("D0", "spikedata")
   sttc = ws.get("D0", "sttc_matrix")

If you only need a single item and want to avoid loading the entire
workspace into memory, use :meth:`~spikelab.AnalysisWorkspace.load_item`:

.. code-block:: python

   sttc = AnalysisWorkspace.load_item("results/workspace", "D0", "sttc_matrix")


Listing and Retrieving
-----------------------

Several methods let you browse the workspace contents without loading the
actual data:

.. code-block:: python

   # List all top-level namespaces
   namespaces = ws.list_namespaces()
   print(namespaces)   # ['D0', 'D3', 'D10', 'D30', 'D50', 'all']

   # List keys within a specific namespace
   keys = ws.list_keys("D0")
   print(keys)         # ['spikedata', 'sttc_matrix', 'fr_corr_matrix', ...]

   # List keys across all namespaces
   all_keys = ws.list_keys()   # returns {ns: [keys]} dict
   for ns, ks in all_keys.items():
       print(f"  {ns}: {len(ks)} items")

   # Get summary info for a single item (without loading the object)
   info = ws.get_info("D0", "sttc_matrix")
   print(info)
   # {'type': 'PairwiseCompMatrix', 'shape': (120, 120), 'created_at': ..., 'note': '...'}

To retrieve the actual object:

.. code-block:: python

   obj = ws.get("D0", "sttc_matrix")   # returns the object, or None if not found

Other management operations:

.. code-block:: python

   # Rename a key
   ws.rename("D0", "sttc_matrix", "sttc_delt20")

   # Add or update a note
   ws.add_note("D0", "sttc_delt20", "STTC with delt=20 ms, 120 units")

   # Delete a single item
   ws.delete("D0", "sttc_delt20")

   # Delete an entire namespace
   ws.delete("scratch")


Lazy Loading
-------------

For large datasets where loading everything into RAM is impractical, use
:class:`~spikelab.workspace.workspace.LazyAnalysisWorkspace`.  It has the
same API as the regular workspace, but each ``store`` call immediately writes
the object to a temporary HDF5 file and releases it from memory, and each
``get`` call reads the object back from disk.

.. code-block:: python

   from spikelab.workspace.workspace import LazyAnalysisWorkspace

   ws = LazyAnalysisWorkspace(name="large_experiment")

   # Store writes to disk immediately — the object is not kept in RAM
   ws.store("D0", "spikedata", sd, note="Baseline recording")
   ws.store("D0", "burst_rss", rss, note="Burst-aligned RateSliceStack")

   # get reads from disk on every call
   rss = ws.get("D0", "burst_rss")

   # save copies the backing file to the target path
   ws.save("results/workspace")

   # Load also returns a LazyAnalysisWorkspace when the file is large
   ws = LazyAnalysisWorkspace.load("results/workspace")

All other operations
-- ``list_namespaces``, ``list_keys``, ``describe``, ``get_info`` -- work
from an in-memory index and do not trigger disk reads.

A typical workflow is to use the lazy workspace during long-running compute
scripts (where intermediate results accumulate and would exhaust RAM) and
switch to the regular workspace for interactive exploration:

.. code-block:: python

   # During computation — use lazy to keep RAM under control
   ws = LazyAnalysisWorkspace(name="compute_session")

   for condition in conditions:
       sd = load_my_data(condition)
       rss = sd.align_to_events(tburst, pre_ms=250, post_ms=500, kind="rate")
       corr_stack, _ = rss.get_slice_to_slice_unit_corr_from_stack()

       ws.store(condition, "spikedata", sd)
       ws.store(condition, "burst_rss", rss)
       ws.store(condition, "burst_corr", corr_stack)

   ws.save("results/workspace")

   # Later, for interactive exploration — load everything into RAM
   ws = AnalysisWorkspace.load("results/workspace")
   sd = ws.get("D0", "spikedata")


Merging Workspaces
------------------

When analyses are run separately (e.g. by different scripts or on different
machines), you can combine their results into a single workspace with
:meth:`~spikelab.workspace.workspace.AnalysisWorkspace.merge_from`:

.. code-block:: python

   ws_main = AnalysisWorkspace(name="combined")

   ws_a = AnalysisWorkspace.load("results/analysis_a/workspace")
   ws_b = AnalysisWorkspace.load("results/analysis_b/workspace")

   result = ws_main.merge_from(ws_a)
   print(f"Merged {result['merged']} items from A")

   result = ws_main.merge_from(ws_b)
   print(f"Merged {result['merged']} items from B, skipped {result['skipped']}")

   ws_main.save("results/combined/workspace")

By default, existing keys are kept and incoming duplicates are skipped. Pass
``overwrite=True`` to replace existing items with the incoming values instead.
