.. _hippie_classification:

Cell-Type Classification with HIPPIE
=====================================

SpikeLab has optional integration with `HIPPIE`_, a pretrained multimodal
generative model for neuron classification.  HIPPIE encodes each neuron's
**waveform**, **interspike-interval distribution**, and **autocorrelogram**
into a shared 30-D latent space, then uses UMAP + HDBSCAN for unsupervised
cell-type discovery.

.. _HIPPIE: https://huggingface.co/Jesusgf23/hippie

Installation
------------

HIPPIE is an optional dependency — install it alongside SpikeLab::

    pip install "spikelab[hippie]"

This pulls in PyTorch, HuggingFace Hub, umap-learn, and hdbscan in addition
to the HIPPIE package itself.  PyTorch with CUDA must be installed separately
if GPU inference is desired.

.. note::

   Nothing in the base SpikeLab install is affected.  The HIPPIE adapter is
   never imported unless you explicitly call it.

Data requirements
-----------------

HIPPIE requires three features per neuron:

* **Average waveform** — stored as ``avg_waveform`` in ``neuron_attributes``
* **Spike trains** — always present in a ``SpikeData`` object
* **Recording technology** — passed as ``tech_id`` at call time

The waveform is the only thing that may need preparation.  The three
pipelines below cover the most common starting points.

.. _hippie_pipeline_a:

Pipeline A — Kilosort output + raw ``.bin`` file
-------------------------------------------------

This is the typical Neuropixels + Kilosort4 workflow.  Kilosort gives you
spike times only; waveforms are extracted from the raw voltage trace
afterward.

.. note::

   Attaching raw data to a ``SpikeData`` object is currently a Python-only
   step — there is no MCP tool for it.  Use this path when scripting directly.

.. code-block:: python

    import numpy as np
    from spikelab.data_loaders import load_spikedata_from_kilosort
    from spikelab.spikedata.hippie_adapter import classify_neurons

    # 1. Load spike times from Kilosort output directory
    sd = load_spikedata_from_kilosort(
        folder_path="/path/to/kilosort_output/",
        fs_Hz=30000,                       # Neuropixels default
        cluster_info_tsv="cluster_info.tsv",
        include_noise=False,
    )

    # 2. Attach the raw voltage recording
    #    Shape must be (n_channels, n_samples).
    #    Use np.memmap for large files to avoid loading everything into RAM.
    raw = np.memmap(
        "/path/to/recording.ap.bin",
        dtype="int16",
        mode="r",
        shape=(385, n_samples),            # adjust n_channels and n_samples
    )
    sd.raw_data = raw.astype(np.float32)
    sd.raw_time = 30.0                     # sampling rate in kHz (30 000 Hz)

    # 3. Extract average waveforms for all units in one call.
    #    store=True writes avg_waveform into neuron_attributes automatically.
    sd.get_waveform_traces(
        unit=None,                         # None = all units
        ms_before=1.0,
        ms_after=2.0,
        store=True,
    )

    # 4. Run HIPPIE: embed → UMAP → HDBSCAN
    result = classify_neurons(
        sd,
        tech_id="neuropixels",             # or tech_id=0
        run_umap=True,
        run_hdbscan=True,
        hdbscan_kwargs={"min_cluster_size": 5},
    )

    # 5. Store results back into neuron_attributes
    sd.set_neuron_attribute("hippie_cluster",   result["cluster_labels"])
    sd.set_neuron_attribute("hippie_umap_x",    result["umap_coords"][:, 0])
    sd.set_neuron_attribute("hippie_umap_y",    result["umap_coords"][:, 1])
    sd.set_neuron_attribute("hippie_embedding", result["embeddings"])

    n_clusters = (result["cluster_labels"] >= 0).sum()
    print(f"{sd.N} neurons → {n_clusters} clustered, "
          f"{(result['cluster_labels'] < 0).sum()} noise")

What ``get_waveform_traces`` does in step 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every unit it finds the peak channel (from ``neuron_to_channel_map``),
extracts a short voltage snippet around each spike, averages them, and
stores the average as ``neuron_attributes[i]["avg_waveform"]``.  The adapter
then reads those stored values — no raw data is needed after this point.


.. _hippie_pipeline_b:

Pipeline B — NWB file with raw traces
--------------------------------------

NWB files produced by SpikeInterface, the Allen Brain Atlas pipeline, or
similar tools often embed both spike times and raw traces in a single file.
This is the only path that works end-to-end from the **MCP / agent interface**.

Python
~~~~~~

.. code-block:: python

    from spikelab.data_loaders import load_spikedata_from_nwb
    from spikelab.spikedata.hippie_adapter import classify_neurons

    sd = load_spikedata_from_nwb("/path/to/recording.nwb")

    # Extract waveforms for all units
    sd.get_waveform_traces(unit=None, ms_before=1.0, ms_after=2.0, store=True)

    result = classify_neurons(sd, tech_id="neuropixels")
    sd.set_neuron_attribute("hippie_cluster",   result["cluster_labels"])
    sd.set_neuron_attribute("hippie_umap_x",    result["umap_coords"][:, 0])
    sd.set_neuron_attribute("hippie_umap_y",    result["umap_coords"][:, 1])
    sd.set_neuron_attribute("hippie_embedding", result["embeddings"])

MCP / agent
~~~~~~~~~~~

Give an agent these prompts in order:

.. code-block:: text

    1. "Load the NWB file at /path/to/recording.nwb"

    2. "Extract waveforms for all N units with 1 ms before and 2 ms after the spike"
       (the agent will call get_waveform_traces once per unit)

    3. "Classify the neurons using HIPPIE with tech_id 0 (neuropixels)"

    4. "How many clusters did HIPPIE find?  List cluster IDs and neuron counts."

.. note::

   The MCP ``get_waveform_traces`` tool extracts one unit at a time.
   For a recording with many units the agent needs to call it N times before
   HIPPIE can run.  See :ref:`hippie_mcp_gap` below.


.. _hippie_pipeline_c:

Pipeline C — Waveforms already available
-----------------------------------------

If ``avg_waveform`` is already in ``neuron_attributes`` — e.g. loaded from an
HDF5 workspace, set manually from an upstream pipeline, or computed in a
previous session — skip straight to classification:

.. code-block:: python

    from spikelab.spikedata.hippie_adapter import classify_neurons

    # sd already has avg_waveform in neuron_attributes
    result = classify_neurons(sd, tech_id="neuropixels")

    sd.set_neuron_attribute("hippie_cluster",   result["cluster_labels"])
    sd.set_neuron_attribute("hippie_umap_x",    result["umap_coords"][:, 0])
    sd.set_neuron_attribute("hippie_umap_y",    result["umap_coords"][:, 1])
    sd.set_neuron_attribute("hippie_embedding", result["embeddings"])

To check whether waveforms are already present before trying:

.. code-block:: python

    waves = sd.get_neuron_attribute("avg_waveform")
    if waves is None or any(w is None for w in waves):
        print("Waveforms missing — run get_waveform_traces first")
    else:
        print(f"Waveforms ready for {sd.N} units")

Quick start (waveforms already present)
----------------------------------------

.. code-block:: python

    from spikelab.spikedata.hippie_adapter import classify_neurons

    result = classify_neurons(
        sd,
        tech_id="neuropixels",   # or 0, 1, 2, 3  — see Technology IDs below
        run_umap=True,
        run_hdbscan=True,
    )

    sd.set_neuron_attribute("hippie_cluster",   result["cluster_labels"])
    sd.set_neuron_attribute("hippie_umap_x",    result["umap_coords"][:, 0])
    sd.set_neuron_attribute("hippie_umap_y",    result["umap_coords"][:, 1])
    sd.set_neuron_attribute("hippie_embedding", result["embeddings"])

Return values
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Shape
     - Description
   * - ``embeddings``
     - ``(N, 30)``
     - Latent z_mean vectors from the HIPPIE encoder
   * - ``umap_coords``
     - ``(N, 2)``
     - 2-D UMAP projection (present when ``run_umap=True``)
   * - ``cluster_labels``
     - ``(N,)``
     - HDBSCAN cluster IDs; ``-1`` = noise / unclustered
       (present when ``run_hdbscan=True``)

Technology IDs
~~~~~~~~~~~~~~

The pretrained checkpoint was trained on recordings from four technology
families.  Pass the matching ``tech_id`` for best results:

.. list-table::
   :header-rows: 1
   :widths: 10 25

   * - ``tech_id``
     - Technology
   * - ``0`` / ``"neuropixels"``
     - Neuropixels probes *(default)*
   * - ``1`` / ``"silicon_probe"``
     - Silicon probes (non-Neuropixels)
   * - ``2`` / ``"juxtacellular"``
     - Juxtacellular recordings
   * - ``3`` / ``"tetrodes"``
     - Tetrode recordings

Advanced options
----------------

Tuning UMAP and HDBSCAN
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result = classify_neurons(
        sd,
        tech_id=0,
        umap_kwargs={"n_neighbors": 15, "min_dist": 0.05},
        hdbscan_kwargs={"min_cluster_size": 10, "min_samples": 5},
    )

Embeddings only (no clustering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful when you want to inspect the latent space before deciding on
clustering parameters:

.. code-block:: python

    result = classify_neurons(sd, run_umap=False, run_hdbscan=False)
    embeddings = result["embeddings"]   # (N, 30)

Using the HIPPIE API directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full control over preprocessing or batching, import
:class:`hippie.inference.HIPPIEClassifier` directly:

.. code-block:: python

    from hippie import HIPPIEClassifier

    clf = HIPPIEClassifier.from_pretrained("Jesusgf23/hippie", device="cpu")

    # Inputs must be preprocessed — see hippie_adapter.extract_features()
    # for the exact normalization applied to each modality.
    embeddings = clf.get_embeddings(wave, isi, acg, tech_id=0)
    coords      = clf.umap_reduce(embeddings, n_neighbors=30)
    labels      = clf.hdbscan_cluster(coords, min_cluster_size=5)

    # Load from a local checkpoint instead of HuggingFace
    clf2 = HIPPIEClassifier.from_checkpoint("./my_trained_model.ckpt")

Using via the MCP server
------------------------

The ``classify_neurons_hippie`` tool is available in the SpikeLab MCP
server once ``spikelab[hippie]`` is installed.  After the tool runs, it
writes ``hippie_embedding``, ``hippie_umap_x``, ``hippie_umap_y``, and
``hippie_cluster`` directly into ``neuron_attributes``, making them
accessible to all downstream tools.

Example agent prompts
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    "Classify the neurons in this recording using HIPPIE."

    "Run HIPPIE cell-type classification with tech_id 1 (silicon probe)."

    "Embed the neurons with HIPPIE and cluster with HDBSCAN, minimum cluster size 10."

    "What cell types did HIPPIE find? List the cluster IDs and neuron counts."

    "Plot the HIPPIE UMAP coloured by cluster label."

.. _hippie_mcp_gap:

Known limitation: MCP waveform extraction is per-unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current ``get_waveform_traces`` MCP tool extracts waveforms for a
**single unit** per call.  For a recording with N neurons, an agent must
call it N times before ``classify_neurons_hippie`` can run.

Workaround until a bulk ``extract_all_waveforms`` tool is added:

* Use Pipeline A or C in Python, where ``get_waveform_traces(unit=None)``
  processes all units in one call.
* Or pre-compute waveforms in Python and save the workspace; the agent can
  then load it and run ``classify_neurons_hippie`` directly.

How it works
------------

1. **Feature extraction** — For each neuron, SpikeLab computes:

   * *Waveform* (50 samples, min-max normalized to [-1, 1])
   * *ISI histogram* (100 log-spaced bins from 1 ms to 5 s, log(x+1)
     transformed, then min-max normalized)
   * *Autocorrelogram* (100 bins, 0–100 ms, min-max normalized)

2. **Encoding** — Three modality-specific ResNet18 encoders project each
   neuron's features into a shared 30-D latent space, conditioned on the
   recording technology (``tech_id``).

3. **UMAP** — The 30-D embeddings are projected to 2-D using cosine-distance
   UMAP for visualization and clustering.

4. **HDBSCAN** — Density clusters are found in the 2-D UMAP space.
   Neurons that do not belong to any cluster receive label ``-1``.

Checkpoint
----------

The pretrained model (``hippie_techcond_v1.ckpt``) is hosted at
`huggingface.co/Jesusgf23/hippie`_.  It is downloaded automatically on
first use and cached locally (HuggingFace default cache, or override with
``cache_dir``).  The file is 293 MB; subsequent calls use the local cache.

.. _huggingface.co/Jesusgf23/hippie: https://huggingface.co/Jesusgf23/hippie

The model was pretrained on 11 labeled electrophysiology datasets spanning
mouse, rat, and macaque across multiple brain regions and recording
technologies.
