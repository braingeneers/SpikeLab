# HIPPIE Agent Prompts for SpikeLab

Personal reference for running HIPPIE cell-type classification and VAE compression
via an AI agent (e.g., Claude Code) that has access to the SpikeLab MCP server.

---

## 1. Pretrained HIPPIE Classifier

### Pipeline A — Kilosort output + raw .bin file

```
I have Kilosort spike-sorting results at /path/to/kilosort_output/ and the
raw binary recording at /path/to/recording.bin with 384 channels sampled at
30,000 Hz.  The electrode geometry is neuropixels.

Using the SpikeLab MCP server:
1. Load the Kilosort output into a new workspace called "my_session".
2. Extract average waveforms from the raw binary for every unit
   (window: 1 ms before to 2 ms after each spike, max 500 spikes per unit).
3. Classify neurons with the pretrained HIPPIE model
   (tech_id="neuropixels", run_umap=True, run_hdbscan=True, min_cluster_size=5).
4. Return the cluster label counts and save embeddings + UMAP coordinates
   back into the workspace under the "hippie" namespace.
```

### Pipeline B — NWB file

```
I have an NWB file at /path/to/session.nwb that contains spike times and
pre-sorted units with waveforms stored under processing/ecephys.

Using the SpikeLab MCP server:
1. Load the NWB file into a workspace called "nwb_session".
2. Classify neurons with the pretrained HIPPIE model
   (tech_id="neuropixels", run_umap=True, run_hdbscan=True).
3. Add hippie_cluster, hippie_embedding, hippie_umap_x, hippie_umap_y as
   neuron attributes back to the workspace.
4. Show me the cluster label distribution.
```

### Pipeline C — Pre-computed waveforms already in SpikeData

```
I already have a SpikeData object saved at /path/to/session.pkl that includes
avg_waveform in neuron_attributes.

Using the SpikeLab MCP server:
1. Load the pickle into workspace "precomputed".
2. Run classify_neurons_hippie with tech_id="neuropixels" and default UMAP/HDBSCAN settings.
3. Write results back to the workspace and show me the number of neurons per cluster.
```

---

## 2. Unsupervised VAE (train on your own data, no conditioning)

### Train a new VAE on your recordings

```
I have a SpikeData object at workspace "my_session" namespace "sorted" with
avg_waveform in neuron_attributes.

Using the SpikeLab MCP server:
1. Train an unsupervised multimodal VAE on the neurons in that workspace
   using train_vae_hippie:
   - output_dir: ./vae_checkpoints/my_session
   - z_dim: 30
   - n_epochs: 100
   - batch_size: 256
   - val_fraction: 0.1
2. Tell me the best validation loss achieved and the path to the saved checkpoint.
```

### Compress neurons with a trained checkpoint

```
I have a trained VAE checkpoint at ./vae_checkpoints/my_session/vae_best.ckpt
and a SpikeData object in workspace "new_session" namespace "sorted".

Using the SpikeLab MCP server:
1. Run compress_neurons_hippie with that checkpoint on the workspace.
2. Use run_umap=True and run_hdbscan=True with min_cluster_size=5.
3. Write vae_embedding, vae_umap_x, vae_umap_y, vae_cluster into the workspace.
4. Return the cluster label counts.
```

---

## 3. Run Both Models and Compare

```
I have a SpikeData workspace "comparison_session" namespace "sorted" with
avg_waveform in neuron_attributes (neuropixels recording).

Using the SpikeLab MCP server, run both classification pipelines and compare:

Step 1 — Pretrained HIPPIE:
  classify_neurons_hippie(tech_id="neuropixels", run_umap=True, run_hdbscan=True,
                           min_cluster_size=5)

Step 2 — Unsupervised VAE (train from scratch):
  train_vae_hippie(output_dir="./vae_ckpt", z_dim=30, n_epochs=100)
  compress_neurons_hippie(checkpoint_path="./vae_ckpt/vae_best.ckpt",
                           run_umap=True, run_hdbscan=True, min_cluster_size=5)

Step 3 — Report:
  - Number of HIPPIE clusters vs VAE clusters
  - Number of noise-labeled neurons (-1) in each
  - Suggest which to use for downstream analysis given the cluster coherence
```

---

## 4. Key Notes

| Item | Detail |
|------|--------|
| `tech_id` options | `"neuropixels"` (0), `"silicon_probe"` (1), `"juxtacellular"` (2), `"tetrodes"` (3) |
| HIPPIE checkpoint | Auto-downloaded from `Jesusgf23/hippie` on HuggingFace (~293 MB, cached after first use) |
| VAE checkpoint | Saved locally to `output_dir/vae_best.ckpt` |
| `avg_waveform` required | Must be present in `neuron_attributes` before calling either pipeline |
| MCP per-unit limitation | `get_waveform_traces` MCP tool is per-unit; for bulk waveform extraction ask the agent to loop over units or use the Python API directly |
| HIPPIE install | `pip install spikelab[hippie]` before running any of the above |
