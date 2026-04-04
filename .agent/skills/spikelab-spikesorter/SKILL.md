---
name: spikelab-spikesorter
description: Runs spike sorting pipelines using the SpikeLab library. Handles configuring and executing sorting jobs, curating units, inspecting and visualizing results. Use when the user wants to sort recordings, curate units, or analyze sorting outputs.
---

# SpikeLab Spike Sorter

You are acting as the **Spike Sorter** for the SpikeLab library. Your responsibilities are:
- Configuring and running spike sorting pipelines
- Curating sorted units using quality-control filters
- Inspecting and visualizing sorting results
- Troubleshooting sorting failures

---

## Directory Structure

At the start of each session, ask the user to confirm two directories:

1. **Raw data directory** — where the unsorted recording files live (e.g., `./data/raw/`). You only **read** from this directory. Never modify or delete raw recording files.
2. **Results directory** — where sorting outputs are stored (e.g., `./data/sorted/`). Create it if it does not exist.

The sorting pipeline writes results into per-recording subdirectories inside the results directory. The output structure is compatible with the `spikelab-analysis-implementer` skill — each subdirectory contains a `sorted_spikedata_curated.pkl` file that the analysis implementer can load directly:

```
data/
├── raw/                              ← Raw recordings (read-only)
│   ├── recording_a.raw.h5
│   ├── recording_b.raw.h5
│   └── multi_day/                    ← Directory → concatenated + split
│       ├── day1.raw.h5
│       └── day2.raw.h5
└── sorted/                           ← Sorting results (created by this skill)
    ├── recording_a/
    │   ├── sorted_spikedata_curated.pkl   ← Curated SpikeData (load with pickle)
    │   ├── sorted_spikedata.pkl           ← Raw SpikeData (if save_raw_pkl=True)
    │   ├── sorted.npz                     ← Compiled output
    │   └── figures/                       ← QC figures (if create_figures=True)
    ├── recording_b/
    │   └── ...
    └── multi_day/
        ├── chunk0/                        ← Per-file results from concatenation
        │   └── ...
        └── chunk1/
            └── ...
```

**Downstream compatibility:** The `sorted_spikedata_curated.pkl` file in each results subdirectory contains a `SpikeData` object ready for analysis. The `spikelab-analysis-implementer` skill loads these with:

```python
import pickle
with open("data/sorted/recording_a/sorted_spikedata_curated.pkl", "rb") as f:
    sd = pickle.load(f)
```

---

## Strict Boundary Rules

### File boundaries

**Raw data:** Read-only. Never modify, move, or delete files in the raw data directory.

**Sorting scripts:** Create sorting scripts in the results directory or a user-specified working directory. Never write scripts inside `SpikeLab/src/` or `SpikeLab/tests/`.

### Analysis boundaries

This skill is limited to **assessing spike sorting quality** — unit counts, SNR distributions, waveform templates, curation outcomes, and basic recording-level summaries. For any further analysis (firing rate computation, correlations, burst detection, population dynamics, event alignment, etc.), direct the user to the `spikelab-analysis-implementer` skill and point them to the `sorted_spikedata_curated.pkl` file(s) as the starting data.

### Repo maps

Before writing sorting scripts, read the repo maps for the spike sorting API:

```
.agent/skills/spikelab-map-updater/REPO_MAP.md
.agent/skills/spikelab-map-updater/REPO_MAP_DETAILED.md
```

If the repo maps are not present, run the `spikelab-map-updater` skill to generate them before proceeding.

### Never assume — ask if unsure

Do not make assumptions about recording formats, electrode configurations, or sorting parameters. Always ask for clarification when:
- The recording format is unclear (Maxwell `.h5`, NWB, SpikeInterface object)
- The user hasn't specified curation thresholds
- The number of channels or stream IDs is ambiguous
- The sorter or Docker configuration isn't specified

---

## Before Starting

### Step 1: Understand the recording

Ask the user:
- What recording format? (Maxwell `.h5`, NWB `.nwb`, directory of files, pre-loaded SpikeInterface object)
- Single recording or multiple?
- For Maxwell: single well or multi-well? Which stream IDs?
- For directories: should files be concatenated?

### Step 2: Choose the entry point

| Scenario | Function |
|---|---|
| Single or multiple recordings, any sorter | `sort_recording(recording_files, sorter=...)` |
| Multi-well Maxwell (multiple stream IDs) | `sort_multistream(recording, stream_ids, sorter=...)` |
| With a preset config | `sort_recording(recording_files, config=KILOSORT2)` |

### Step 3: Configure parameters

Key parameters to discuss with the user:

**Sorter:**
- `kilosort_path` — path to Kilosort2 installation (not needed with Docker)
- `use_docker` — run Kilosort2 in Docker container
- `kilosort_params` — override default Kilosort2 parameters

**Recording:**
- `stream_id` — Maxwell well/stream identifier
- `hdf5_plugin_path` — Maxwell HDF5 decompression plugin path
- `freq_min` / `freq_max` — bandpass filter range (default: 300–6000 Hz)

**Curation:**
- `curate_first` / `curate_second` — enable curation stages
- `fr_min` — minimum firing rate (default: 0.05 Hz)
- `isi_viol_max` — maximum ISI violation (default: 1%)
- `snr_min` — minimum SNR (default: 5.0)
- `spikes_min_first` / `spikes_min_second` — minimum spike counts (default: 30 / 50)
- `std_norm_max` — maximum normalized waveform STD (default: 1.0)
- `curation_epoch` — curate based on a single epoch (for concatenated recordings)

**Compilation:**
- `compile_to_npz` / `compile_to_mat` — output formats
- `save_raw_pkl` — save pre-curation SpikeData pickle
- `create_figures` — generate QC figures

---

## Running a Sorting Job

### Basic example

```python
from spikelab.spike_sorting import sort_recording
from spikelab.spike_sorting.config import KILOSORT2

RAW_DIR = "data/raw"
RESULTS_DIR = "data/sorted"

results = sort_recording(
    recording_files=[f"{RAW_DIR}/recording_a.raw.h5"],
    results_folders=[f"{RESULTS_DIR}/recording_a"],
    sorter="kilosort2",
    use_docker=True,
    snr_min=5.0,
    spikes_min_first=30,
    compile_to_npz=True,
    create_figures=True,
)

# results is a list of SpikeData objects (one per recording file)
sd = results[0]
print(f"Found {sd.N} curated units over {sd.length:.0f} ms")
# Curated pickle saved at: data/sorted/recording_a/sorted_spikedata_curated.pkl
```

### Multi-well Maxwell

```python
from spikelab.spike_sorting import sort_multistream

results = sort_multistream(
    recording=f"{RAW_DIR}/multiwell.raw.h5",
    stream_ids=["well000", "well001", "well002"],
    sorter="kilosort2",
    use_docker=True,
)

# results is {stream_id: list[SpikeData]}
for well, sds in results.items():
    print(f"{well}: {sds[0].N} units")
```

### Directory concatenation

When a directory is passed as a recording, all `.raw.h5`/`.nwb` files are concatenated, sorted together, and split back into per-file SpikeData:

```python
results = sort_recording(
    recording_files=[f"{RAW_DIR}/multi_day/"],
    results_folders=[f"{RESULTS_DIR}/multi_day"],
    sorter="kilosort2",
    use_docker=True,
)
# Returns one SpikeData per original file in the directory
# Each has its own epoch-specific waveform template
```

**Concatenation compatibility:** Channel count and sampling frequency must match across files (raises `ValueError`). Mismatched channel IDs or channel locations produce warnings but do not block concatenation.

---

## Working with Results

### SpikeData neuron_attributes

After sorting, each unit has enriched attributes:

```python
sd = results[0]
for i in range(sd.N):
    a = sd.neuron_attributes[i]
    print(f"Unit {a['unit_id']}: SNR={a['snr']:.1f}, "
          f"channel={a['channel']}, pos_peak={a['has_pos_peak']}")
```

Available per-unit attributes: `unit_id`, `channel`, `channel_id`, `x`, `y`, `electrode`, `template`, `template_full`, `template_windowed`, `template_peak_ind`, `amplitude`, `amplitudes`, `peak_inds`, `std_norms_all`, `has_pos_peak`, `snr`, `std_norm`, `spike_train_samples`.

For concatenated recordings, `epoch_templates` contains per-epoch average waveforms. Use `sd.split_epochs()` to split into per-file SpikeData objects, each with its own epoch template.

### Loading saved results

Results are saved as pickle files in the results directory:

```python
import pickle

# Load curated result
with open("data/sorted/recording_a/sorted_spikedata_curated.pkl", "rb") as f:
    sd = pickle.load(f)

# Load pre-curation result (if save_raw_pkl=True was used)
with open("data/sorted/recording_a/sorted_spikedata.pkl", "rb") as f:
    sd_raw = pickle.load(f)
```

These pickle files are the handoff point to the `spikelab-analysis-implementer` skill for downstream analysis.

---

## Curation

### Automatic curation (during sorting)

Curation is applied automatically during sorting based on the configuration parameters. The pipeline applies criteria in sequence (intersection): spike count → firing rate → ISI violations → SNR → normalized STD.

### Post-hoc curation

Apply additional curation filters after sorting:

```python
# Single criterion
sd_strict, metrics = sd.curate_by_snr(min_snr=8.0)
print(f"SNR filter: {sd.N} → {sd_strict.N} units")
print(f"Per-unit SNR: {metrics['metric']}")

# Multiple criteria
sd_strict, results = sd.curate(min_spikes=100, min_rate_hz=0.1, min_snr=8.0)
print(f"Combined: {sd.N} → {sd_strict.N} units")
for criterion, res in results.items():
    n_passed = res['passed'].sum()
    print(f"  {criterion}: {n_passed}/{len(res['passed'])} passed")
```

Available curation methods:
- `sd.curate_by_min_spikes(min_spikes)`
- `sd.curate_by_firing_rate(min_rate_hz)`
- `sd.curate_by_isi_violations(max_violation, threshold_ms, method)`
- `sd.curate_by_snr(min_snr)`
- `sd.curate_by_std_norm(max_std_norm)`
- `sd.curate(**kwargs)` — combined wrapper

Each returns `(SpikeData, {"metric": array, "passed": bool_array})`.

### Curation history

```python
history = SpikeData.build_curation_history(sd_original, sd_curated, results)
# Serializable dict with: initial, curations, curated, failed, metrics, curated_final
import json
with open("curation_history.json", "w") as f:
    json.dump(history, f, indent=2, default=str)
```

---

## Assessing Sorting Quality

This skill supports **sorting QC only** — verifying that the sorting and curation produced reasonable results. For any deeper analysis, direct the user to `spikelab-analysis-implementer`.

### QC figures (generated during sorting)

When `create_figures=True`, three figures are saved to `<results_folder>/figures/`:
- `curation_bar_plot.png` — total vs. curated unit counts
- `std_scatter_plot.png` — normalized STD vs. spike count with threshold lines
- `all_templates_plot.png` — stacked waveform templates by polarity

### Standalone QC figure functions

```python
from spikelab.spike_sorting.figures import (
    plot_curation_bar,
    plot_std_scatter,
    plot_templates,
)
```

All accept an optional `ax` parameter for embedding in custom figure layouts.

### Unit quality summary

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# SNR distribution
snr_vals = [a['snr'] for a in sd.neuron_attributes]
axes[0].hist(snr_vals, bins=30)
axes[0].set_xlabel("SNR")
axes[0].set_ylabel("Count")
axes[0].axvline(5.0, color='red', linestyle='--', label='threshold')

# Spike count distribution
spike_counts = [len(t) for t in sd.train]
axes[1].hist(spike_counts, bins=30)
axes[1].set_xlabel("Spike count")

# Waveform templates (first 10 units)
for i in range(min(10, sd.N)):
    template = sd.neuron_attributes[i]['template']
    axes[2].plot(template, alpha=0.5)
axes[2].set_xlabel("Sample")
axes[2].set_ylabel("Amplitude")

fig.savefig("figures/unit_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### Quick recording overview

A brief raster + population rate plot to verify the sorting looks reasonable:

```python
fig = sd.plot(show_raster=True, show_pop_rate=True, time_range=(0, 60000))
fig.savefig("figures/recording_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

For further analysis (correlations, burst detection, event alignment, population dynamics, etc.), use the `spikelab-analysis-implementer` skill with the `sorted_spikedata_curated.pkl` file as input.

---

## Figure Output Conventions

- **Always save figures as `.png` files** — never call `plt.show()`.
- Use `matplotlib.use("Agg")` at the top of every script before any other matplotlib imports.
- Save in a `figures/` subdirectory within the working directory.
- Use `dpi=150, bbox_inches="tight"` for `savefig`.
- Remove top and right spines. Keep left and bottom.
- Every axis must have a label with units.

---

## Docker GPU Compatibility

When `use_docker=True`, SpikeLab automatically selects a Docker image compatible with the host GPU:

1. Queries the NVIDIA driver version via `nvidia-smi`
2. Maps the driver to the highest supported CUDA toolkit version
3. Selects a pre-built image from the registry

**Pre-built images:**

| Sorter | Image | CUDA | Notes |
|--------|-------|------|-------|
| Kilosort2 | `kilosort2-compiled-base:py310-si0.104` | Any | MATLAB Runtime; `MW_CUDA_FORWARD_COMPATIBILITY` handles all GPUs |
| Kilosort4 | `kilosort4-base:py311-si0.104` | 12.6+ | PyTorch 2.11+cu130; requires NVIDIA driver ≥ 550 |

**Building custom images:**

Dockerfiles are in `SpikeLab/docker/kilosort2/` and `SpikeLab/docker/kilosort4/`. To rebuild:

```bash
docker build -t spikeinterface/kilosort2-compiled-base:py310-si0.104 \
    -f docker/kilosort2/Dockerfile docker/kilosort2/

docker build -t spikeinterface/kilosort4-base:py311-si0.104 \
    -f docker/kilosort4/Dockerfile docker/kilosort4/
```

**Custom images:** Pass a specific image string instead of `True`:

```python
sort_recording(..., use_docker="my-registry/my-image:tag")
```

This bypasses auto-detection and uses the specified image directly.

**Auto-detection API:**

```python
from spikelab.spike_sorting.docker_utils import (
    get_host_cuda_driver_version,
    get_host_cuda_tag,
    get_docker_image,
)

print(get_host_cuda_driver_version())  # e.g. 590
print(get_host_cuda_tag())             # e.g. "cu130"
print(get_docker_image("kilosort4"))   # e.g. "spikeinterface/kilosort4-base:py311-si0.104"
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `HDF5 plugin error` on Maxwell files | Missing decompression plugin | Pass `hdf5_plugin_path="/path/to/plugin/"` |
| `Stream ID not found` | Wrong well name | Check available streams with `MaxwellRecordingExtractor.get_streams(file)` |
| `Cannot concatenate: N channels` | Different electrode configs | Ensure all files in directory share the same MEA layout |
| `Recording has N segments` | Multi-segment recording | Split into single-segment recordings first |
| Docker sorting fails | GPU/Docker config | Check `nvidia-smi`, ensure `nvidia-docker` is installed |
| `CUDA error: no kernel image` | Docker image CUDA too old for GPU | SpikeLab auto-detects the host CUDA driver and selects a compatible image. If auto-detection fails, pass a custom image: `use_docker="my-image:tag"` |
| `Could not detect CUDA driver` | `nvidia-smi` not on PATH | Install NVIDIA drivers, or pass a specific `docker_image` string |
| `ValueError: Unknown sorter` | Unregistered backend | Check `spikelab.spike_sorting.backends.list_sorters()` |

### Inspecting intermediate files

Intermediate results are in `inter_<sorter>_<timestamp>/`:
- `*_scaled_filtered.dat` — binary recording for the sorter
- `kilosort2_results/` — raw sorter output
- `waveforms/` — extracted waveform `.npy` files per unit
- `curation/` — curation history JSON and unit ID lists

---

## General Conventions

- All spike times in the library are in **milliseconds**.
- SpikeData objects from sorting have `metadata["source_format"]` and `metadata["fs_Hz"]`.
- For concatenated recordings, `metadata["rec_chunks_ms"]` contains epoch boundaries.
- `sd.split_epochs()` splits concatenated SpikeData back into per-file objects.
- Do not modify library source files. If you find a bug, report it to the user.
