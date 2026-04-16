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
src/spikelab/agent/skills/spikelab-map-updater/REPO_MAP.md
src/spikelab/agent/skills/spikelab-map-updater/REPO_MAP_DETAILED.md
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

Available sorters (see `spikelab.spike_sorting.backends.list_sorters()`):
- `"kilosort2"` — MATLAB-based. Runs locally with a real MATLAB + Kilosort2 install (pass `kilosort_path`), or in Docker using a pre-built image that bundles the compiled MATLAB Runtime (no MATLAB license needed).
- `"kilosort4"` — Pure Python via PyTorch. Runs locally (`pip install kilosort` + CUDA-enabled PyTorch) or in Docker.

Preset configs (from `spikelab.spike_sorting.config`): `KILOSORT2`, `KILOSORT2_DOCKER`, `KILOSORT4`, `KILOSORT4_DOCKER`.

### Step 3: Configure parameters

Key parameters to discuss with the user:

**Sorter:**
- `sorter` — `"kilosort2"` or `"kilosort4"`
- `use_docker` — run the sorter inside a Docker container (auto-selects compatible image)
- `kilosort_path` — path to a local Kilosort2 source installation (only for `sorter="kilosort2"` without Docker)
- `kilosort_params` — override default sorter parameters (passed as-is to the underlying sorter)

**Recording:**
- `stream_id` — Maxwell well/stream identifier
- `hdf5_plugin_path` — Maxwell HDF5 decompression plugin path
- `freq_min` / `freq_max` — bandpass filter range (default: 300–6000 Hz)
- `first_n_mins` — sort only the first N minutes of the recording
- `start_time_s` / `end_time_s` — sort a specific time window in seconds (see "Sorting a time slice" below)
- `rec_chunks_s` — list of `(start_s, end_s)` tuples to sort multiple disjoint time windows
- `rec_chunks` — frame-based version of `rec_chunks_s` (advanced; requires manual sample-rate math)

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

**Figures:**
- `create_figures` — generate QC figures: quality distributions (pre-curation), curation bar, STD scatter, all templates, raster + pop rate (default: False)
- `create_unit_figures` — generate per-unit figures: ISI histogram, waveform footprint, max-channel overlay with individual traces; sorted into `curated/` and `failed/` subdirs after curation (default: False, requires `create_figures=True`)

---

## Running a Sorting Job

### Basic example (Kilosort2 via Docker)

```python
from spikelab.spike_sorting import sort_recording

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

### Kilosort4 (local or Docker)

```python
# Local — requires `pip install kilosort` and PyTorch with CUDA
results = sort_recording(
    recording_files=[f"{RAW_DIR}/recording_a.raw.h5"],
    results_folders=[f"{RESULTS_DIR}/recording_a_ks4"],
    sorter="kilosort4",
    snr_min=5.0,
    compile_to_npz=True,
)

# Docker — no local KS4 / PyTorch installation needed
results = sort_recording(
    recording_files=[f"{RAW_DIR}/recording_a.raw.h5"],
    results_folders=[f"{RESULTS_DIR}/recording_a_ks4"],
    sorter="kilosort4",
    use_docker=True,
)
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

### Sorting a time slice

Pass times in seconds — the sampling rate is read from the recording:

```python
# Single window
sort_recording(..., start_time_s=180, end_time_s=300)

# First N minutes (shortcut)
sort_recording(..., first_n_mins=5)

# Multiple disjoint windows (concatenated; split later via sd.split_epochs())
sort_recording(..., rec_chunks_s=[(0, 60), (300, 360), (600, 660)])
```

`start_time_s` defaults to 0 and `end_time_s` to the recording duration. Time-based params cannot be combined with the frame-based `rec_chunks`.

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

### QC figures script

After sorting completes, **always run the figure generation script**:

```bash
conda run -n spikelab python SpikeLab/scripts/generate_sorting_figures.py <results_folder>
```

This generates all QC figures in `<results_folder>/figures/`:

| Figure | Description |
|---|---|
| `curation_bar_plot.png` | Total vs. curated unit counts |
| `std_scatter_plot.png` | Normalized STD vs. spike count with curation thresholds |
| `all_templates_plot.png` | Stacked waveform templates by polarity |
| `quality_distributions.png` | 4-panel histogram: SNR, firing rate, spike count, ISI violations (**all units pre-curation**, with threshold lines) |
| `raster_pop_rate_first30s.png` | Raster + population rate for the first 30 s |
| `units/curated/unit_NNNN.png` | Per-unit (passed curation): ISI histogram (0–100 ms) + waveform footprint (|peak| > 8 µV) + max-channel overlay with individual traces |
| `units/failed/unit_NNNN.png` | Per-unit (failed curation): same 3-panel layout |

**Per-unit figures and quality distributions are generated automatically during the sorting pipeline** (before curation, while individual spike waveforms are still on disk and all units are available). This ensures the distributions always include all pre-curation units. After curation, per-unit figures are sorted into `curated/` and `failed/` subdirectories. Each per-unit figure has 3 panels: ISI histogram (0–100 ms), average waveform footprint at electrode positions, and a max-channel overlay showing individual spike traces (grey) with the mean waveform (red).

The post-hoc script (`generate_sorting_figures.py`) generates the remaining figures (curation bar, STD scatter, templates, raster). Use `--skip-per-unit` to skip per-unit figures when running post-hoc (they are already generated during the pipeline), or `--amp-thresh-uv N` to change the footprint amplitude threshold.

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

### Post-sorting report

**Always generate a Markdown report after every sorting run** and write it to `<results_folder>/sorting_report.md`. This is part of the default workflow — do not wait for the user to ask. The report should combine information from:

1. **The sorting script** that launched the job — extract the actual call to `sort_recording`/`sort_multistream` and list every parameter passed (sorter, use_docker, curation thresholds, time slicing, etc.). Do not infer defaults; only list what the user explicitly set.
2. **The log file** (`<results_folder>/sorting_*.log`) — parse these fields:
   - Environment header: host, Python/SI/SpikeLab versions, Docker image, GPU name + memory, RAM total, disk available, memory limit
   - Recording info: path, file size, sampling rate, channel count, duration, time slicing applied
   - Pipeline stages with timestamps (from the `[YYYY-MM-DD HH:MM:SS]` banners)
   - Curation line: `Curation: N_raw → N_curated units (N_removed removed)`
   - Final status line, wall time, resources at finish (RAM/GPU/disk)
3. **The results files** — load `sorted_spikedata_curated.pkl` and report:
   - Unit count (raw and curated), total spike count, mean/median firing rate
   - SNR distribution (mean, median, min, max)
   - Spikes-per-unit distribution (mean, median, min, max)
   - ISI violation percentages (mean, max)

The report must reference the **full path to the source log file** so the user can dig into raw output if needed.

**Report structure** (adapt as needed, but keep Curation Outcome at the top so it's the first thing the user sees):

```markdown
# Sorting Report — <rec_name>

## Curation Outcome
- Raw units: N
- Curated units: N (N removed)
- Total spikes: N
- Mean FR: X.XX Hz
- Median spikes/unit: N
- Mean SNR: X.X

## Overview
- Recording: `<path>` (<channels> ch, <fs> Hz, <duration> min)
- Sorter: `<sorter>` (Docker: <yes/no>)
- Status: <COMPLETED | FAILED | KILLED>
- Wall time: <X> min <Y> s
- Log file: `<absolute_path_to_sorting_YYMMDD_HHMMSS.log>`

## Script Settings
- Script: `<script_path>`
- Parameters explicitly set:
  - `sorter="kilosort2"`
  - `use_docker=True`
  - `snr_min=5.0`
  - ...

## Environment
| Field | Value |
|---|---|
| Host | ... |
| Python | ... |
| SI version | ... |
| SpikeLab | ... |
| Docker image | ... |
| GPU | ... |
| RAM total | ... |
| Memory limit | ... |

## Pipeline Timing
| Stage | Timestamp | Duration |
|---|---|---|
| LOADING RECORDING | ... | ... |
| SPIKE SORTING | ... | ... |
| EXTRACTING WAVEFORMS | ... | ... |
| GENERATING PER-UNIT FIGURES | ... | ... |
| CURATION | ... | ... |
| COMPILING RESULTS | ... | ... |
| DONE | ... | (total) |

Compute duration for each step as the difference between its timestamp and the next step's timestamp. Parse timestamps from the ``[YYYY-MM-DD HH:MM:SS]`` banners in the log.

## Unit Quality Distributions
(include brief tables or bullet lists — refer to unit_quality.png figure if generated)

## Resources at Finish
| Metric | Value |
|---|---|
| RAM available | ... |
| GPU memory | ... |
| Disk avail | ... |

## Output Files
- `sorted_spikedata_curated.pkl` — <size>
- `sorted.npz` — <size>
- `figures/` — <list of generated QC figures if any>
```

Keep the report factual — don't interpret whether the results are "good" unless specific thresholds were clearly violated (e.g., zero curated units, extreme ISI violations). For interpretation, direct the user to review the QC figures and the unit quality distributions.

Locate the latest `sorting_*.log` in the results folder (most recent mtime), identify the script path from the log header (`Script:` line), load the pkl, and write `sorting_report.md` in the same folder.

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
| Kilosort2 `Matrix dimensions must agree` in splitting step | Data-dependent KS2 bug on high-density wells that produce very high template counts (>~1000 clusters); fails after `Finished splitting. Found N splits, checked M/M clusters, nccg K` | Raise the second-pass detection threshold: `kilosort_params={"projection_threshold": [10, 8]}` (default is `[10, 4]`). This reduces the number of spikes extracted in the second pass, which lowers the template count and avoids the splitting bug. Retry without other changes. If still failing, try `[12, 8]`. **Retry automatically** — do not escalate to the user on a first hit; only escalate if the bumped threshold also fails. |

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
