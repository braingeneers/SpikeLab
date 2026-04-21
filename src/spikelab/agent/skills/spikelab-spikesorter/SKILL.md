---
name: spikelab-spikesorter
description: Runs spike sorting pipelines using the SpikeLab library (Kilosort2, Kilosort4, RT-Sort). Handles configuring and executing sorting jobs, curating units, inspecting and visualizing results. For stimulation experiments, runs artifact removal and stim-aligned sorting via sort_stim_recording. Use when the user wants to sort recordings, curate units, or analyze sorting outputs.
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
- **Is this a stimulation experiment?** Stimulation recordings contain large stimulation artifacts caused by electrical stimulation of the tissue. If the user mentions stimulation, or if you observe large artifact patterns in the data, the workflow is different — use the two-step RT-Sort + `sort_stim_recording` pipeline (see "Stimulation-aware sorting" below). Ask for: the intrinsic activity recording (for training sequences), the stim recording, and the logged stim times.

### Step 2: Choose the entry point

| Scenario | Function |
|---|---|
| Single or multiple recordings, any sorter | `sort_recording(recording_files, sorter=...)` |
| Multi-well Maxwell (multiple stream IDs) | `sort_multistream(recording, stream_ids, sorter=...)` |
| Stimulation recording (artifact removal + stim-aligned sorting) | `sort_stim_recording(stim_recording, rt_sort, stim_times_ms, ...)` |

Available sorters (see `spikelab.spike_sorting.backends.list_sorters()`):
- `"kilosort2"` — MATLAB-based. Runs locally with a real MATLAB + Kilosort2 install (pass `kilosort_path`), or in Docker using a pre-built image that bundles the compiled MATLAB Runtime (no MATLAB license needed).
- `"kilosort4"` — Pure Python via PyTorch. Runs locally (`pip install kilosort` + CUDA-enabled PyTorch) or in Docker.
- `"rt_sort"` — Deep-learning-based propagation sequence sorter (van der Molen, Lim et al. 2024, PLOS ONE). Requires PyTorch with CUDA, `diptest`, `scikit-learn`, and `tqdm`. No Docker support. The trained RTSort object is persisted to disk for reuse in stimulation-aware sorting (see "Stimulation-Aware Sorting" below).

Preset configs (from `spikelab.spike_sorting.config`): `KILOSORT2`, `KILOSORT2_DOCKER`, `KILOSORT4`, `KILOSORT4_DOCKER`, `RT_SORT_MEA`, `RT_SORT_NEUROPIXELS`.

### Step 3: Configure parameters

Key parameters to discuss with the user:

**Sorter:**
- `sorter` — `"kilosort2"`, `"kilosort4"`, or `"rt_sort"`
- `use_docker` — run the sorter inside a Docker container (auto-selects compatible image; not available for RT-Sort)
- `kilosort_path` — path to a local Kilosort2 source installation (only for `sorter="kilosort2"` without Docker)
- `kilosort_params` — override default sorter parameters (passed as-is to the underlying sorter)

**RT-Sort specific** (only used when `sorter="rt_sort"`):
- `rt_sort_probe` — `"mea"` (default) or `"neuropixels"` — selects the bundled pretrained detection model
- `rt_sort_device` — `"cuda"` (default) or `"cpu"`
- `rt_sort_save_pickle` — persist the trained RTSort object for reuse in stim sorting (default: True)
- `rt_sort_params` — override dict for fine-grained tuning (e.g. `{"stringent_thresh": 0.2, "inner_radius": 60}`)
- `rt_sort_recording_window_ms` — `(start_ms, end_ms)` window applied to **both** detection and `sort_offline`.
- `rt_sort_detection_window_s` — narrow the detection window to only the first N seconds; `sort_offline` still covers the full recording. Decouples the memory-heavy detection phase from total recording duration. Recommended default: `180` (3 min) — long enough to express the active unit set on typical MEA preparations, short enough to fit dense probes in a ~16 GB RAM budget. Extend only for very low-activity preps.

**Waveform extraction (all sorters)**:
- `streaming_waveforms` — per-unit streaming extraction + template computation (default: `True`). Bounds peak RAM to a single unit's waveform buffer (~100 MB on MaxOne) regardless of unit count.
- `save_waveform_files` — when `streaming_waveforms=True`, controls whether per-unit waveform `.npy` files are kept on disk (default: `True`). Set to `False` for the tightest low-RAM operation — templates and metrics still go to `template_cache`; downstream code that reads `get_computed_template(...)` still works.

See `RTSortConfig` in `REPO_MAP_DETAILED.md` for the full parameter list (`rt_sort_model_path`, `rt_sort_num_processes`, `rt_sort_recording_window_ms`, etc.).

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

### RT-Sort (propagation-based sorting)

RT-Sort uses a DL detection model and propagation patterns to sort spikes. Same pipeline as Kilosort (load → sort → waveforms → SpikeData → curate → compile). Requires `torch` (CUDA), `diptest`, `scikit-learn`, `tqdm`.

```python
results = sort_recording(
    recording_files=[f"{RAW_DIR}/recording_a.raw.h5"],
    results_folders=[f"{RESULTS_DIR}/recording_a_rtsort"],
    sorter="rt_sort",
    rt_sort_device="cuda",
    snr_min=5.0,
)
# RTSort object saved at: inter_*/sorter_output/rt_sort.pickle
```

Using a preset: `sort_recording(..., config=RT_SORT_NEUROPIXELS)`.

### Stimulation-aware sorting

Two-step workflow: train sequences on intrinsic activity, then sort a stim recording.

**Step 1** — Sort a baseline recording with RT-Sort:

```python
results = sort_recording(
    recording_files=[f"{RAW_DIR}/intrinsic_activity.raw.h5"],
    results_folders=[f"{RESULTS_DIR}/intrinsic"],
    sorter="rt_sort",
    rt_sort_save_pickle=True,  # default — saves rt_sort.pickle for reuse
)
```

**Step 2** — Sort the stimulation recording using those trained sequences:

```python
from spikelab.spike_sorting.stim_sorting import sort_stim_recording

stim_slices = sort_stim_recording(
    stim_recording=f"{RAW_DIR}/stim_recording.raw.h5",
    rt_sort="path/to/inter/sorter_output/rt_sort.pickle",
    stim_times_ms=logged_stim_times,
    pre_ms=50,
    post_ms=200,
    peak_mode="down_edge",          # biphasic anodic-first: align to up→down transition
    n_reference_channels=8,         # top-K summed reference for clean derivatives
)
# Returns SpikeSliceStack aligned to corrected stim times
```

The pipeline recenters logged stim times to actual artifact peaks, removes artifacts using per-event polynomial detrend (preserves spikes — they're too fast for the smooth polynomial to capture), sorts with the pre-trained sequences, and aligns to corrected stim events. Sequential stim protocols (bursts, paired-pulse) are handled by dynamically extending the blanking region.

**Recentering alignment (`peak_mode`)** — pick the alignment target that matches your stim protocol:

| `peak_mode` | Reference trace | Lands on | When to use |
|---|---|---|---|
| `"abs_max"` (default) | per-sample `max_ch |V|` | sample with largest ‖voltage‖ | Monophasic pulses; backward-compat with older pipelines |
| `"pos_peak"` | top-K summed | largest +V | Monophasic anodic |
| `"neg_peak"` | top-K summed | most negative V | Monophasic cathodic |
| `"down_edge"` | top-K summed | first + → − zero crossing between the positive peak (searched in `prewindow_ms` before the negative peak) and the negative peak | **Biphasic anodic-first** — the AP trigger point is the up→down current reversal, not either phase's peak |
| `"up_edge"` | top-K summed | symmetric down-up crossing | Biphasic cathodic-first |

`n_reference_channels` (default 8) controls how many highest-amplitude channels are summed to form the signed reference trace; summing preserves phase (coherent across artifact channels, cancels uncorrelated noise) and yields cleaner derivatives for the edge modes. `prewindow_ms` (default 5.0) is the radius of the opposite-polarity search before the primary peak.

**Saturation threshold (`saturation_threshold`)** — when `None` and a recording object is available, a gain-anchored threshold is derived from `recording.get_channel_gains()` combined with the observed amplitude distribution. If no clipping is detected (< 100 samples pinned at the maximum), the threshold returns `+inf` — meaning **no samples get blanked**, and the polynomial detrend handles everything. This matches the "only blank completely saturated electrodes" semantics: high-amplitude artifacts that never hit the ADC rail are recoverable by detrend and should not be destroyed. To force a specific rail, pass the value explicitly (e.g. `saturation_threshold=5500.0`). To fall back to the legacy 99.9-quantile heuristic, call `remove_stim_artifacts` directly without `recording=`.

Components are also available individually:

```python
from spikelab.spike_sorting.stim_sorting import recenter_stim_times, remove_stim_artifacts

corrected = recenter_stim_times(
    traces, logged_times, fs_Hz=20000,
    peak_mode="down_edge", n_reference_channels=8,
)
cleaned, blanked = remove_stim_artifacts(
    traces, corrected, fs_Hz=20000,
    recording=rec_si,  # enables gain-anchored auto threshold + "no clip → no blank"
)
```

See `REPO_MAP_DETAILED.md` for the full `sort_stim_recording` and `remove_stim_artifacts` parameter signatures.

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

### Quick recording overview

```python
fig = sd.plot(show_raster=True, show_pop_rate=True, time_range=(0, 60000))
fig.savefig("figures/recording_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

For further analysis (correlations, burst detection, event alignment, population dynamics, etc.), use the `spikelab-analysis-implementer` skill with the `sorted_spikedata_curated.pkl` file as input.

### Post-sorting report

**Always generate a Markdown report after every sorting run** and write it to `<results_folder>/sorting_report.md`. Do not wait for the user to ask.

**Data sources:** (1) the sorting script — list every parameter explicitly passed, (2) the log file (`sorting_*.log`) — environment, pipeline stage timestamps, curation line, wall time, (3) the results pickle — unit counts, spike counts, SNR/FR/ISI distributions.

**Structure:** Put **Curation Outcome** (raw → curated unit counts, total spikes, mean FR, mean SNR) at the top so it's the first thing the user sees. Then: Overview (recording info, sorter, status, wall time, log path), Script Settings (explicitly set params only), Environment table, Pipeline Timing table (parse `[YYYY-MM-DD HH:MM:SS]` banners), Unit Quality Distributions, Resources at Finish, Output Files.

Keep the report factual. Don't interpret quality unless specific thresholds were clearly violated (zero curated units, extreme ISI violations). Reference the full log path so the user can dig into raw output.

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
| `ImportError: RT-Sort backend requires...` | Missing RT-Sort dependencies | Install: `pip install torch diptest scikit-learn tqdm h5py`. For torch, match your CUDA version: https://pytorch.org/get-started/locally/ |
| RT-Sort CUDA out of memory | Recording too large for GPU VRAM | Reduce `rt_sort_recording_window_ms` to a shorter window, or use `rt_sort_device="cpu"` (slow) |
| Host RAM-bound on RT-Sort with long recordings | Detection holds the full filtered recording + model state | Use `rt_sort_detection_window_s=180` (detect once on 3 min, sort_offline still covers full recording). Keep `streaming_waveforms=True` (default). |
| OOM during waveform extraction with many units | High-unit-count sorts without streaming | Ensure `streaming_waveforms=True` (default). For extreme cases also set `save_waveform_files=False` so only templates are persisted. |
| Pickling error during RT-Sort parallel clustering | Windows multiprocessing (spawn vs fork) | Set `rt_sort_num_processes=1` to use sequential processing |
| Stim peri-event alignment looks offset for biphasic pulses | Default `peak_mode="abs_max"` lands on the largest-amplitude phase, not the current-reversal moment | For biphasic anodic-first pulses pass `peak_mode="down_edge"` (or `"up_edge"` for cathodic-first). Aligns to the + → − zero crossing between the two phases — the AP trigger point. |
| `remove_stim_artifacts` blanks zero samples despite large artifacts | New gain-anchored threshold returns `+inf` when no ADC clipping is detected (< 10 samples pinned at max) | Expected behavior when artifacts stay below the ADC rail — polynomial detrend handles them. If you genuinely need to blank, pass an explicit `saturation_threshold=<µV>` or lower `min_clip_samples` in `_saturation_threshold_from_recording`. |
| Stim artifact removal leaves residual | Polynomial order too low or artifact window too short | Increase `artifact_window_ms` (e.g. 15-20) or try `poly_order=4` (but >5 risks fitting spikes) |

### Inspecting intermediate files

Intermediate results are in `inter_<sorter>_<timestamp>/`:
- `*_scaled_filtered.dat` — binary recording for the sorter
- `kilosort2_results/` — raw sorter output
- `waveforms/` — extracted waveform `.npy` files per unit
- `curation/` — curation history JSON and unit ID lists

For RT-Sort, the intermediate folder also contains:
- `scaled_traces.npy` — cached voltage traces (float16)
- `model_outputs.npy` — DL detection model predictions
- `rt_sort.pickle` — serialized RTSort object (for Phase 2 stim sorting reuse)
- `sorting.npz` — cached NumpySorting for fast reload on rerun
- `root_elecs.npy` — per-unit root electrode indices

---

## General Conventions

- All spike times in the library are in **milliseconds**.
- SpikeData objects from sorting have `metadata["source_format"]` and `metadata["fs_Hz"]`.
- For concatenated recordings, `metadata["rec_chunks_ms"]` contains epoch boundaries.
- `sd.split_epochs()` splits concatenated SpikeData back into per-file objects.
- Do not modify library source files. If you find a bug, report it to the user.
