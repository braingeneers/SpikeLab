---
name: spikelab-spikesorter
description: Runs spike sorting pipelines using the SpikeLab library (Kilosort2, Kilosort4, RT-Sort). Handles configuring and executing sorting jobs, curating units, inspecting and visualizing results. For stimulation experiments, runs artifact removal via preprocess_stim_artifacts / sort_stim_recording and then the appropriate sorter. Use when the user wants to sort recordings, curate units, or analyze sorting outputs.
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

### Execution mode split (local vs remote cluster)

For compute-intensive workflows, always pick an execution mode explicitly:

- **Local path** (default): use this skill directly with `sort_recording(..., use_docker=True)` when the user intends to run on the current workstation. Most users will use this path.
- **Remote cluster path**: only when the user explicitly requests cluster execution (e.g., "run on NRP", "deploy to cluster", "submit a batch job"). Keep sorter parameter selection in this skill, then read `src/spikelab/batch_jobs/INSTRUCTIONS.md` for the deployment workflow.

Do not suggest remote execution unless the user asks for it — many users do not have access to cloud compute.

When handing off to the batch job workflow, pass:
- chosen sorter (`kilosort2` or `kilosort4`)
- key curation thresholds (`snr_min`, `spikes_min_first`, etc.)
- desired CPU/GPU image profile for the batch container
- output path expectations (profile-configured S3 prefix or user override)

### Repo maps

Before writing sorting scripts, read the repo maps for the spike sorting API. Both files live in `agent/skills/spikelab-map-updater/` inside the installed `spikelab` package. Find the package directory with:

```bash
python -c "import spikelab; print(spikelab.__path__[0])"
```

For editable installs this is `<clone>/SpikeLab/src/spikelab/`; for PyPI installs it is `<env>/site-packages/spikelab/`. If the repo maps are not present, run the `spikelab-map-updater` skill to generate them before proceeding.

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
- **Is this a stimulation experiment?** Stimulation recordings contain large stimulation artifacts caused by electrical stimulation of the tissue. If the user mentions stimulation, or if you observe large artifact patterns in the data, the workflow branches on whether there is a usable intrinsic-activity baseline:
  - **With a baseline recording:** use the two-step RT-Sort + `sort_stim_recording` pipeline (see "Stimulation-aware sorting" below). Ask for the intrinsic activity recording (for training sequences), the stim recording, and the logged stim times.
  - **No baseline available** (or short stim-only recording): clean the stim recording with `preprocess_stim_artifacts` and then pass the cleaned recording into the normal `sort_recording(..., sorter="kilosort2"/"kilosort4")` entry point (see "Stim-sorting without an intrinsic-activity baseline" below). Ask for the stim recording and the logged stim times.

### Step 2: Choose the entry point

| Scenario | Function |
|---|---|
| Single or multiple recordings, any sorter | `sort_recording(recording_files, sorter=...)` |
| Multi-well Maxwell (multiple stream IDs) | `sort_multistream(recording, stream_ids, sorter=...)` |
| Stimulation recording, with intrinsic-activity baseline | `sort_stim_recording(stim_recording, rt_sort, stim_times_ms, ...)` |
| Stimulation recording, no baseline (KS2/KS4 on cleaned traces) | `cleaned, meta = preprocess_stim_artifacts(rec, stim_times_ms, output_path=...)` → `sort_recording([cleaned], sorter="kilosort2")` |

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
- `isi_viol_max` — maximum ISI violation, in the units of `isi_violation_method` (default: `0.01`). With `method="percent"` (default) the value is a **fraction** in `[0, 1]` — `0.01` means ≤ 1% of spikes are ISI violations, `0.05` ≤ 5%. (Legacy callers passing values `≥ 1.0` with `method="percent"` are auto-divided by 100 with a `DeprecationWarning` — `1.0` still works and is treated as 1%.) With `method="hill"` it is the Hill et al. (2011) contamination ratio (>1 = highly contaminated).
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

**Pipeline safeguards (opt-in):**
- `canary_first_n_s` — when > 0, run the configured backend on the first N seconds of each recording before launching the full sort, catching MEX-compile / model-load / Docker-image / preprocessing failures in seconds rather than hours. Default: `0.0` (disabled). Recommended for long sorts on flaky configs (e.g. 30 s).
- `docker_image_expected_digest` — optional `sha256:...` digest the operator expects the local Docker image to match. The actual digest is always recorded in `config_used.json` and the sorting report; this knob only emits a **warning** (no failure) when the local digest differs. Default: `None`. Use to pin reproducibility against mutable image tags.

---

## Running a Sorting Job

### Remote cluster handoff

If the user explicitly requests cluster execution:

1. Finalize sorter parameters in this skill.
2. Generate or update the run command that should execute inside the container.
3. Read `src/spikelab/batch_jobs/INSTRUCTIONS.md` and follow its workflow:
   - temporary image build/push steps
   - `spikelab-batch-jobs render-job ...`
   - `spikelab-batch-jobs deploy-job ... --image-profile <cpu|gpu>`
4. Return to this skill for quality review after artifacts are produced.

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

#### Stim-sorting without an intrinsic-activity baseline

When there is no baseline recording to train RT-Sort sequences on, use the one-shot `preprocess_stim_artifacts` wrapper to produce a cleaned `BaseRecording` and hand that to the normal `sort_recording` entry point with any sorter (KS2/KS4/etc.):

```python
from spikelab.spike_sorting.stim_sorting import preprocess_stim_artifacts
from spikelab.spike_sorting import sort_recording

cleaned_rec, stim_meta = preprocess_stim_artifacts(
    recording=stim_rec,               # SpikeInterface BaseRecording of the stim file
    stim_times_ms=logged_times_ms,
    output_path=f"{RESULTS_DIR}/cleaned.dat",  # float32 binary; required for Docker sorters
    # method="polynomial" is the default — preserves spikes in the
    # 0-10 ms post-stim window. Switch to method="blank" only if the
    # `poly_clamp_factor` warning fires on >~5 % of events (see below).
    artifact_window_ms=10.0,
    recenter=True, max_offset_ms=50.0,
)
# cleaned_rec is a BinaryRecordingExtractor — dumpable through SI's JSON
# encoder, which is required for any Docker-based sorter (KS2/KS4).
results = sort_recording(
    recording_files=[cleaned_rec],
    results_folders=[f"{RESULTS_DIR}/sorted"],
    sorter="kilosort2",
    use_docker=True,
)
# stim_meta: stim_times_ms_{logged,corrected}, recenter_offsets_ms,
#            blanked_fraction, blanked_fraction_per_channel
```

When to prefer this over `sort_stim_recording`:
- no intrinsic-activity file is available to train sequences on;
- the recording is short and stim-dense (no usable pre-stim baseline within the file itself);
- you want KS2/KS4-style global sorting rather than sequence-based assignment.

**Method choice — `"polynomial"` (default) vs. `"blank"`.** Polynomial detrend is the default for `preprocess_stim_artifacts` and `remove_stim_artifacts`, and is what you want unless the analysis is genuinely indifferent to the 0–10 ms post-stim window (early evoked responses, polysynaptic recurrence, paired-pulse facilitation, drug-vs-baseline rate comparisons all live in that window). The polynomial blanks ADC-clipped samples automatically (same as `"blank"` does) *and* fits a low-order polynomial to the post-saturation tail to recover spikes riding on it — so it's strictly a superset of what blanking does, never worse on a per-segment basis except for compute time. The 600 mV MaxOne incident that motivated the cautious-blank guidance was a polynomial-divergence-on-the-tail problem, not an "polynomial doesn't blank saturated samples" problem; `remove_stim_artifacts` now ships with a `poly_clamp_factor=10.0` sanity clamp that catches divergence per (channel, fit segment) — any post-subtract segment exceeding `10 × saturation_threshold` is blanked and a one-shot warning is emitted at end-of-call. **Switch to `"blank"` when** (a) the clamp warning fires on more than ~5 % of events — at that scale a uniform blank produces a more consistent dataset than mixing fits and clamp-fallback blanks per event; (b) curation rejects an unusually high fraction of raw units, especially via the ISI-violation gate (a real-data validation on the 600 mV MaxOne recording showed polynomial dropping curated units 43 → 18 because polynomial residuals near each artifact onset were detected as spurious 4 Hz-locked spikes, contaminating real units' spike trains and doubling the ISI-violation failure rate from 17 % to 32 %; the clamp didn't fire because the fits weren't divergent, just imperfect enough to trip KS2's threshold detector); or (c) you want a single defensible "always-clean" processing path for a publication figure regardless of whether polynomial would have helped. **Diagnostic heuristic:** if you sort the same recording twice (once polynomial, once blank) and the polynomial run yields substantially fewer curated units while ISI-violation failures dominate, polynomial residuals are contaminating curation — switch to blank for that recording. To disable the clamp pass `poly_clamp_factor=None` (typically only for comparing pre-clamp vs post-clamp behaviour during diagnostics).

`preprocess_stim_artifacts` returns a `NumpyRecording` (in-memory) when `output_path` is omitted — fine for non-Docker sorters and iterative debugging, but NOT dumpable for Docker.

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

See `REPO_MAP_DETAILED.md` for the full `sort_stim_recording`, `preprocess_stim_artifacts`, and `remove_stim_artifacts` parameter signatures.

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

**The pipeline auto-generates `<results_folder>/sorting_report.md` after every recording.** No manual report writing is required. The report bundles the curation outcome, environment, pipeline timing, non-default settings, unit quality stats, output file listing, warnings, and (on failure) the full traceback + last-200-lines context.

To inspect a run, point the user at the per-recording results folder. The artefacts written there are:

- `sorting_report.md` — **primary human-readable report.** Auto-generated after every recording; safe to delete the Tee log once it's written.
- `recording_report.json` — machine-readable per-recording status (status, error class, retries, log path, peak resource usage).
- `config_used.json` — full snapshot of the `SortingPipelineConfig` used; diff against defaults for reproducibility.
- `watchdog_events.jsonl` — only present when any watchdog crossed warn/abort. Timestamped JSONL of pressure events.
- `disk_exhaustion_report.json` — only present when the disk watchdog tripped. Includes free space, projection, and top consumers.
- `gpu_snapshot_at_trip.txt` — only present when the host-RAM or GPU watchdog tripped. `nvidia-smi` + `torch.cuda.memory_summary` per device.
- `sorting_<YYMMDD_HHMMSS>.log` — full Tee-mirrored stdout. Lifecycle governed by `tee_log_policy` (see below).

The full artefact inventory and lifecycle table is in **Pipeline Resource Management → Output artefacts** below.

The Tee log lifecycle is governed by `ExecutionConfig.tee_log_policy` (defaults to `"delete_on_success"` — the log is removed only **after** the Markdown report writes successfully). **Failed sorts always preserve the Tee log** regardless of policy, so the traceback is never lost.

If a user wants to regenerate or update a report manually (e.g. after editing `recording_report.json` by hand, or to refresh the report against a more recent log):

```python
from spikelab.spike_sorting.report import generate_sorting_report

generate_sorting_report("data/sorted/recording_a")
```

This re-reads the log + JSON + pickle and rewrites the Markdown file. It does **not** trigger the `tee_log_policy` deletion — that only happens during the live `sort_recording` flow.

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
| Kilosort4 | `kilosort4-base:py311-si0.104` | 12.6+ | PyTorch+cu126 wheels; requires NVIDIA driver ≥ 550. Used for both cu126 and cu130 hosts (cu126 wheels run fine on cu130 drivers). |

**Drivers below 525** (Kepler/Fermi GPUs, or very old enterprise systems): KS4 Docker is unworkable — modern PyTorch has dropped support for those GPU architectures, so no custom image will help. Recommend KS2 Docker first (its image selection is driver-agnostic and the bundled MATLAB Runtime supports drivers back to ~450); if that also fails, fall back to local KS2 with a real MATLAB install (`sorter="kilosort2"`, `kilosort_path="..."`, no `use_docker`). If neither works the GPU is too old — switch hardware or upgrade the driver.

**Building custom images for unsupported CUDA versions:**

If sorting raises `RuntimeError: No compatible Docker image for '{sorter}' with CUDA {cuda_tag}`, the host GPU driver is too old for the pre-built images. Build a custom image by changing the PyTorch CUDA wheel in the Dockerfile:

1. Open `SpikeLab/docker/kilosort4/Dockerfile`.
2. Find the `pip install` line that installs PyTorch:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
3. Replace the CUDA suffix in the URL with the detected tag from the error (e.g., `cu118`, `cu121`, `cu124`):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<cuda_tag>
   ```
   Available PyTorch CUDA wheels: `cu118`, `cu121`, `cu124`, `cu126`. Check https://download.pytorch.org/whl/ for the latest list.
4. Also remove the `nvidia-*-cu11` uninstall lines if building for CUDA 11.8 (the base image already has the correct packages).
5. Build and tag the image:
   ```bash
   docker build -t spikeinterface/kilosort4-base:py311-si0.104-<cuda_tag> \
       -f SpikeLab/docker/kilosort4/Dockerfile SpikeLab/docker/kilosort4/
   ```
6. Pass the custom image to `sort_recording`:
   ```python
   sort_recording(..., use_docker="spikeinterface/kilosort4-base:py311-si0.104-<cuda_tag>")
   ```

Kilosort2 does not need custom CUDA-version images — its MATLAB Runtime ships bundled CUDA libraries, and `MW_CUDA_FORWARD_COMPATIBILITY` lets one image work on any GPU newer than the runtime's build target. It still requires a reasonably modern NVIDIA driver (≥ 525).

**Rebuilding the default images:**

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
    get_local_image_digest,
)

print(get_host_cuda_driver_version())  # e.g. 590
print(get_host_cuda_tag())             # e.g. "cu130"
print(get_docker_image("kilosort4"))   # e.g. "spikeinterface/kilosort4-base:py311-si0.104"
print(get_local_image_digest("spikeinterface/kilosort4-base:py311-si0.104"))
# e.g. "sha256:9f1c..."
```

**Image digest pinning (warn-only).** The pipeline auto-records the local image's digest (`get_local_image_digest(image_tag)`) into `config_used.json` and `sorting_report.md` whenever `use_docker=True`, so two sorts months apart can be diffed at the bit level instead of only by mutable image tag. To enforce a specific digest, set `docker_image_expected_digest="sha256:..."`. When the local digest differs the pipeline emits a one-line **warning** (it does **not** fail) — the recorded digest is the source of truth, the expected-digest knob is a tripwire for unintentional drift. Returns `None` if the digest cannot be resolved (image absent, daemon down, neither `docker` python lib nor the `docker` CLI available); pinning is then silently skipped.

---

## Pipeline Resource Management

`sort_recording` applies an extensive set of automatic safeguards. They are sorter-agnostic except where noted — KS2, KS4, and RT-Sort all benefit. Every safeguard is configurable via `ExecutionConfig`; defaults are sensible for a 32–64 GB workstation.

### Pre-loop preflight checks

Run before any recording is sorted. Findings are printed; `preflight_strict=True` flips warnings into hard failures.

| Check | Protects against | Adjustable knob |
|---|---|---|
| Free disk on intermediate / results folders | Sort filling the volume mid-run | `preflight_min_free_inter_gb=20.0`, `preflight_min_free_results_gb=2.0` |
| Available host RAM | Starting a sort that immediately swaps | `preflight_min_available_ram_gb=4.0` |
| Free GPU VRAM (when sorter uses GPU) | Starting on a GPU another process owns | `preflight_min_free_vram_gb=2.0` |
| `HDF5_PLUGIN_PATH` directory exists when configured | Maxwell decoder plugin load failures | `RecordingConfig.hdf5_plugin_path` |
| `.wslconfig` memory ceiling (Windows + Docker) | WSL2 VM growing beyond host capacity | n/a (warns operator to set `[wsl2] memory=`) |
| RT-Sort intermediate-disk projection | Multi-hundred-GB intermediate files filling the volume | n/a (computed from recording dims) |
| Recording path existence + extension | Typos / missing files | n/a (warn / fail) |
| `RLIMIT_NOFILE` / `RLIMIT_NPROC` (POSIX) | Sort running out of FDs / fork slots mid-run | n/a (warn — operator must `ulimit` higher) |
| SpikeInterface version inside tested range | Surprise SI API changes | n/a (warn) |
| Sorter runtime dependencies | Wrong env / missing install / Docker daemon down — fails fast in milliseconds rather than seconds-to-minutes into the sort. Probes per backend: KS2 host (matlab on PATH + `master_kilosort.m`), KS4 host (`import kilosort` + version inside [4.0, 5.0)), RT-Sort (`torch` + `torch.cuda.is_available()` when device=cuda, plus `diptest`/`sklearn`/`h5py`/`tqdm`), KS2/KS4 Docker (daemon ping + image cached locally) | n/a — fail-level finding `sorter_dependency_missing`; KS4 out-of-range version emits `kilosort4_version_outside_tested_range` (warn) |
| GPU device exists | Configured `cuda:N` not present on the host (typo or dev box swap) — surfaced upfront instead of as an opaque CUDA invalid-device error mid-sort | n/a — fail-level finding `gpu_device_not_present`; only runs when the sorter is GPU-backed |
| Recording sample rate inside sorter window | Rate that puts the sorter out-of-distribution. KS2/KS4 window is `[10, 50] kHz`; **RT-Sort is rate-locked** — bundled MEA model trained at 20 kHz, Neuropixels at 30 kHz, ±0.5% tolerance. Only inspects pre-loaded recordings (path-only inputs are skipped) | n/a — warn-level finding `sample_rate_out_of_window`; flips to fail under `preflight_strict=True` |

The whole preflight subsystem can be disabled via `preflight=False`.

### Live watchdogs (run alongside the sort)

Five daemon-thread monitors wrap the per-recording sort. Each emits warn / abort events to `<results_folder>/watchdog_events.jsonl`. On abort, each writes a diagnostic artefact and either kills a registered subprocess (KS2 MATLAB, Docker container) or invokes a kill callback (`_thread.interrupt_main` → `os._exit` for in-process KS4 / RT-Sort).

| Watchdog | Trip condition | Defaults | Adjustable knobs |
|---|---|---|---|
| Host memory | system-wide `psutil.virtual_memory().percent` | warn 85%, abort 92% | `host_ram_watchdog`, `host_ram_warn_pct`, `host_ram_abort_pct`, `host_ram_poll_interval_s` |
| GPU memory | per-device VRAM used % (only the in-use device) | warn 85%, abort 95% | `gpu_watchdog`, `gpu_warn_pct`, `gpu_abort_pct`, `gpu_poll_interval_s` |
| Disk usage | free GB on intermediate-folder volume | warn ≤ 5 GB, abort ≤ 1 GB | `disk_watchdog`, `disk_warn_free_gb`, `disk_abort_free_gb`, `disk_poll_interval_s` |
| Log inactivity | Tee log mtime stagnation, recording-aware tolerance | base 600 s + 30 s/min, max 7200 s, in-process kill grace 10 s | `sorter_inactivity_timeout`, `sorter_inactivity_base_s`, `sorter_inactivity_per_min_s`, `sorter_inactivity_max_s`, `sorter_inactivity_in_process_grace_s` |
| I/O stall | volume read+write byte counter stagnation | stall 300 s, poll 10 s | `io_stall_watchdog`, `io_stall_s`, `io_stall_poll_interval_s` |

### Pipeline canary (opt-in)

When `ExecutionConfig.canary_first_n_s > 0`, the pipeline clones the live config, restricts the recording window to `[0, canary_first_n_s]` seconds, disables curation / exporters / figures / the post-sort report, and runs the same backend on the same recording into `<inter_path>/_canary/` **before** committing to the full sort. The MEX compile (KS2), model load (KS4 / RT-Sort), Docker container start, and the sorter's first preprocessing pass all execute under realistic conditions — that is the point. A multi-hour sort that would have failed at minute 1 fails at second 30 instead.

Failure handling is asymmetric on purpose:
- **Classified failure** in the canary (`InsufficientActivityError`, `BiologicalSortFailure`, `EnvironmentSortFailure`, `ResourceSortFailure`) — propagated as the recording's classified result; the full sort is **never launched**.
- **Unexpected / non-classified failure** in the canary (e.g. canary itself OOMing on a tiny window) — logged but **not** propagated; the full sort proceeds and the live watchdogs handle resource shape at runtime.

Recommend enabling it (`canary_first_n_s=30`) for long sorts on potentially-flaky configurations: first-time Docker images, freshly compiled MEX binaries, new CUDA drivers, untried sorter param overrides. Off by default because the smoke test adds ~30 s of startup overhead per recording. Recordings shorter than `canary_first_n_s` are silently skipped.

### Memory enforcement

| Layer | Where | What it does | Adjustable |
|---|---|---|---|
| `RLIMIT_DATA` | Linux / macOS | Kernel-enforced heap cap at 80% of host RAM | hard-coded 80% |
| Windows Job Object | Windows + pywin32 installed | Kernel-enforced process memory cap (mirrors `RLIMIT_DATA`) | tracks `host_ram_abort_pct` |
| Userspace watchdog | All platforms | Warn + graceful abort with diagnostic capture | see Host memory watchdog above |
| Docker `mem_limit` | Docker sorters | Container kernel limit at 80% of host RAM | hard-coded 80% |

### Subprocess / container kills

Lifecycle of registered kill targets:

- **KS2 MATLAB Popen** — registered with both the host-memory watchdog and the inactivity watchdog; killed via `Popen.terminate()` → `Popen.kill()` after a 5 s grace.
- **Docker container** (KS2 / KS4 with `use_docker=True`) — registered with the host-memory watchdog and a container-aware inactivity watchdog; killed via `container.stop(timeout=2)` → `container.kill()`. Auto-unregistered via `weakref.finalize` when SI releases the container.
- **In-process KS4 / RT-Sort** — kill path is `_thread.interrupt_main()` first, then `os._exit(1)` after `sorter_inactivity_in_process_grace_s` (default 10 s). The graceful interrupt covers Python-level pauses; the `os._exit` fallback covers stuck CUDA kernels and numba `@njit(parallel=True)` deadlocks that don't return to Python.

### Per-recording sort lock

`<inter_path>/.spikelab_sort.lock` is created atomically at sort start and removed at end. Two `sort_recording` calls against the same intermediate folder cannot proceed simultaneously — the second raises `ConcurrentSortError` with a clear message. Stale locks from crashed sorts (PID no longer alive) are reclaimed automatically.

### OOM auto-retry

When a sort fails with `GPUOutOfMemoryError`, the pipeline scales the relevant per-sorter knob and retries:

| Sorter | Knob halved on retry |
|---|---|
| KS2 | `NT` (rounded down to a multiple of 32; refuses below 1024) |
| KS4 | `batch_size` (refuses below 1024) |
| RT-Sort | `num_processes` (refuses at 1) |

Adjustable via `oom_retry_max=1` and `oom_retry_factor=0.5`. The scaled config is reset after the recording so subsequent recordings start fresh.

### Atomic writes

All result files (`sorted_spikedata_curated.pkl`, `sorted_spikedata.pkl`, `recording_report.json`, `disk_exhaustion_report.json`, `sorting_report.md`) are written via the `<file>.tmp` + `os.replace` pattern, so an `os._exit` mid-write cannot corrupt them.

### Quality-of-life

- `prevent_system_sleep` (Windows-only) — calls `SetThreadExecutionState` to prevent sleep / hibernation during a sort. On by default; `prevent_system_sleep=False` to disable.
- `cleanup_temp_files` — sweeps marker-prefixed temp files in `$TMPDIR` / `%TEMP%` on **clean** exit. Failed sorts leave temp files behind for diagnosis. On by default; `cleanup_temp_files=False` to disable.

### Output artefacts (always check these for diagnostics)

Every per-recording results folder contains:

| File | When written | Purpose |
|---|---|---|
| `sorting_report.md` | Always (auto-generated post-sort) | **Primary human-readable report.** Curation outcome, environment, pipeline timing, non-default settings, unit quality stats, output files, warnings, and (on failure) full traceback + last-200-lines context. |
| `recording_report.json` | Always | Machine-readable status: status, error class, wall time, retries, log path, peak resource usage. |
| `config_used.json` | Always | Snapshot of the full `SortingPipelineConfig` used, for diff against defaults / reproducibility. When `use_docker=True`, also includes the resolved `docker_image_digest` field — the actual `sha256:...` of the local image at sort time, so a sort can be replayed bit-for-bit even if the image tag drifts. |
| `watchdog_events.jsonl` | When any watchdog crossed warn / abort | Timestamped audit log of pressure events. |
| `sorting_<YYMMDD_HHMMSS>.log` | Always (subject to `tee_log_policy`) | Full pipeline stdout — environment banner, per-stage progress, traceback on failure, closing summary. |
| `disk_exhaustion_report.json` | Only when disk watchdog tripped | Free space, projection, top consumers, suggested actions. |
| `gpu_snapshot_at_trip.txt` | Only when host-RAM or GPU watchdog tripped | `nvidia-smi` output + `torch.cuda.memory_summary` for each device. |

`tee_log_policy` controls the Tee log lifecycle on **success**:
- `"keep"` — leave the log untouched.
- `"gzip_on_success"` — compress to `.log.gz`.
- `"delete_on_success"` (default) — remove. The condensed `sorting_report.md` carries the full failure context for failed sorts, so deletion only fires after a successful post-sort report has been generated. **Failed sorts always preserve the log.**

### Disabling / overriding safeguards

Every safeguard is independently togglable via `ExecutionConfig`. To pass through `sort_recording` directly, use the same name as a kwarg:

```python
sort_recording(
    recording_files=[...],
    sorter="kilosort4",
    use_docker=True,
    # turn down the host-RAM watchdog for an undersized box
    host_ram_warn_pct=70.0,
    host_ram_abort_pct=80.0,
    # silence the Windows sleep prevention
    prevent_system_sleep=False,
    # keep all Tee logs even on success
    tee_log_policy="keep",
)
```

Or with a preset config:

```python
from spikelab.spike_sorting.config import KILOSORT4
cfg = KILOSORT4.override(
    gpu_abort_pct=98.0,
    oom_retry_max=2,
    sorter_inactivity_max_s=14400.0,  # raise to 4h cap for very long sorts
)
sort_recording(recording_files=[...], config=cfg)
```

### Inspecting watchdog history

After a sort, parse `<results_folder>/watchdog_events.jsonl`:

```python
import json
events = [
    json.loads(line)
    for line in open("data/sorted/rec1/watchdog_events.jsonl")
]
host_warns = [e for e in events if e["watchdog"] == "host_memory" and e["event"] == "warn"]
peak_used = max((e["used_pct"] for e in host_warns), default=None)
```

Each `RecordingResult` in the `SortRunReport` returned by `sort_recording` already includes `peak_host_ram_pct` / `peak_gpu_used_pct` / `min_disk_free_gb` derived from this file.

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
| Host RAM-bound on RT-Sort with long recordings | Detection holds the full filtered recording + model state | Use `rt_sort_detection_window_s=180` (detect once on 3 min, sort_offline still covers full recording). Keep `streaming_waveforms=True` (default). On Linux/macOS the pipeline's `RLIMIT_DATA` cap will surface a clean `MemoryError` before the kernel OOM killer; on Windows monitor RAM manually since the cap is not enforced. |
| OOM during waveform extraction with many units | High-unit-count sorts without streaming | Ensure `streaming_waveforms=True` (default). For extreme cases also set `save_waveform_files=False` so only templates are persisted. The pipeline's heap cap (POSIX only) raises `MemoryError` rather than letting the OS kill the process. |
| `MemoryError` during local sort | Heap cap reached 80% of host RAM (POSIX `RLIMIT_DATA`) | Reduce concurrency (`n_jobs`), shorten the time window (`first_n_mins`, `rt_sort_detection_window_s`), or run a smaller chunk per call. The cap is a guard against runaway numpy/torch allocations; hitting it indicates the workload genuinely needs more RAM. |
| `HostMemoryWatchdogError` (status `oom_host_ram`) | Userspace watchdog tripped because system memory crossed `host_ram_abort_pct` (default 92%) | Same remediation as `MemoryError`. The trip wrote `gpu_snapshot_at_trip.txt` next to the recording report — inspect it to identify whether the sort or another process owned the memory. To raise the abort threshold, set `host_ram_abort_pct=95.0`. |
| `GpuMemoryWatchdogError` (status `oom_gpu`) | GPU VRAM crossed `gpu_abort_pct` (default 95%) | OOM-retry handles the simple case automatically (halves `NT` / `batch_size` / `num_processes`). If retries are exhausted, reduce the per-batch knob manually or use a larger-memory GPU. The trip wrote a `gpu_snapshot_at_trip.txt`. |
| `SorterTimeoutError` (status `sorter_timeout`) | Tee log mtime stagnant beyond the recording-aware tolerance | Inspect the log up to the trip moment for the proximate cause (CUDA hang, MATLAB JVM deadlock, mex kernel failure). To increase tolerance for unusually quiet sorts, raise `sorter_inactivity_base_s` and/or `sorter_inactivity_max_s`. |
| `DiskExhaustionError` (status `disk_exhausted`) | Disk watchdog tripped — free disk dropped below `disk_abort_free_gb` (default 1 GB) | Inspect `<results_folder>/disk_exhaustion_report.json` for top file consumers and projected need. Free disk and rerun. To run from a larger volume: set `intermediate_folders` explicitly. |
| `IOStallError` (status `io_stall`) | Volume's read+write byte counter stagnant for `io_stall_s` (default 300 s) | Likely a hung NFS / SMB / S3-fuse mount. Verify the storage is responding (`ls`, `df`). To extend tolerance for slow but live mounts: raise `io_stall_s`. To disable: `io_stall_watchdog=False`. |
| `ConcurrentSortError` (status `concurrent_sort`) | Another `sort_recording` call holds the lock at `<inter_path>/.spikelab_sort.lock` | Wait for the running sort to finish or use a different `intermediate_folders` path. If the lock is stale (e.g. holder PID is dead but the file lingered after a crash), the next acquire will reclaim automatically; if it doesn't, delete the lock file manually and rerun. |
| `[host memory watchdog] WARNING: system memory at 87.x%` printed repeatedly | System memory creeping up but not yet at abort threshold | Expected behaviour — the warning is the watchdog asking you to free memory. If it consistently warns without ever aborting, lower `host_ram_warn_pct` to silence the noise, or raise `host_ram_abort_pct` to give more headroom. |
| `[inactivity watchdog] active: ... tolerance=600.0s` printed at sort start with concern | Some buffered Python sorters (KS4 in-process, RT-Sort) can have multi-second silent stretches | The default 600 s base + 30 s/min recording-aware scaling already absorbs realistic buffered pauses. If you genuinely need more tolerance for an unusually quiet workflow, raise `sorter_inactivity_base_s`. |
| `[sort lock] reclaiming stale lock at ...` printed at sort start | Previous sort crashed without releasing its lock; the holder PID is no longer alive | Informational — the new sort proceeds normally after reclaim. No action needed. |
| Preflight: `low_rlimit_nofile` warning | POSIX `ulimit -n` is below 4096 — RT-Sort and KS4 worker pools may exhaust FDs | Raise the limit before launching: `ulimit -n 65536`. For systemd services, set `LimitNOFILE=65536` in the unit file. |
| Preflight: `wslconfig_missing` / `wslconfig_no_memory` (Windows + Docker only) | WSL2 VM has no host-side memory ceiling | Create / edit `%USERPROFILE%\.wslconfig` with a `[wsl2]` section and `memory=<N>GB` (~75% of physical RAM), then run `wsl --shutdown` and restart Docker Desktop. |
| Preflight: `sorter_dependency_missing` (fail) | Wrong env / missing install / Docker daemon down. Fires for: KS2 host (no `matlab` on PATH or `KILOSORT_PATH` not pointing at a directory containing `master_kilosort.m`), KS4 host (`import kilosort` fails), RT-Sort (any of `torch`/`diptest`/`sklearn`/`h5py`/`tqdm` not importable, or `device='cuda'` with `torch.cuda.is_available()=False`), KS2/KS4 Docker (daemon ping fails). Warn variant `sorter_dependency_missing` (Docker-only) for a missing locally-cached image. | Activate the right conda env or `pip install` the missing package. For KS2 host, set `KILOSORT_PATH` (or `SorterConfig.sorter_path`) to a Kilosort2 checkout. For Docker, start Docker Desktop / `systemctl start docker`, or pre-pull the image with `docker pull <tag>`. As a fallback, switch to a host-path sorter via `use_docker=False`, or to Docker via `use_docker=True`. |
| Preflight: `gpu_device_not_present` (fail) | `torch_device="cuda:N"` (KS4) or `RTSortConfig.device="cuda:N"` (RT-Sort) names an index not present on the host (typo or dev box swap) | List available indices via `python -c "import torch; print(torch.cuda.device_count())"` and pick one in range. Most workstations only have `cuda:0`. |
| Preflight: `sample_rate_out_of_window` (warn) | Recording sample rate is outside the sorter's tested window. KS2/KS4 window is `[10, 50] kHz` — most clinical/research MEAs are inside it. **For RT-Sort this is the loud one:** the bundled detection model is rate-locked (MEA at 20 kHz, Neuropixels at 30 kHz, ±0.5% jitter). Feeding any other rate puts the model out of distribution and silently degrades quality even when nothing crashes. | Resample the recording to within the supported window, or pick a sorter whose window matches the recording's native rate. KS2/KS4 are the rate-flexible choices. To suppress for an unusual but-known-safe rate, run with `preflight_strict=False` (default) — the warning prints but the sort proceeds. |
| `Canary aborted full sort due to <ClassifiedError>` printed at sort start | The opt-in canary (`canary_first_n_s > 0`) ran the configured backend on the first N seconds and surfaced a classified failure (`InsufficientActivityError`, `BiologicalSortFailure`, `EnvironmentSortFailure`, or `ResourceSortFailure`). The full sort was **never launched** — the recording's result reflects the canary's exception. | Address the underlying classified failure (same remediation as if the full sort had hit it). Once fixed, rerun. To bypass the canary for a quick retry without disabling it globally, pass `canary_first_n_s=0`. Non-classified canary failures (e.g. canary-itself OOM) are logged but do **not** abort the full sort. |
| `[docker image] expected digest sha256:... but local is sha256:...` warning | `docker_image_expected_digest` was set and the local image's digest differs from the configured expected value. Warn-only by design — recorded digest is the source of truth. | Either re-pull the image to match the expected digest (`docker pull <tag>`) or update `docker_image_expected_digest` to the new digest if the change is intentional. Inspect `config_used.json` for the recorded `docker_image_digest` field. |
| Pickling error during RT-Sort parallel clustering | Windows multiprocessing (spawn vs fork) | Set `rt_sort_num_processes=1` to use sequential processing |
| Stim peri-event alignment looks offset for biphasic pulses | Default `peak_mode="abs_max"` lands on the largest-amplitude phase, not the current-reversal moment | For biphasic anodic-first pulses pass `peak_mode="down_edge"` (or `"up_edge"` for cathodic-first). Aligns to the + → − zero crossing between the two phases — the AP trigger point. |
| `remove_stim_artifacts` blanks zero samples despite large artifacts | New gain-anchored threshold returns `+inf` when no ADC clipping is detected (< 10 samples pinned at max) | Expected behavior when artifacts stay below the ADC rail — polynomial detrend handles them. If you genuinely need to blank, pass an explicit `saturation_threshold=<µV>` or lower `min_clip_samples` in `_saturation_threshold_from_recording`. |
| Stim artifact removal leaves residual | Polynomial order too low or artifact window too short | Increase `artifact_window_ms` (e.g. 15-20) or try `poly_order=4` (but >5 risks fitting spikes) |
| `UserWarning: remove_stim_artifacts: polynomial fit diverged on N segment(s)` | Polynomial fit extrapolated wildly across saturated samples; the `poly_clamp_factor` clamp blanked those segments instead of leaving 10+ V residuals in the trace | Expected at high stim amplitudes (e.g. 600 mV on MaxOne). The clamp keeps output safe, but if you see N > a few percent of stim events, switch to `method="blank"` for the whole recording — the polynomial isn't earning its keep there. |
| `UserWarning: recenter_stim_times: median \|offset\| = X ms exceeds warn_offset_ms` | Logged stim times have a fixed delay vs. the actual artifacts in the recording (commonly: hardware/log clock skew, wrong time column, or unit mismatch ms vs s vs samples) | Verify: read the first stim time from the log, find the corresponding artifact sample in the trace, confirm the offset is consistent across a few events. If the systematic shift is real and acceptable, pass `warn_offset_ms=None` to silence; if it's a bug in the log, fix the log loading code. |
| MaxOne `.raw.h5` file fails to load with `ValueError: signal_channels do not have unique ids for stream 0` and a `falling back to spikelab.spike_sorting.maxwell_io.load_maxwell_native()` notice | mxw v25.x firmware writes a `settings/mapping` table with duplicate channel IDs that neo's `MaxwellRawIO` rejects | The library auto-falls back to the native loader; no action needed. To use the loader directly: `from spikelab.spike_sorting.maxwell_io import load_maxwell_native; rec = load_maxwell_native(path)`. For multi-well files call `list_maxwell_wells(path)` first. |

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
