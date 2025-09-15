# IntegratedAnalysisTools

[![SpikeData Tests](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml)

A monorepo for a suite of analysis tools supporting automated closed-loop experimentation and data analysis in neuroscience and related fields.

---

## Repository Structure

- **spikedata/**  
  Core module for spike train data representation, manipulation, and analysis.
- **data_loaders/**  
  Utilities to load various file formats (HDF5, NWB, KiloSort/Phy, SpikeInterface) into `SpikeData`.
  Includes exporters in `data_loaders/data_exporters.py` to write `SpikeData` back to these formats.
- *(other modules to be documented as they are added)*

---

## SpikeData Class Overview (`spikedata/spikedata.py`)

The `SpikeData` class provides a unified, extensible interface for representing, manipulating, and analyzing neuronal spike train data. It is designed to support a wide range of neuroscience data analysis workflows, with a focus on clarity, performance, and interoperability. The 2025-09 refactor streamlines the API, focusing on core spike train operations and removing legacy or niche features.

---

### Class: `SpikeData`

**High-Level Description:**
- Represents a collection of spike trains, one per neuron, as lists of numpy arrays.
- Supports loading from various formats (indices/times, rasters, events, Neo objects).
- Stores metadata, neuron attributes, and optional raw timeseries data.
- Provides methods for binning, rate calculation, interspike interval analysis, subsetting, concatenation, time slicing, latency analysis, and spike time tiling coefficient (STTC) computation.

---

#### Methods

##### `binned(bin_size)`
> **Return a binned spike count array for all neurons.**  
> - **Parameters:**  
>   - `bin_size` (float): Bin width in ms.
> - **Returns:**  
>   - `np.ndarray`: 2D array (neurons × bins) of spike counts.

##### `raster(bin_size=1.0)`
> **Return a dense binary raster (neurons × time bins) of spike events.**  
> - **Parameters:**  
>   - `bin_size` (float, optional): Bin width in ms (default: 1.0).
> - **Returns:**  
>   - `np.ndarray`: Binary matrix of shape (neurons, bins).

##### `sparse_raster(bin_size=1.0)`
> **Return a sparse matrix representation of the spike raster.**  
> - **Parameters:**  
>   - `bin_size` (float, optional): Bin width in ms (default: 1.0).
> - **Returns:**  
>   - `scipy.sparse.csr_matrix`: Sparse binary raster.

##### `rates(bin_size=100.0)`
> **Compute firing rates for each neuron in specified bins.**  
> - **Parameters:**  
>   - `bin_size` (float): Bin width in ms.
> - **Returns:**  
>   - `np.ndarray`: Firing rates (Hz) per neuron per bin.

##### `binned_meanrate(bin_size=100.0)`
> **Compute the mean firing rate for each neuron over the entire recording.**  
> - **Parameters:**  
>   - `bin_size` (float): Bin width in ms.
> - **Returns:**  
>   - `np.ndarray`: Mean firing rate (Hz) per neuron.

##### `interspike_intervals()`
> **Return a list of interspike intervals for each neuron.**  
> - **Returns:**  
>   - `List[np.ndarray]`: Each array contains ISIs for one neuron.

##### `resampled_isi(num_samples=1000)`
> **Return a resampled distribution of interspike intervals for all neurons.**  
> - **Parameters:**  
>   - `num_samples` (int): Number of samples to draw.
> - **Returns:**  
>   - `np.ndarray`: Pooled, resampled ISIs.

##### `subset(neuron_indices)`
> **Return a new SpikeData object containing only the specified neurons.**  
> - **Parameters:**  
>   - `neuron_indices` (list or array): Indices of neurons to include.
> - **Returns:**  
>   - `SpikeData`: Subsetted object.

##### `append(other)`
> **Append spike trains from another SpikeData object.**  
> - **Parameters:**  
>   - `other` (`SpikeData`): Another SpikeData instance.
> - **Returns:**  
>   - `None`

##### `concatenate_spike_data(others)`
> **Concatenate multiple SpikeData objects along the neuron axis.**  
> - **Parameters:**  
>   - `others` (list of `SpikeData`): Objects to concatenate.
> - **Returns:**  
>   - `SpikeData`: Concatenated object.

##### `subtime(start, stop)`
> **Return a new SpikeData object with spikes restricted to a time window.**  
> - **Parameters:**  
>   - `start` (float): Start time (ms).
>   - `stop` (float): Stop time (ms).
> - **Returns:**  
>   - `SpikeData`: Time-sliced object.

##### `frames(frame_size, overlap=0)`
> **Yield SpikeData objects for consecutive time frames.**  
> - **Parameters:**  
>   - `frame_size` (float): Frame duration (ms).
>   - `overlap` (float, optional): Overlap between frames (ms).
> - **Yields:**  
>   - `SpikeData`: For each frame.

##### `latencies(times, window_ms=100.0)`
> **Compute latencies from given times to nearest spikes in each train within a window.**  
> - **Parameters:**  
>   - `times` (list or array): Reference times.
>   - `window_ms` (float): Maximum latency window (ms).
> - **Returns:**  
>   - `List[List[float]]`: Latencies per neuron per reference time.

##### `latencies_to_index(i, window_ms=100.0)`
> **Compute latencies from all spikes in neuron `i` to all other neurons.**  
> - **Parameters:**  
>   - `i` (int): Index of reference neuron.
>   - `window_ms` (float): Maximum latency window (ms).
> - **Returns:**  
>   - `List[List[float]]`: Latencies per neuron.

##### `spike_time_tiling(i, j, delt=20.0)`
> **Compute the spike time tiling coefficient (STTC) between two neurons.**  
> - **Parameters:**  
>   - `i` (int): Index of first neuron.
>   - `j` (int): Index of second neuron.
>   - `delt` (float): Window size (ms).
> - **Returns:**  
>   - `float`: STTC value.

##### `spike_time_tilings(delt=20.0)`
> **Compute the pairwise STTC matrix for all neuron pairs.**  
> - **Parameters:**  
>   - `delt` (float): Window size (ms).
> - **Returns:**  
>   - `np.ndarray`: STTC matrix (neurons × neurons).

##### `to_hdf5(filepath, ..., style='ragged')`
> **Export spike data to an HDF5 file using one of four styles.**  
> - **Styles:** `raster`, `ragged`, `group`, `paired`  
> - **Key params:**  
>   - `raster_bin_size_ms` (for `raster`)  
>   - `spike_times_unit` and `fs_Hz` (for `ragged` with `'samples'`)  
>   - `group_per_unit`, `group_time_unit` (for `group`)  
>   - `idces_dataset`, `times_dataset`, `times_unit` (for `paired`)  
>   - Optional `raw_dataset`, `raw_time_dataset`, `raw_time_unit` to write continuous data/time  
> - Times are converted from internal ms to requested units.

##### `to_nwb(filepath, spike_times_dataset='spike_times', spike_times_index_dataset='spike_times_index', group='units')`
> **Export spike data to a minimal NWB-compatible file (HDF5 backend).**  
> - Writes `/units/spike_times` and `/units/spike_times_index` in seconds  
> - Round-trippable with `load_spikedata_from_nwb(..., prefer_pynwb=False)`

##### `to_kilosort(folder, fs_Hz, spike_times_file='spike_times.npy', spike_clusters_file='spike_clusters.npy', time_unit='samples', cluster_ids=None)`
> **Export spike data to KiloSort/Phy format.**  
> - Produces `spike_times.npy` and `spike_clusters.npy`  
> - `time_unit`: `'samples'` (default, requires `fs_Hz`), `'ms'`, or `'s'`  
> - `cluster_ids` maps SpikeData unit indices to arbitrary cluster IDs

---

### Standalone Utilities

#### `spike_time_tiling(tA, tB, delt=20.0, length=None)`
> **Compute the spike time tiling coefficient (STTC) between two spike trains.**  
> - **Parameters:**  
>   - `tA`, `tB` (array-like): Sorted spike times for each train.
>   - `delt` (float): Window size (ms).
>   - `length` (float, optional): Total duration (ms).
> - **Returns:**  
>   - `float`: STTC value.

#### `butter_filter(data, lowcut=None, highcut=None, fs=20000.0, order=5)`
> **Apply a digital Butterworth filter to raw data.**  
> - **Parameters:**  
>   - `data` (array-like): Input signal.
>   - `lowcut` (float, optional): Low cutoff frequency (Hz).
>   - `highcut` (float, optional): High cutoff frequency (Hz).
>   - `fs` (float): Sampling rate (Hz).
>   - `order` (int): Filter order.
> - **Returns:**  
>   - `np.ndarray`: Filtered signal.

---

#### `randomize(ar, swap_per_spike=5)`
> **Randomize a binary spike raster while preserving row/column sums.**  
> - **Parameters:**  
>   - `ar` (array-like): Binary matrix (neurons × time or time × neurons).  
>   - `swap_per_spike` (int): Target number of successful degree-preserving swaps per spike.  
> - **Returns:**  
>   - `np.ndarray`: Randomized binary matrix with same shape and marginals.  
>
> Example:
```python
import numpy as np
from spikedata import randomize

# neurons × time raster
raster = (np.random.rand(50, 1000) < 0.02).astype(float)
rnd = randomize(raster, swap_per_spike=5)

assert np.allclose(raster.sum(axis=0), rnd.sum(axis=0))  # column sums preserved
assert np.allclose(raster.sum(axis=1), rnd.sum(axis=1))  # row sums preserved
```

#### `get_pop_rate(t_spk_mat, SQUARE_WIDTH, GAUSS_SIGMA)`
> **Compute population firing rate by smoothing summed spike counts.**  
> - **Parameters:**  
>   - `t_spk_mat` (array-like): Time-major spike matrix (T × N), values 0/1 or counts.  
>   - `SQUARE_WIDTH` (int): Moving-average window width (samples), 0 to disable.  
>   - `GAUSS_SIGMA` (float): Gaussian sigma (samples) for additional smoothing, 0 to disable.  
> - **Returns:**  
>   - `np.ndarray`: Population rate vector of length T.  
>
> Example:
```python
import numpy as np
from spikedata import get_pop_rate

# Build T × N spike matrix
T, N = 1000, 64
t_spk_mat = (np.random.rand(T, N) < 0.01).astype(float)

pop_rate = get_pop_rate(t_spk_mat, SQUARE_WIDTH=5, GAUSS_SIGMA=2)
```

#### `get_bursts(pop_rate, pop_rate_acc, THR_BURST, MIN_BURST_DIFF, BURST_EDGE_MULT_THRESH)`
> **Detect bursts from a population rate trace using peak detection and amplitude-scaled edges.**  
> - **Parameters:**  
>   - `pop_rate` (array-like): Population rate vector (length T).  
>   - `pop_rate_acc` (array-like): Optional accumulator with same length T for peak localization; pass an empty list to skip.  
>   - `THR_BURST` (float): Multiplier on RMS(pop_rate) for peak height threshold.  
>   - `MIN_BURST_DIFF` (int): Minimum distance (samples) between consecutive peaks.  
>   - `BURST_EDGE_MULT_THRESH` (float): Edge threshold as a fraction of each burst’s peak amplitude.  
> - **Returns:**  
>   - `(tburst, edges, peak_amp)`: peak times, edge indices per burst, and amplitudes.  
>
> Example:
```python
import numpy as np
from spikedata import get_bursts

# Suppose pop_rate is computed from get_pop_rate(...)
pop_rate = np.zeros(500)
pop_rate[95:106] = np.array([0,2,4,6,8,10,8,6,4,2,0])
pop_rate[295:306] = np.array([0,3,6,9,12,15,12,9,6,3,0])

tburst, edges, peak_amp = get_bursts(
    pop_rate,
    pop_rate_acc=[],
    THR_BURST=0.5,
    MIN_BURST_DIFF=10,
    BURST_EDGE_MULT_THRESH=0.2,
)

# tburst: indices near burst peaks; edges[i] brackets burst i
```

---

### API Changes in 2025-09 Refactor

**Removed/Deprecated:**
- NEST/MuscleBeachTools integration (`NestIDNeuronAttributes`, `from_nest`, `from_mbt_neurons`)
- ISI analytics (`isi_skewness`, `isi_log_histogram`, `isi_threshold_cma`)
- Burst/avalanche/DCC analysis (`burstiness_index`, `avalanches`, `avalanche_duration_size`, `deviation_from_criticality`, `DCCResult`, `_p_and_alpha`)
- Randomization utilities (`randomized`, `randomize_raster`, `randomize_raster_greedy`, `randomize_raster_okun`, `_okun_swap`, `best_effort_sample`)
- Population/correlation/histogram utilities (`population_firing_rate`, `fano_factors`, `pearson`, `cumulative_moving_average`, `burst_detection`)

**Reorganization:**
- STTC helpers (`_sttc_ta`, `_sttc_na`) are now colocated with `spike_time_tiling` for transparency and maintainability.

---

### Migration & Usage Tips

- **Population Firing Rate:**  
  Use `SpikeData.binned(bin_size)` and apply external smoothing:
  ```python
  bins = sd.binned(10)
  smoothed = np.convolve(bins / 10, np.ones(5), 'same') / 5
  ```

- **Pairwise Correlations:**  
  Compute on the dense raster output using NumPy/SciPy:
  ```python
  r = sd.raster(1.0)
  corr = np.corrcoef(r)
  ```

- **Burst/Avalanche/DCC Analysis:**  
  These features are no longer included; use or develop dedicated modules for these analyses.

---

For detailed API documentation, see the docstrings in `spikedata/spikedata.py`.  
If you are migrating from an older version, consult the notes in the module docstring for guidance on replacing removed features.

---

## Data Loaders Overview (`data_loaders/data_loaders.py`)

These helpers convert common neurophysiology formats into `SpikeData`. Times are normalized to milliseconds.

Import convenience:
```python
from data_loaders import (
    load_spikedata_from_hdf5,
    load_spikedata_from_hdf5_raw_thresholded,
    load_spikedata_from_nwb,
    load_spikedata_from_kilosort,
    load_spikedata_from_spikeinterface,
    load_spikedata_from_spikeinterface_recording,
)
```

### HDF5 (generic)

#### Function: `load_spikedata_from_hdf5`
**High-Level Description:**
- Load spike trains from an HDF5 file using one of four input styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can be attached. Times are converted to milliseconds.

**Parameters:**
- **filepath** (`str`): Path to the HDF5 file.
- **raster_dataset** (`str | None`): Dataset path for a 2D raster/counts matrix (units × time). Use with `raster_bin_size_ms`.
- **raster_bin_size_ms** (`float | None`): Bin size (ms) for `raster_dataset`.
- **spike_times_dataset** (`str | None`): Dataset path for flat concatenated spike times (ragged array style).
- **spike_times_index_dataset** (`str | None`): Dataset path for end indices per unit (ragged array style).
- **spike_times_unit** (`'s' | 'ms' | 'samples'`): Unit for `spike_times_dataset`.
- **fs_Hz** (`float | None`): Sampling frequency (Hz). Required when any unit is `'samples'`.
- **group_per_unit** (`str | None`): Group path containing one dataset per unit with that unit's spike times.
- **group_time_unit** (`'s' | 'ms' | 'samples'`): Unit for datasets under `group_per_unit`.
- **idces_dataset** (`str | None`): Dataset path for unit indices (paired arrays style).
- **times_dataset** (`str | None`): Dataset path for spike times (paired arrays style).
- **times_unit** (`'s' | 'ms' | 'samples'`): Unit for `times_dataset`.
- **raw_dataset** (`str | None`): Dataset path for optional raw analog data (e.g., channels × time).
- **raw_time_dataset** (`str | None`): Dataset path for time vector corresponding to `raw_dataset`.
- **raw_time_unit** (`'s' | 'ms' | 'samples'`): Unit of `raw_time_dataset`.
- **length_ms** (`float | None`): Recording duration; inferred from last spike if not provided.
- **metadata** (`Mapping[str, object] | None`): Extra metadata to attach; `source_file` is added automatically.

**Returns:**
- **`SpikeData`**: Spike trains in milliseconds; may include attached `raw_data` and `raw_time`.

**Raises:**
- **ValueError**: If not exactly one input style is specified, or missing required arguments for a style, or invalid time units.
- **ImportError**: If `h5py` is unavailable.

**Behavior and Notes:**
- Exactly one of the four styles must be provided. Raster uses `SpikeData.from_raster`; paired arrays use `SpikeData.from_idces_times`; ragged arrays and group-per-unit build per-unit trains by splitting and converting to ms. Optional raw arrays are attached if both `raw_dataset` and `raw_time_dataset` are provided.

```python
sd = load_spikedata_from_hdf5(
    "data.h5",
    spike_times_dataset="/units/spike_times",
    spike_times_index_dataset="/units/spike_times_index",
    spike_times_unit="s",  # 's' | 'ms' | 'samples' (requires fs_Hz)
    fs_Hz=20000.0,
)
```

Supported styles (specify exactly one):
- Raster (units × time): `raster_dataset` + `raster_bin_size_ms`
- Flat ragged arrays (NWB-like): `spike_times_dataset` + `spike_times_index_dataset`
- Group-per-unit: `group_per_unit` (each child holds spike times)
- Paired (indices, times): `idces_dataset` + `times_dataset` + `times_unit`

Optional raw attachments: `raw_dataset` + `raw_time_dataset` with `raw_time_unit` in `s/ms/samples` (needs `fs_Hz` for samples).

### HDF5 thresholding

#### Function: `load_spikedata_from_hdf5_raw_thresholded`
**High-Level Description:**
- Threshold-and-detect spikes from an HDF5 dataset of raw traces shaped (channels × time) or (time × channels). Returns a `SpikeData` built from detected spikes per channel.

**Parameters:**
- **filepath** (`str`): Path to the HDF5 file.
- **dataset** (`str`): Dataset path containing raw traces.
- **fs_Hz** (`float`): Sampling frequency (Hz).
- **threshold_sigma** (`float`, default `5.0`): Threshold in units of per-channel standard deviation.
- **filter** (`dict | bool`, default `True`): If `True`, apply default Butterworth bandpass; if a `dict`, passed as filter configuration; if `False`, no filtering.
- **hysteresis** (`bool`, default `True`): Use rising-edge detection if `True`.
- **direction** (`'both' | 'up' | 'down'`, default `'both'`): Polarity of detection.

**Returns:**
- **`SpikeData`**: Detected spike trains per channel in milliseconds.

**Raises:**
- **ImportError**: If `h5py` is unavailable.
- **ValueError**: Propagated from detection if invalid arguments are provided.

```python
sd = load_spikedata_from_hdf5_raw_thresholded(
    "raw.h5", dataset="/raw", fs_Hz=20000.0,
    threshold_sigma=5.0, filter=True, hysteresis=True, direction="both",
)
```

### NWB (Units)

#### Function: `load_spikedata_from_nwb`
**High-Level Description:**
- Load spike trains from an NWB file's Units table. Prefers `pynwb`; falls back to `h5py` to read `/units/spike_times` and `/units/spike_times_index`. Times are in seconds and converted to milliseconds.

**Parameters:**
- **filepath** (`str`): Path to the NWB file.
- **prefer_pynwb** (`bool`, default `True`): If `True`, try `pynwb` first; on failure, fall back to `h5py` with a warning.
- **length_ms** (`float | None`): Recording duration; inferred from last spike if not provided.

**Returns:**
- **`SpikeData`**: Spike trains in milliseconds; metadata includes `source_file` and `format='NWB'`.

**Raises:**
- **ValueError**: If the file lacks a Units table or spike_times datasets.
- **ImportError**: If `h5py` is unavailable for fallback.

```python
sd = load_spikedata_from_nwb("recording.nwb", prefer_pynwb=True)
```
Falls back to `h5py` if `pynwb` is unavailable.

### KiloSort / Phy

#### Function: `load_spikedata_from_kilosort`
**High-Level Description:**
- Load clusters from KiloSort/Phy outputs: `spike_times.npy` and `spike_clusters.npy`. Optionally parse a TSV to filter clusters by label. Spike times are converted from samples or seconds to milliseconds.

**Parameters:**
- **folder** (`str`): Path to KiloSort/Phy output directory.
- **fs_Hz** (`float`): Sampling frequency (Hz) when `time_unit='samples'`.
- **spike_times_file** (`str`, default `'spike_times.npy'`): File with spike times.
- **spike_clusters_file** (`str`, default `'spike_clusters.npy'`): File with cluster assignments.
- **cluster_info_tsv** (`str | None`): Optional TSV file with cluster metadata (`group`/`KSLabel`, `cluster_id`/`id`).
- **time_unit** (`'samples' | 's' | 'ms'`, default `'samples'`): Unit of `spike_times.npy`.
- **include_noise** (`bool`, default `False`): If `False`, keep only clusters labeled `good`/`mua` when TSV provided; if `True`, keep all clusters.
- **length_ms** (`float | None`): Recording duration; inferred from last spike if not provided.

**Returns:**
- **`SpikeData`**: Spike trains grouped by cluster; metadata contains `cluster_ids`, `fs_Hz`, and `source_folder`.

**Raises:**
- **ValueError**: If `spike_times` and `spike_clusters` lengths mismatch.

```python
sd = load_spikedata_from_kilosort(
    "path/to/ks/",
    fs_Hz=30000.0,
    cluster_info_tsv="cluster_info.tsv",  # optional
)
```

### SpikeInterface

#### Function: `load_spikedata_from_spikeinterface`
**High-Level Description:**
- Convert a SpikeInterface `SortingExtractor`-like object to `SpikeData` by retrieving unit spike trains and converting sample indices to milliseconds using the sorting's sampling frequency.

**Parameters:**
- **sorting** (object): Exposes `get_unit_ids()`, `get_sampling_frequency()`, and `get_unit_spike_train(...)`.
- **sampling_frequency** (`float | None`): Override sampling frequency (Hz); if `None`, use `sorting.get_sampling_frequency()`.
- **unit_ids** (`Sequence[int | str] | None`): Subset of unit IDs to include; if `None`, include all.
- **segment_index** (`int`, default `0`): Segment index for multi-segment sortings.

**Returns:**
- **`SpikeData`**: Spike trains in milliseconds; metadata includes `source_format='SpikeInterface'`, `unit_ids`, and `fs_Hz`.

**Raises:**
- **TypeError**: If `sorting` lacks required methods.
- **ValueError**: If sampling frequency is not positive.

#### Function: `load_spikedata_from_spikeinterface_recording`
**High-Level Description:**
- Convert a SpikeInterface `BaseRecording`-like object into `SpikeData` by thresholding raw traces. Automatically orients the trace matrix to (channels × time) using a robust heuristic.

**Parameters:**
- **recording** (object): Provides `get_traces(segment_index=...)` returning a 2D array, `get_sampling_frequency()` or `sampling_frequency` attribute, and optionally `get_num_channels()`.
- **segment_index** (`int`, default `0`): Segment index to read traces from.
- **threshold_sigma** (`float`, default `5.0`): Threshold in units of per-channel standard deviation.
- **filter** (`dict | bool`, default `False`): If `True`, apply default bandpass; if `dict`, pass as filter config; if `False`, no filtering.
- **hysteresis** (`bool`, default `True`): Use rising-edge detection if `True`.
- **direction** (`'both' | 'up' | 'down'`, default `'both'`): Detection polarity.

**Returns:**
- **`SpikeData`**: Detected spike trains per channel in milliseconds.

**Raises:**
- **ValueError**: If sampling frequency is not positive or traces are not 2D.

From Sorting:
```python
sd = load_spikedata_from_spikeinterface(sorting)
```

From BaseRecording (thresholding):
```python
sd = load_spikedata_from_spikeinterface_recording(recording, threshold_sigma=5.0)
```

---

### Notes
- Times are stored in milliseconds in `SpikeData`.
- Optional dependencies are imported lazily (e.g., `h5py`, `pynwb`, `pandas`).
- See `tests/test_dataloaders.py` for runnable examples and edge cases.

---

## Data Exporters Overview (`data_loaders/data_exporters.py`)

These helpers write `SpikeData` out to common formats. Times are converted from internal milliseconds to requested units. You can call the standalone functions or the convenience instance methods on `SpikeData`.

Import convenience:
```python
from data_loaders.data_exporters import (
    export_spikedata_to_hdf5,
    export_spikedata_to_nwb,
    export_spikedata_to_kilosort,
)
```

### HDF5 (generic)

#### Function: `export_spikedata_to_hdf5(sd, filepath, *, style='ragged', ...)`
High-Level Description:
- Export spike trains to HDF5 using one of four styles: raster matrix, flat ragged arrays (NWB-like), group-per-unit datasets, or paired indices/times arrays. Optional raw analog arrays and time bases can also be exported.

Parameters (selected):
- `style` (`'raster'|'ragged'|'group'|'paired'`): Output organization style.
- Raster: `raster_dataset`, `raster_bin_size_ms`
- Ragged: `spike_times_dataset`, `spike_times_index_dataset`, `spike_times_unit` (`'s'|'ms'|'samples'`), `fs_Hz` (required for `'samples'`)
- Group: `group_per_unit`, `group_time_unit` (`'s'|'ms'|'samples'`), `fs_Hz` if `'samples'`
- Paired: `idces_dataset`, `times_dataset`, `times_unit` (`'s'|'ms'|'samples'`), `fs_Hz` if `'samples'`
- Optional raw: `raw_dataset`, `raw_time_dataset`, `raw_time_unit` (`'s'|'ms'|'samples'`), `fs_Hz` for `'samples'`

Example:
```python
sd.to_hdf5(
    "out.h5",
    style="ragged",
    spike_times_unit="s",
)
```

### NWB (Units)

#### Function: `export_spikedata_to_nwb(sd, filepath, *, spike_times_dataset='spike_times', spike_times_index_dataset='spike_times_index', group='units')`
High-Level Description:
- Write ragged spike times to `/units/spike_times` and `/units/spike_times_index` in seconds, sufficient for round-tripping with the NWB loader (h5py-based path).

Example:
```python
sd.to_nwb("out.nwb")
```

### KiloSort / Phy

#### Function: `export_spikedata_to_kilosort(sd, folder, *, fs_Hz, spike_times_file='spike_times.npy', spike_clusters_file='spike_clusters.npy', time_unit='samples', cluster_ids=None)`
High-Level Description:
- Create `spike_times.npy` and `spike_clusters.npy` suitable for KiloSort/Phy. Times default to integer samples at `fs_Hz`; can also write `'ms'` or `'s'`.

Example:
```python
sd.to_kilosort("ks_out", fs_Hz=30000.0)
```

Notes:
- Requires `h5py` for HDF5/NWB exports. Install with `pip install h5py`.
- See `tests/test_dataexporters.py` for runnable examples and edge cases.