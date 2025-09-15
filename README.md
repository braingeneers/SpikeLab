# IntegratedAnalysisTools

[![SpikeData Tests](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml)

A monorepo for a suite of analysis tools supporting automated closed-loop experimentation and data analysis in neuroscience and related fields.

---

## Repository Structure

- **spikedata/**  
  Core module for spike train data representation, manipulation, and analysis.
- **data_loaders/**  
  Utilities to load various file formats (HDF5, NWB, KiloSort/Phy, SpikeInterface) into `SpikeData`.
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

```python
sd = load_spikedata_from_hdf5_raw_thresholded(
    "raw.h5", dataset="/raw", fs_Hz=20000.0,
    threshold_sigma=5.0, filter=True, hysteresis=True, direction="both",
)
```

### NWB (Units)

```python
sd = load_spikedata_from_nwb("recording.nwb", prefer_pynwb=True)
```
Falls back to `h5py` if `pynwb` is unavailable.

### KiloSort / Phy

```python
sd = load_spikedata_from_kilosort(
    "path/to/ks/",
    fs_Hz=30000.0,
    cluster_info_tsv="cluster_info.tsv",  # optional
)
```

### SpikeInterface

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
