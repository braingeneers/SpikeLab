# IntegratedAnalysisTools

[![SpikeData Tests](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml?query=branch%3Amain) [![Black Formatting](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml/badge.svg)](https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml)

A monorepo for a suite of analysis tools supporting automated closed-loop experimentation and data analysis in neuroscience and related fields.

---

## Repository Structure

- **spikedata/**  
  Core module for spike train data representation, manipulation, and analysis.
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
