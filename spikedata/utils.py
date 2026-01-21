<<<<<<< HEAD
from typing import Optional, Literal, Union
=======
from typing import Optional, List
>>>>>>> d2e0e10 (add neuron attributes checker to neuron atts)

import numpy as np
from scipy import ndimage, signal
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


__all__ = [
    "get_sttc",
    "swap",
    "randomize",
    "trough_between",
<<<<<<< HEAD
    "TimeUnit",
    "ensure_h5py",
    "times_from_ms",
    "to_ms",
=======
    "extract_waveforms",
>>>>>>> 4e66959 (Enhanced get_traces with bandpass filtering, storage, and improved API)
]
TimeUnit = Literal["ms", "s", "samples"]

try:
    import h5py
except ImportError:
    h5py = None


def get_sttc(tA, tB, delt=20.0, length: Optional[float] = None):
    """
    Calculate the spike time tiling coefficient between two spike trains.

    Formula:
    STTC = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0 + (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0

    Parameters:
    tA (list): List of spike times for the first spike train
    tB (list): List of spike times for the second spike train
    delt (float): Time window in milliseconds (default: 20.0)
    length (float): Total duration in milliseconds (optional)

    Returns:
    sttc (float): Spike time tiling coefficient between the two spike trains

    [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
        comparison of methods and application to the study of retinal waves. Journal of
        Neuroscience 34:43, 14288–14303 (2014).
    """
    if length is None:
        length = float(max(tA[-1], tB[-1]))

    if len(tA) == 0 or len(tB) == 0:
        return 0.0

    TA = _sttc_ta(tA, delt, length) / length
    TB = _sttc_ta(tB, delt, length) / length
    return _spike_time_tiling(tA, tB, TA, TB, delt)


def _spike_time_tiling(tA, tB, TA, TB, delt):
    """Internal helper method for the second half of STTC calculation."""
    if len(tA) == 0 or len(tB) == 0:
        return 0
    PA = _sttc_na(tA, tB, delt) / len(tA)
    PB = _sttc_na(tB, tA, delt) / len(tB)

    aa = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0
    bb = (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0
    return (aa + bb) / 2


def _sttc_ta(tA, delt: float, tmax: float) -> float:
    """
    Helper function for STTC: calculate the total amount of time within a range delt of spikes within tA.
    """
    if len(tA) == 0:
        return 0.0

    base = min(delt, tA[0]) + min(delt, tmax - tA[-1])
    return base + np.minimum(np.diff(tA), 2 * delt).sum()


def _sttc_na(tA, tB, delt: float) -> int:
    """Helper function for STTC: Calculate the number of spikes in tA within delt of any spike in tB."""
    if len(tB) == 0:
        return 0
    tA, tB = np.asarray(tA), np.asarray(tB)

    # Find the closest spike in B after spikes in A.
    iB = np.searchsorted(tB, tA)

    # Clip to ensure legal indexing, then check the spike at that
    # index and its predecessor to see which is closer.
    np.clip(iB, 1, len(tB) - 1, out=iB)
    dt_left = np.abs(tB[iB] - tA)
    dt_right = np.abs(tB[iB - 1] - tA)

    # Return how many of those spikes are actually within delt.
    return (np.minimum(dt_left, dt_right) <= delt).sum()


def _resampled_isi(spikes, times, sigma_ms):
    """
    Helper method for calculating the firing rate of a spike train at specific times,
    based on the reciprocal inter-spike interval.

    Parameters:
    spikes (list): List of spike times
    times (list): List of times
    sigma_ms (float): Standard deviation in milliseconds

    Returns:
    fr (numpy.ndarray): Firing rate at specific times

    Notes:
    - Assumed to have been sampled halfway between any two given spikes, interpolated, and then
    smoothed by a Gaussian kernel with the given width.
    """
    if len(spikes) == 0:
        return np.zeros_like(times)
    elif len(spikes) == 1:
        return np.ones_like(times) / spikes[0]
    else:
        x = 0.5 * (spikes[:-1] + spikes[1:])
        y = 1 / np.diff(spikes)
        fr = np.interp(times, x, y)
        if len(np.atleast_1d(fr)) < 2:
            return fr

        dt_ms = times[1] - times[0]
        sigma = sigma_ms / dt_ms
        if sigma > 0:
            return ndimage.gaussian_filter1d(fr, sigma)
        else:
            return fr


def _train_from_i_t_list(idces, times, N):
    """
    Helper method for SpikeData constructors: Given lists of spike times and unit indices,
    produces a list where each entry contains the spike times for the corresponding unit.

    Parameters:
    idces (list): List of spike indices
    times (list): List of spike times
    N (int): Number of units

    Returns:
    ret (list): List whose ith entry is a list of the spike times of the ith unit
    """
    idces, times = np.asarray(idces), np.asarray(times)
    if N is None:
        N = idces.max() + 1

    ret = []
    for i in range(N):
        ret.append(times[idces == i])
    return ret


def butter_filter(
    data,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    fs=20000.0,
    order=5,
):
    """
    A digital butterworth filter. Type is based on input value.

    Parameters:
    data (array_like): Data to be filtered
    lowcut (float): Low cutoff frequency. If None or 0, highcut must be a number.
    highcut (float): High cutoff frequency. If None, lowpass must be a non-zero number.
    fs (float): Sample rate
    order (int): Order of the filter

    Returns:
    filtered_traces (numpy.ndarray): The filtered output with the same shape as data

    Notes:
    - If lowcut and highcut are both given, this filter is bandpass.
    - In this case, lowcut must be smaller than highcut.
    """
    if lowcut is None and highcut is None:
        raise ValueError(
            "Need at least a low cutoff (lowcut) or high cutoff (highcut) frequency!"
        )
    elif lowcut is None and highcut is not None:
        filter_type = "lowpass"
        Wn = highcut / fs * 2
    elif lowcut is not None and highcut is None:
        filter_type = "highpass"
        Wn = lowcut / fs * 2
    else:
        if lowcut >= highcut:
            raise ValueError("lowcut must be smaller than highcut")
        filter_type = "bandpass"
        band = [lowcut, highcut]
        Wn = [e / fs * 2 for e in band]

    filter_coeff = signal.iirfilter(
        order, Wn, analog=False, btype=filter_type, output="sos"
    )
    filtered_traces = signal.sosfiltfilt(filter_coeff, data)
    return filtered_traces


def swap(ar, idxs):
    """
    Attempt one double-edge swap in a binary spike raster while preserving per-row and per-column sums.

    Parameters:
    ar (numpy.ndarray): Binary spike raster
    idxs (tuple): Tuple of numpy arrays containing the indices of the spikes

    Returns:
    success (bool): True if a swap was performed

    Notes:
    - The swap chooses two existing spike positions (i0, j0) and (i1, j1) and,
    if the off-diagonal positions (i0, j1) and (i1, j0) are both empty and the indices are distinct,
    swaps them so that spikes move to those positions.
    """
    idx0 = np.random.randint(len(idxs[0]))
    idx1 = np.random.randint(len(idxs[0]))
    i0, j0 = idxs[0][idx0], idxs[1][idx0]
    i1, j1 = idxs[0][idx1], idxs[1][idx1]
    if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
        return False
    ar[i0, j0] = ar[i1, j1] = 0.0
    ar[i0, j1] = ar[i1, j0] = 1.0
    idxs[0][idx0], idxs[1][idx0] = i0, j1
    idxs[0][idx1], idxs[1][idx1] = i1, j0
    return True


def randomize(ar, swap_per_spike=5):
    """
    Randomize a binary spike raster using degree-preserving double-edge swaps.

    Parameters:
    ar (array_like): Binary matrix shaped (neurons, time) or (time, neurons). Values should be 0/1.
    swap_per_spike (int): Target number of successful swaps per spike.

    Returns:
    randomized_raster (numpy.ndarray): Randomized binary matrix with the same shape and row/column sums.
    """
    ar = np.array(ar, dtype=float, copy=True)
    idxs = np.where(ar == 1.0)
    n_spikes = int(np.sum(ar))
    attempts = int((swap_per_spike + 1) * n_spikes)
    cnt_swap = 0
    for _ in range(attempts):
        if swap(ar, idxs):
            cnt_swap += 1

    if cnt_swap < swap_per_spike * n_spikes:
        for _ in range(attempts):
            if swap(ar, idxs):
                cnt_swap += 1

    if cnt_swap < swap_per_spike * n_spikes:
        print(
            "ERROR: Not sufficient succesfull swaps, only {} of {} required".format(
                cnt_swap, swap_per_spike * n_spikes
            )
        )

    return ar


def trough_between(i0, i1, pop_rate):
    """
    Helper function for get_bursts(). Finds the minimum value (trough) between two indices.

    Parameters:
    i0, i1 (int): Time bin indices of the bursts
    pop_rate (np.ndarray[float64]): Smoothed population spiking data in spikes per bin

    Returns:
    (int): Time bin index of minimum value (trough) between peaks
    """
    L, R = int(i0), int(i1)

    if R - L <= 1:
        return None

    seg = pop_rate[L:R]

    return L + int(np.argmin(seg))


def compute_cross_correlation_with_lag(ref_rate, comp_rate, max_lag=0):
    """
    Compute normalized cross correlation with lag information.



    Parameters:
    -----------
    - ref_rate (array): Reference firing rate signal
    - comp_rate (array): Comparison firing rate signal
    - max_lag (int): Maximum lag in frames to search for similarity.
                     If None, lag is set to 0.

    Returns:
    --------
    - max_corr (float): Maximum correlation coefficient
    - max_lag_idx (int): Lag (in frames) at which maximum correlation occurs
    """
    if max_lag is None:
        max_lag = 0

    # Fast path for zero lag (no time shift)
    if max_lag == 0:
        max_corr = np.sum(ref_rate * comp_rate) / np.sqrt(
            np.sum(ref_rate**2) * np.sum(comp_rate**2)
        )
        return max_corr, 0
    # r is the correlation between ref and comp. Each value is sum of elementwise products
    # for each possible lag and it is normalized so each value is between -1 and 1
    r = signal.correlate(ref_rate, comp_rate, mode="same") / np.sqrt(
        # Below is the normalziation method. You take signal's correaltion of itself, and
        # take the center value which is lag = 0, and use that for normalizing
        signal.correlate(ref_rate, ref_rate, mode="same")[int(len(ref_rate) / 2)]
        * signal.correlate(comp_rate, comp_rate, mode="same")[int(len(comp_rate) / 2)]
    )

    center = int(len(r) / 2)

    # Search within max_lag window
    search_start = max(0, center - max_lag)
    search_end = min(len(r), center + max_lag + 1)
    search_window = r[search_start:search_end]

    max_corr = np.max(search_window)
    max_lag_idx = np.argmax(search_window) + search_start - center

    return max_corr, max_lag_idx


def compute_cosine_similarity_with_lag(ref_rate, comp_rate, max_lag=0):
    """
    Compute cosine similarity with lag information.

    Parameters:
    -----------
    ref_rate (array): Reference firing rate signal
    comp_rate (array): Comparison firing rate signal
    max_lag (int): Maximum lag in frames to search for similarity.
                   If None, lag is set to 0.

    Returns:
    --------
    max_sim (float): Maximum cosine similarity coefficient
    max_lag_idx (int): Lag (in frames) at which maximum similarity occurs
    """
    from sklearn.metrics.pairwise import cosine_similarity

    ref_rate = np.array(ref_rate).flatten()
    comp_rate = np.array(comp_rate).flatten()

    # Handle None case (convert to 0)
    if max_lag is None:
        max_lag = 0

    if max_lag == 0:
        # Only check zero lag
        sim = cosine_similarity(ref_rate.reshape(1, -1), comp_rate.reshape(1, -1))[0, 0]
        return sim, 0
    lag_range = range(-max_lag, max_lag + 1)

    similarities = []
    valid_lags = []

    # Compute cosine similarity at each lag, and makes a case for negative, positive or no lag
    for lag in lag_range:
        if lag < 0:
            # comp_rate leads ref_rate (shift comp_rate left, or ref_rate right)
            ref_segment = ref_rate[-lag:]
            comp_segment = comp_rate[:lag]
        elif lag > 0:
            # ref_rate leads comp_rate (shift ref_rate left, or comp_rate right)
            ref_segment = ref_rate[:-lag]
            comp_segment = comp_rate[lag:]
        else:
            # No lag
            ref_segment = ref_rate
            comp_segment = comp_rate

        # Skip if segments are too short
        if len(ref_segment) > 0 and len(comp_segment) > 0:
            sim = cosine_similarity(
                ref_segment.reshape(1, -1), comp_segment.reshape(1, -1)
            )[0, 0]
            similarities.append(sim)
            valid_lags.append(lag)

    # Find maximum similarity and corresponding lag
    similarities = np.array(similarities)
    valid_lags = np.array(valid_lags)

    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    max_lag_idx = valid_lags[max_idx]

    return max_sim, max_lag_idx


def extract_lower_triangle_features(matrix_3d):
    """
    Extract lower triangle (excluding diagonal) from each correlation matrix in a 3D array.

    Parameters:
    -----------
    matrix_3d (array or PairwiseCompMatrixStack): 3D correlation matrix of shape (n, n, S) or PairwiseCompMatrixStack object.
        Where n is the matrix dimension and S is the number of slices/samples.

    Returns:
    --------
    features (array): 2D matrix of shape (S, F) each row contains lower triangle values for that correlation matrix
                      F = n*(n-1)/2 (number of unique pairs or more simply the number of values in lower triangle)
    """
    # Handle structured types
    if hasattr(matrix_3d, "stack") and isinstance(matrix_3d.stack, np.ndarray):
        matrix_3d = matrix_3d.stack

    if matrix_3d.ndim != 3:
        raise ValueError(f"Input must be a 3D array (or stack), got {matrix_3d.ndim}D")

    if matrix_3d.shape[0] != matrix_3d.shape[1]:
        raise ValueError(
            "The input 3D matrix must have shape (n, n, S) where the first two dimensions are equal."
        )
    num_items = matrix_3d.shape[0]  # n
    num_samples = matrix_3d.shape[2]  # S

    # Get lower triangle indices
    lower_tri_idx = np.tril_indices(num_items, k=-1)

    # Extract all lower triangles at once (vectorized)
    # matrix_3d[lower_tri_idx[0], lower_tri_idx[1], :] gives shape (F, S), transpose to (S, F)
    features = matrix_3d[lower_tri_idx[0], lower_tri_idx[1], :].T
    return features


def PCA_reduction(matrix_2d, n_components=2):
    """
    Compute PCA dimensionality reduction on axis 1 of a 2d matrix

    Parameters:
    -----------
    matrix_2d (array): 2D matrix where values must be int, float, or bool

    Returns:
    --------
    pca_result (array): 2D matrix of shape (rows, n_components)
    """

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(matrix_2d)

    return pca_result


<<<<<<< HEAD
<<<<<<< HEAD
def ensure_h5py():
    """Ensure h5py is available for HDF5-based exporters."""
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5/NWB exporters. `pip install h5py`."
        )


def times_from_ms(
    times_ms: np.ndarray, unit: TimeUnit, fs_Hz: Optional[float]
) -> Union[np.ndarray, float, int]:
    """Convert times from milliseconds to the requested unit."""
    if unit == "ms":
        return times_ms.astype(float)
    if unit == "s":
        return times_ms.astype(float) / 1e3
    if unit == "samples":
        if not fs_Hz or fs_Hz <= 0:
            raise ValueError("fs_Hz must be provided and > 0 when unit='samples'")
        # Use round-to-nearest to produce integer samples
        return np.rint(times_ms.astype(float) * (fs_Hz / 1e3)).astype(int)
    raise ValueError(f"Unknown time unit '{unit}' (expected 's','ms','samples')")


def to_ms(values: np.ndarray, unit: str, fs_Hz: Optional[float]) -> np.ndarray:
    """Convert a vector of times to milliseconds."""
    if unit == "ms":
        return values.astype(float)
    if unit == "s":
        return values.astype(float) * 1e3
    if unit == "samples":
        if not fs_Hz or fs_Hz <= 0:
            raise ValueError("fs_Hz must be provided and > 0 when unit='samples'")
        return values.astype(float) / fs_Hz * 1e3
    raise ValueError(f"Unknown time unit '{unit}' (expected 's','ms','samples')")
=======

=======
>>>>>>> 5b275ed (add test for neuronattributes)
def check_neuron_attributes(
    neuron_attributes: List[dict], n_neurons: Optional[int] = None
) -> List[dict]:
    """
    Check a list of dictionaries for use as neuron_attributes to verify that keys and values are consistent.

    Parameters:
        neuron_attributes: List of dictionaries containing neuron attributes.
        n_neurons: Expected number of neurons. If provided, validates the list length.

    Returns:
        A list of dictionaries where all dictionaries have valid keys and values.

    Notes:
    - If some dictionaries are missing keys that others have, a ValueError is raised
      describing the inconsistent keys.
    """
    if not isinstance(neuron_attributes, list):
        raise ValueError("neuron_attributes must be a list")
    if n_neurons is not None and len(neuron_attributes) != n_neurons:
        raise ValueError(
            f"neuron_attributes has {len(neuron_attributes)} items, expected {n_neurons}"
        )
    for i, attr in enumerate(neuron_attributes):
        if not isinstance(attr, dict):
            raise ValueError(f"neuron_attributes[{i}] must be a dict")

    if not neuron_attributes:
        return []

    all_keys = set().union(*(attr.keys() for attr in neuron_attributes))
    if not all_keys:
        return [d.copy() for d in neuron_attributes]

    missing = {
        i: all_keys - attr.keys()
        for i, attr in enumerate(neuron_attributes)
        if attr.keys() != all_keys
    }
    if missing:
        parts = [f"Neuron {i} missing: {keys}" for i, keys in sorted(missing.items())]
        raise ValueError(f"Inconsistent neuron_attributes keys. {'; '.join(parts)}.")

    return [{key: attr.get(key) for key in all_keys} for attr in neuron_attributes]
<<<<<<< HEAD
>>>>>>> d2e0e10 (add neuron attributes checker to neuron atts)
=======


def extract_waveforms(
    raw_data: np.ndarray,
    spike_times_ms: np.ndarray,
    fs_kHz: float,
    ms_before: float = 1.0,
    ms_after: float = 2.0,
    channel_indices: Optional[List[int]] = None,
    bandpass: Optional[tuple] = None,
    filter_order: int = 3,
) -> np.ndarray:
    """
    Extract waveform snippets from raw data at specified spike times.

    Parameters:
        raw_data: Raw voltage data with shape (num_channels, num_samples).
        spike_times_ms: Array of spike times in milliseconds.
        fs_kHz: Sampling rate in kHz.
        ms_before: Milliseconds before each spike time (default: 1.0).
        ms_after: Milliseconds after each spike time (default: 2.0).
        channel_indices: Channel indices to extract. If None, extracts all.
        bandpass: Optional (lowcut_Hz, highcut_Hz) for bandpass filtering.
        filter_order: Butterworth filter order (default: 3).

    Returns:
        3D array (num_channels, num_samples, num_spikes). Empty if no valid spikes.
    """
    if raw_data.size == 0:
        raise ValueError("raw_data is empty")

    n_channels_total, n_time_samples = raw_data.shape

    if channel_indices is None:
        channel_indices = list(range(n_channels_total))
    n_channels = len(channel_indices)

    before_samples = round(ms_before * fs_kHz)
    after_samples = round(ms_after * fs_kHz)
    n_samples = before_samples + after_samples

    if bandpass is not None:
        lowcut, highcut = bandpass
        data_to_extract = butter_filter(
            raw_data,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs_kHz * 1000,
            order=filter_order,
        )
    else:
        data_to_extract = raw_data

    if len(spike_times_ms) == 0:
        return np.zeros((n_channels, n_samples, 0), dtype=raw_data.dtype)

    waveforms = []
    for spike_time_ms in spike_times_ms:
        spike_sample = round(spike_time_ms * fs_kHz)
        start = spike_sample - before_samples
        end = spike_sample + after_samples

        if start < 0 or end > n_time_samples:
            continue

        waveforms.append(data_to_extract[channel_indices, start:end])

    if len(waveforms) == 0:
        return np.zeros((n_channels, n_samples, 0), dtype=raw_data.dtype)

    return np.array(waveforms).transpose(1, 2, 0)
>>>>>>> 4e66959 (Enhanced get_traces with bandpass filtering, storage, and improved API)
