import math
import warnings
from typing import Optional, List, Literal, Union, Dict, Any

import numpy as np
from itertools import groupby as _groupby

from scipy import ndimage, signal
from scipy.stats import norm

__all__ = [
    "get_sttc",
    "swap",
    "randomize",
    "trough_between",
    "TimeUnit",
    "ensure_h5py",
    "times_from_ms",
    "to_ms",
    "extract_waveforms",
    "check_neuron_attributes",
    "get_channels_for_unit",
    "compute_avg_waveform",
    "get_valid_spike_times",
    "waveforms_by_channel",
    "extract_unit_waveforms",
    "consecutive_durations",
    "gplvm_state_entropy",
    "gplvm_continuity_prob",
    "gplvm_average_state_probability",
    "shuffle_z_score",
    "shuffle_percentile",
    "slice_trend",
    "slice_stability",
]
TimeUnit = Literal["ms", "s", "samples"]

try:  # optional, only needed for HDF5/NWB exporters
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

# Optional dependencies for manifold learning and graph-based clustering.
try:  # optional, only needed for UMAP-based reductions
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None  # type: ignore

try:  # optional, only needed for graph/community detection
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

try:  # optional, only needed for Louvain community detection
    import community as community_louvain  # type: ignore
except ImportError:  # pragma: no cover
    community_louvain = None  # type: ignore


def get_sttc(
    tA, tB, delt=20.0, length: Optional[float] = None, start_time: float = 0.0
):
    """
    Calculate the spike time tiling coefficient between two spike trains.

    Formula:
    STTC = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0 + (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0

    Parameters:
    tA (list): List of spike times for the first spike train.
    tB (list): List of spike times for the second spike train.
    delt (float): Time window in milliseconds (default: 20.0).
    length (float): Total duration in milliseconds (optional). If None,
        inferred from the latest spike time after shifting.
    start_time (float): Time origin of the spike trains (default 0.0).
        Spike times are shifted by ``-start_time`` before computation so
        that the STTC edge corrections work correctly for event-centered
        data with negative spike times.

    Returns:
    sttc (float): Spike time tiling coefficient between the two spike trains.

    [1] Cutts & Eglen. Detecting pairwise correlations in spike trains: An objective
        comparison of methods and application to the study of retinal waves. Journal of
        Neuroscience 34:43, 14288–14303 (2014).
    """
    if delt <= 0:
        raise ValueError(f"delt must be positive, got {delt}")

    if len(tA) == 0 or len(tB) == 0:
        return 0.0

    # Shift both trains by -start_time so they are 0-based. This ensures
    # _sttc_ta edge corrections work correctly for event-centered data.
    tA = np.asarray(tA, dtype=float) - start_time
    tB = np.asarray(tB, dtype=float) - start_time

    if length is None:
        length = float(max(tA[-1], tB[-1]))

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

    base = min(delt, tA[0]) + min(delt, max(0, tmax - tA[-1]))
    return base + np.minimum(np.diff(tA), 2 * delt).sum()


def _sttc_na(tA, tB, delt: float) -> int:
    """Helper function for STTC: Calculate the number of spikes in tA within delt of any spike in tB."""
    if len(tB) == 0:
        return 0
    tA, tB = np.asarray(tA), np.asarray(tB)

    if len(tB) == 1:
        return int((np.abs(tA - tB[0]) <= delt).sum())

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
    fr (numpy.ndarray): Firing rate at specific times. Same size as times

    Notes:
    - Assumed to have been sampled halfway between any two given spikes, interpolated, and then
    smoothed by a Gaussian kernel with the given width.
    """

    if len(spikes) == 0 or len(spikes) == 1:
        # Need at least 2 spikes to do get inter-spike interval
        return np.zeros_like(times)
    if len(times) < 2:
        raise ValueError("times has less than 2 values. Input more times")

    spikes = np.array(spikes)
    times = np.array(times)

    # Remove duplicate spike times (BUG-002)
    unique_spikes = np.unique(spikes)
    if len(unique_spikes) < len(spikes):
        warnings.warn(
            f"{len(spikes) - len(unique_spikes)} duplicate spike time(s) removed "
            f"before ISI computation.",
            RuntimeWarning,
        )
        spikes = unique_spikes
    if len(spikes) < 2:
        return np.zeros_like(times)

    # Reject duplicate time grid values (BUG-003)
    if len(np.unique(times)) < len(times):
        raise ValueError(
            "times array contains duplicate values. "
            "Provide an evenly-spaced grid with unique time points."
        )

    # Compute inter spike intervals (piece 1 logic)
    isi = np.diff(spikes)
    isi = np.insert(isi, 0, 0)  # Add spacer for first spike

    # Compute instantaneous firing rates (1/isi, in Hz assuming ms units)
    isi_rate = np.zeros_like(isi, dtype=float)
    isi_rate[1:] = 1.0 / isi[1:] * 1000

    # Create temporary result array matching times resolution
    t_start, t_end = times[0], times[-1]
    dt_ms = times[1] - times[0]
    n_bins = int(round((t_end - t_start) / dt_ms)) + 1
    isi_rate_temp = np.zeros(n_bins)

    # Assign rates to bins between spikes (piece 1 logic)
    for i in range(1, len(spikes)):
        start_bin = int(round((spikes[i - 1] - t_start) / dt_ms))
        end_bin = int(round((spikes[i] - t_start) / dt_ms))
        if start_bin < n_bins:
            isi_rate_temp[start_bin : min(end_bin, n_bins)] = isi_rate[i]

    # Interpolate to exact times grid (if needed)
    fr = np.interp(times, t_start + dt_ms * np.arange(n_bins), isi_rate_temp)

    # Apply Gaussian smoothing
    if len(fr) < 2:
        return fr

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


def swap(ar, idxs, rng):
    """
    Attempt one double-edge swap in a binary spike raster while preserving per-row and per-column sums.

    Parameters:
    -----------
    - ar (numpy.ndarray): Binary spike raster
    - idxs (tuple): Tuple of numpy arrays containing the indices of the spikes
    - rng (numpy.random.Generator): Random number generator for reproducibility.

    Returns:
    --------
    - success (bool): True if a swap was performed

    Notes:
    ------
    - The swap chooses two existing spike positions (i0, j0) and (i1, j1) and,
    if the off-diagonal positions (i0, j1) and (i1, j0) are both empty and the indices are distinct,
    swaps them so that spikes move to those positions.
    """
    # idx0 = np.random.randint(len(idxs[0]))
    # idx1 = np.random.randint(len(idxs[0]))
    idx0 = rng.integers(len(idxs[0]))
    idx1 = rng.integers(len(idxs[0]))
    i0, j0 = idxs[0][idx0], idxs[1][idx0]
    i1, j1 = idxs[0][idx1], idxs[1][idx1]
    if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
        return False
    ar[i0, j0] = ar[i1, j1] = 0.0
    ar[i0, j1] = ar[i1, j0] = 1.0
    idxs[0][idx0], idxs[1][idx0] = i0, j1
    idxs[0][idx1], idxs[1][idx1] = i1, j0
    return True


def randomize(ar, swap_per_spike=5, seed=None):
    """
    Randomize a binary spike raster using degree-preserving double-edge swaps.

    Parameters:
    -----------
    - ar (array_like): Binary matrix shaped (neurons, time) or (time, neurons). Values should be 0/1.
    - swap_per_spike (int): Target number of successful swaps per spike.
    - seed (int): This is the random seed number. If you want repeatability during experiments, set the seed number.

    Returns:
    --------
    -randomized_raster (numpy.ndarray): Randomized binary matrix with the same shape and row/column sums.

    Notes:
    ------
    - Shuffling is done in a manner where each neuron's average firing rate is preserved, but the specific time_bin in it spikes is shuffled.
    - Shuffling is done in a manner where each time bin's population rate is preserved, but the specific units active in each time bin are shuffled.
    - Ever spike swap involves 2 different spikes so on average, ever spike will get swapped 2*swap_per_spike times

    Ref:
    ----
    - Okun, M. et al. Population rate dynamics and multineuron firing patterns in sensory cortex. J. Neurosci. 32, 17108–17119 (2012)
    """
    rng = np.random.default_rng(seed)

    ar = np.array(ar, dtype=float, copy=True)
    unique_vals = np.unique(ar)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            "randomize() requires a binary (0/1) raster. "
            f"Found values: {unique_vals}"
        )
    idxs = np.where(ar == 1.0)
    n_spikes = int(np.sum(ar))
    attempts = int((swap_per_spike + 1) * n_spikes)
    cnt_swap = 0
    for _ in range(attempts):
        if swap(ar, idxs, rng):
            cnt_swap += 1

    if cnt_swap < swap_per_spike * n_spikes:
        for _ in range(attempts):
            if swap(ar, idxs, rng):
                cnt_swap += 1

    if cnt_swap < swap_per_spike * n_spikes:
        warnings.warn(
            "Not sufficient successful swaps, only {} of {} required".format(
                cnt_swap, swap_per_spike * n_spikes
            ),
            RuntimeWarning,
        )

    return ar.astype(int)


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

    # Handle zero-norm vectors:
    # - Both zero → undefined (NaN)
    # - One zero, one not → uncorrelated (0.0)
    ref_norm = np.sum(ref_rate**2)
    comp_norm = np.sum(comp_rate**2)
    if ref_norm == 0 and comp_norm == 0:
        return np.nan, 0
    if ref_norm == 0 or comp_norm == 0:
        return 0.0, 0
    norm_product = ref_norm * comp_norm

    # Fast path for zero lag (no time shift)
    if max_lag == 0:
        max_corr = np.sum(ref_rate * comp_rate) / np.sqrt(norm_product)
        return max_corr, 0
    # r is the correlation between ref and comp. Each value is sum of elementwise products
    # for each possible lag and it is normalized so each value is between -1 and 1
    # Normalization: autocorrelation at zero lag for each signal
    auto_ref = signal.correlate(ref_rate, ref_rate, mode="same")[len(ref_rate) // 2]
    auto_comp = signal.correlate(comp_rate, comp_rate, mode="same")[len(comp_rate) // 2]
    denom = auto_ref * auto_comp
    if denom <= 0:
        return 0.0, 0
    r = signal.correlate(ref_rate, comp_rate, mode="same") / np.sqrt(denom)

    center = len(r) // 2

    # Search within max_lag window
    search_start = max(0, center - max_lag)
    search_end = min(len(r), center + max_lag + 1)
    search_window = r[search_start:search_end]

    max_corr = np.max(search_window)
    max_lag_idx = np.argmax(search_window) + search_start - center

    return max_corr, max_lag_idx


def _cosine_sim(a, b):
    """Cosine similarity between two 1-D vectors. NaN if both zero-norm, 0.0 if one is."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 and norm_b == 0.0:
        return np.nan
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


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
    ref_rate = np.array(ref_rate).flatten()
    comp_rate = np.array(comp_rate).flatten()

    # Handle None case (convert to 0)
    if max_lag is None:
        max_lag = 0

    if max_lag == 0:
        # Only check zero lag
        return _cosine_sim(ref_rate, comp_rate), 0
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
            similarities.append(_cosine_sim(ref_segment, comp_segment))
            valid_lags.append(lag)

    # Find maximum similarity and corresponding lag
    similarities = np.array(similarities)
    valid_lags = np.array(valid_lags)

    if np.all(np.isnan(similarities)):
        return np.nan, 0

    max_idx = np.nanargmax(similarities)
    max_sim = similarities[max_idx]
    max_lag_idx = valid_lags[max_idx]

    return max_sim, max_lag_idx


def PCA_reduction(matrix_2d, n_components=2):
    """
    Compute PCA dimensionality reduction on axis 1 of a 2d matrix.

    Parameters:
    -----------
    matrix_2d (array): 2D matrix of shape (samples, features) where values
        must be int, float, or bool.
    n_components (int): Number of principal components to retain (default: 2).

    Returns:
    --------
    embedding (array): 2D matrix of shape (samples, n_components).
    explained_variance_ratio (array): 1D array of shape (n_components,) with the
        fraction of total variance explained by each component.
    components (array): 2D matrix of shape (n_components, features) with the
        principal axes (loadings) — each row is one PC expressed in the
        original feature space.
    """

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "PCA_reduction requires the optional dependency 'scikit-learn'. "
            "Install it with `pip install scikit-learn`."
        )

    max_components = min(matrix_2d.shape)
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds "
            f"min(n_samples, n_features)={max_components}"
        )

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(matrix_2d)

    return embedding, pca.explained_variance_ratio_, pca.components_


def _clamp_umap_n_neighbors(n_samples: int, n_neighbors: int) -> int:
    """
    umap-learn requires n_neighbors strictly less than n_samples / 2.
    Clamp user/default values so small datasets do not raise at fit time.
    """
    if n_samples < 2:
        return 1
    max_nn = max(1, int(math.ceil(n_samples / 2)) - 1)
    return min(max(int(n_neighbors), 2), max_nn)


def UMAP_reduction(
    matrix_2d,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
    **umap_kwargs: Any,
):
    """
    Compute UMAP dimensionality reduction on a 2D matrix.

    Parameters
    ----------
    matrix_2d : array-like, shape (n_samples, n_features)
        Input data. Each row is a sample, each column is a feature.

    n_components : int, default=2
        Dimension of the embedded space.

    n_neighbors : int, default=15
        Size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation.

    min_dist : float, default=0.1
        Controls how tightly UMAP packs points together in the low-dimensional space.

    metric : str, default="euclidean"
        Distance metric used in the input space.

    random_state : int or None, default=None
        Random seed for reproducibility.

    **umap_kwargs :
        Additional keyword arguments passed to ``umap.UMAP``.

    Returns
    -------
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional embedding of the data.
    trustworthiness_score : float
        Trustworthiness of the embedding (0 to 1). Measures how well local
        neighborhoods in the high-dimensional space are preserved in the
        embedding. Requires scikit-learn; returns NaN if unavailable.
    """
    if umap is None:
        raise ImportError(
            "UMAP_reduction requires the optional dependency 'umap-learn'. "
            "Install it with `pip install umap-learn`."
        )

    matrix_2d = np.asarray(matrix_2d)
    n_neighbors = _clamp_umap_n_neighbors(matrix_2d.shape[0], n_neighbors)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **umap_kwargs,
    )
    embedding = reducer.fit_transform(matrix_2d)

    try:
        from sklearn.manifold import trustworthiness

        tw = float(trustworthiness(matrix_2d, embedding, n_neighbors=n_neighbors))
    except ImportError:
        tw = float("nan")

    return embedding, tw


def UMAP_graph_communities(
    matrix_2d,
    n_components: int = 2,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
    **umap_kwargs: Any,
):
    """
    Run UMAP and Louvain community detection on the UMAP connectivity graph.

    This helper keeps UMAP_reduction simple while providing an optional
    graph-based clustering approach that builds on UMAP's internal graph.

    Parameters
    ----------
    matrix_2d : array-like, shape (n_samples, n_features)
        Input data. Each row is a sample, each column is a feature.

    n_components : int, default=2
        Dimension of the embedded space.

    resolution : float, default=1.0
        Resolution parameter for the Louvain community detection algorithm.
        Higher values -> more, smaller communities. Lower values -> fewer,
        larger communities.

    n_neighbors, min_dist, metric, random_state, **umap_kwargs :
        Passed through to the underlying UMAP_reduction / umap.UMAP.

    Returns
    -------
    embedding : ndarray, shape (n_samples, n_components)
        Low-dimensional UMAP embedding.

    labels : ndarray, shape (n_samples,)
        Integer community label for each sample.

    trustworthiness_score : float
        Trustworthiness of the embedding (0 to 1). Returns NaN if
        scikit-learn is not available.
    """
    # First compute the UMAP embedding and fitted mapper using the same
    # configuration as UMAP_reduction.
    if umap is None:
        raise ImportError(
            "UMAP_graph_communities requires the optional dependency 'umap-learn'. "
            "Install it with `pip install umap-learn`."
        )
    if nx is None:
        raise ImportError(
            "UMAP_graph_communities requires the optional dependency 'networkx'. "
            "Install it with `pip install networkx`."
        )
    if community_louvain is None:
        raise ImportError(
            "UMAP_graph_communities requires the optional dependency "
            "'python-louvain'. Install it with `pip install python-louvain`."
        )

    matrix_2d = np.asarray(matrix_2d)
    n_neighbors = _clamp_umap_n_neighbors(matrix_2d.shape[0], n_neighbors)

    mapper = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **umap_kwargs,
    ).fit(matrix_2d)

    # UMAP's internal connectivity graph -> NetworkX graph
    # Use a compatibility shim so both old and new NetworkX versions work.
    if hasattr(nx, "from_scipy_sparse_array"):
        G = nx.from_scipy_sparse_array(mapper.graph_)
    else:
        G = nx.from_scipy_sparse_matrix(mapper.graph_)

    # Louvain community detection on the graph
    clustering = community_louvain.best_partition(G, resolution=resolution)

    # Convert dict {node_idx: community_id} -> label array
    # Use the fitted mapper's embedding to determine n_samples so that
    # callers can pass in any array-like that UMAP accepts (not just ndarrays).
    n_samples = mapper.embedding_.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    for node, c_id in clustering.items():
        labels[node] = c_id

    try:
        from sklearn.manifold import trustworthiness

        tw = float(
            trustworthiness(matrix_2d, mapper.embedding_, n_neighbors=n_neighbors)
        )
    except ImportError:
        tw = float("nan")

    return mapper.embedding_, labels, tw


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
      indicating which neuron entries have inconsistent keys.
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


def get_channels_for_unit(
    unit_idx: int,
    channels: Optional[Union[int, List[int]]],
    neuron_to_channel: dict,
    n_channels_total: int,
) -> List[int]:
    """
    Determine which channels to extract for a given unit.

    Parameters:
        unit_idx: Index of the unit.
        channels: Channel specification. None uses neuron_to_channel_map or all channels;
                  int for single channel; list for multiple; empty list for mapped channel.
        neuron_to_channel: Mapping from unit indices to channel indices.
        n_channels_total: Total number of channels in the raw data.

    Returns:
        List of channel indices to extract.

    Raises:
        ValueError: If channels argument is invalid type.
    """
    if channels is None:
        if unit_idx in neuron_to_channel:
            return [neuron_to_channel[unit_idx]]
        return list(range(n_channels_total))
    elif isinstance(channels, int):
        return [channels]
    elif isinstance(channels, list):
        if len(channels) == 0:
            if unit_idx in neuron_to_channel:
                return [neuron_to_channel[unit_idx]]
            return list(range(n_channels_total))
        return channels
    raise ValueError(f"Invalid channels argument: {channels}")


def compute_avg_waveform(
    waveforms: np.ndarray,
    channel_indices: List[int],
    dtype: np.dtype,
) -> np.ndarray:
    """
    Compute the average waveform from extracted waveforms.

    Parameters:
        waveforms: 3D array of shape (num_channels, num_samples, num_spikes).
        channel_indices: List of channel indices used for extraction.
        dtype: Data type for the output array if waveforms is empty.

    Returns:
        2D array of shape (num_channels, num_samples) containing the average waveform.
    """
    if waveforms.shape[2] > 0:
        return waveforms.mean(axis=2)
    else:
        return np.zeros(
            (len(channel_indices), waveforms.shape[1]),
            dtype=dtype,
        )


def get_valid_spike_times(
    spike_times_ms: np.ndarray,
    fs_kHz: float,
    ms_before: float,
    ms_after: float,
    n_time_samples: int,
) -> np.ndarray:
    """
    Filter spike times to only those within valid bounds of the raw data.

    Parameters:
        spike_times_ms: Array of spike times in milliseconds.
        fs_kHz: Sampling rate in kHz.
        ms_before: Milliseconds before each spike time.
        ms_after: Milliseconds after each spike time.
        n_time_samples: Total number of time samples in the raw data.

    Returns:
        Array of valid spike times in milliseconds.
    """
    before_samples = round(ms_before * fs_kHz)
    after_samples = round(ms_after * fs_kHz)
    valid_spike_times = []
    for spike_time_ms in spike_times_ms:
        spike_sample = round(spike_time_ms * fs_kHz)
        start = spike_sample - before_samples
        end = spike_sample + after_samples
        if start >= 0 and end <= n_time_samples:
            valid_spike_times.append(spike_time_ms)
    return np.array(valid_spike_times)


def waveforms_by_channel(
    waveforms: np.ndarray, channel_indices: List[int]
) -> Dict[int, np.ndarray]:
    """
    Convert a waveform stack into a per-channel dict.

    Parameters:
        waveforms: 3D array shaped (num_channels, num_samples, num_spikes).
        channel_indices: List of channel indices corresponding to waveforms axis 0.

    Returns:
        Dict mapping channel index -> 2D array shaped (num_samples, num_spikes).
    """
    if waveforms.ndim != 3:
        raise ValueError(f"waveforms must be 3D, got shape {waveforms.shape}")
    if len(channel_indices) != waveforms.shape[0]:
        raise ValueError(
            "channel_indices length must match waveforms.shape[0] "
            f"({len(channel_indices)} != {waveforms.shape[0]})"
        )
    # Note: waveforms[ch_i] is (num_samples, num_spikes) for that channel.
    return {ch: waveforms[i, :, :] for i, ch in enumerate(channel_indices)}


def extract_unit_waveforms(
    unit_idx: int,
    spike_times_ms: np.ndarray,
    raw_data: np.ndarray,
    fs_kHz: float,
    ms_before: float,
    ms_after: float,
    channels: Optional[Union[int, List[int]]],
    neuron_to_channel: dict,
    bandpass: Optional[tuple] = None,
    filter_order: int = 3,
    return_channel_waveforms: bool = False,
    return_avg_waveform: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract waveforms and compute statistics for a single unit.

    This function orchestrates the full waveform extraction pipeline:
    1. Resolves which channels to extract based on user input and neuron mapping
    2. Extracts raw voltage snippets around each spike time
    3. Computes the mean waveform across all spikes
    4. Filters spike times to only those with valid extraction windows

    Parameters:
        unit_idx: Index of the unit being extracted.
        spike_times_ms: Array of spike times in milliseconds for this unit.
        raw_data: Raw voltage data with shape (num_channels, num_samples).
        fs_kHz: Sampling rate in kHz.
        ms_before: Milliseconds before each spike time.
        ms_after: Milliseconds after each spike time.
        channels: Channel specification. None uses neuron_to_channel mapping or all
                  channels; int for single channel; list for multiple; empty list
                  for mapped channel.
        neuron_to_channel: Mapping from unit indices to channel indices.
        bandpass: Optional (lowcut_Hz, highcut_Hz) for bandpass filtering.
        filter_order: Butterworth filter order (default: 3).

    Returns:
        (waveforms, meta) where:
            - waveforms: 3D array (num_channels, num_samples, num_spikes)
            - meta: dict containing per-unit metadata (no raw waveforms):
                - channels: List[int] of channel indices used
                - spike_times_ms: np.ndarray of valid spike times
                - avg_waveform: 2D array (num_channels, num_samples), or None if disabled
                - channel_waveforms: Optional dict[channel -> (num_samples, num_spikes)]
    """
    n_channels_total = raw_data.shape[0]
    n_time_samples = raw_data.shape[1]

    # Resolve which channels to extract based on user input and neuron mapping
    # Priority: explicit channels arg > neuron_to_channel mapping > all channels
    channel_indices = get_channels_for_unit(
        unit_idx, channels, neuron_to_channel, n_channels_total
    )

    # Extract raw voltage snippets around each spike time (num_channels, num_samples, num_spikes)
    waveforms = extract_waveforms(
        raw_data=raw_data,
        spike_times_ms=spike_times_ms,
        fs_kHz=fs_kHz,
        ms_before=ms_before,
        ms_after=ms_after,
        channel_indices=channel_indices,
        bandpass=bandpass,
        filter_order=filter_order,
    )

    # Compute mean waveform across spikes if requested.
    # Note: this mean is across spikes (axis=2), not across channels.
    avg_waveform = (
        compute_avg_waveform(waveforms, channel_indices, raw_data.dtype)
        if return_avg_waveform
        else None
    )

    # Filter spike times to only those with valid extraction windows
    # (i.e., spikes not too close to recording start/end)
    valid_spike_times = get_valid_spike_times(
        spike_times_ms, fs_kHz, ms_before, ms_after, n_time_samples
    )

    meta: Dict[str, Any] = {
        "channels": channel_indices,
        "spike_times_ms": valid_spike_times,
        "avg_waveform": avg_waveform,
    }

    # Optionally provide a per-channel view for convenience:
    # channel -> (num_samples, num_spikes)
    if return_channel_waveforms:
        meta["channel_waveforms"] = waveforms_by_channel(waveforms, channel_indices)

    return waveforms, meta


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


def consecutive_durations(signal, threshold, mode="above", min_dur=1):
    """
    Compute the lengths of consecutive runs in a 1-D signal that satisfy a threshold condition.

    Scans *signal* for contiguous stretches of bins that are above (>=) or
    below (<) *threshold*, returns an array of their durations, and optionally
    filters out runs shorter than *min_dur*.

    Parameters:
        signal (array_like): 1-D numeric array (e.g. continuity probability
            time series from a GPLVM).
        threshold (float): Threshold value for the condition.
        mode (str): ``"above"`` keeps runs where ``signal >= threshold``;
            ``"below"`` keeps runs where ``signal < threshold``.
        min_dur (int): Minimum run length to keep. Runs shorter than this
            are discarded.

    Returns:
        durations (np.ndarray): 1-D integer array of run lengths that satisfy
            the condition and are at least *min_dur* bins long. May be empty.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")

    if mode == "above":
        condition = signal >= threshold
    elif mode == "below":
        condition = signal < threshold
    else:
        raise ValueError("mode must be 'above' or 'below'")

    # Compute lengths of consecutive True runs
    durations = np.array(
        [sum(1 for _ in group) for key, group in _groupby(condition) if key],
        dtype=int,
    )

    if durations.size > 0:
        durations = durations[durations >= min_dur]

    return durations


def gplvm_state_entropy(posterior_latent_marg):
    """
    Compute Shannon entropy of the latent state distribution at each time bin.

    Parameters:
        posterior_latent_marg (np.ndarray): Marginal posterior over latent
            states with shape ``(T, K)`` where *T* is the number of time bins
            and *K* is the number of latent states. Typically obtained from
            ``SpikeData.fit_gplvm()["decode_res"]["posterior_latent_marg"]``.

    Returns:
        entropy (np.ndarray): 1-D array of shape ``(T,)`` with the Shannon
            entropy (in nats) for each time bin.
    """
    from scipy.stats import entropy as _entropy

    posterior_latent_marg = np.asarray(posterior_latent_marg)
    if posterior_latent_marg.ndim != 2:
        raise ValueError(
            f"posterior_latent_marg must be 2-D (T, K), got shape "
            f"{posterior_latent_marg.shape}"
        )
    return _entropy(posterior_latent_marg, axis=1)


def gplvm_continuity_prob(decode_res):
    """
    Extract the continuity (non-jump) probability time series from a GPLVM decode result.

    The continuity probability at each time bin is the marginal posterior
    probability that the dynamics remained continuous (i.e. did not jump)
    between the previous and current time bin.

    Parameters:
        decode_res (dict): Decoded latent state dictionary as returned by
            ``SpikeData.fit_gplvm()["decode_res"]``. Must contain the key
            ``"posterior_dynamics_marg"`` with shape ``(T, D)`` where the
            first column (index 0) holds the continuity probability.

    Returns:
        continuity_prob (np.ndarray): 1-D array of shape ``(T,)`` with the
            continuity probability at each time bin.
    """
    if not isinstance(decode_res, dict):
        raise TypeError("decode_res must be a dict from SpikeData.fit_gplvm()")
    if "posterior_dynamics_marg" not in decode_res:
        raise KeyError(
            "decode_res must contain 'posterior_dynamics_marg'. "
            "Pass the 'decode_res' dict from SpikeData.fit_gplvm()."
        )
    dynamics = np.asarray(decode_res["posterior_dynamics_marg"])
    if dynamics.ndim != 2 or dynamics.shape[1] < 1:
        raise ValueError(
            f"posterior_dynamics_marg must be 2-D with at least 1 column, "
            f"got shape {dynamics.shape}"
        )
    return dynamics[:, 0]


def gplvm_average_state_probability(posterior_latent_marg):
    """
    Compute the average probability of each latent state across all time bins.

    Parameters:
        posterior_latent_marg (np.ndarray): Marginal posterior over latent
            states with shape ``(T, K)`` where *T* is the number of time bins
            and *K* is the number of latent states. Typically obtained from
            ``SpikeData.fit_gplvm()["decode_res"]["posterior_latent_marg"]``.

    Returns:
        avg_prob (np.ndarray): 1-D array of shape ``(K,)`` with the mean
            probability of each latent state, averaged over all time bins.
    """
    posterior_latent_marg = np.asarray(posterior_latent_marg)
    if posterior_latent_marg.ndim != 2:
        raise ValueError(
            f"posterior_latent_marg must be 2-D (T, K), got shape "
            f"{posterior_latent_marg.shape}"
        )
    return np.mean(posterior_latent_marg, axis=0)


def _get_attr(obj, key, default):
    """Get an attribute from a dict-like or object-like neuron attribute entry."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _validate_time_start_to_end(
    times_start_to_end, warn_negative_start=False, recording_range=None
):
    """
    Validates that the list of (start, end) tuples has the same duration and is in
    proper format for the object constructor.

    Parameters:
    -----------
    times_start_to_end (list): Each entry must be a tuple (start, end).
    warn_negative_start (bool): If True, emit a warning for windows with
        negative start times (default False). Useful when times are expected
        to be absolute recording positions.
    recording_range (tuple or None): If provided, a ``(rec_start, rec_end)``
        tuple defining the valid time range. Any window that extends outside
        this range raises ``ValueError``. If None (default), no range check
        is performed.

    Returns:
    --------
    valid_time_tuples (list): Sorted list of valid (start, end) tuples.
                                Negative-start windows are preserved.
    """
    if not isinstance(times_start_to_end, list):
        raise TypeError("times must be a list of tuples")
    time_diff_check = []
    valid_time_tuples = []
    times_start_to_end = sorted(times_start_to_end)
    for i, time_window in enumerate(times_start_to_end):
        if not isinstance(time_window, tuple):
            raise TypeError(f"Element {i} of times is not a tuple: {time_window}")
        if len(time_window) != 2:
            raise TypeError(
                f"Element {i} of times must be a tuple of length 2 (start, end): "
                f"{time_window}"
            )
        if not (
            isinstance(time_window[0], (int, float, np.number))
            and isinstance(time_window[1], (int, float, np.number))
        ):
            raise TypeError(
                f"Start and end times in element {i} must be numbers: {time_window}"
            )
        if time_window[0] > time_window[1]:
            raise ValueError(
                f"Start time must not exceed end time in element {i}: {time_window}"
            )
        if time_window[0] == time_window[1]:
            warnings.warn(
                f"Zero-duration time window in element {i}: {time_window}. "
                "Treating as an empty slice.",
                UserWarning,
            )
        if warn_negative_start and time_window[0] < 0:
            warnings.warn(
                f"Time window {i} has negative start ({time_window[0]}). "
                "If these are absolute recording times, negative values are "
                "unexpected. For event-centered data constructed via "
                "time_peaks + time_bounds, this is normal.",
                UserWarning,
            )
        if recording_range is not None:
            rec_start, rec_end = recording_range
            if time_window[0] < rec_start or time_window[1] > rec_end:
                raise ValueError(
                    f"Time window {i} ({time_window[0]}, {time_window[1]}) "
                    f"extends outside the recording range "
                    f"[{rec_start}, {rec_end}]."
                )
        time_diff_check.append(time_window[1] - time_window[0])
        valid_time_tuples.append(time_window)
        if len(set(time_diff_check)) > 1:
            raise ValueError("All time windows must have the same length")
    return valid_time_tuples


def _rank_order_correlation_from_timing(
    timing_matrix,
    min_overlap=3,
    min_overlap_frac=None,
    n_shuffles=100,
    seed=1,
):
    """
    Compute Spearman rank-order correlation of unit timing between all slice pairs.

    Shared implementation used by both SpikeSliceStack.rank_order_correlation
    and RateSliceStack.rank_order_correlation.

    Parameters:
        timing_matrix (np.ndarray): Array of shape (U, S) with timing values
            per unit per slice. NaN entries mark inactive units.
        min_overlap (int): Minimum units active in both slices (default: 3).
        min_overlap_frac (float or None): Minimum fraction of total units
            active in both slices. Effective threshold is
            max(min_overlap, ceil(min_overlap_frac * U)).
        n_shuffles (int): Shuffle iterations for z-scoring (default: 100).
            0 = raw correlations. Values 1-4 are rejected.
        seed (int or None): Random seed for shuffle reproducibility.

    Returns:
        corr_matrix (PairwiseCompMatrix): (S, S) Spearman correlation or z-score matrix.
        av_corr (float): Average over valid lower-triangle pairs.
        overlap_matrix (PairwiseCompMatrix): (S, S) fraction of units active in both slices.
    """
    from scipy.stats import spearmanr

    # Import here to avoid circular import at module level
    from .pairwise import PairwiseCompMatrix

    if 0 < n_shuffles < 5:
        raise ValueError(
            f"n_shuffles must be 0 (no shuffling) or >= 5, got {n_shuffles}"
        )

    timing_matrix = np.asarray(timing_matrix)
    if timing_matrix.ndim != 2:
        raise ValueError(
            f"timing_matrix must be 2-D (U, S), got shape {timing_matrix.shape}"
        )

    num_units = timing_matrix.shape[0]
    effective_min = min_overlap
    if min_overlap_frac is not None:
        frac_count = int(np.ceil(min_overlap_frac * num_units))
        effective_min = max(effective_min, frac_count)

    rng = np.random.default_rng(seed)
    num_slices = timing_matrix.shape[1]
    corr = np.full((num_slices, num_slices), np.nan)
    overlap = np.zeros((num_slices, num_slices), dtype=int)
    if n_shuffles == 0:
        np.fill_diagonal(corr, 1.0)

    for i in range(num_slices):
        overlap[i, i] = int(np.sum(~np.isnan(timing_matrix[:, i])))

    for i in range(num_slices):
        for j in range(i + 1, num_slices):
            valid = ~np.isnan(timing_matrix[:, i]) & ~np.isnan(timing_matrix[:, j])
            n_valid = int(np.sum(valid))
            overlap[i, j] = n_valid
            overlap[j, i] = n_valid

            if n_valid < effective_min:
                continue

            a = timing_matrix[valid, i]
            b = timing_matrix[valid, j]
            rho, _ = spearmanr(a, b)

            if n_shuffles == 0:
                corr[i, j] = rho
                corr[j, i] = rho
            else:
                null_rhos = np.empty(n_shuffles)
                for k in range(n_shuffles):
                    b_shuffled = rng.permutation(b)
                    null_rhos[k], _ = spearmanr(a, b_shuffled)
                null_mean = np.mean(null_rhos)
                null_std = np.std(null_rhos)
                if null_std > 0:
                    z = (rho - null_mean) / null_std
                else:
                    z = np.nan
                corr[i, j] = z
                corr[j, i] = z

    lower_tri = np.tril_indices(num_slices, k=-1)
    av_corr = float(np.nanmean(corr[lower_tri]))

    overlap_frac = (
        overlap.astype(float) / num_units if num_units > 0 else overlap.astype(float)
    )

    return (
        PairwiseCompMatrix(matrix=corr),
        av_corr,
        PairwiseCompMatrix(matrix=overlap_frac),
    )


# ---------------------------------------------------------------------------
# Slice comparison utilities
# ---------------------------------------------------------------------------


def shuffle_z_score(observed, shuffle_distribution):
    """
    Z-score an observed value against a shuffle null distribution.

    Parameters:
        observed (scalar or np.ndarray): The metric computed on the real data.
        shuffle_distribution (np.ndarray): Shape ``(N, ...)`` array of the
            same metric computed on N shuffled datasets (e.g. from
            ``SpikeSliceStack.apply`` on a shuffle stack built by
            ``SpikeData.spike_shuffle_stack``).

    Returns:
        z (np.ndarray): Z-score ``(observed - mean) / std`` computed along
            axis 0. Same shape as *observed*.

    Notes:
        - Intended for determining whether an observed metric is significantly
          different from what degree-preserving shuffled data produces.
        - Elements where the shuffle standard deviation is zero will be NaN.
    """
    shuffle_distribution = np.asarray(shuffle_distribution)
    mean = np.nanmean(shuffle_distribution, axis=0)
    std = np.nanstd(shuffle_distribution, axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    z = (np.asarray(observed) - mean) / safe_std
    z = np.where(std == 0, np.nan, z)
    return z


def shuffle_percentile(observed, shuffle_distribution):
    """
    Compute the percentile rank of an observed value within a shuffle distribution.

    Parameters:
        observed (scalar or np.ndarray): The metric computed on the real data.
        shuffle_distribution (np.ndarray): Shape ``(N, ...)`` array of the
            same metric computed on N shuffled datasets.

    Returns:
        pct (np.ndarray): Fraction of shuffle values ≤ observed, computed
            along axis 0. Values in [0, 1]. Same shape as *observed*.

    Notes:
        - Non-parametric alternative to ``shuffle_z_score``; gives the rank
          of the observed value within the null distribution without assuming
          normality.
    """
    shuffle_distribution = np.asarray(shuffle_distribution)
    observed = np.asarray(observed)
    return np.mean(shuffle_distribution <= observed, axis=0)


def slice_trend(values, times=None):
    """
    Fit a linear trend to a metric computed across ordered slices.

    Parameters:
        values (np.ndarray): Shape ``(S,)`` array of metric values, one per
            slice, in temporal order.
        times (np.ndarray | None): Shape ``(S,)`` array of slice midpoints
            in milliseconds. If None, integer indices ``0 .. S-1`` are used.

    Returns:
        slope (float): Linear regression slope. Units are metric-change per
            millisecond (if *times* provided) or per slice index.
        p_value (float): Two-sided p-value for the null hypothesis that the
            slope is zero.

    Notes:
        - Intended for detecting systematic drift of a metric over the course
          of a recording. Apply to the output of ``SpikeSliceStack.apply`` on
          a frames stack built by ``SpikeData.frames``. A significant
          positive or negative slope indicates non-stationarity.
        - Uses ``scipy.stats.linregress``.
    """
    from scipy.stats import linregress

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            f"values must be 1-D, got shape {values.shape}. "
            "For higher-dimensional metrics, reduce to a scalar per slice "
            "before calling slice_trend."
        )
    if times is None:
        times = np.arange(len(values), dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    mask = ~np.isnan(values) & ~np.isnan(times)
    n_valid = int(np.sum(mask))
    if n_valid < 2:
        raise ValueError(
            "slice_trend requires at least 2 non-NaN (value, time) pairs; "
            f"got {n_valid} after omitting NaNs."
        )
    result = linregress(times[mask], values[mask])
    return result.slope, result.pvalue


def slice_stability(values):
    """
    Compute the coefficient of variation of a metric across slices.

    Parameters:
        values (np.ndarray): Shape ``(S,)`` or ``(S, ...)`` array of metric
            values from ``SpikeSliceStack.apply``.

    Returns:
        cv (np.ndarray or float): Coefficient of variation ``std / |mean|``
            computed along axis 0. Scalar when input is ``(S,)``.

    Notes:
        - Intended for summarising how much a metric varies across slices
          (frames, trials, or shuffles). Low CV indicates a stable metric;
          high CV indicates instability or sensitivity to the slicing.
        - Elements where the mean is zero will be NaN.
    """
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    abs_mean = np.abs(mean)
    safe_mean = np.where(abs_mean == 0, 1.0, abs_mean)
    cv = std / safe_mean
    cv = np.where(abs_mean == 0, np.nan, cv)
    return float(cv) if cv.ndim == 0 else cv
