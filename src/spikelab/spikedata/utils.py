import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
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

try:
    import h5py
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


# ---------------------------------------------------------------------------
# Thread-pool parallelisation helpers
# ---------------------------------------------------------------------------


def _resolve_n_jobs(n_jobs):
    """Resolve an n_jobs parameter to a concrete worker count.

    Parameters:
        n_jobs (int or None): Desired parallelism. -1 means all cores, None or
            1 means serial execution, negative values count from cpu_count.

    Returns:
        n_workers (int): Positive integer worker count (1 = serial).
    """
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1:
        cores = os.cpu_count() or 1
        return max(1, cores + 1 + n_jobs)
    return n_jobs


def get_sttc(
    tA, tB, delt=20.0, length: Optional[float] = None, start_time: float = 0.0
):
    """Calculate the spike time tiling coefficient between two spike trains.

    Parameters:
        tA (list): List of spike times for the first spike train.
        tB (list): List of spike times for the second spike train.
        delt (float): Time window in milliseconds (default: 20.0).
        length (float or None): Total duration in milliseconds. If None,
            inferred from the latest spike time after shifting, which may
            underestimate the true recording duration if the last spike does
            not fall near the end. Pass the actual recording length for
            unbiased STTC.
        start_time (float): Time origin of the spike trains (default 0.0).
            Spike times are shifted by ``-start_time`` before computation so
            that the STTC edge corrections work correctly for event-centered
            data with negative spike times.

    Returns:
        sttc (float): Spike time tiling coefficient between the two spike
            trains.

    Notes:
        Formula: STTC = ((PA - TB) / (1 - PA * TB) + (PB - TA) / (1 - PB * TA)) / 2

        [1] Cutts & Eglen. Detecting pairwise correlations in spike trains:
        An objective comparison of methods and application to the study of
        retinal waves. Journal of Neuroscience 34:43, 14288-14303 (2014).
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
    """Calculate the total amount of time within a range delt of spikes within tA."""
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
    # Uses inclusive <= (common implementation practice) rather than strict <
    # from Cutts & Eglen (2014). For continuous spike times the difference is
    # negligible; for binned data it may slightly increase coincidence counts.
    return (np.minimum(dt_left, dt_right) <= delt).sum()


def _resampled_isi(spikes, times, sigma_ms):
    """Calculate the firing rate of a spike train at specific times using the reciprocal inter-spike interval.

    Parameters:
        spikes (list): List of spike times.
        times (list): List of times.
        sigma_ms (float): Standard deviation in milliseconds.

    Returns:
        fr (np.ndarray): Firing rate at specific times. Same size as times.

    Notes:
        Assumed to have been sampled halfway between any two given spikes,
        interpolated, and then smoothed by a Gaussian kernel with the given
        width.
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

    # Assign rates to bins between spikes.
    # Note: int(round(...)) bin assignment can shift spikes at exact bin
    # boundaries to adjacent bins — a known sub-ms precision limitation.
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
    """Given lists of spike times and unit indices, produce a list of per-unit spike times.

    Parameters:
        idces (list): List of spike indices.
        times (list): List of spike times.
        N (int): Number of units.

    Returns:
        ret (list): List whose ith entry is a list of the spike times of the
            ith unit.
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
    """Apply a digital Butterworth filter. Filter type is based on input values.

    Parameters:
        data (array_like): Data to be filtered.
        lowcut (float or None): Low cutoff frequency. If None or 0, highcut
            must be a number.
        highcut (float or None): High cutoff frequency. If None, lowcut must
            be a non-zero number.
        fs (float): Sample rate.
        order (int): Order of the filter.

    Returns:
        filtered_traces (np.ndarray): The filtered output with the same shape
            as data.

    Notes:
        If lowcut and highcut are both given, this filter is bandpass. In
        this case, lowcut must be smaller than highcut.
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
    """Attempt one double-edge swap in a binary spike raster while preserving per-row and per-column sums.

    Parameters:
        ar (np.ndarray): Binary spike raster.
        idxs (tuple): Tuple of numpy arrays containing the indices of the
            spikes.
        rng (np.random.Generator): Random number generator for
            reproducibility.

    Returns:
        success (bool): True if a swap was performed.

    Notes:
        Both ``ar`` and ``idxs`` are mutated in-place for performance.

        The swap chooses two existing spike positions (i0, j0) and (i1, j1)
        and, if the off-diagonal positions (i0, j1) and (i1, j0) are both
        empty and the indices are distinct, swaps them so that spikes move
        to those positions.
    """
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
    """Randomize a binary spike raster using degree-preserving double-edge swaps.

    Parameters:
        ar (array_like): Binary matrix shaped (neurons, time) or
            (time, neurons). Values should be 0/1.
        swap_per_spike (int): Target number of successful swaps per spike.
        seed (int or None): Random seed number. Set for repeatability during
            experiments.

    Returns:
        randomized_raster (np.ndarray): Randomized binary matrix with the
            same shape and row/column sums.

    Notes:
        Shuffling preserves each neuron's average firing rate but shuffles
        which time bins it spikes in. Each time bin's population rate is also
        preserved but the specific units active are shuffled. Every spike
        swap involves 2 different spikes so on average every spike will get
        swapped 2 * swap_per_spike times.

        Okun, M. et al. Population rate dynamics and multineuron firing
        patterns in sensory cortex. J. Neurosci. 32, 17108-17119 (2012).
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
    """Find the minimum value (trough) between two indices in a population rate array.

    Parameters:
        i0 (int): Time bin index of the first burst.
        i1 (int): Time bin index of the second burst.
        pop_rate (np.ndarray): Smoothed population spiking data in spikes
            per bin.

    Returns:
        trough_idx (int or None): Time bin index of minimum value (trough)
            between peaks. None if the indices are adjacent.
    """
    L, R = int(i0), int(i1)

    if R - L <= 1:
        return None

    seg = pop_rate[L:R]

    return L + int(np.argmin(seg))


def compute_cross_correlation_with_lag(ref_rate, comp_rate, max_lag=0):
    """Compute normalized cross-correlation with lag information.

    Parameters:
        ref_rate (array): Reference firing rate signal.
        comp_rate (array): Comparison firing rate signal.
        max_lag (int or None): Maximum lag in frames to search for
            similarity. If None, lag is set to 0.

    Returns:
        max_corr (float): Maximum correlation coefficient.
        max_lag_idx (int): Lag (in frames) at which maximum correlation
            occurs.
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

    # Fast path for zero lag: direct dot-product normalisation (equivalent to
    # the general path below with max_lag=0, but avoids scipy.signal overhead).
    if max_lag == 0:
        max_corr = np.sum(ref_rate * comp_rate) / np.sqrt(norm_product)
        return max_corr, 0
    # General path: use scipy.signal.correlate and normalise by the geometric
    # mean of the zero-lag autocorrelations (equivalent to dividing by the
    # product of L2 norms, but computed via correlate for consistency).
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
    """Compute cosine similarity with lag information.

    Parameters:
        ref_rate (array): Reference firing rate signal.
        comp_rate (array): Comparison firing rate signal.
        max_lag (int or None): Maximum lag in frames to search for
            similarity. If None, lag is set to 0.

    Returns:
        max_sim (float): Maximum cosine similarity coefficient.
        max_lag_idx (int): Lag (in frames) at which maximum similarity
            occurs.
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
    """Compute PCA dimensionality reduction on axis 1 of a 2D matrix.

    Parameters:
        matrix_2d (array): 2D matrix of shape (samples, features) where
            values must be int, float, or bool.
        n_components (int): Number of principal components to retain
            (default: 2).

    Returns:
        embedding (np.ndarray): 2D matrix of shape
            ``(samples, n_components)``.
        explained_variance_ratio (np.ndarray): 1D array of shape
            ``(n_components,)`` with the fraction of total variance
            explained by each component.
        components (np.ndarray): 2D matrix of shape
            ``(n_components, features)`` with the principal axes
            (loadings) -- each row is one PC expressed in the original
            feature space.
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
    """Clamp n_neighbors so small datasets do not raise at UMAP fit time."""
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
    """Compute UMAP dimensionality reduction on a 2D matrix.

    Parameters:
        matrix_2d (array_like): Input data of shape
            ``(n_samples, n_features)``. Each row is a sample, each column
            is a feature.
        n_components (int): Dimension of the embedded space.
        n_neighbors (int): Size of local neighborhood used for manifold
            approximation.
        min_dist (float): Controls how tightly UMAP packs points together in
            the low-dimensional space.
        metric (str): Distance metric used in the input space.
        random_state (int or None): Random seed for reproducibility.
        **umap_kwargs: Additional keyword arguments passed to
            ``umap.UMAP``.

    Returns:
        embedding (np.ndarray): Low-dimensional embedding of shape
            ``(n_samples, n_components)``.
        trustworthiness_score (float): Trustworthiness of the embedding
            (0 to 1). Measures how well local neighborhoods in the
            high-dimensional space are preserved. Returns NaN if
            scikit-learn is unavailable.
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
    """Run UMAP and Louvain community detection on the UMAP connectivity graph.

    This helper keeps UMAP_reduction simple while providing an optional
    graph-based clustering approach that builds on UMAP's internal graph.

    Parameters:
        matrix_2d (array_like): Input data of shape
            ``(n_samples, n_features)``. Each row is a sample, each column
            is a feature.
        n_components (int): Dimension of the embedded space.
        resolution (float): Resolution parameter for the Louvain community
            detection algorithm. Higher values produce more, smaller
            communities. Lower values produce fewer, larger communities.
        n_neighbors (int): Passed through to ``umap.UMAP``.
        min_dist (float): Passed through to ``umap.UMAP``.
        metric (str): Passed through to ``umap.UMAP``.
        random_state (int or None): Passed through to ``umap.UMAP``.
        **umap_kwargs: Additional keyword arguments passed to
            ``umap.UMAP``.

    Returns:
        embedding (np.ndarray): Low-dimensional UMAP embedding of shape
            ``(n_samples, n_components)``.
        labels (np.ndarray): Integer community label for each sample, shape
            ``(n_samples,)``.
        trustworthiness_score (float): Trustworthiness of the embedding
            (0 to 1). Returns NaN if scikit-learn is not available.
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
    """Raise ``ImportError`` if *h5py* is not installed."""
    if h5py is None:
        raise ImportError(
            "h5py is required for this operation. " "Install it with: pip install h5py"
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
        return np.rint(times_ms.astype(float) * (fs_Hz / 1e3)).astype(np.int64)
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
    """Check a list of dictionaries for use as neuron_attributes to verify that keys and values are consistent.

    Parameters:
        neuron_attributes (list of dict): List of dictionaries containing
            neuron attributes.
        n_neurons (int or None): Expected number of neurons. If provided,
            validates the list length.

    Returns:
        result (list of dict): A list of dictionaries where all dictionaries
            have valid keys and values.

    Notes:
        If some dictionaries are missing keys that others have, a ValueError
        is raised indicating which neuron entries have inconsistent keys.
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
    """Determine which channels to extract for a given unit.

    Parameters:
        unit_idx (int): Index of the unit.
        channels (int, list of int, or None): Channel specification. None
            uses neuron_to_channel mapping or all channels; int for single
            channel; list for multiple; empty list for mapped channel.
        neuron_to_channel (dict): Mapping from unit indices to channel
            indices.
        n_channels_total (int): Total number of channels in the raw data.

    Returns:
        result (list of int): Channel indices to extract.

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
    """Compute the average waveform from extracted waveforms.

    Parameters:
        waveforms (np.ndarray): 3D array of shape
            ``(num_channels, num_samples, num_spikes)``.
        channel_indices (list of int): List of channel indices used for
            extraction.
        dtype (np.dtype): Data type for the output array if waveforms is
            empty.

    Returns:
        avg (np.ndarray): 2D array of shape ``(num_channels, num_samples)``
            containing the average waveform.
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
    """Filter spike times to only those within valid bounds of the raw data.

    Parameters:
        spike_times_ms (np.ndarray): Array of spike times in milliseconds.
        fs_kHz (float): Sampling rate in kHz.
        ms_before (float): Milliseconds before each spike time.
        ms_after (float): Milliseconds after each spike time.
        n_time_samples (int): Total number of time samples in the raw data.

    Returns:
        valid (np.ndarray): Array of valid spike times in milliseconds.
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
    """Convert a waveform stack into a per-channel dict.

    Parameters:
        waveforms (np.ndarray): 3D array shaped
            ``(num_channels, num_samples, num_spikes)``.
        channel_indices (list of int): List of channel indices corresponding
            to waveforms axis 0.

    Returns:
        result (dict): Mapping of channel index to 2D array shaped
            ``(num_samples, num_spikes)``.
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
    """Extract waveforms and compute statistics for a single unit.

    This function orchestrates the full waveform extraction pipeline:
    resolves channels, extracts raw voltage snippets around each spike
    time, computes the mean waveform, and filters spike times to valid
    extraction windows.

    Parameters:
        unit_idx (int): Index of the unit being extracted.
        spike_times_ms (np.ndarray): Array of spike times in milliseconds
            for this unit.
        raw_data (np.ndarray): Raw voltage data with shape
            ``(num_channels, num_samples)``.
        fs_kHz (float): Sampling rate in kHz.
        ms_before (float): Milliseconds before each spike time.
        ms_after (float): Milliseconds after each spike time.
        channels (int, list of int, or None): Channel specification. None
            uses neuron_to_channel mapping or all channels; int for single
            channel; list for multiple; empty list for mapped channel.
        neuron_to_channel (dict): Mapping from unit indices to channel
            indices.
        bandpass (tuple or None): Optional ``(lowcut_Hz, highcut_Hz)`` for
            bandpass filtering.
        filter_order (int): Butterworth filter order (default: 3).
        return_channel_waveforms (bool): If True, include per-channel
            waveforms in the metadata dict.
        return_avg_waveform (bool): If True, compute and include the
            average waveform in the metadata dict.

    Returns:
        waveforms (np.ndarray): 3D array
            ``(num_channels, num_samples, num_spikes)``.
        meta (dict): Per-unit metadata containing ``channels``,
            ``spike_times_ms``, ``avg_waveform``, and optionally
            ``channel_waveforms``.
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
    """Extract waveform snippets from raw data at specified spike times.

    Parameters:
        raw_data (np.ndarray): Raw voltage data with shape
            ``(num_channels, num_samples)``.
        spike_times_ms (np.ndarray): Array of spike times in milliseconds.
        fs_kHz (float): Sampling rate in kHz.
        ms_before (float): Milliseconds before each spike time.
        ms_after (float): Milliseconds after each spike time.
        channel_indices (list of int or None): Channel indices to extract.
            If None, extracts all.
        bandpass (tuple or None): Optional ``(lowcut_Hz, highcut_Hz)`` for
            bandpass filtering.
        filter_order (int): Butterworth filter order (default: 3).

    Returns:
        waveforms (np.ndarray): 3D array
            ``(num_channels, num_samples, num_spikes)``. Empty if no valid
            spikes.
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
    """Validate that a list of (start, end) tuples has the same duration and is in proper format.

    Parameters:
        times_start_to_end (list): Each entry must be a tuple
            ``(start, end)``.
        warn_negative_start (bool): If True, emit a warning for windows with
            negative start times (default False). Useful when times are
            expected to be absolute recording positions.
        recording_range (tuple or None): If provided, a
            ``(rec_start, rec_end)`` tuple defining the valid time range.
            Any window that extends outside this range raises
            ``ValueError``. If None (default), no range check is performed.

    Returns:
        valid_time_tuples (list): Sorted list of valid ``(start, end)``
            tuples. Negative-start windows are preserved.
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

    if len(time_diff_check) > 1:
        diffs = np.array(time_diff_check)
        if not np.allclose(diffs, diffs[0], atol=1e-6, rtol=0):
            raise ValueError("All time windows must have the same length")
    return valid_time_tuples


def _rank_order_correlation_from_timing(
    timing_matrix,
    min_overlap=3,
    min_overlap_frac=None,
    n_shuffles=100,
    seed=1,
    n_jobs=-1,
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
        n_jobs (int): Number of threads for parallel computation. -1 uses all
            cores (default), 1 disables parallelism, None is serial.

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

    num_slices = timing_matrix.shape[1]
    corr = np.full((num_slices, num_slices), np.nan)
    overlap = np.zeros((num_slices, num_slices), dtype=int)
    if n_shuffles == 0:
        np.fill_diagonal(corr, 1.0)

    for i in range(num_slices):
        overlap[i, i] = int(np.sum(~np.isnan(timing_matrix[:, i])))

    # Pre-compute validity masks and extract data for each pair
    pairs = [(i, j) for i in range(num_slices) for j in range(i + 1, num_slices)]

    # Each pair needs its own independent RNG for reproducibility
    ss = np.random.SeedSequence(seed)
    pair_seeds = ss.spawn(len(pairs))

    def _compute_pair(args):
        (i, j), child_seed = args
        valid = ~np.isnan(timing_matrix[:, i]) & ~np.isnan(timing_matrix[:, j])
        n_valid = int(np.sum(valid))

        if n_valid < effective_min:
            return i, j, n_valid, np.nan

        a = timing_matrix[valid, i]
        b = timing_matrix[valid, j]
        rho, _ = spearmanr(a, b)

        if n_shuffles == 0:
            return i, j, n_valid, rho

        rng = np.random.default_rng(child_seed)
        null_rhos = np.empty(n_shuffles)
        for k in range(n_shuffles):
            null_rhos[k], _ = spearmanr(a, rng.permutation(b))
        null_mean = np.mean(null_rhos)
        null_std = np.std(null_rhos)
        z = (rho - null_mean) / null_std if null_std > 0 else np.nan
        return i, j, n_valid, z

    work_items = list(zip(pairs, pair_seeds))

    n_workers = _resolve_n_jobs(n_jobs)
    if n_workers > 1 and len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = pool.map(_compute_pair, work_items)
    else:
        results = map(_compute_pair, work_items)

    for i, j, n_valid, value in results:
        overlap[i, j] = n_valid
        overlap[j, i] = n_valid
        corr[i, j] = value
        corr[j, i] = value

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


# ---------------------------------------------------------------------------
# Sorter comparison helpers
# ---------------------------------------------------------------------------


def _count_matching_spikes(times1, times2, delta):
    """Count the number of matching spikes between two sorted spike trains.

    Two spikes are considered matching if they occur within *delta* of each
    other and belong to different trains. When multiple candidates exist the
    earliest pair wins (greedy left-to-right).

    Algorithm adapted from SpikeInterface ``count_matching_events``.

    Parameters:
        times1 (np.ndarray): Sorted spike times for train 1.
        times2 (np.ndarray): Sorted spike times for train 2.
        delta (float): Maximum allowed temporal distance for a match.

    Returns:
        n_matches (int): Number of matched spike pairs.
    """
    times1 = np.asarray(times1)
    times2 = np.asarray(times2)
    if len(times1) == 0 or len(times2) == 0:
        return 0

    i = 0
    j = 0
    n_matches = 0
    n1 = len(times1)
    n2 = len(times2)

    while i < n1 and j < n2:
        dt = times1[i] - times2[j]
        if abs(dt) <= delta:
            n_matches += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    return n_matches
def _compute_agreement_score(train1, train2, delta):
    """Compute spike-train agreement between two spike trains.

    Parameters:
        train1 (np.ndarray): Sorted spike times for train 1.
        train2 (np.ndarray): Sorted spike times for train 2.
        delta (float): Maximum allowed temporal distance for a match.

    Returns:
        agreement (float): Jaccard-style agreement score
            ``n_matches / (n1 + n2 - n_matches)``.
        frac_1 (float): Fraction of train1 spikes that were matched.
        frac_2 (float): Fraction of train2 spikes that were matched.
    """
    n1 = len(train1)
    n2 = len(train2)
    if n1 == 0 and n2 == 0:
        return 0.0, 0.0, 0.0
    n_matches = _count_matching_spikes(train1, train2, delta)
    denom = n1 + n2 - n_matches
    agreement = n_matches / denom if denom > 0 else 0.0
    frac_1 = n_matches / n1 if n1 > 0 else 0.0
    frac_2 = n_matches / n2 if n2 > 0 else 0.0
    return agreement, frac_1, frac_2


def _compute_footprint(neuron_attrs, f_rel_to_trough, n_channels):
    """Build a spatial waveform footprint array for one unit.

    The footprint is a 2-D array of shape ``(n_channels, n_samples)`` where
    ``n_samples = f_rel_to_trough[0] + f_rel_to_trough[1] + 1``. The
    template waveform is placed at the unit's main channel row, and
    neighbouring-channel templates are placed at their respective rows, all
    aligned to the trough of the main template.

    Parameters:
        neuron_attrs (dict): Neuron attribute dictionary containing:
            ``template`` (1-D ndarray), ``neighbor_templates`` (2-D ndarray),
            ``channel`` (int), ``neighbor_channels`` (1-D ndarray).
        f_rel_to_trough (tuple of int): ``(pre, post)`` number of samples
            before and after the trough to include.
        n_channels (int): Total number of channels on the probe.

    Returns:
        fp (np.ndarray): Footprint array of shape
            ``(n_channels, f_rel_to_trough[0] + f_rel_to_trough[1] + 1)``.
    """
    n_samples = f_rel_to_trough[0] + f_rel_to_trough[1] + 1
    fp = np.zeros((n_channels, n_samples))

    template = np.asarray(neuron_attrs["template"])
    nb_templates = np.asarray(neuron_attrs["neighbor_templates"])
    channel = int(neuron_attrs["channel"])
    nb_channels = np.asarray(neuron_attrs["neighbor_channels"])

    t_i = int(np.argmin(template))

    sel_start = max(0, t_i - f_rel_to_trough[0])
    sel_end = min(len(template) - 1, t_i + f_rel_to_trough[1])

    pre_seg = template[sel_start:t_i]
    post_seg = template[t_i : sel_end + 1]

    paste_start = f_rel_to_trough[0] - len(pre_seg)
    paste_end = f_rel_to_trough[0] + len(post_seg)

    fp[channel, paste_start : f_rel_to_trough[0]] = pre_seg
    fp[channel, f_rel_to_trough[0] : paste_end] = post_seg

    for nb_i in range(1, len(nb_channels)):
        pre_nb = nb_templates[nb_i, sel_start:t_i]
        post_nb = nb_templates[nb_i, t_i : sel_end + 1]
        ch = int(nb_channels[nb_i])
        if ch < n_channels:
            fp[ch, paste_start : f_rel_to_trough[0]] = pre_nb
            fp[ch, f_rel_to_trough[0] : paste_end] = post_nb

    return fp


def _compute_footprint_similarity(fp1, fp2, max_lag=5):
    """Compute the best cosine similarity between two footprints over lag shifts.

    Both footprints are flattened and cosine similarity is evaluated at
    integer lags from ``-max_lag`` to ``+max_lag``. The maximum similarity
    across all lags is returned.

    Parameters:
        fp1 (np.ndarray): Footprint array (n_channels, n_samples).
        fp2 (np.ndarray): Footprint array (n_channels, n_samples), same
            shape as *fp1*.
        max_lag (int): Maximum lag in samples to search (default 5).

    Returns:
        best_sim (float): Highest cosine similarity across all tested lags.
    """
    fp1_flat = fp1.ravel()
    fp2_flat = fp2.ravel()

    if len(fp1_flat) != len(fp2_flat):
        raise ValueError(
            f"Footprints must have the same size after flattening, "
            f"got {len(fp1_flat)} and {len(fp2_flat)}"
        )

    best = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            vec1 = fp1_flat[-lag:]
            vec2 = fp2_flat[: len(fp1_flat) + lag]
        elif lag > 0:
            vec1 = fp1_flat[: len(fp1_flat) - lag]
            vec2 = fp2_flat[lag:]
        else:
            vec1 = fp1_flat
            vec2 = fp2_flat
        sim = _cosine_sim(vec1, vec2)
        if not np.isnan(sim) and sim > best:
            best = sim

    return float(best) if best > -np.inf else np.nan
