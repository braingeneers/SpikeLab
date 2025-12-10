from typing import Optional

import numpy as np
from scipy import ndimage, signal
from scipy.stats import norm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

__all__ = [
    "spike_time_tiling",
    "swap",
    "randomize",
    "get_pop_rate",
    "get_bursts",
]


def spike_time_tiling(tA, tB, delt=20.0, length: Optional[float] = None):
    """
    Calculate the spike time tiling coefficient [1] between two spike trains. STTC is a
    metric for correlation between spike trains with some improved intuitive properties
    compared to the Pearson correlation coefficient. Spike trains are lists of spike
    times sorted in ascending order.

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
    "Internal helper method for the second half of STTC calculation."
    if len(tA) == 0 or len(tB) == 0:
        return 0
    PA = _sttc_na(tA, tB, delt) / len(tA)
    PB = _sttc_na(tB, tA, delt) / len(tB)

    aa = (PA - TB) / (1 - PA * TB) if PA * TB != 1 else 0
    bb = (PB - TA) / (1 - PB * TA) if PB * TA != 1 else 0
    return (aa + bb) / 2


def _sttc_ta(tA, delt: float, tmax: float) -> float:
    """
    Helper function for spike time tiling coefficients: calculate the total amount of
    time within a range delt of spikes within the given sorted list of spike times tA.
    """
    if len(tA) == 0:
        return 0.0

    base = min(delt, tA[0]) + min(delt, tmax - tA[-1])
    return base + np.minimum(np.diff(tA), 2 * delt).sum()


def _sttc_na(tA, tB, delt: float) -> int:
    """
    Helper function for spike time tiling coefficients: given two sorted lists of spike
    times, calculate the number of spikes in spike train A within delt of any spike in
    spike train B.
    """
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
    based on the reciprocal inter-spike interval. It is assumed to have been sampled
    halfway between any two given spikes, interpolated, and then smoothed by a Gaussian
    kernel with the given width.
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
    Helper method for SpikeData constructors: given lists of spike times and indices,
    produce a list whose ith entry is a list of the spike times of the ith unit.
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

    Inputs:
        data: array_like data to be filtered
        lowcut: low cutoff frequency. If None or 0, highcut must be a number.
                Filter is lowpass.
        highcut: high cutoff frequency. If None, lowpass must be a non-zero number.
                 Filter is highpass.
        If lowcut and highcut are both give, this filter is bandpass.
        In this case, lowcut must be smaller than highcut.
        fs: sample rate
        order: order of the filter

    Returns:
        The filtered output with the same shape as data
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
    Attempt one double-edge swap in a binary spike raster while preserving
    per-row and per-column sums.

    The swap chooses two existing spike positions (i0, j0) and (i1, j1) and,
    if the off-diagonal positions (i0, j1) and (i1, j0) are both empty and the
    indices are distinct, swaps them so spikes move to those positions.

    Returns True if a swap was performed, otherwise False.
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

    Parameters
    ----------
    ar : array_like
        Binary matrix shaped (neurons, time) or (time, neurons). Values should be 0/1.
    swap_per_spike : int
        Target number of successful swaps per spike.

    Returns
    -------
    ndarray
        Randomized binary matrix with the same shape and row/column sums.
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
    Compute population firing rate by smoothing the summed spike counts.

    First apply a moving-average (square) window, then optionally apply a Gaussian
    smoothing window parameterized by GAUSS_SIGMA (in samples).
    """
    if SQUARE_WIDTH > 0:
        square_smooth_summed_spike = np.convolve(
            np.sum(t_spk_mat, axis=1),
            np.ones(SQUARE_WIDTH) / SQUARE_WIDTH,
            mode="same",
        )
    else:
        square_smooth_summed_spike = np.sum(t_spk_mat, axis=1)

    if GAUSS_SIGMA > 0:
        gauss_window = norm.pdf(
            np.arange(-3 * GAUSS_SIGMA, 3 * GAUSS_SIGMA + 1), 0, GAUSS_SIGMA
        )
        pop_rate = np.convolve(
            square_smooth_summed_spike,
            gauss_window / np.sum(gauss_window),
            mode="same",
        )
    else:
        pop_rate = square_smooth_summed_spike

    return pop_rate


def get_bursts(
    pop_rate, pop_rate_acc, THR_BURST, MIN_BURST_DIFF, BURST_EDGE_MULT_THRESH
):
    """
    Detect bursts from a population rate vector using thresholded peak finding and
    amplitude-scaled edge detection.

    Returns (tburst, edges, peak_amp).
    """
    pop_rms = np.sqrt(np.mean(np.square(pop_rate)))

    peaks, _ = signal.find_peaks(
        pop_rate, height=pop_rms * THR_BURST, distance=MIN_BURST_DIFF
    )
    peak_amp = pop_rate[peaks]

    edges = np.full((len(peaks), 2), np.nan)
    tburst = np.full(len(peaks), np.nan)

    for burst in range(len(peaks)):
        frames_below_thresh = np.where(
            pop_rate < peak_amp[burst] * BURST_EDGE_MULT_THRESH
        )[0]
        rel_frames = peaks[burst] - frames_below_thresh

        if (
            len(rel_frames) == 0
            or len(rel_frames[rel_frames > 0]) == 0
            or len(rel_frames[rel_frames < 0]) == 0
        ):
            continue

        rel_burst_start = np.min(rel_frames[rel_frames > 0])
        rel_burst_end = np.max(rel_frames[rel_frames < 0])

        edges[burst, :] = [peaks[burst] - rel_burst_start, peaks[burst] - rel_burst_end]

        if len(pop_rate_acc) == len(pop_rate):
            segment = pop_rate_acc[int(edges[burst, 0]) : int(edges[burst, 1])]
            acc_peak = np.argmax(segment)
            peak_val = np.max(segment)
            tburst[burst] = acc_peak + edges[burst, 0]
            peak_amp[burst] = peak_val
        else:
            tburst[burst] = peaks[burst]

    edges = edges[~np.isnan(tburst), :]
    peak_amp = peak_amp[~np.isnan(tburst)]
    tburst = tburst[~np.isnan(tburst)]

    return tburst, edges, peak_amp


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
    matrix_3d (array): 3D correaltion matrix of shape (B, N, N) [b, :, :] is a symmetric N×N matrix
                       (this is just an example, can also be N x B x B. It just must be a 3d correlation matrix)

    Returns:
    --------
    features (array): 2D matrix of shape (B, F) each row contains lower triangle values for that correlation matrix
                      F = N*(N-1)/2 (number of unique pairs or more simply the number of values in lower traingle)
    """
    num_samples = matrix_3d.shape[0]  # B
    num_items = matrix_3d.shape[1]  # N

    # Get lower triangle indices
    lower_tri_idx = np.tril_indices(num_items, k=-1)

    # Extract all lower triangles at once (vectorized)
    features = matrix_3d[:, lower_tri_idx[0], lower_tri_idx[1]]
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
