from typing import Optional

import numpy as np
from scipy import ndimage, signal
from scipy.stats import norm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

__all__ = [
    "get_sttc",
    "swap",
    "randomize",
    "get_pop_rate",
    "get_bursts",
]


def get_sttc(tA, tB, delt=20.0, length: Optional[float] = None):
    """
    Calculate the spike time tiling coefficient between two spike trains.

    Parameters:
    tA (list): List of spike times for the first spike train
    tB (list): List of spike times for the second spike train
    delt (float): Time window in milliseconds (default: 20.0)
    length (float): Total duration in milliseconds (optional)

    Returns:
    sttc (float): Spike time tiling coefficient between the two spike trains

    Notes:
    - STTC is a metric for correlation between spike trains with some improved intuitive properties
    compared to the Pearson correlation coefficient.
    """
    if length is None:
        length = float(max(tA[-1], tB[-1]))

    if len(tA) == 0 or len(tB) == 0:
        return 0.0

    TA = _sttc_ta(tA, delt, length) / length
    TB = _sttc_ta(tB, delt, length) / length
    return _spike_time_tiling(tA, tB, TA, TB, delt)


def _spike_time_tiling(tA, tB, TA, TB, delt):
    """
    Internal helper method for the second half of STTC calculation.

    Parameters:
    tA (list): List of spike times for the first spike train
    tB (list): List of spike times for the second spike train
    TA (float): Total amount of time within a range delt of spikes within the given sorted list of spike times tA
    TB (float): Total amount of time within a range delt of spikes within the given sorted list of spike times tB
    delt (float): Time window in milliseconds

    Returns:
    sttc (float): Spike time tiling coefficient between the two spike trains
    """
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

    Parameters:
    tA (list): List of spike times for the first spike train
    delt (float): Time window in milliseconds
    tmax (float): Total duration in milliseconds

    Returns:
    ta (float): Total amount of time within a range delt of spikes within the given sorted list of spike times tA
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

    Parameters:
    tA (list): List of spike times for the first spike train
    tB (list): List of spike times for the second spike train
    delt (float): Time window in milliseconds

    Returns:
    num_spikes (int): Number of spikes in spike train A within delt of any spike in spike train B
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
    - If lowcut and highcut are both give, this filter is bandpass.
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
    Compute population firing rate by smoothing the summed spike counts.

    First apply a moving-average (square) window, then optionally apply a Gaussian
    smoothing window parameterized by GAUSS_SIGMA (in samples).

    Parameters:
    t_spk_mat (numpy.ndarray): Time-major spike matrix (T × N), values 0/1 or counts
    SQUARE_WIDTH (int): Moving-average window width (samples), 0 to disable
    GAUSS_SIGMA (float): Gaussian sigma (samples) for additional smoothing, 0 to disable

    Returns:
    pop_rate (numpy.ndarray): Population rate vector of length T
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

    Parameters:
    pop_rate (numpy.ndarray): Population rate vector (length T)
    pop_rate_acc (numpy.ndarray): Optional accumulator with same length T for peak localization; pass an empty list to skip
    THR_BURST (float): Multiplier on RMS(pop_rate) for peak height threshold
    MIN_BURST_DIFF (int): Minimum distance (samples) between consecutive peaks
    BURST_EDGE_MULT_THRESH (float): Edge threshold as a fraction of each burst's peak amplitude

    Returns:
    tburst (numpy.ndarray): Peak times
    edges (numpy.ndarray): Edge indices per burst
    peak_amp (numpy.ndarray): Peak amplitudes
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
