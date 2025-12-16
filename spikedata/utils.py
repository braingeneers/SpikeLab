from typing import Optional

import numpy as np
from scipy import ndimage, signal
from scipy.stats import norm

__all__ = [
    "get_sttc",
    "swap",
    "randomize",
    "trough_between",
]


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
