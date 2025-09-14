from typing import Optional

import numpy as np
from scipy import ndimage, signal

__all__ = [
    "spike_time_tiling",
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

    Refactor 2025-09: behavior unchanged; helpers colocated below for clarity.
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

    Refactor 2025-09: definition colocated here for clarity; behavior unchanged.
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

    Refactor 2025-09: definition colocated here for clarity; behavior unchanged.
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
