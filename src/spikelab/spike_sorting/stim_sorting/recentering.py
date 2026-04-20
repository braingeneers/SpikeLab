"""Stimulation time recentering.

Finds the actual stimulation artifact onset near each logged stim time
by detecting the electrode-saturation peak in the raw voltage traces.
Stimulation typically causes immediate electrode saturation, making the
artifact peak trivially identifiable as the sample with the largest
absolute voltage across all channels within a search window.
"""

import numpy as np


def recenter_stim_times(
    traces,
    stim_times_ms,
    fs_Hz,
    max_offset_ms=50.0,
):
    """Find actual stimulation artifact times near logged stim times.

    For each logged stim time, searches a window of +/-``max_offset_ms``
    in the raw voltage traces and returns the time of the sample with
    the largest absolute voltage across all channels.  This corrects
    for timing offsets between the stimulation hardware trigger log
    and the actual artifact in the recording.

    Parameters:
        traces (np.ndarray): Raw voltage traces, shape
            ``(channels, samples)``.
        stim_times_ms (array-like): Logged stimulation event times in
            milliseconds.  Need not be sorted.
        fs_Hz (float): Sampling frequency in Hz.
        max_offset_ms (float): Radius of the search window around
            each logged stim time, in milliseconds.  Default 50.0.

    Returns:
        corrected_ms (np.ndarray): Corrected stim times in
            milliseconds, same length as ``stim_times_ms``.  Events
            whose search window extends outside the recording are
            clipped to the recording boundary.

    Notes:
        - When multiple stim events have overlapping search windows,
          each is recentered independently.
    """
    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)
    n_samples = traces.shape[1]

    offset_samples = int(np.round(max_offset_ms * fs_Hz / 1000.0))

    # Max absolute voltage across channels at each sample
    abs_max = np.max(np.abs(traces), axis=0)

    corrected = np.empty_like(stim_times_ms)
    for i, t_ms in enumerate(stim_times_ms):
        center = int(np.round(t_ms * fs_Hz / 1000.0))
        lo = max(0, center - offset_samples)
        hi = min(n_samples, center + offset_samples + 1)

        peak_sample = lo + int(np.argmax(abs_max[lo:hi]))
        corrected[i] = peak_sample / fs_Hz * 1000.0

    return corrected
