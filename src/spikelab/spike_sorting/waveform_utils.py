"""Reusable waveform utilities for spike sorting pipelines.

These functions extract the per-spike centering, polarity classification,
and max-channel detection logic so they can be shared across different
sorter backends (Kilosort2, Kilosort4, etc.).

All functions operate on plain numpy arrays or SpikeInterface objects —
they do not depend on any specific sorter's output format.
"""

from typing import Any, Dict

import numpy as np


def classify_polarity(
    templates: np.ndarray, pos_peak_thresh: float = 2.0
) -> np.ndarray:
    """Classify each unit as positive-peak or negative-peak.

    Compares the maximum positive deflection against the maximum negative
    deflection across all channels.  A unit is classified as positive-peak
    when its positive amplitude exceeds ``pos_peak_thresh`` times the
    absolute negative amplitude.

    Parameters:
        templates (np.ndarray): Template array with shape
            ``(n_units, n_samples, n_channels)``.
        pos_peak_thresh (float): Ratio threshold.  A unit is positive-peak
            when ``max_positive >= pos_peak_thresh * abs(max_negative)``.

    Returns:
        use_pos_peak (np.ndarray): Boolean array of shape ``(n_units,)``.
            True for positive-peak units.
    """
    # Per-unit minimum across (samples, channels) → most negative value
    neg_peaks = np.min(templates, axis=1)  # (n_units, n_channels)
    neg_values = np.min(neg_peaks, axis=1)  # (n_units,)

    # Per-unit maximum across (samples, channels) → most positive value
    pos_peaks = np.max(templates, axis=1)  # (n_units, n_channels)
    pos_values = np.max(pos_peaks, axis=1)  # (n_units,)

    return pos_values >= pos_peak_thresh * np.abs(neg_values)


def get_max_channels(templates: np.ndarray, use_pos_peak: np.ndarray) -> np.ndarray:
    """Find the channel with the largest amplitude per unit.

    For positive-peak units, selects the channel with the highest
    positive peak.  For negative-peak units, selects the channel with
    the deepest negative trough.

    Parameters:
        templates (np.ndarray): Template array with shape
            ``(n_units, n_samples, n_channels)``.
        use_pos_peak (np.ndarray): Boolean array of shape ``(n_units,)``.
            True for positive-peak units (from ``classify_polarity``).

    Returns:
        chans_max (np.ndarray): Integer array of shape ``(n_units,)``
            with the max-amplitude channel index per unit.
    """
    neg_peaks = np.min(templates, axis=1)  # (n_units, n_channels)
    neg_chans = neg_peaks.argmin(axis=1)  # (n_units,)

    pos_peaks = np.max(templates, axis=1)  # (n_units, n_channels)
    pos_chans = pos_peaks.argmax(axis=1)  # (n_units,)

    return np.where(use_pos_peak, pos_chans, neg_chans)


def compute_half_window_sizes(
    templates: np.ndarray, chans_max: np.ndarray, window_size_scale: float = 0.75
) -> np.ndarray:
    """Compute the half-window size for per-spike peak finding.

    For each unit, measures the distance from the template center to the
    last zero-crossing on the max channel, then scales by
    *window_size_scale* to produce a conservative search window that
    reduces the risk of locking onto adjacent peaks.

    Parameters:
        templates (np.ndarray): Template array with shape
            ``(n_units, n_samples, n_channels)``.
        chans_max (np.ndarray): Max-channel index per unit, shape
            ``(n_units,)``.
        window_size_scale (float): Fraction of the measured distance to
            use as the half-window.  Smaller values reduce false peak
            detections but may miss the true peak if it is offset.

    Returns:
        half_windows (np.ndarray): Integer array of shape ``(n_units,)``
            with the half-window size in samples per unit.
    """
    # Extract each unit's template on its max channel: (n_units, n_samples)
    n_units = templates.shape[0]
    n_samples = templates.shape[1]
    templates_max_ch = templates[np.arange(n_units), :, chans_max]

    template_mid = n_samples // 2
    half_windows = np.zeros(n_units, dtype=int)

    for i in range(n_units):
        t = templates_max_ch[i, :]
        # Find where the template amplitude drops below 1% of peak
        # before the midpoint.  Works for both zero-padded (KS2) and
        # dense (KS4) templates — matches KilosortSortingExtractor logic.
        peak_amp = np.abs(t).max()
        threshold = peak_amp * 0.01
        small_before_mid = np.flatnonzero(np.abs(t[:template_mid]) < threshold)
        if len(small_before_mid) > 0:
            size = template_mid - small_before_mid[-1]
        else:
            # No sub-threshold region found — use full half
            size = template_mid
        half_windows[i] = max(1, int(size * window_size_scale))

    return half_windows


def center_spike_times(
    recording: Any,
    spike_times_by_unit: Dict[int, np.ndarray],
    chans_max: Any,
    use_pos_peak: Any,
    half_window_sizes: Any,
    segment_index: int = 0,
) -> Dict[int, np.ndarray]:
    """Refine spike times by recentering each spike on the actual voltage peak.

    For each spike, reads the raw voltage trace on the unit's max channel
    within a local window around the sorter-reported spike time, finds the
    true peak (maximum for positive-peak units, minimum for negative-peak
    units), and returns the corrected spike time.  This is a **per-spike**
    correction — each spike can shift by a different amount.

    Parameters:
        recording: SpikeInterface BaseRecording.  Must support
            ``get_traces(start_frame, end_frame, segment_index)``.
        spike_times_by_unit (dict): ``{unit_id: np.ndarray}`` mapping
            each unit to its spike times in samples (sorted).
        chans_max (dict or array-like): Max-channel index per unit.
            Accessed as ``chans_max[unit_id]``.
        use_pos_peak (dict or array-like): Boolean polarity per unit.
            Accessed as ``use_pos_peak[unit_id]``.
        half_window_sizes (dict or array-like): Half-window size in
            samples per unit (from ``compute_half_window_sizes``).
            Accessed as ``half_window_sizes[unit_id]``.
        segment_index (int): Recording segment index.

    Returns:
        centered (dict): ``{unit_id: np.ndarray}`` with corrected spike
            times in samples.  Same keys and lengths as
            *spike_times_by_unit*.

    Notes:
        - Spikes too close to the recording boundary to fit the search
          window are left unchanged.
        - When multiple samples share the peak value, the middle index
          is chosen to avoid systematic bias.
        - This function reads raw traces in bulk per unit for efficiency
          but does not store waveforms — use a waveform extractor
          afterward with the corrected times.
    """
    n_samples_total = recording.get_num_samples(segment_index=segment_index)
    centered = {}

    for unit_id, spike_times in spike_times_by_unit.items():
        hw = int(half_window_sizes[unit_id])
        chan = int(chans_max[unit_id])
        is_pos = bool(use_pos_peak[unit_id])

        corrected = spike_times.copy()

        if len(spike_times) == 0:
            centered[unit_id] = corrected
            continue

        # Load a contiguous trace block covering all spikes for this unit
        margin = hw + 1
        block_start = int(max(0, spike_times[0] - margin))
        block_end = int(min(n_samples_total, spike_times[-1] + margin + 1))

        traces = recording.get_traces(
            start_frame=block_start,
            end_frame=block_end,
            segment_index=segment_index,
        )
        trace_chan = traces[:, chan]
        max_trace_idx = len(trace_chan) - 1

        for i, st in enumerate(spike_times):
            st_local = int(st) - block_start

            win_left = max(st_local - hw, 0)
            win_right = min(st_local + hw, max_trace_idx)
            win_size = win_right - win_left + 1
            window = trace_chan[win_left:win_right + 1]

            if len(window) == 0:
                continue

            if is_pos:
                peak_val = np.max(window)
            else:
                peak_val = np.min(window)

            peak_indices = np.flatnonzero(window == peak_val)
            offset = peak_indices[peak_indices.size // 2] - win_size // 2
            corrected[i] = int(st) + offset

        centered[unit_id] = corrected

    return centered
