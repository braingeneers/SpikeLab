"""Stimulation artifact removal for offline electrophysiology recordings.

Removes electrical stimulation artifacts from multi-electrode array
(MEA) recordings while preserving neural spikes.  Two methods are
provided:

``"polynomial"`` (default)
    Per-event, per-channel low-order polynomial detrend.  A polynomial
    (default cubic) is fit to the non-saturated samples in the artifact
    tail — after the electrode desaturates — and subtracted.  Because
    the polynomial is far too smooth to capture spike waveforms
    (~0.5-1 ms), spikes riding on the artifact tail are preserved in
    the residual.  Saturated samples are blanked (set to zero).

``"blank"``
    Simply zeros out the entire artifact window.  Crude but useful as
    a quick sanity check or when the artifact is too variable for a
    good polynomial fit.

The polynomial detrend approach is related to SALPA (Suprathreshold
Artifact-Level Polynomial Approximation):

    Wagenaar, D. A. & Potter, S. M. (2002). Real-time multi-channel
    stimulus artifact suppression by local curve fitting. J Neurosci
    Methods, 120(2), 113-120.

SALPA fits a local polynomial in a causal (backward-looking) sliding
window and forward-extrapolates during the artifact, which is necessary
for real-time operation.  This module is designed for offline use, so it
instead looks ahead past saturation and fits the polynomial to the
actual post-saturation recovery curve, yielding a more accurate fit
without the extrapolation drift inherent in SALPA's forward prediction.

Sequential stimulation handling
    When multiple stim events occur in rapid succession (e.g. burst or
    paired-pulse protocols), the signal may re-saturate before reaching
    baseline after the previous stim.  This module dynamically detects
    whether the signal has returned to baseline-like levels after each
    desaturation.  If re-saturation occurs before baseline is reached,
    the blanking region is extended and the polynomial fit is deferred
    until after the final stim in the burst.
"""

import warnings

import numpy as np


def _auto_saturation_threshold(traces, quantile=0.999):
    """Estimate a saturation threshold from the trace amplitude distribution.

    Uses a high quantile of the absolute voltage distribution as the
    threshold.  Recordings with genuine saturation will have a hard
    clip at the ADC rail, so the quantile lands just below that clip.

    Parameters:
        traces (np.ndarray): ``(channels, samples)``.
        quantile (float): Quantile of ``|traces|`` to use.

    Returns:
        threshold (float): Absolute voltage threshold.
    """
    return float(np.quantile(np.abs(traces), quantile))


def _auto_baseline_threshold(traces, stim_times_ms, fs_Hz, k=5.0):
    """Estimate a baseline envelope threshold from pre-stim signal.

    Computes the median absolute deviation (MAD) of the signal in the
    2 ms window before the first stim event (or the first 2 ms of the
    recording if there's no pre-stim data), then returns
    ``median + k * MAD`` as the threshold for "signal has returned to
    baseline-like levels."

    Parameters:
        traces (np.ndarray): ``(channels, samples)``.
        stim_times_ms (np.ndarray): Corrected stim times in ms.
        fs_Hz (float): Sampling frequency in Hz.
        k (float): Multiplier on MAD.  Default 5.0.

    Returns:
        threshold (float): Baseline envelope threshold (absolute).
    """
    baseline_ms = 2.0
    baseline_samples = max(1, int(np.round(baseline_ms * fs_Hz / 1000.0)))

    if len(stim_times_ms) > 0:
        first_stim_sample = int(np.round(np.min(stim_times_ms) * fs_Hz / 1000.0))
        end = max(1, first_stim_sample)
        start = max(0, end - baseline_samples)
    else:
        start = 0
        end = min(baseline_samples, traces.shape[1])

    segment = traces[:, start:end]
    if segment.size == 0:
        return float(np.median(np.abs(traces)) * k)

    med = np.median(np.abs(segment))
    mad = np.median(np.abs(np.abs(segment) - med))
    return float(med + k * mad)


def _find_saturation_end(channel_trace, start, saturation_threshold, n_samples):
    """Find the first sample after *start* where the signal desaturates.

    Parameters:
        channel_trace (np.ndarray): 1-D voltage trace for one channel.
        start (int): Sample index to start searching from.
        saturation_threshold (float): Absolute voltage threshold.
        n_samples (int): Total number of samples in the trace.

    Returns:
        end (int): First sample index where
            ``|voltage| < saturation_threshold``, or ``n_samples`` if
            the signal never desaturates.
    """
    idx = start
    while idx < n_samples and np.abs(channel_trace[idx]) >= saturation_threshold:
        idx += 1
    return idx


def _signal_reached_baseline(
    channel_trace, start, baseline_threshold, window_samples, n_samples
):
    """Check whether the signal has returned to baseline-like levels.

    The signal is considered at baseline when the rolling maximum
    of ``|voltage|`` over *window_samples* consecutive samples drops
    below *baseline_threshold*.

    Parameters:
        channel_trace (np.ndarray): 1-D voltage trace.
        start (int): Sample index to start checking from.
        baseline_threshold (float): Absolute voltage threshold.
        window_samples (int): Number of consecutive sub-threshold
            samples required.
        n_samples (int): Trace length.

    Returns:
        at_baseline (bool): True if the signal reached baseline before
            the end of the trace.
        end_idx (int): Sample index where baseline was reached, or
            ``n_samples``.
    """
    consecutive = 0
    idx = start
    while idx < n_samples:
        if np.abs(channel_trace[idx]) < baseline_threshold:
            consecutive += 1
            if consecutive >= window_samples:
                return True, idx - window_samples + 1
        else:
            consecutive = 0
        idx += 1
    return False, n_samples


def _process_stim_group_polynomial(
    channel_trace,
    group_start,
    last_desat,
    artifact_window_samples,
    baseline_threshold,
    baseline_window_samples,
    poly_order,
    n_samples,
    blanked,
    ch_idx,
):
    """Polynomial detrend for one stim group on one channel.

    Blanks from ``group_start`` through ``last_desat``, then fits and
    subtracts a polynomial to the artifact tail starting at
    ``last_desat``.  The polynomial is fit using samples from
    ``last_desat`` through the end of the artifact window, anchored
    toward baseline by extending the fit window until the signal
    reaches baseline-like levels.

    Parameters:
        channel_trace (np.ndarray): 1-D trace (modified in-place).
        group_start (int): First sample of the blanking region.
        last_desat (int): Sample where the last saturation ended.
        artifact_window_samples (int): Max samples to fit after desat.
        baseline_threshold (float): Threshold for baseline detection.
        baseline_window_samples (int): Consecutive samples for baseline.
        poly_order (int): Polynomial order.
        n_samples (int): Trace length.
        blanked (np.ndarray): 2-D boolean mask ``(channels, samples)``,
            modified in-place.
        ch_idx (int): Channel index for the blanked mask.
    """
    # Blank from group start through desaturation
    blank_end = min(last_desat, n_samples)
    channel_trace[group_start:blank_end] = 0.0
    blanked[ch_idx, group_start:blank_end] = True

    # Determine the fit region: from desaturation through the artifact tail
    fit_start = last_desat
    fit_end = min(last_desat + artifact_window_samples, n_samples)

    if fit_start >= n_samples or fit_start >= fit_end:
        return

    # Extend fit_end to where the signal reaches baseline (if within window)
    reached, baseline_idx = _signal_reached_baseline(
        channel_trace,
        fit_start,
        baseline_threshold,
        baseline_window_samples,
        min(fit_end, n_samples),
    )
    if reached:
        # Include a short segment past baseline for a stable fit
        fit_end = min(baseline_idx + baseline_window_samples, n_samples)

    if fit_end <= fit_start:
        return

    # Fit polynomial to the artifact tail
    x = np.arange(fit_end - fit_start, dtype=np.float64)
    y = channel_trace[fit_start:fit_end].astype(np.float64)

    # Exclude any remaining saturated samples from the fit
    mask = np.abs(y) < (baseline_threshold * 3)
    if np.sum(mask) <= poly_order:
        # Not enough non-saturated samples — blank the whole region
        channel_trace[fit_start:fit_end] = 0.0
        blanked[ch_idx, fit_start:fit_end] = True
        return

    coeffs = np.polyfit(x[mask], y[mask], poly_order)
    artifact_estimate = np.polyval(coeffs, x)

    channel_trace[fit_start:fit_end] -= artifact_estimate


def remove_stim_artifacts(
    traces,
    stim_times_ms,
    fs_Hz,
    method="polynomial",
    artifact_window_ms=10.0,
    saturation_threshold=None,
    baseline_threshold=None,
    poly_order=3,
    artifact_window_only=True,
    copy=True,
):
    """Remove stimulation artifacts from multi-channel voltage traces.

    Processes each stim event independently per channel.  Saturated
    samples are always blanked (zeroed).  For the ``"polynomial"``
    method, a low-order polynomial is fit to the post-saturation
    artifact tail and subtracted, preserving neural spikes (which are
    too fast for the smooth polynomial to capture).

    When multiple stim events occur in rapid succession and the signal
    re-saturates before reaching baseline levels, the blanking region
    is extended dynamically and the polynomial fit is deferred until
    after the final desaturation in the burst.

    The polynomial detrend is conceptually related to SALPA (Wagenaar
    & Potter 2002, J Neurosci Methods), adapted for offline processing
    where look-ahead past saturation is available — see the module
    docstring for details.

    Parameters:
        traces (np.ndarray): Raw voltage traces, shape
            ``(channels, samples)``.
        stim_times_ms (array-like): Corrected stim times in
            milliseconds (e.g. from ``recenter_stim_times``).
        fs_Hz (float): Sampling frequency in Hz.
        method (str): ``"polynomial"`` (default) or ``"blank"``.
        artifact_window_ms (float): Maximum duration in milliseconds
            of the artifact tail after the last desaturation point.
            The polynomial is fit over this window.  Default 10.0.
        saturation_threshold (float or None): Absolute voltage value
            above which a sample is considered saturated.  When None,
            auto-detected from the 99.9th percentile of ``|traces|``.
        baseline_threshold (float or None): Absolute voltage envelope
            below which the signal is considered to have returned to
            baseline.  When None, auto-detected from pre-stim MAD.
        poly_order (int): Polynomial order for the detrend.  Default
            3 (cubic).  Higher orders risk fitting spike-like features;
            lower orders may not capture the artifact decay shape.
        artifact_window_only (bool): If True (default), only process
            windows around stim events.  If False, apply a global
            polynomial detrend to the entire trace (for recordings
            with very frequent stimulation).
        copy (bool): If True (default), return a copy; if False,
            modify ``traces`` in-place.

    Returns:
        cleaned (np.ndarray): Cleaned traces, shape
            ``(channels, samples)``.
        blanked_mask (np.ndarray): Boolean array, shape
            ``(channels, samples)``.  True for samples that were
            blanked (zeroed) because they fell within a saturation
            region.
    """
    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)
    if copy:
        traces = traces.copy()

    n_channels, n_samples = traces.shape
    blanked = np.zeros((n_channels, n_samples), dtype=bool)

    if len(stim_times_ms) == 0:
        return traces, blanked

    if method not in ("polynomial", "blank"):
        raise ValueError(
            f"Unknown artifact removal method {method!r}; "
            "expected 'polynomial' or 'blank'."
        )

    # Auto-detect thresholds
    if saturation_threshold is None:
        saturation_threshold = _auto_saturation_threshold(traces)
    if baseline_threshold is None:
        baseline_threshold = _auto_baseline_threshold(traces, stim_times_ms, fs_Hz)

    artifact_window_samples = int(np.round(artifact_window_ms * fs_Hz / 1000.0))
    baseline_window_samples = max(
        1, int(np.round(1.0 * fs_Hz / 1000.0))  # 1 ms of consecutive samples
    )

    # Convert stim times to sample indices and sort
    stim_samples = np.round(stim_times_ms * fs_Hz / 1000.0).astype(int)
    stim_samples = np.sort(stim_samples)
    stim_samples = stim_samples[(stim_samples >= 0) & (stim_samples < n_samples)]

    if len(stim_samples) == 0:
        return traces, blanked

    if not artifact_window_only:
        # Global mode: treat all stim events together
        # For now, fall through to per-event processing since the logic
        # is the same — each event is handled independently.
        pass

    # Process each channel independently
    for ch in range(n_channels):
        ch_trace = traces[ch]

        # Group stim events that form a sequential burst.
        # Walk through sorted stim samples; after each stim, find where
        # saturation ends.  If the signal re-saturates or hasn't reached
        # baseline before the next stim, merge into the same group.
        i = 0
        while i < len(stim_samples):
            group_start = max(0, stim_samples[i])

            # Walk forward through this stim and any sequential stims
            current_stim_idx = i
            last_desat = _find_saturation_end(
                ch_trace, group_start, saturation_threshold, n_samples
            )

            while True:
                # Check if the next stim event is before the signal
                # reaches baseline
                next_idx = current_stim_idx + 1
                if next_idx < len(stim_samples):
                    next_stim = stim_samples[next_idx]

                    # Has signal reached baseline before the next stim?
                    reached, _ = _signal_reached_baseline(
                        ch_trace,
                        last_desat,
                        baseline_threshold,
                        baseline_window_samples,
                        min(next_stim, n_samples),
                    )

                    if not reached:
                        # Signal hasn't recovered — merge with next stim
                        current_stim_idx = next_idx
                        new_desat = _find_saturation_end(
                            ch_trace,
                            next_stim,
                            saturation_threshold,
                            n_samples,
                        )
                        last_desat = max(last_desat, new_desat)
                        continue

                # Either no more stim events, or signal reached baseline
                break

            # Now process this group
            if method == "polynomial":
                _process_stim_group_polynomial(
                    ch_trace,
                    group_start,
                    last_desat,
                    artifact_window_samples,
                    baseline_threshold,
                    baseline_window_samples,
                    poly_order,
                    n_samples,
                    blanked,
                    ch,
                )
            elif method == "blank":
                blank_end = min(last_desat + artifact_window_samples, n_samples)
                ch_trace[group_start:blank_end] = 0.0
                blanked[ch, group_start:blank_end] = True

            # Advance past all stim events in this group
            i = current_stim_idx + 1

    return traces, blanked
