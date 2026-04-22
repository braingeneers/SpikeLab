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


def _saturation_threshold_from_recording(
    recording, traces, frac=0.95, min_clip_samples=10
):
    """Derive a saturation threshold (µV) from gain metadata + observed extremes.

    Returns ``+inf`` (i.e. "do not blank anything") when the recording
    shows no evidence of ADC clipping.  Otherwise returns
    ``frac * round(max_uV / gain_uV_per_bit) * gain_uV_per_bit`` — the
    observed rail rounded to a whole number of raw ADC bits and pulled
    in by ``frac``.

    Saturation detection: a hard ADC clip produces many samples pinned
    at the rail (a flat top in the amplitude histogram).  A single
    large spike produces exactly one sample at the maximum.  We count
    samples within one raw bit of ``max(|traces|)``; if fewer than
    ``min_clip_samples``, we treat the recording as unsaturated and
    return ``+inf``.  This means "blank only completely saturated
    electrodes" semantics: high-amplitude artifacts that didn't reach
    the rail are left intact for the polynomial detrend.

    Why use the gain at all (vs. raw ``frac * max(|traces|)``):
      * Read from the recording's ``get_channel_gains()``, which on a
        SpikeInterface chain is propagated from the underlying integer
        extractor (e.g. ``MaxwellRecordingExtractor``).
      * Rounding the observed max to a whole number of raw bits anchors
        the threshold to a hardware-meaningful value, not a
        floating-point artefact of preprocessing arithmetic — two
        recordings of the same probe at the same gain settings produce
        the same threshold.
      * The "within one raw bit of max" tolerance for the clip-detection
        count is also gain-anchored, not arbitrary.

    Parameters:
        recording (BaseRecording or None): SpikeInterface recording
            exposing ``get_channel_gains()``.  When ``None`` or no
            gains available, falls back to a 1.0 µV/bit assumption.
        traces (np.ndarray): ``(channels, samples)`` already scaled to µV.
        frac (float): Fraction of the rail to use as threshold.  Default
            ``0.95`` — leaves ~5% margin so samples within the top bit
            of the rail still count as saturated.
        min_clip_samples (int): Minimum number of samples within one raw
            bit of ``max(|traces|)`` required to consider the recording
            saturated.  Below this, the recording is treated as
            unsaturated and the function returns ``+inf``.  Default
            ``10`` — high enough to ignore single-spike maxima and
            small numbers of outlier samples, low enough to catch even
            very sparse stimulation protocols.

    Returns:
        threshold (float, µV) — finite if saturation detected, ``+inf``
        if not.
    """
    abs_traces = np.abs(traces)
    observed_max_uV = float(np.max(abs_traces))

    # Resolve gain
    gain_uV_per_bit = 1.0
    if recording is not None:
        try:
            gains = recording.get_channel_gains()
        except (AttributeError, NotImplementedError):
            gains = None
        if gains is not None and len(gains) > 0:
            gain_uV_per_bit = max(1e-9, float(np.max(np.abs(gains))))

    # Saturation detection: how many samples sit at or just below the
    # observed max?  A hard clip pins many samples there; a single big
    # spike is just one sample.
    n_at_rail = int(np.sum(abs_traces >= observed_max_uV - gain_uV_per_bit))
    if n_at_rail < min_clip_samples:
        return float("inf")

    observed_rail_bits = round(observed_max_uV / gain_uV_per_bit)
    return frac * observed_rail_bits * gain_uV_per_bit


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


def _find_saturation_end_from_mask(mask_ch, start, n_samples):
    """Variant of ``_find_saturation_end`` driven by a pre-computed
    clip mask rather than an amplitude threshold.

    When the caller has raw (pre-filter) traces available, it is more
    correct to identify saturated samples from the raw signal and pass
    a boolean mask here — bandpass filtering of a stim artifact
    produces ringing whose amplitude can exceed the raw ADC rail even
    on samples that weren't actually clipped.
    """
    idx = start
    while idx < n_samples and mask_ch[idx]:
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


_MIN_DESCENT_SAMPLES = 2  # min samples between fit_start and neg-peak to split


def _polyfit_and_subtract(channel_trace, blanked, ch_idx, lo, hi, poly_order):
    """Fit a polynomial to ``channel_trace[lo:hi]`` (excluding blanked
    samples) and subtract it in-place.

    If too few non-blanked samples remain to support the fit (e.g.
    because Fit 1 of the auto-split landed on a very short descent
    window between the recentered stim time and the negative peak),
    the region is left untouched rather than blanked — those samples
    are not saturated, just covered by a window too small for a
    reliable polynomial of this order.
    """
    if hi <= lo:
        return
    x = np.arange(hi - lo, dtype=np.float64)
    y = channel_trace[lo:hi].astype(np.float64)
    mask = ~blanked[ch_idx, lo:hi]
    if np.sum(mask) <= poly_order:
        return
    coeffs = np.polyfit(x[mask], y[mask], poly_order)
    channel_trace[lo:hi] -= np.polyval(coeffs, x)


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
    pre_artifact_samples=0,  # accepted for API stability, currently unused
    clip_mask_ch=None,  # accepted for API stability, currently unused
):
    """Polynomial detrend for one stim group on one channel.

    Workflow per stim group:
      1. Blank ``[group_start, last_desat)`` (any genuine ADC clip).
      2. Determine the fit window
         ``[fit_start = last_desat, fit_end = last_desat + artifact_window]``,
         extending ``fit_end`` to where the signal returns to baseline.
      3. Locate the negative peak (``argmin``) inside the window, and
         the subsequent positive peak (``argmax`` after the negative
         peak) inside the window.
      4. Split the fit at the meaningful peaks and run an independent
         polynomial on each segment:

         * **3-fit split** (descent + ascent + decay) when both a
           descent of ≥ ``_MIN_DESCENT_SAMPLES`` and an ascent of
           ≥ ``_MIN_DESCENT_SAMPLES`` exist:
              - Fit A: ``[fit_start, neg_peak]`` — descent.
              - Fit B: ``[neg_peak, pos_peak]`` — ascent through zero
                up to the post-artifact positive overshoot.
              - Fit C: ``[pos_peak, fit_end]`` — decay back to
                baseline (the original implementation).
           This is the typical biphasic anodic-first case sorted with
           ``peak_mode="down_edge"``: the post-stim signal goes down,
           up through zero, may overshoot, and decays.

         * **2-fit split** (descent + tail) when there is a descent
           but no meaningful ascent before ``fit_end``:
              - Fit A: ``[fit_start, neg_peak]``
              - Fit B+C: ``[neg_peak, fit_end]`` — single tail fit.

         * **Single fit** when there's no descent (stim time is
           already at or essentially at the negative peak — e.g.
           ``peak_mode="abs_max"`` or ``"neg_peak"``):
              - Fit C: ``[fit_start, fit_end]``.

      Each segment is monotonic-ish, so a low-order polynomial (cubic)
      fits each well; one polynomial trying to fit the full
      down-up-down shape would have to interpolate two inflection
      points and leaves residuals.
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
        # Anchor the fit polynomial to a span of clean baseline
        # samples beyond the artifact tail.  Without this anchor the
        # cubic had freedom to curl in the trailing region and left a
        # small step at the boundary between the subtracted region
        # and the un-touched baseline tail.  Extending 3 ms past
        # ``baseline_idx`` (≈1 ms for the detection window + 2 ms of
        # additional anchor) gives the polynomial enough "known-
        # baseline" points to be pulled naturally toward zero at the
        # boundary without over-extending the fit.
        fit_end = min(baseline_idx + 3 * baseline_window_samples, n_samples)

    if fit_end <= fit_start:
        return

    # Locate the negative peak in the fit window, then the subsequent
    # positive peak.  Both indices are computed on the un-modified
    # trace before any subtraction so the splits are stable.
    neg_peak_offset = int(np.argmin(channel_trace[fit_start:fit_end]))
    neg_peak_sample = fit_start + neg_peak_offset

    if neg_peak_sample + 1 < fit_end:
        pos_peak_offset_after = int(
            np.argmax(channel_trace[neg_peak_sample + 1 : fit_end])
        )
        pos_peak_sample = neg_peak_sample + 1 + pos_peak_offset_after
    else:
        pos_peak_sample = neg_peak_sample  # no room for a subsequent peak

    descent_samples = neg_peak_offset
    ascent_samples = pos_peak_sample - neg_peak_sample

    has_descent = descent_samples >= _MIN_DESCENT_SAMPLES
    has_ascent = ascent_samples >= _MIN_DESCENT_SAMPLES

    if has_descent and has_ascent:
        # 3-fit split: descent + ascent + decay
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            fit_start,
            neg_peak_sample + 1,
            poly_order,
        )
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            neg_peak_sample + 1,
            pos_peak_sample + 1,
            poly_order,
        )
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            pos_peak_sample + 1,
            fit_end,
            poly_order,
        )
    elif has_descent:
        # 2-fit split: descent + tail (no positive overshoot found)
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            fit_start,
            neg_peak_sample + 1,
            poly_order,
        )
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            neg_peak_sample + 1,
            fit_end,
            poly_order,
        )
    else:
        # No descent — stim already at neg peak; single fit.
        _polyfit_and_subtract(
            channel_trace,
            blanked,
            ch_idx,
            fit_start,
            fit_end,
            poly_order,
        )


def _global_polynomial_detrend(
    channel_trace,
    window_samples,
    overlap_samples,
    saturation_threshold,
    poly_order,
    n_samples,
    blanked,
    ch_idx,
):
    """Sliding-window polynomial detrend applied to an entire channel.

    Divides the trace into overlapping windows, fits a polynomial to
    the non-saturated samples in each window, and subtracts the fit.
    Overlap regions are blended with a linear crossfade to avoid
    discontinuities at window boundaries.  Saturated samples are
    blanked.

    Parameters:
        channel_trace (np.ndarray): 1-D trace (modified in-place).
        window_samples (int): Window length in samples.
        overlap_samples (int): Overlap between consecutive windows.
        saturation_threshold (float): Absolute voltage saturation level.
        poly_order (int): Polynomial order for the detrend.
        n_samples (int): Trace length.
        blanked (np.ndarray): 2-D boolean mask ``(channels, samples)``,
            modified in-place.
        ch_idx (int): Channel index for the blanked mask.
    """
    step = window_samples - overlap_samples
    if step < 1:
        step = 1

    # Pre-compute the output buffer so we can blend overlaps
    output = np.zeros(n_samples, dtype=np.float64)
    weight = np.zeros(n_samples, dtype=np.float64)

    start = 0
    while start < n_samples:
        end = min(start + window_samples, n_samples)
        seg = channel_trace[start:end].astype(np.float64)
        seg_len = end - start

        # Mark saturated samples
        sat_mask = np.abs(seg) >= saturation_threshold
        if np.any(sat_mask):
            blanked[ch_idx, start:end] |= sat_mask

        fit_mask = ~sat_mask & ~np.isnan(seg)
        if np.sum(fit_mask) > poly_order:
            x = np.arange(seg_len, dtype=np.float64)
            coeffs = np.polyfit(x[fit_mask], seg[fit_mask], poly_order)
            artifact_estimate = np.polyval(coeffs, x)
            detrended = seg - artifact_estimate
        else:
            # Not enough non-saturated samples — zero out
            detrended = np.zeros(seg_len)
            blanked[ch_idx, start:end] = True

        # Zero saturated samples in the detrended output
        detrended[sat_mask] = 0.0

        # Build a blending window (linear ramps in overlap regions)
        w = np.ones(seg_len)
        if start > 0 and overlap_samples > 0:
            ramp_len = min(overlap_samples, seg_len)
            w[:ramp_len] = np.linspace(0, 1, ramp_len)
        if end < n_samples and overlap_samples > 0:
            ramp_len = min(overlap_samples, seg_len)
            w[-ramp_len:] = np.linspace(1, 0, ramp_len)

        output[start:end] += detrended * w
        weight[start:end] += w

        start += step

    # Normalize by blending weights
    nonzero = weight > 0
    channel_trace[nonzero] = output[nonzero] / weight[nonzero]
    channel_trace[~nonzero] = 0.0


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
    *,
    recording=None,
    raw_traces=None,
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

            Note: when the post-stim window contains a clear descent
            from the recentered stim time to a subsequent negative
            peak (typical for biphasic anodic-first pulses sorted with
            ``peak_mode="down_edge"``), the fit is automatically split
            into two independent polynomials at the negative peak —
            one for ``[stim_time, neg_peak]`` (the descent) and one
            for ``[neg_peak, baseline_recovery]`` (the tail).  When
            the recentered stim time IS the negative peak (e.g.
            ``peak_mode="abs_max"`` or ``"neg_peak"``), no descent
            exists and a single fit is used.  This is automatic; no
            user knob.
        saturation_threshold (float or None): Absolute voltage value
            above which a sample is considered saturated.  When None,
            auto-detected — preferring gain-anchored detection from
            ``recording`` metadata when supplied (see ``recording``
            kwarg below), falling back to the 99.9th percentile of
            ``|traces|`` otherwise.
        raw_traces (np.ndarray or None): Optional pre-bandpass traces,
            same shape as ``traces``, used as the source of truth for
            saturation detection.  Bandpass filtering of a stim
            artifact produces ringing whose filtered amplitude can
            exceed the raw ADC rail even on unsaturated samples, so
            auto-detection from ``traces`` (filtered) both over-
            reports (ringing overshoot) and under-reports (group-delay
            smoothing) clips.  When provided, the threshold is derived
            from ``raw_traces`` and the clip mask is built from
            ``np.abs(raw_traces) >= threshold``; the filtered ``traces``
            are blanked at those same sample indices and polynomial-
            detrended around them.
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

    if traces.shape[0] == 0 or traces.shape[1] == 0:
        raise ValueError(
            f"traces must have at least one channel and one sample, "
            f"got shape {traces.shape}"
        )

    # Pick the source of truth for saturation detection.  Prefer raw
    # (pre-bandpass) traces when provided — filter ringing after a stim
    # artifact can drive filtered samples past the raw ADC rail even
    # when nothing was actually clipped, so detecting on the filtered
    # signal both over-reports clips (filter overshoot) and under-
    # reports them (group delay + smoothing).  ``raw_traces`` is
    # typically the un-filtered ``ScaleRecording`` output extracted by
    # the caller; for a filtered-only path pass nothing and the
    # filtered ``traces`` will be used.
    detection_traces = raw_traces if raw_traces is not None else traces

    # Auto-detect thresholds.  Prefer gain-anchored detection when a
    # recording object is provided — anchors the threshold to actual
    # ADC bit boundaries and returns +inf when no clipping is detected,
    # so non-saturated recordings are left alone.  Falls back to the
    # quantile-based heuristic when no recording metadata is available.
    if saturation_threshold is None:
        if recording is not None:
            saturation_threshold = _saturation_threshold_from_recording(
                recording, detection_traces
            )
        else:
            saturation_threshold = _auto_saturation_threshold(detection_traces)

    # Pre-compute the clip mask once, from detection_traces.  All
    # downstream saturation checks read this mask instead of re-
    # computing ``|trace| >= threshold`` against the filtered signal.
    # When ``saturation_threshold`` is ``+inf`` (no clipping detected)
    # the mask is all-False and all blanking logic short-circuits.
    clip_mask = np.abs(detection_traces) >= saturation_threshold
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
        # Global mode: apply a sliding-window polynomial detrend to the
        # entire recording.  Useful when stimulation is so frequent that
        # artifact windows overlap or cover most of the trace, or when
        # stim timing information is unavailable.
        overlap_samples = artifact_window_samples // 2
        for ch in range(n_channels):
            if method == "polynomial":
                _global_polynomial_detrend(
                    traces[ch],
                    artifact_window_samples,
                    overlap_samples,
                    saturation_threshold,
                    poly_order,
                    n_samples,
                    blanked,
                    ch,
                )
            elif method == "blank":
                # Global blank: blank only the saturated samples
                sat = clip_mask[ch]
                traces[ch, sat] = 0.0
                blanked[ch, sat] = True
        return traces, blanked

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
            last_desat = _find_saturation_end_from_mask(
                clip_mask[ch], group_start, n_samples
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
                        new_desat = _find_saturation_end_from_mask(
                            clip_mask[ch],
                            next_stim,
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
