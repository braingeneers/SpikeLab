"""Stimulation time recentering.

Finds the actual stimulation artifact onset near each logged stim time
by detecting a chosen alignment point in the raw voltage traces:

* ``"abs_max"`` (default): sample with the largest ``|voltage|`` across
  channels — appropriate for monophasic pulses where there is a single
  artifact peak.
* ``"pos_peak"`` / ``"neg_peak"``: sample with the largest positive or
  most negative voltage in a top-K summed reference trace.
* ``"down_edge"``: up→down transition of a biphasic anodic-first pulse.
  First finds the negative peak in the search window, then the positive
  peak within a preceding ``prewindow_ms``, then returns the first
  positive-to-negative zero-crossing between them (falling back to the
  steepest negative slope if the signal does not cross zero).  This is
  the moment at which the stim current reverses direction — the AP
  trigger point for biphasic anodic-first protocols.
* ``"up_edge"``: symmetric version for biphasic cathodic-first pulses.

For the signed modes (``pos_peak``, ``neg_peak``, ``down_edge``,
``up_edge``) the reference trace is the *sum* of the top-K highest-
amplitude channels rather than the per-sample max.  Summing preserves
phase information (biphasic transitions add coherently across nearby
channels that see the same artifact; uncorrelated noise cancels) and
yields cleaner derivatives for edge detection.
"""

import warnings

import numpy as np


def _build_reference_trace(traces, n_reference_channels):
    """Return a single reference trace by summing the top-K channels
    by peak ``|voltage|``.

    Parameters:
        traces (np.ndarray): ``(channels, samples)``.
        n_reference_channels (int): K.  Clamped to ``[1, n_channels]``.

    Returns:
        reference (np.ndarray): Signed ``(samples,)`` array.
    """
    chan_amps = np.max(np.abs(traces), axis=1)
    k = max(1, min(int(n_reference_channels), traces.shape[0]))
    top_k_idx = np.argpartition(chan_amps, -k)[-k:]
    return np.sum(traces[top_k_idx], axis=0)


def _find_down_edge(reference, lo, hi, prewindow_ms, fs_Hz, neg_peak=None):
    """Find the up→down transition in a biphasic pulse.

    Algorithm:
      1. Find the negative peak in ``reference[lo:hi]`` (or use the
         caller-supplied ``neg_peak`` for multi-peak recentering).
      2. Find the positive peak in the window
         ``[max(lo, neg_peak - prewindow_samples), neg_peak)``.
      3. Transition = first positive-to-negative zero-crossing in
         ``reference[pos_peak:neg_peak + 1]``.
      4. If the signal does not cross zero (e.g. DC offset), fall back
         to the sample with the steepest negative slope in the same
         interval.
      5. If the pre-window is empty (negative peak at ``lo``), return
         the negative peak.
    """
    if neg_peak is None:
        neg_peak = lo + int(np.argmin(reference[lo:hi]))

    prewindow_samples = max(1, int(round(prewindow_ms * fs_Hz / 1000.0)))
    pre_lo = max(lo, neg_peak - prewindow_samples)
    pre_hi = neg_peak  # exclusive
    if pre_hi <= pre_lo:
        return neg_peak
    pos_peak = pre_lo + int(np.argmax(reference[pre_lo:pre_hi]))

    segment = reference[pos_peak : neg_peak + 1]
    # Sign transitions: +V followed by -V (or zero).  np.diff(sign) is
    # strictly negative at a + → - crossing.
    signs = np.sign(segment)
    sign_diffs = np.diff(signs)
    crossings = np.where(sign_diffs < 0)[0]
    if crossings.size > 0:
        return pos_peak + int(crossings[0])

    # Fallback: steepest negative slope inside the pos→neg interval.
    diffs = np.diff(segment)
    if diffs.size == 0:
        return neg_peak
    return pos_peak + int(np.argmin(diffs))


def _find_up_edge(reference, lo, hi, prewindow_ms, fs_Hz, pos_peak=None):
    """Symmetric to ``_find_down_edge`` for biphasic cathodic-first.

    Finds the positive peak (or uses caller-supplied ``pos_peak``), then
    the negative peak in a pre-window before it, then the first
    negative-to-positive zero-crossing between them.
    """
    if pos_peak is None:
        pos_peak = lo + int(np.argmax(reference[lo:hi]))

    prewindow_samples = max(1, int(round(prewindow_ms * fs_Hz / 1000.0)))
    pre_lo = max(lo, pos_peak - prewindow_samples)
    pre_hi = pos_peak
    if pre_hi <= pre_lo:
        return pos_peak
    neg_peak = pre_lo + int(np.argmin(reference[pre_lo:pre_hi]))

    segment = reference[neg_peak : pos_peak + 1]
    signs = np.sign(segment)
    sign_diffs = np.diff(signs)
    crossings = np.where(sign_diffs > 0)[0]
    if crossings.size > 0:
        return neg_peak + int(crossings[0])

    diffs = np.diff(segment)
    if diffs.size == 0:
        return pos_peak
    return neg_peak + int(np.argmax(diffs))


_VALID_PEAK_MODES = ("abs_max", "pos_peak", "neg_peak", "down_edge", "up_edge")
_VALID_MULTI_PEAK_SELECT = ("first", "last")


def _multi_peak_anchor(
    reference,
    lo,
    hi,
    peak_mode,
    multi_peak_select,
    multi_peak_threshold,
    multi_peak_min_separation_ms,
    fs_Hz,
):
    """Pick a single anchor sample from possibly multiple peaks in ``reference[lo:hi]``.

    Used for stimulation trains: the search window is wide enough to span
    several pulses, and a simple ``argmax``/``argmin`` would arbitrarily
    pick whichever pulse happened to have the largest amplitude.  This
    helper finds **all** local peaks in the window whose amplitude is at
    least ``multi_peak_threshold * max_peak_amplitude``, then returns the
    first (or last) one — guaranteeing alignment to a chosen end of the
    train regardless of pulse-to-pulse amplitude variation.

    The "search signal" used for peak detection depends on the
    ``peak_mode``:
      * ``abs_max``: ``|reference|`` — peaks of unsigned amplitude.
      * ``pos_peak``, ``up_edge``: positive lobe — peaks of ``reference``.
      * ``neg_peak``, ``down_edge``: negative lobe — peaks of
        ``-reference``.

    For ``down_edge`` / ``up_edge``, the returned anchor is the sample of
    the main artifact peak (negative or positive resp.) of the chosen
    pulse — not the zero-crossing.  The caller threads this anchor into
    :func:`_find_down_edge` / :func:`_find_up_edge` to compute the
    zero-crossing relative to that anchor.

    Returns:
        anchor_sample (int): Sample index in the global ``reference``
            array (i.e. already offset by ``lo``).  Falls back to single-
            peak ``argmax``/``argmin`` if no peaks are found above
            threshold (e.g. very short window or pure noise).
    """
    from scipy.signal import find_peaks  # lazy import

    segment = reference[lo:hi]
    if segment.size == 0:
        return lo

    if peak_mode == "abs_max":
        search = np.abs(segment)
    elif peak_mode in ("pos_peak", "up_edge"):
        search = np.maximum(segment, 0.0)
    else:  # neg_peak, down_edge
        search = -np.minimum(segment, 0.0)

    distance_samples = max(1, int(round(multi_peak_min_separation_ms * fs_Hz / 1000.0)))
    if search.max() <= 0:
        # All-zero or all-wrong-sign window → degrade to argmax/argmin
        if peak_mode in ("neg_peak", "down_edge"):
            return lo + int(np.argmin(segment))
        return lo + int(np.argmax(segment))

    threshold_abs = multi_peak_threshold * float(search.max())
    peak_idxs, _ = find_peaks(search, height=threshold_abs, distance=distance_samples)

    if peak_idxs.size == 0:
        # No interior local maxima above threshold (e.g. monotonic ramp).
        # Fall back to the single best sample in the window.
        if peak_mode in ("neg_peak", "down_edge"):
            return lo + int(np.argmin(segment))
        return lo + int(np.argmax(segment))

    if multi_peak_select == "first":
        chosen_local = int(peak_idxs[0])
    else:  # last
        chosen_local = int(peak_idxs[-1])
    return lo + chosen_local


def recenter_stim_times(
    traces,
    stim_times_ms,
    fs_Hz,
    max_offset_ms=50.0,
    *,
    peak_mode="abs_max",
    n_reference_channels=8,
    prewindow_ms=5.0,
    warn_offset_ms=3.0,
    multi_peak=False,
    multi_peak_select="first",
    multi_peak_threshold=0.6,
    multi_peak_min_separation_ms=2.0,
):
    """Find actual stimulation artifact times near logged stim times.

    For each logged stim time, searches a window of ``±max_offset_ms``
    in the raw voltage traces and returns the sample at the alignment
    point selected by ``peak_mode``.  This corrects for timing offsets
    between the stimulation hardware trigger log and the artifact in
    the recording.

    Parameters:
        traces (np.ndarray): Raw voltage traces, shape
            ``(channels, samples)``.
        stim_times_ms (array-like): Logged stimulation event times in
            milliseconds.  Need not be sorted.
        fs_Hz (float): Sampling frequency in Hz.
        max_offset_ms (float): Radius of the search window around
            each logged stim time, in milliseconds.  Default 50.0.
        peak_mode (str): Alignment target.  One of:
            * ``"abs_max"`` (default): largest ``|voltage|`` across
              channels.  Backward-compatible with the pre-``peak_mode``
              API.
            * ``"pos_peak"``: largest positive voltage in the top-K
              summed reference trace.
            * ``"neg_peak"``: most negative voltage in the top-K
              summed reference.
            * ``"down_edge"``: up→down transition for biphasic
              anodic-first pulses (see module docstring).
            * ``"up_edge"``: down→up transition for biphasic
              cathodic-first pulses.
        n_reference_channels (int): Number of highest-amplitude
            channels summed to build the signed reference trace for
            non-``abs_max`` modes.  Default ``8``.  Ignored for
            ``abs_max``.
        prewindow_ms (float): For ``down_edge`` / ``up_edge``, radius
            of the pre-window in which to search for the preceding
            opposite-polarity peak.  Default ``5.0``.
        warn_offset_ms (float or None): When the median ``|corrected -
            logged|`` shift exceeds this threshold, emit a
            ``UserWarning``.  A large systematic shift usually means a
            fixed hardware-vs-log delay, a wrong time column in the
            stim log, or a unit mismatch (ms vs s vs samples) rather
            than genuine jitter.  Set to ``None`` to silence.  Default
            ``3.0`` ms — well above one-sample jitter at 20–30 kHz.
        multi_peak (bool): Opt-in support for multi-pulse stim trains.
            When ``True``, the search window is treated as potentially
            containing multiple stimulation pulses (e.g. a 100 Hz
            train), and the alignment target is the **first** or
            **last** qualifying pulse rather than the strongest one.
            Default ``False`` — preserves backward-compatible single-
            peak behavior.
        multi_peak_select (str): When ``multi_peak=True``, which
            qualifying peak to lock onto.  ``"first"`` (default) =
            first pulse onset (matches "first-pulse alignment" used
            for train PSTHs).  ``"last"`` = last pulse onset (useful
            for studying after-train rebound).  Ignored when
            ``multi_peak=False``.
        multi_peak_threshold (float): When ``multi_peak=True``, only
            peaks whose amplitude is at least this fraction of the
            largest peak in the search window are considered "real
            pulses".  Default ``0.6`` — accepts pulses up to 40% weaker
            than the strongest while still rejecting noise.
        multi_peak_min_separation_ms (float): When ``multi_peak=True``,
            the minimum spacing between candidate peaks.  Prevents
            multi-sample peaks of a single pulse from being counted as
            separate pulses.  Default ``2.0`` ms — well below any
            sensible inter-pulse interval (5 ms = 200 Hz; 10 ms =
            100 Hz).

    Returns:
        corrected_ms (np.ndarray): Corrected stim times in
            milliseconds, same length as ``stim_times_ms``.  Events
            whose search window extends outside the recording are
            clipped to the recording boundary.

    Notes:
        * When multiple stim events have overlapping search windows,
          each is recentered independently.
        * For monophasic pulses the ``*_edge`` modes degrade
          gracefully: the pre-window search returns the opposite
          polarity's noise peak and the zero-crossing fallback lands
          near the onset of the single artifact — but ``pos_peak`` /
          ``neg_peak`` will give cleaner results in that case.
        * For single-pulse stim, ``multi_peak=True`` degrades to the
          original single-peak behavior (only one peak in the window
          is above threshold; first==last).  Set it always-on if you
          mix single-pulse and train conditions in one recording.
    """
    if peak_mode not in _VALID_PEAK_MODES:
        raise ValueError(
            f"Unknown peak_mode {peak_mode!r}; " f"expected one of {_VALID_PEAK_MODES}"
        )
    if multi_peak and multi_peak_select not in _VALID_MULTI_PEAK_SELECT:
        raise ValueError(
            f"Unknown multi_peak_select {multi_peak_select!r}; "
            f"expected one of {_VALID_MULTI_PEAK_SELECT}"
        )
    if multi_peak and not (0.0 < multi_peak_threshold <= 1.0):
        raise ValueError(
            f"multi_peak_threshold must be in (0, 1]; got {multi_peak_threshold}"
        )

    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)
    n_samples = traces.shape[1]
    offset_samples = int(np.round(max_offset_ms * fs_Hz / 1000.0))

    # Reference trace: unsigned max-of-abs for abs_max (backward compat),
    # signed top-K sum for all other modes.
    if peak_mode == "abs_max":
        reference = np.max(np.abs(traces), axis=0)
    else:
        reference = _build_reference_trace(traces, n_reference_channels)

    corrected = np.empty_like(stim_times_ms)
    for i, t_ms in enumerate(stim_times_ms):
        center = int(np.round(t_ms * fs_Hz / 1000.0))
        lo = max(0, center - offset_samples)
        hi = min(n_samples, center + offset_samples + 1)

        if multi_peak:
            anchor = _multi_peak_anchor(
                reference,
                lo,
                hi,
                peak_mode,
                multi_peak_select,
                multi_peak_threshold,
                multi_peak_min_separation_ms,
                fs_Hz,
            )
            if peak_mode == "abs_max":
                peak_sample = anchor
            elif peak_mode == "pos_peak":
                peak_sample = anchor
            elif peak_mode == "neg_peak":
                peak_sample = anchor
            elif peak_mode == "down_edge":
                # anchor IS the negative-peak sample of the chosen pulse;
                # pre-window search runs ahead of it for the pos peak.
                peak_sample = _find_down_edge(
                    reference, lo, hi, prewindow_ms, fs_Hz, neg_peak=anchor
                )
            else:  # up_edge
                peak_sample = _find_up_edge(
                    reference, lo, hi, prewindow_ms, fs_Hz, pos_peak=anchor
                )
        else:
            if peak_mode == "abs_max":
                peak_sample = lo + int(np.argmax(reference[lo:hi]))
            elif peak_mode == "pos_peak":
                peak_sample = lo + int(np.argmax(reference[lo:hi]))
            elif peak_mode == "neg_peak":
                peak_sample = lo + int(np.argmin(reference[lo:hi]))
            elif peak_mode == "down_edge":
                peak_sample = _find_down_edge(reference, lo, hi, prewindow_ms, fs_Hz)
            else:  # up_edge
                peak_sample = _find_up_edge(reference, lo, hi, prewindow_ms, fs_Hz)

        corrected[i] = peak_sample / fs_Hz * 1000.0

    if warn_offset_ms is not None and len(stim_times_ms) > 0:
        median_abs_offset_ms = float(np.median(np.abs(corrected - stim_times_ms)))
        if median_abs_offset_ms > warn_offset_ms:
            warnings.warn(
                f"recenter_stim_times: median |offset| = "
                f"{median_abs_offset_ms:.2f} ms over {len(stim_times_ms)} events "
                f"exceeds warn_offset_ms ({warn_offset_ms} ms).  This is well "
                f"above one-sample jitter and usually indicates a fixed "
                f"hardware-vs-log delay, a wrong time column in the stim log, "
                f"or a unit mismatch (ms vs s vs samples).  Verify against a "
                f"test pulse, then either accept the shift or pass "
                f"warn_offset_ms=None to silence.",
                UserWarning,
                stacklevel=2,
            )

    return corrected
