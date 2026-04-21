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


def _find_down_edge(reference, lo, hi, prewindow_ms, fs_Hz):
    """Find the up→down transition in a biphasic pulse.

    Algorithm:
      1. Find the negative peak in ``reference[lo:hi]``.
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


def _find_up_edge(reference, lo, hi, prewindow_ms, fs_Hz):
    """Symmetric to ``_find_down_edge`` for biphasic cathodic-first.

    Finds the positive peak, then the negative peak in a pre-window
    before it, then the first negative-to-positive zero-crossing
    between them.
    """
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


def recenter_stim_times(
    traces,
    stim_times_ms,
    fs_Hz,
    max_offset_ms=50.0,
    *,
    peak_mode="abs_max",
    n_reference_channels=8,
    prewindow_ms=5.0,
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
    """
    if peak_mode not in _VALID_PEAK_MODES:
        raise ValueError(
            f"Unknown peak_mode {peak_mode!r}; " f"expected one of {_VALID_PEAK_MODES}"
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

    return corrected
