"""Stimulation-aware spike sorting pipeline.

Applies pre-trained RT-Sort sequences (from the Phase 1 vanilla sort)
to a stimulation recording and returns per-event peri-stim
``SpikeSliceStack``.

The pipeline is **per-event-chunked** by default: only the peri-event
time window around each stim event (with buffers for recentering and
artifact removal) is ever materialised.  Peak RAM scales with a
single chunk's working set (typically ~100-200 MB on a 1018-ch MaxOne
recording) rather than with total recording duration.  The full
chunked path is used whenever the caller passes a path or a lazy
SpikeInterface recording.  Pre-materialised ``np.ndarray`` input
falls back to the legacy full-recording path (caller controls memory).

Per chunk, the pipeline:
  1. Read the chunk's filtered traces (from the top-level lazy
     recording) and the pre-filter traces (walking up to the first
     non-filter parent), DC-centering the latter per channel.
  2. Recenter the stim event time(s) within the chunk using the
     configured ``peak_mode`` (``"down_edge"`` for biphasic anodic-
     first pulses, etc.).
  3. Remove artifacts (auto 2- or 3-way polynomial split at the
     negative peak / subsequent positive peak, or single fit).
  4. Run ``RTSort.sort_offline`` on the cleaned chunk.
  5. Extract the peri-event ``[-pre_ms, +post_ms]`` slice per event.
  6. Drop the chunk; accumulate the per-event ``SpikeData`` slices.

Events whose chunks would overlap (e.g., burst / paired-pulse
protocols) are grouped into a single chunk so sort_offline sees them
together.
"""

from pathlib import Path

import numpy as np


# Extra margin beyond the peri-event + recentering + artifact-removal
# window.  Gives RT-Sort's detection model a few ms to warm up before
# the peri-event region starts, so the first few samples after the
# sort_offline reset are not in the output window.  The algorithm's
# internal buffer_size is 100 samples (~5 ms at 20 kHz), so 30 ms is
# a comfortable warmup — small enough that chunks for stim at 2 Hz
# (500 ms apart) do not merge into one big chunk.
_CHUNK_WARMUP_MS = 30.0


def sort_stim_recording(
    stim_recording,
    rt_sort,
    stim_times_ms,
    pre_ms,
    post_ms,
    fs_Hz=None,
    *,
    artifact_method="polynomial",
    artifact_window_ms=10.0,
    saturation_threshold=None,
    baseline_threshold=None,
    poly_order=3,
    artifact_window_only=True,
    max_stim_offset_ms=50.0,
    peak_mode="abs_max",
    n_reference_channels=8,
    prewindow_ms=5.0,
    model=None,
    model_path=None,
    recording_window_ms=None,
    verbose=True,
):
    """Sort spikes in a stimulation recording using pre-trained RT-Sort sequences.

    Takes a raw stimulation recording and a trained ``RTSort`` object
    (or path to a saved one produced by
    ``sort_recording(..., sorter="rt_sort")``), removes stimulation
    artifacts, runs offline spike sorting, and returns a
    ``SpikeSliceStack`` of sorted spikes aligned to the corrected
    stimulation event times.

    **Memory model.**  When ``stim_recording`` is a path or a lazy
    SpikeInterface recording, the pipeline processes one *per-event
    time chunk* at a time (peak RAM ≈ one chunk's working set,
    typically 100-200 MB on MaxOne — independent of recording
    duration).  When ``stim_recording`` is a pre-materialised
    ``np.ndarray``, the full-recording path is used instead (caller
    has already paid the memory cost).

    Parameters:
        stim_recording: The stimulation recording.  Can be:
            - ``str`` or ``Path`` to a recording file (Maxwell .h5 or
              NWB).  Chunked path.
            - A SpikeInterface ``BaseRecording`` object.  Chunked path.
            - ``np.ndarray`` of shape ``(channels, samples)``.
              Full-recording path (no chunking possible).
        rt_sort: The trained RT-Sort object or path to its pickle.
        stim_times_ms (array-like): Logged stimulation event times in
            milliseconds.
        pre_ms (float): Output peri-event window radius before each
            stim event, in milliseconds.
        post_ms (float): Output peri-event window radius after each
            stim event, in milliseconds.
        fs_Hz (float or None): Sampling frequency in Hz.  Required
            for ndarray input; inferred from the recording object
            otherwise.
        artifact_method (str): ``"polynomial"`` (default) or
            ``"blank"``.  Passed to ``remove_stim_artifacts``.
        artifact_window_ms (float): Max artifact tail duration after
            the last desaturation.  Default 10.0.
        saturation_threshold (float or None): Saturation voltage
            threshold.  None auto-detects (gain-anchored from
            recording metadata if available).
        baseline_threshold (float or None): Baseline envelope
            threshold.  None auto-detects from pre-stim MAD.
        poly_order (int): Polynomial order for detrend.  Default 3.
        artifact_window_only (bool): Only process around stim events.
            Default True.
        max_stim_offset_ms (float): Search window radius for stim
            time recentering.  Default 50.0.
        peak_mode (str): Alignment target for ``recenter_stim_times``.
            One of ``"abs_max"`` (default), ``"pos_peak"``,
            ``"neg_peak"``, ``"down_edge"``, ``"up_edge"``.  For
            biphasic anodic-first pulses where the AP is triggered at
            the up→down current reversal, use ``"down_edge"``.
        n_reference_channels (int): Top-K highest-amplitude channels
            summed to form the signed reference trace for non-
            ``abs_max`` peak modes.  Default 8.
        prewindow_ms (float): For ``down_edge`` / ``up_edge``, radius
            of the pre-window before the primary peak.  Default 5.0.
        model (ModelSpikeSorter or None): Detection model instance for
            ``load_rt_sort`` when ``rt_sort`` is a path.
        model_path (str or Path or None): Path to a detection model
            folder for ``load_rt_sort`` when ``rt_sort`` is a path.
        recording_window_ms (tuple or None): ``(start_ms, end_ms)``
            sub-window to restrict processing to.  Only events whose
            peri-event window falls entirely within this range are
            sorted.  ``None`` processes the full recording.
        verbose (bool): Print progress messages.  Default True.

    Returns:
        stim_slices (SpikeSliceStack): Event-aligned spike slice stack
            with one slice per (corrected) stim event.  Each slice
            spans ``[-pre_ms, +post_ms]`` relative to the stim time.
    """
    from ..rt_sort_runner import load_rt_sort  # noqa: F401  (validates install)

    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)

    # --- Load RTSort once --------------------------------------------
    rt_sort_obj = _load_rt_sort(rt_sort, model, model_path, verbose)

    # --- Dispatch on input type --------------------------------------
    if isinstance(stim_recording, np.ndarray):
        return _sort_stim_full_recording(
            traces=stim_recording,
            recording_obj=None,
            fs_Hz=fs_Hz,
            rt_sort_obj=rt_sort_obj,
            stim_times_ms=stim_times_ms,
            pre_ms=pre_ms,
            post_ms=post_ms,
            artifact_method=artifact_method,
            artifact_window_ms=artifact_window_ms,
            saturation_threshold=saturation_threshold,
            baseline_threshold=baseline_threshold,
            poly_order=poly_order,
            artifact_window_only=artifact_window_only,
            max_stim_offset_ms=max_stim_offset_ms,
            peak_mode=peak_mode,
            n_reference_channels=n_reference_channels,
            prewindow_ms=prewindow_ms,
            recording_window_ms=recording_window_ms,
            verbose=verbose,
        )

    # Path or BaseRecording → chunked path.
    if isinstance(stim_recording, (str, Path)):
        if verbose:
            print(f"Opening recording {stim_recording} (lazy, for chunked reads)...")
        from ..recording_io import load_single_recording

        rec = load_single_recording(stim_recording)
    else:
        rec = stim_recording

    return _sort_stim_chunked(
        recording=rec,
        rt_sort_obj=rt_sort_obj,
        stim_times_ms=stim_times_ms,
        pre_ms=pre_ms,
        post_ms=post_ms,
        artifact_method=artifact_method,
        artifact_window_ms=artifact_window_ms,
        saturation_threshold=saturation_threshold,
        baseline_threshold=baseline_threshold,
        poly_order=poly_order,
        artifact_window_only=artifact_window_only,
        max_stim_offset_ms=max_stim_offset_ms,
        peak_mode=peak_mode,
        n_reference_channels=n_reference_channels,
        prewindow_ms=prewindow_ms,
        recording_window_ms=recording_window_ms,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Chunked path — the default when we have a lazy recording
# ---------------------------------------------------------------------------


def _sort_stim_chunked(
    recording,
    rt_sort_obj,
    stim_times_ms,
    pre_ms,
    post_ms,
    *,
    artifact_method,
    artifact_window_ms,
    saturation_threshold,
    baseline_threshold,
    poly_order,
    artifact_window_only,
    max_stim_offset_ms,
    peak_mode,
    n_reference_channels,
    prewindow_ms,
    recording_window_ms,
    verbose,
):
    from ...spikedata.spikedata import SpikeData  # noqa: F401
    from ...spikedata.spikeslicestack import SpikeSliceStack
    from .artifact_removal import remove_stim_artifacts
    from .recentering import recenter_stim_times

    fs_Hz = float(recording.get_sampling_frequency())
    n_total_samples = int(recording.get_num_samples())
    raw_parent = _find_prefilter_parent(recording)

    # Chunk window budget — each chunk must encompass:
    #   pre_ms  + max_stim_offset_ms + _CHUNK_WARMUP_MS   (before each event)
    #   post_ms + artifact_window_ms + _CHUNK_WARMUP_MS   (after each event)
    # Note: ``max_stim_offset_ms`` is the search radius for *recentering*,
    # which only applies before the logged stim time (we look in the
    # recording to find where the artifact actually is).  It is NOT
    # needed in the post-window.
    chunk_pre_ms = pre_ms + max_stim_offset_ms + _CHUNK_WARMUP_MS
    chunk_post_ms = post_ms + artifact_window_ms + _CHUNK_WARMUP_MS

    # Filter events by the recording_window_ms + actual recording bounds.
    if recording_window_ms is not None:
        rwin_lo, rwin_hi = recording_window_ms
    else:
        rwin_lo, rwin_hi = 0.0, n_total_samples / fs_Hz * 1000.0
    rec_lo_ms = 0.0
    rec_hi_ms = n_total_samples / fs_Hz * 1000.0

    valid_mask = (
        (stim_times_ms - pre_ms >= rwin_lo)
        & (stim_times_ms + post_ms <= rwin_hi)
        & (stim_times_ms - chunk_pre_ms >= rec_lo_ms - 1e-6)
        & (stim_times_ms + chunk_post_ms <= rec_hi_ms + 1e-6)
    )
    n_dropped = int(np.sum(~valid_mask))
    if n_dropped > 0 and verbose:
        print(
            f"  Dropping {n_dropped} event(s) whose peri-event window would "
            f"extend outside the recording / recording_window_ms bounds"
        )
    global_event_indices = np.flatnonzero(valid_mask)
    if len(global_event_indices) == 0:
        raise ValueError(
            "No stim events left after filtering for recording bounds."
        )
    kept_times_ms = stim_times_ms[global_event_indices]

    # Group adjacent events into chunks.
    groups = _group_stim_events_into_chunks(
        kept_times_ms, chunk_pre_ms, chunk_post_ms
    )
    if verbose:
        print(
            f"Chunking {len(kept_times_ms)} stim events into {len(groups)} "
            f"time chunk(s)  (chunk window ≈ "
            f"{chunk_pre_ms + chunk_post_ms:.0f} ms per event)"
        )

    # Process chunks in order; accumulate per-event results keyed by
    # original stim_times_ms index so we can emit the final slice stack
    # in the caller's event order.
    per_event_sd = [None] * len(kept_times_ms)
    per_event_global_corr_ms = np.empty(len(kept_times_ms), dtype=np.float64)

    for chunk_idx, group in enumerate(groups):
        group_kept = np.asarray(group, dtype=int)  # indices into kept_times_ms
        group_events_ms = kept_times_ms[group_kept]

        chunk_lo_ms = float(group_events_ms[0] - chunk_pre_ms)
        chunk_hi_ms = float(group_events_ms[-1] + chunk_post_ms)
        start_frame = max(0, int(np.floor(chunk_lo_ms * fs_Hz / 1000.0)))
        end_frame = min(
            n_total_samples, int(np.ceil(chunk_hi_ms * fs_Hz / 1000.0))
        )
        chunk_start_ms = start_frame * 1000.0 / fs_Hz
        chunk_len_ms = (end_frame - start_frame) * 1000.0 / fs_Hz

        if verbose:
            print(
                f"  chunk {chunk_idx + 1}/{len(groups)}: "
                f"{len(group_events_ms)} event(s), "
                f"{chunk_len_ms:.0f} ms ({end_frame - start_frame} samples)"
            )

        # Load filtered chunk traces.
        chunk_traces = recording.get_traces(
            start_frame=start_frame, end_frame=end_frame, return_scaled=True
        ).T.astype(np.float32, copy=False)

        # Load matching pre-filter raw chunk and DC-center per channel
        # (in-place to avoid a transient duplicate allocation).
        chunk_raw = None
        if raw_parent is not None:
            chunk_raw = raw_parent.get_traces(
                start_frame=start_frame, end_frame=end_frame, return_scaled=True
            ).T.astype(np.float32, copy=False)
            chunk_raw -= np.median(chunk_raw, axis=1, keepdims=True)

        # Recenter stim times within this chunk (event times in local
        # chunk coordinates, ms from the chunk start).
        local_event_ms = group_events_ms - chunk_start_ms
        local_corrected_ms = recenter_stim_times(
            chunk_traces,
            local_event_ms,
            fs_Hz,
            max_offset_ms=max_stim_offset_ms,
            peak_mode=peak_mode,
            n_reference_channels=n_reference_channels,
            prewindow_ms=prewindow_ms,
        )

        # Remove artifacts on the chunk (in-place — we don't need the
        # pre-cleaning trace after this).
        chunk_cleaned, _ = remove_stim_artifacts(
            chunk_traces,
            local_corrected_ms,
            fs_Hz,
            method=artifact_method,
            artifact_window_ms=artifact_window_ms,
            saturation_threshold=saturation_threshold,
            baseline_threshold=baseline_threshold,
            poly_order=poly_order,
            artifact_window_only=artifact_window_only,
            copy=False,
            recording=recording,
            raw_traces=chunk_raw,
        )
        del chunk_raw

        # Sort offline on the cleaned chunk — reset is default True, so
        # each chunk sort is independent.
        sorting = rt_sort_obj.sort_offline(
            recording=chunk_cleaned,
            recording_window_ms=None,
            return_spikeinterface_sorter=True,
            verbose=False,
        )
        # ``sort_offline`` on an ndarray returns a NumpySorting with
        # no associated recording, so we must pass n_samples explicitly.
        chunk_n_samples = chunk_cleaned.shape[1]
        chunk_sd = _sorting_to_spikedata(sorting, fs_Hz, n_samples=chunk_n_samples)
        del chunk_cleaned, chunk_traces, sorting

        # Extract peri-event SpikeData per event in this chunk.
        for i_in_group, kept_idx in enumerate(group_kept):
            t_local_corr = float(local_corrected_ms[i_in_group])
            t_global_corr = chunk_start_ms + t_local_corr
            per_event_global_corr_ms[kept_idx] = t_global_corr

            slice_lo_ms = t_local_corr - pre_ms
            slice_hi_ms = t_local_corr + post_ms

            # ``subtime(start, end, shift_to=peak)`` returns a SpikeData
            # with spike times relative to ``peak`` — so spike times
            # land in [-pre_ms, +post_ms] with start_time = -pre_ms.
            # Matches the output of ``SpikeData.align_to_events``.
            ev_sd = chunk_sd.subtime(slice_lo_ms, slice_hi_ms, shift_to=t_local_corr)
            per_event_sd[kept_idx] = ev_sd

        del chunk_sd

    # Guard: all kept events must have produced a slice.
    missing = [i for i, sd in enumerate(per_event_sd) if sd is None]
    if missing:
        raise RuntimeError(
            f"Internal error: {len(missing)} kept events did not receive a "
            f"peri-event slice (indices into kept set: {missing[:5]}...)"
        )

    # Build the final SpikeSliceStack.  ``times_start_to_end`` is in
    # absolute recording ms (using the chunked-recentered event times).
    times_start_to_end = [
        (float(t - pre_ms), float(t + post_ms))
        for t in per_event_global_corr_ms
    ]

    # Resolve neuron_attributes from the RTSort if available (so the
    # stack exposes per-unit metadata without duplicating it across
    # slices — ``SpikeSliceStack`` strips per-slice attrs by default).
    neuron_attributes = None
    if per_event_sd and per_event_sd[0].neuron_attributes is not None:
        neuron_attributes = per_event_sd[0].neuron_attributes

    stim_slices = SpikeSliceStack(
        spike_stack=list(per_event_sd),
        times_start_to_end=times_start_to_end,
        neuron_attributes=neuron_attributes,
    )

    if verbose:
        n_slices = len(stim_slices.spike_stack)
        print(
            f"Produced SpikeSliceStack with {n_slices} slices "
            f"(peri-event windows [-{pre_ms}, +{post_ms}] ms)"
        )
    return stim_slices


def _group_stim_events_into_chunks(times_ms, chunk_pre_ms, chunk_post_ms):
    """Group event indices whose peri-event chunk windows overlap.

    Given sorted-or-unsorted event times, returns a list of lists of
    indices (into the input array) such that:

      * events within a group have overlapping chunk windows
        ``[t - chunk_pre_ms, t + chunk_post_ms]`` — so they are
        naturally processed together in one chunk;
      * no two groups have overlapping chunk windows.

    Group boundaries are decided by sorted times; within each group
    indices are returned in time order (matching the sort-order of
    ``times_ms``).  Input indices, not times, are returned — the
    caller keeps its own mapping.
    """
    times_ms = np.asarray(times_ms, dtype=np.float64)
    if len(times_ms) == 0:
        return []
    order = np.argsort(times_ms, kind="stable")
    gap_threshold = chunk_pre_ms + chunk_post_ms
    groups = [[int(order[0])]]
    for idx in order[1:]:
        prev_idx = groups[-1][-1]
        if times_ms[idx] - times_ms[prev_idx] > gap_threshold:
            groups.append([int(idx)])
        else:
            groups[-1].append(int(idx))
    return groups


# ---------------------------------------------------------------------------
# Full-recording path — kept for ndarray inputs (caller has already
# materialised the traces).
# ---------------------------------------------------------------------------


def _sort_stim_full_recording(
    traces,
    recording_obj,
    fs_Hz,
    rt_sort_obj,
    stim_times_ms,
    pre_ms,
    post_ms,
    *,
    artifact_method,
    artifact_window_ms,
    saturation_threshold,
    baseline_threshold,
    poly_order,
    artifact_window_only,
    max_stim_offset_ms,
    peak_mode,
    n_reference_channels,
    prewindow_ms,
    recording_window_ms,
    verbose,
):
    """Full-recording path — processes the entire recording in one go.

    Used when the caller passed a pre-materialised ``np.ndarray``; for
    lazy recordings the chunked path is preferred (much lower peak
    RAM).
    """
    from .artifact_removal import remove_stim_artifacts
    from .recentering import recenter_stim_times

    if traces.ndim != 2:
        raise ValueError(
            f"Expected 2-D array (channels, samples), got shape {traces.shape}."
        )
    if fs_Hz is None:
        raise ValueError("fs_Hz is required when stim_recording is a numpy array.")
    fs_Hz = float(fs_Hz)

    if verbose:
        print("Recentering stim times (full-recording path)...")
    corrected_stim_ms = recenter_stim_times(
        traces,
        stim_times_ms,
        fs_Hz,
        max_offset_ms=max_stim_offset_ms,
        peak_mode=peak_mode,
        n_reference_channels=n_reference_channels,
        prewindow_ms=prewindow_ms,
    )
    if verbose:
        offsets = corrected_stim_ms - stim_times_ms
        print(
            f"  Stim time corrections: "
            f"mean={np.mean(offsets):.2f} ms  "
            f"max={np.max(np.abs(offsets)):.2f} ms"
        )
        print(f"Removing artifacts (method={artifact_method!r})...")

    cleaned, blanked_mask = remove_stim_artifacts(
        traces,
        corrected_stim_ms,
        fs_Hz,
        method=artifact_method,
        artifact_window_ms=artifact_window_ms,
        saturation_threshold=saturation_threshold,
        baseline_threshold=baseline_threshold,
        poly_order=poly_order,
        artifact_window_only=artifact_window_only,
        copy=True,
        recording=recording_obj,
        raw_traces=None,
    )
    if verbose:
        print(f"  {100.0 * np.mean(blanked_mask):.1f}% of samples blanked")
        print("Running RT-Sort offline sorting on cleaned traces...")

    sorting = rt_sort_obj.sort_offline(
        recording=cleaned,
        recording_window_ms=recording_window_ms,
        return_spikeinterface_sorter=True,
        verbose=verbose,
    )
    # sort_offline on an ndarray produces a NumpySorting without an
    # associated recording; pass n_samples explicitly.
    sd = _sorting_to_spikedata(sorting, fs_Hz, n_samples=cleaned.shape[1])
    if verbose:
        print(f"  {sd.N} units, {sum(len(t) for t in sd.train)} total spikes")
        print(
            f"Aligning to {len(corrected_stim_ms)} stim events "
            f"(window: -{pre_ms} to +{post_ms} ms)..."
        )
    stim_slices = sd.align_to_events(corrected_stim_ms, pre_ms, post_ms, kind="spike")
    if verbose:
        print(f"  Produced SpikeSliceStack with {len(stim_slices.spike_stack)} slices")
    return stim_slices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_prefilter_parent(recording):
    """Walk a SpikeInterface preprocessing chain upward until we leave
    any bandpass/highpass/lowpass filter and return the first non-filter
    parent.  For the standard SpikeLab chain
    ``BandpassFilterRecording → ScaleRecording → …`` this returns the
    ``ScaleRecording`` — float32 uV traces without filter ringing.

    Returns ``None`` if the top-level recording is already non-filter
    (nothing to walk) or if the chain cannot be traversed.
    """
    if recording is None:
        return None
    cls_name = type(recording).__name__
    if "Filter" not in cls_name:
        return None

    cur = recording
    visited: set = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if "Filter" not in type(cur).__name__:
            return cur
        kwargs = getattr(cur, "_kwargs", None)
        if not isinstance(kwargs, dict):
            return None
        parent = kwargs.get("recording") or kwargs.get("parent_recording")
        if parent is None:
            return None
        cur = parent
    return None


def _load_rt_sort(rt_sort, model, model_path, verbose):
    """Load or return an RTSort object."""
    if isinstance(rt_sort, (str, Path)):
        if verbose:
            print(f"Loading RTSort from {rt_sort}...")
        from ..rt_sort_runner import load_rt_sort as _load

        return _load(Path(rt_sort), model=model, model_path=model_path)
    return rt_sort


def _sorting_to_spikedata(sorting, fs_Hz, n_samples=None):
    """Convert a NumpySorting to a SpikeData (lightweight, no waveforms).

    Converts spike times from samples to milliseconds and builds a
    minimal SpikeData.  No waveform extraction or curation is
    performed — the assumption is that the RTSort sequences were
    already curated during the Phase 1 vanilla sorting.

    Parameters:
        sorting: SpikeInterface ``NumpySorting``.
        fs_Hz (float): Sampling frequency in Hz.
        n_samples (int or None): Duration of the sort in samples.
            When a sorting was produced by ``RTSort.sort_offline`` on
            a bare ndarray (as in the chunked path), it has no
            associated recording, so ``sorting.get_num_samples()``
            raises.  Pass the chunk's sample count explicitly in that
            case.  When None, falls back to ``sorting.get_num_samples()``
            (which requires the sorting to have an associated
            recording).
    """
    from ...spikedata.spikedata import SpikeData

    unit_ids = sorting.get_unit_ids()
    train = []
    for uid in unit_ids:
        spike_samples = sorting.get_unit_spike_train(uid)
        spike_ms = spike_samples.astype(np.float64) / fs_Hz * 1000.0
        train.append(spike_ms)

    if n_samples is None:
        n_samples = sorting.get_num_samples()
    length_ms = n_samples / fs_Hz * 1000.0
    return SpikeData(train, length=length_ms)
