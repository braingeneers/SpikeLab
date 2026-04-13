"""Stimulation-aware spike sorting pipeline.

Orchestrates the full workflow for sorting spikes in a stimulation
recording using a pre-trained RT-Sort model:

1. Load the recording (if a path).
2. Recenter logged stim times to the actual artifact peaks.
3. Remove stimulation artifacts (polynomial detrend or blanking).
4. Load the trained RTSort object (from Phase 1 vanilla sorting).
5. Run ``sort_offline`` on the cleaned traces.
6. Convert the sorted spike trains to a ``SpikeData``.
7. Align to corrected stim times → ``SpikeSliceStack``.
"""

from pathlib import Path

import numpy as np


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

    Parameters:
        stim_recording: The stimulation recording.  Can be:
            - ``np.ndarray`` of shape ``(channels, samples)``
            - ``str`` or ``Path`` to a recording file (Maxwell .h5 or
              NWB), loaded via SpikeInterface
            - A SpikeInterface ``BaseRecording`` object
        rt_sort: The trained RT-Sort object.  Can be:
            - An ``RTSort`` instance (from ``detect_sequences``)
            - ``str`` or ``Path`` to a ``rt_sort.pickle`` file
              (produced by the Phase 1 vanilla sorting pipeline)
        stim_times_ms (array-like): Logged stimulation event times in
            milliseconds.
        pre_ms (float): Window duration before each stim event in
            milliseconds, for the output ``SpikeSliceStack``.
        post_ms (float): Window duration after each stim event in
            milliseconds.
        fs_Hz (float or None): Sampling frequency in Hz.  Required
            when ``stim_recording`` is a numpy array; inferred from
            the recording object otherwise.
        artifact_method (str): ``"polynomial"`` (default) or
            ``"blank"``.  Passed to ``remove_stim_artifacts``.
        artifact_window_ms (float): Max artifact tail duration after
            the last desaturation.  Default 10.0.
        saturation_threshold (float or None): Saturation voltage
            threshold.  None auto-detects.
        baseline_threshold (float or None): Baseline envelope
            threshold.  None auto-detects.
        poly_order (int): Polynomial order for detrend.  Default 3.
        artifact_window_only (bool): Only process around stim events.
            Default True.
        max_stim_offset_ms (float): Search window radius for stim
            time recentering.  Default 50.0.
        model (ModelSpikeSorter or None): Detection model instance for
            ``load_rt_sort`` when ``rt_sort`` is a path.
        model_path (str or Path or None): Path to a detection model
            folder for ``load_rt_sort`` when ``rt_sort`` is a path.
        recording_window_ms (tuple or None): ``(start_ms, end_ms)``
            sub-window to restrict ``sort_offline`` to.
        verbose (bool): Print progress messages.  Default True.

    Returns:
        stim_slices (SpikeSliceStack): Event-aligned spike slice stack
            with one slice per (corrected) stim event.  Each slice
            spans ``[-pre_ms, +post_ms]`` relative to the stim time.
    """
    from ..rt_sort_runner import load_rt_sort
    from .artifact_removal import remove_stim_artifacts
    from .recentering import recenter_stim_times

    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)

    # --- Step 1: Load recording ----------------------------------------
    traces, fs_Hz, recording_obj = _load_recording(stim_recording, fs_Hz, verbose)

    # --- Step 2: Recenter stim times -----------------------------------
    if verbose:
        print("Recentering stim times...")
    corrected_stim_ms = recenter_stim_times(
        traces, stim_times_ms, fs_Hz, max_offset_ms=max_stim_offset_ms
    )
    if verbose:
        offsets = corrected_stim_ms - stim_times_ms
        print(
            f"  Stim time corrections: "
            f"mean={np.mean(offsets):.2f} ms, "
            f"max={np.max(np.abs(offsets)):.2f} ms"
        )

    # --- Step 3: Remove artifacts --------------------------------------
    if verbose:
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
    )
    if verbose:
        pct_blanked = 100.0 * np.mean(blanked_mask)
        print(f"  {pct_blanked:.1f}% of samples blanked")

    # --- Step 4: Load RTSort -------------------------------------------
    rt_sort_obj = _load_rt_sort(rt_sort, model, model_path, verbose)

    # --- Step 5: Sort offline ------------------------------------------
    if verbose:
        print("Running RT-Sort offline sorting on cleaned traces...")
    recording_for_sort = _traces_to_recording(cleaned, fs_Hz, recording_obj)

    sorting = rt_sort_obj.sort_offline(
        recording=recording_for_sort,
        recording_window_ms=recording_window_ms,
        return_spikeinterface_sorter=True,
        verbose=verbose,
    )

    # --- Step 6: Convert to SpikeData ----------------------------------
    sd = _sorting_to_spikedata(sorting, fs_Hz)
    if verbose:
        print(f"  {sd.N} units, {sum(len(t) for t in sd.train)} total spikes")

    # --- Step 7: Align to events → SpikeSliceStack ---------------------
    if verbose:
        print(
            f"Aligning to {len(corrected_stim_ms)} stim events "
            f"(window: -{pre_ms} to +{post_ms} ms)..."
        )
    stim_slices = sd.align_to_events(corrected_stim_ms, pre_ms, post_ms, kind="spike")
    if verbose:
        n_slices = len(stim_slices.spike_stack)
        print(f"  Produced SpikeSliceStack with {n_slices} slices")

    return stim_slices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_recording(stim_recording, fs_Hz, verbose):
    """Load a recording into (traces, fs_Hz, recording_obj) form.

    Parameters:
        stim_recording: numpy array, path, or BaseRecording.
        fs_Hz (float or None): Sampling freq (required for arrays).
        verbose (bool): Print progress.

    Returns:
        traces (np.ndarray): ``(channels, samples)``.
        fs_Hz (float): Sampling frequency.
        recording_obj: Original BaseRecording or None.
    """
    if isinstance(stim_recording, np.ndarray):
        if fs_Hz is None:
            raise ValueError("fs_Hz is required when stim_recording is a numpy array.")
        if stim_recording.ndim != 2:
            raise ValueError(
                f"Expected 2-D array (channels, samples), "
                f"got shape {stim_recording.shape}."
            )
        return stim_recording, float(fs_Hz), None

    if isinstance(stim_recording, (str, Path)):
        if verbose:
            print(f"Loading recording from {stim_recording}...")
        from ..recording_io import load_single_recording

        rec = load_single_recording(stim_recording)
        traces = rec.get_traces(return_scaled=True).T  # (channels, samples)
        return traces, float(rec.get_sampling_frequency()), rec

    # Assume BaseRecording-like object
    traces = stim_recording.get_traces(return_scaled=True).T
    return traces, float(stim_recording.get_sampling_frequency()), stim_recording


def _load_rt_sort(rt_sort, model, model_path, verbose):
    """Load or return an RTSort object.

    Parameters:
        rt_sort: RTSort instance or path to pickle.
        model: Detection model instance or None.
        model_path: Model folder path or None.
        verbose (bool): Print progress.

    Returns:
        rt_sort_obj: Ready-to-use RTSort instance.
    """
    if isinstance(rt_sort, (str, Path)):
        if verbose:
            print(f"Loading RTSort from {rt_sort}...")
        from ..rt_sort_runner import load_rt_sort as _load

        return _load(Path(rt_sort), model=model, model_path=model_path)

    # Assume already an RTSort instance
    return rt_sort


def _traces_to_recording(traces, fs_Hz, original_recording):
    """Wrap cleaned traces into a SpikeInterface recording.

    If the original recording is available, tries to use a
    NumpyRecording with matching channel info.  Falls back to a
    plain NumpyRecording otherwise.

    Parameters:
        traces (np.ndarray): ``(channels, samples)``.
        fs_Hz (float): Sampling frequency.
        original_recording: Original BaseRecording or None.

    Returns:
        recording: SpikeInterface BaseRecording wrapping the traces.
    """
    from spikeinterface.core import NumpyRecording

    # NumpyRecording expects (samples, channels)
    traces_T = traces.T.copy()
    rec = NumpyRecording(
        traces_list=[traces_T],
        sampling_frequency=fs_Hz,
    )

    # Copy probe / channel locations from the original if available
    if original_recording is not None:
        try:
            probe = original_recording.get_probe()
            rec = rec.set_probe(probe)
        except Exception:
            pass

    return rec


def _sorting_to_spikedata(sorting, fs_Hz):
    """Convert a NumpySorting to a SpikeData (lightweight, no waveforms).

    Converts spike times from samples to milliseconds and builds a
    minimal SpikeData.  No waveform extraction or curation is
    performed — the assumption is that the RTSort sequences were
    already curated during the Phase 1 vanilla sorting.

    Parameters:
        sorting: SpikeInterface NumpySorting.
        fs_Hz (float): Sampling frequency.

    Returns:
        sd (SpikeData): SpikeData with one unit per sorted sequence.
    """
    from ...spikedata.spikedata import SpikeData

    unit_ids = sorting.get_unit_ids()
    train = []
    for uid in unit_ids:
        spike_samples = sorting.get_unit_spike_train(uid)
        spike_ms = spike_samples.astype(np.float64) / fs_Hz * 1000.0
        train.append(spike_ms)

    n_samples = sorting.get_num_samples()
    length_ms = n_samples / fs_Hz * 1000.0

    return SpikeData(train, length=length_ms)
