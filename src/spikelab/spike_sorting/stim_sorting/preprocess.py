"""High-level stim artifact preprocessing.

``preprocess_stim_artifacts`` wraps :func:`recenter_stim_times` and
:func:`remove_stim_artifacts` into a single function that takes a
SpikeInterface recording and returns a new recording with the artifacts
removed, preserving channel IDs, locations, gains, and offsets.

The returned recording is usable with any downstream sorter.  When
``output_path`` is given the cleaned traces are written to a float32
binary file and the result is a ``BinaryRecordingExtractor`` — dumpable
through SpikeInterface's JSON encoder, which is required for Docker-
based sorters (Kilosort, IronClust).  With no ``output_path`` the
result is an in-memory ``NumpyRecording`` — fine for local sorters and
interactive exploration.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def preprocess_stim_artifacts(
    recording,
    stim_times_ms,
    output_path: Optional[str] = None,
    *,
    method: str = "polynomial",
    artifact_window_ms: float = 10.0,
    recenter: bool = True,
    max_offset_ms: float = 50.0,
    poly_order: int = 3,
    saturation_threshold: Optional[float] = None,
    baseline_threshold: Optional[float] = None,
    artifact_window_only: bool = True,
    return_scaled: bool = False,
    dtype: str = "float32",
) -> Tuple["BaseRecording", dict]:  # noqa: F821
    """Remove stim artifacts and return a new SpikeInterface recording.

    Materialises ``recording.get_traces()`` to an ndarray, optionally
    recenters the stim times to their artifact peaks, runs
    :func:`remove_stim_artifacts`, and wraps the cleaned traces in
    either a ``BinaryRecordingExtractor`` (when ``output_path`` is
    given) or a ``NumpyRecording``.  Channel IDs, locations, gains, and
    offsets are copied from the input recording.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording to clean.  Single-segment only.
    stim_times_ms : array-like
        Logged stim event times in milliseconds (``len(stim_times_ms)``
        may be 0, in which case recentering/artifact removal are
        skipped and the recording is returned unchanged aside from the
        ``BinaryRecordingExtractor`` wrap when ``output_path`` is
        given).
    output_path : str or Path, optional
        When provided, cleaned traces are written as a float32 binary
        (interleaved channels, i.e. shape ``(num_samples, num_channels)``
        on disk) and a ``BinaryRecordingExtractor`` is returned.  Parent
        directories are created as needed.  When ``None`` (default), a
        ``NumpyRecording`` is returned — NOT dumpable for Docker-based
        sorters.
    method : str
        ``"polynomial"`` (default) or ``"blank"`` — see
        :func:`remove_stim_artifacts`.  Polynomial detrend preserves
        spikes in the 0–10 ms post-stim window (the smooth fit can't
        capture a ~1 ms spike feature) and is safe by default thanks
        to ``poly_clamp_factor`` — divergent fits at extreme stim
        amplitudes are caught and downgraded to blank automatically,
        with one summary warning per call.  Use ``"blank"`` only when
        the post-stim window is genuinely irrelevant to the analysis,
        or when the clamp warning fires on a non-trivial fraction of
        events (in which case a uniform blank is cleaner than mixing
        per-event polynomial subtraction with per-event clamp blanks).
    artifact_window_ms : float
        Length of the post-stim artifact window in ms.  Default 10.0.
    recenter : bool
        When True (default), align logged stim times to the actual
        artifact peaks via :func:`recenter_stim_times` before artifact
        removal.  Set False when the supplied times are already
        peak-aligned.
    max_offset_ms : float
        Maximum recentering shift, passed to
        :func:`recenter_stim_times`.  Default 50.0.
    poly_order : int
        Polynomial order for ``method="polynomial"``.  Default 3.
    saturation_threshold, baseline_threshold : float, optional
        Override the auto-detected thresholds used by
        :func:`remove_stim_artifacts`.
    artifact_window_only : bool
        When True (default), only the windows around stim events are
        processed; when False, a global sliding-window detrend is
        applied (useful for very frequent stim protocols).
    return_scaled : bool
        Whether to materialise µV-scaled traces from ``recording``.
        Default False — match the recording's native dtype/units.  Set
        True to force a µV-scaled float output when the recording
        exposes gains/offsets.  Forwarded as ``return_in_uV`` on newer
        SpikeInterface versions and ``return_scaled`` on older ones.
    dtype : str
        dtype of the cleaned output (both for in-memory and on-disk
        representations).  Default ``"float32"``.

    Returns
    -------
    cleaned_recording : BaseRecording
        New SpikeInterface recording with artifacts removed.  Channel
        IDs, locations, gains, and offsets are inherited from the input.
    metadata : dict
        Artifact-removal metadata.  Keys:
          - ``stim_times_ms_logged``: original stim times as passed in
          - ``stim_times_ms_corrected``: recentered stim times (equals
            ``stim_times_ms_logged`` when ``recenter=False``)
          - ``recenter_offsets_ms``: ``corrected - logged`` offsets
          - ``blanked_fraction``: overall fraction of samples blanked
          - ``blanked_fraction_per_channel``: per-channel blanked
            fractions, shape ``(num_channels,)``
    """
    from spikeinterface.core import (
        BaseRecording,
        BinaryRecordingExtractor,
        NumpyRecording,
    )

    from .artifact_removal import remove_stim_artifacts
    from .recentering import recenter_stim_times

    if not isinstance(recording, BaseRecording):
        raise TypeError(
            f"recording must be a SpikeInterface BaseRecording, "
            f"got {type(recording).__name__}"
        )
    if recording.get_num_segments() != 1:
        raise ValueError(
            "preprocess_stim_artifacts only supports single-segment "
            f"recordings, got {recording.get_num_segments()} segments"
        )

    fs_Hz = float(recording.get_sampling_frequency())
    stim_times_ms = np.asarray(stim_times_ms, dtype=np.float64)

    # get_traces returns (num_samples, num_channels); remove_stim_artifacts
    # wants (num_channels, num_samples).  SpikeInterface 0.105+ renamed
    # ``return_scaled`` to ``return_in_uV``; try the new name first.
    try:
        raw = recording.get_traces(return_in_uV=return_scaled)
    except TypeError:
        raw = recording.get_traces(return_scaled=return_scaled)
    traces = np.ascontiguousarray(raw.T, dtype=np.float32)

    if recenter and len(stim_times_ms) > 0:
        stim_times_corrected = recenter_stim_times(
            traces, stim_times_ms, fs_Hz=fs_Hz, max_offset_ms=max_offset_ms
        )
    else:
        stim_times_corrected = stim_times_ms.copy()
    recenter_offsets = stim_times_corrected - stim_times_ms

    extra_kwargs = {}
    if saturation_threshold is not None:
        extra_kwargs["saturation_threshold"] = saturation_threshold
    if baseline_threshold is not None:
        extra_kwargs["baseline_threshold"] = baseline_threshold

    cleaned, blanked_mask = remove_stim_artifacts(
        traces,
        stim_times_corrected,
        fs_Hz=fs_Hz,
        method=method,
        artifact_window_ms=artifact_window_ms,
        poly_order=poly_order,
        artifact_window_only=artifact_window_only,
        copy=False,
        recording=recording,
        **extra_kwargs,
    )

    n_ch, n_samples = cleaned.shape
    channel_ids = np.asarray(recording.get_channel_ids())
    # (channels, samples) -> (samples, channels) for on-disk / NumpyRecording
    cleaned_sc = np.ascontiguousarray(cleaned.T, dtype=dtype)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_sc.tofile(output_path)
        cleaned_rec = BinaryRecordingExtractor(
            file_paths=[str(output_path)],
            sampling_frequency=fs_Hz,
            num_channels=n_ch,
            dtype=dtype,
            channel_ids=channel_ids,
        )
    else:
        cleaned_rec = NumpyRecording(
            traces_list=[cleaned_sc],
            sampling_frequency=fs_Hz,
            channel_ids=channel_ids,
        )

    # Propagate channel metadata.  Each probe attribute is optional; a
    # recording may expose some and not others.
    try:
        locations = recording.get_channel_locations()
    except Exception:
        locations = None
    if locations is not None:
        cleaned_rec.set_channel_locations(locations)

    try:
        gains = recording.get_channel_gains()
    except Exception:
        gains = None
    if gains is not None:
        cleaned_rec.set_channel_gains(np.asarray(gains))

    try:
        offsets_ch = recording.get_channel_offsets()
    except Exception:
        offsets_ch = None
    if offsets_ch is not None:
        cleaned_rec.set_channel_offsets(np.asarray(offsets_ch))

    metadata = dict(
        stim_times_ms_logged=stim_times_ms,
        stim_times_ms_corrected=stim_times_corrected,
        recenter_offsets_ms=recenter_offsets,
        blanked_fraction=float(blanked_mask.mean()),
        blanked_fraction_per_channel=blanked_mask.mean(axis=1),
    )

    return cleaned_rec, metadata
