"""Native MaxWell HDF5 loader for files that neo cannot open.

MaxWell Biosystems' mxw v25.x firmware writes a ``settings/mapping``
table that occasionally contains duplicate entries for some channel
IDs (e.g. 1021 rows mapping 1019 unique channels).  ``neo``'s
``MaxwellRawIO`` requires unique channel IDs and rejects the file with
``ValueError: signal_channels do not have unique ids for stream 0``,
which propagates through SpikeInterface's ``MaxwellRecordingExtractor``.

This loader bypasses neo entirely: it reads the file with ``h5py``,
dedupes the mapping table by keeping the first occurrence per channel
ID, and returns a SpikeInterface ``BaseRecording`` ready for sorting.
For multi-well files, pass the well/recording ID explicitly or call
``list_maxwell_wells`` first to enumerate them.

The fallback also fires automatically inside
:func:`spikelab.spike_sorting.recording_io.load_single_recording` when
``MaxwellRecordingExtractor`` raises the unique-IDs error, so callers
that already use the standard sort entry points get the workaround
transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


def list_maxwell_wells(h5_path: Any) -> List[Tuple[str, str]]:
    """Return ``(well_id, rec_id)`` pairs available in a MaxWell HDF5 file.

    Useful for multi-well experiments where the caller needs to know
    which streams exist before calling :func:`load_maxwell_native`.
    """
    import h5py

    h5_path = Path(h5_path)
    pairs: List[Tuple[str, str]] = []
    with h5py.File(h5_path, "r") as f:
        wells_grp = f.get("wells")
        if wells_grp is None:
            return pairs
        for well_id in sorted(wells_grp.keys()):
            for rec_id in sorted(wells_grp[well_id].keys()):
                if "groups/routed/raw" in wells_grp[well_id][rec_id]:
                    pairs.append((well_id, rec_id))
    return pairs


def load_maxwell_native(
    h5_path: Any,
    well_id: str = "well000",
    rec_id: str = "rec0000",
    *,
    dtype: str = "float32",
    output_path: Optional[Any] = None,
):
    """Load a MaxWell HDF5 recording into a SpikeInterface recording.

    Reads ``wells/<well_id>/<rec_id>/groups/routed/{raw,channels}`` and
    the matching ``settings/{mapping,sampling,lsb}`` block, dedupes the
    mapping table, scales raw uint16 to microvolts, and returns either
    a ``NumpyRecording`` (in-memory) or a ``BinaryRecordingExtractor``
    (file-backed) — the latter is required for Docker-based sorters
    because SpikeInterface's JSON encoder cannot dump in-memory
    ``NumpyRecording`` ndarrays.

    Parameters
    ----------
    h5_path : str or Path
        MaxWell ``.raw.h5`` file.
    well_id : str
        Well group name inside ``/wells/``.  Default ``"well000"``
        (single-well MaxOne).  Use :func:`list_maxwell_wells` to list
        what's actually present.
    rec_id : str
        Recording group name inside the well.  Default ``"rec0000"``.
    dtype : str
        dtype of the returned traces.  Default ``"float32"``.
    output_path : str or Path, optional
        When provided, the µV-scaled traces are written as an
        interleaved ``(num_samples, num_channels)`` binary file and a
        ``BinaryRecordingExtractor`` backed by that file is returned.
        Parent directories are created as needed.  When ``None`` (the
        default), a ``NumpyRecording`` is returned — fine for local
        sorters and interactive use, NOT dumpable for Docker.

    Returns
    -------
    recording : BaseRecording
        Single-segment recording with channel IDs (string forms of the
        routed channel numbers), per-channel locations from the
        mapping table, and gains/offsets set so ``get_traces`` returns
        µV-scaled values.

    Notes
    -----
    * Per-channel median centering is **not** applied — that is a
      preprocessing decision the caller can make on the returned
      recording.
    * Mapping table dedupe keeps the first ``(x, y)`` seen per
      channel.  For the MaxOne files where this matters, the
      duplicates have been observed to share the same coordinates
      anyway.
    """
    import h5py
    from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"MaxWell HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        try:
            rec_grp = f[f"wells/{well_id}/{rec_id}"]
        except KeyError as exc:
            available = list_maxwell_wells(h5_path)
            raise KeyError(
                f"wells/{well_id}/{rec_id} not found in {h5_path}.  "
                f"Available wells/recs: {available}"
            ) from exc

        mapping = rec_grp["settings/mapping"][:]
        fs_Hz = float(rec_grp["settings/sampling"][()][0])
        lsb_volts = float(rec_grp["settings/lsb"][()][0])
        raw_u16 = rec_grp["groups/routed/raw"][:]
        routed_channels = rec_grp["groups/routed/channels"][:]

    n_ch, n_samples = raw_u16.shape

    # Mapping table is stored in declaration order and may contain
    # duplicates — neo trips on these and so we dedupe by first
    # occurrence per channel ID.  For MaxOne files where this matters,
    # observed duplicates have shared (x, y) anyway.
    seen: dict[int, Tuple[float, float]] = {}
    for entry in mapping:
        ch = int(entry["channel"])
        if ch not in seen:
            seen[ch] = (float(entry["x"]), float(entry["y"]))

    channel_ids = np.array([str(int(c)) for c in routed_channels])
    locations = np.array([seen[int(c)] for c in routed_channels], dtype=np.float32)
    if locations.shape != (n_ch, 2):
        raise ValueError(
            f"Channel-location lookup mismatch: routed_channels has "
            f"{n_ch} entries but only {locations.shape[0]} have mapping "
            f"coordinates"
        )

    # Scale uint16 -> microvolts.  The MaxWell convention is
    # ``volts = raw * lsb``, so µV = raw * (lsb * 1e6).
    traces_uv = raw_u16.astype(dtype, copy=False)
    traces_uv *= np.asarray(lsb_volts * 1e6, dtype=dtype)
    # (channels, samples) -> (samples, channels) for SpikeInterface
    traces_si = np.ascontiguousarray(traces_uv.T)
    del raw_u16, traces_uv

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traces_si.tofile(output_path)
        rec = BinaryRecordingExtractor(
            file_paths=[str(output_path)],
            sampling_frequency=fs_Hz,
            num_channels=n_ch,
            dtype=dtype,
            channel_ids=channel_ids,
        )
    else:
        rec = NumpyRecording(
            traces_list=[traces_si],
            sampling_frequency=fs_Hz,
            channel_ids=channel_ids,
        )

    rec.set_channel_locations(locations)
    rec.set_channel_gains(np.ones(n_ch, dtype=np.float32))
    rec.set_channel_offsets(np.zeros(n_ch, dtype=np.float32))
    return rec
