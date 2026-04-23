"""Recording I/O and waveform extraction for the spike sorting pipeline.

Provides functions for loading recordings from various formats (Maxwell
HDF5, NWB, SpikeInterface), concatenating multi-segment recordings, and
extracting waveforms via the WaveformExtractor.

For the full sorting pipeline (sort → build SpikeData → curate → compile),
use ``pipeline.py`` and ``sort_recording()``."""

import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    natsorted = None

try:
    import spikeinterface.core.segmentutils as si_segmentutils
    from spikeinterface.core import BaseRecording
    from spikeinterface.extractors.extractor_classes import (
        MaxwellRecordingExtractor,
        NwbRecordingExtractor,
    )
    from spikeinterface.preprocessing import bandpass_filter
    from spikeinterface.preprocessing.preprocessing_classes import ScaleRecording

    _SI_AVAILABLE = True
except ImportError:  # pragma: no cover
    si_segmentutils = None
    BaseRecording = None
    MaxwellRecordingExtractor = None
    NwbRecordingExtractor = None
    bandpass_filter = None
    ScaleRecording = None
    _SI_AVAILABLE = False

from . import _globals
from .sorting_utils import (
    Stopwatch,
    print_stage,
)
from .waveform_extractor import WaveformExtractor


# Upstream `neo.rawio.maxwellrawio.auto_install_maxwell_hdf5_compression_plugin`
# treats `HDF5_PLUGIN_PATH` as a single directory. HDF5 actually defines it as
# an os.pathsep-separated list (like `PATH`), so when the env var holds multiple
# entries (e.g. `/home/mxwbio/MaxLab/so/:/home/sharf-lab/MaxLab/so`) upstream
# tries to `Path(...).mkdir()` on the compound string and fails. This wrapper
# patches the helper at SpikeLab import time so the fix survives any `neo`
# reinstall/upgrade.
def _patch_neo_maxwell_hdf5_plugin_path_handling() -> None:
    try:
        import platform
        from pathlib import Path
        from urllib.request import urlopen

        import neo.rawio.maxwellrawio as _mwrawio
    except ImportError:
        return

    def auto_install_maxwell_hdf5_compression_plugin(
        hdf5_plugin_path=None, force_download=True
    ):
        if hdf5_plugin_path is None:
            env_value = os.getenv("HDF5_PLUGIN_PATH", None)
            if env_value is not None:
                # HDF5_PLUGIN_PATH follows PATH-style semantics: a list of
                # directories separated by os.pathsep (':' on Linux/macOS,
                # ';' on Windows). Scan each component for an existing
                # libcompression library before downloading.
                for component in env_value.split(os.pathsep):
                    component = component.strip()
                    if not component:
                        continue
                    candidate_dir = Path(component)
                    if platform.system() == "Linux":
                        candidate = candidate_dir / "libcompression.so"
                    elif platform.system() == "Darwin":
                        candidate = candidate_dir / "libcompression.dylib"
                    elif platform.system() == "Windows":
                        candidate = candidate_dir / "compression.dll"
                    else:
                        candidate = None
                    if candidate is not None and candidate.is_file():
                        hdf5_plugin_path = candidate_dir
                        break
                if hdf5_plugin_path is None:
                    # No existing plugin found in any component; fall back to
                    # the first non-empty component as the install target.
                    for component in env_value.split(os.pathsep):
                        component = component.strip()
                        if component:
                            hdf5_plugin_path = Path(component)
                            break
            if hdf5_plugin_path is None:
                hdf5_plugin_path = Path.home() / "hdf5_plugin_path_maxwell"
                os.environ["HDF5_PLUGIN_PATH"] = str(hdf5_plugin_path)
        hdf5_plugin_path = Path(hdf5_plugin_path)
        hdf5_plugin_path.mkdir(exist_ok=True)

        if platform.system() == "Linux":
            remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FLinux%2Flibcompression.so&dl=1"
            local_lib = hdf5_plugin_path / "libcompression.so"
        elif platform.system() == "Darwin":
            if platform.machine() == "arm64":
                remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FMacOS%2FMac_arm64%2Flibcompression.dylib&dl=1"
            else:
                remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FMacOS%2FMac_x86_64%2Flibcompression.dylib&dl=1"
            local_lib = hdf5_plugin_path / "libcompression.dylib"
        elif platform.system() == "Windows":
            remote_lib = "https://share.mxwbio.com/d/7f2d1e98a1724a1b8b35/files/?p=%2FWindows%2Fcompression.dll&dl=1"
            local_lib = hdf5_plugin_path / "compression.dll"

        if not force_download and local_lib.is_file():
            print(
                f"The h5 compression library for Maxwell is already located in {local_lib}!"
            )
            return

        dist = urlopen(remote_lib)
        with open(local_lib, "wb") as f:
            f.write(dist.read())

    setattr(
        _mwrawio,
        "auto_install_maxwell_hdf5_compression_plugin",
        auto_install_maxwell_hdf5_compression_plugin,
    )


_patch_neo_maxwell_hdf5_plugin_path_handling()


def _time_chunks_to_frames(
    start_time_s: Optional[float],
    end_time_s: Optional[float],
    rec_chunks_s: List[Tuple[float, float]],
    fs: float,
    total_duration_s: float,
) -> List[Tuple[int, int]]:
    """Convert time-based slicing parameters to frame tuples.

    Combines ``start_time_s``/``end_time_s`` (single range) and
    ``rec_chunks_s`` (multiple ranges) into a single list of
    ``(start_frame, end_frame)`` tuples in samples.

    Parameters:
        start_time_s: Start time in seconds, or ``None``.
        end_time_s: End time in seconds, or ``None``.
        rec_chunks_s: List of ``(start_s, end_s)`` ranges in seconds.
        fs: Sampling frequency in Hz.
        total_duration_s: Full recording duration in seconds (used to
            clip ``end_time_s`` if it exceeds the recording).

    Returns:
        List of ``(start_frame, end_frame)`` tuples. Empty list when
        no time-based parameters are provided.

    Raises:
        ValueError: If a time range is invalid (negative start or
            start >= end).
    """
    chunks: List[Tuple[int, int]] = []

    if start_time_s is not None or end_time_s is not None:
        start_s = start_time_s if start_time_s is not None else 0.0
        end_s = end_time_s if end_time_s is not None else total_duration_s
        if end_s > total_duration_s:
            print(
                f"'end_time_s' ({end_s}) exceeds recording duration "
                f"({total_duration_s:.2f}s); clipping to the end."
            )
            end_s = total_duration_s
        if start_s < 0 or start_s >= end_s:
            raise ValueError(
                f"Invalid time range: start_time_s={start_s}, "
                f"end_time_s={end_s}. Must satisfy 0 <= start < end."
            )
        chunks.append((int(round(start_s * fs)), int(round(end_s * fs))))

    for start_s, end_s in rec_chunks_s:
        if start_s < 0 or start_s >= end_s:
            raise ValueError(
                f"Invalid chunk in rec_chunks_s: ({start_s}, {end_s}). "
                f"Must satisfy 0 <= start < end."
            )
        if end_s > total_duration_s:
            print(
                f"'rec_chunks_s' entry ({start_s}, {end_s}) exceeds "
                f"recording duration ({total_duration_s:.2f}s); clipping."
            )
            end_s = total_duration_s
        chunks.append((int(round(start_s * fs)), int(round(end_s * fs))))

    return chunks


def load_recording(rec_path: Any) -> BaseRecording:
    """Load a recording, apply optional truncation and coordinate transforms.

    Loads a single recording file via ``load_single_recording``, or all
    recordings in a directory via ``concatenate_recordings``. Then applies
    the module-level configuration: truncation to ``FIRST_N_MINS``, frame
    chunking via ``REC_CHUNKS``, y-coordinate flipping via ``MEA_Y_MAX``,
    and custom gain/offset scaling.

    Parameters:
        rec_path (str, Path, or BaseRecording): Path to a recording file,
            a directory containing ``.raw.h5`` / ``.nwb`` files to
            concatenate, or a pre-loaded ``BaseRecording``.

    Returns:
        rec (BaseRecording): The loaded and optionally transformed
            SpikeInterface recording object.
    """
    print_stage("LOADING RECORDING")
    print(f"Recording path: {rec_path}")
    stopwatch = Stopwatch()
    rec_path = Path(rec_path)
    if rec_path.is_dir():
        rec = concatenate_recordings(rec_path)
    else:
        rec = load_single_recording(rec_path)

    print(f"Recording has {rec.get_num_channels()} channels")

    # Convert time-based slicing parameters (seconds) to frame tuples.
    time_chunks = _time_chunks_to_frames(
        start_time_s=_globals.START_TIME_S,
        end_time_s=_globals.END_TIME_S,
        rec_chunks_s=_globals.REC_CHUNKS_S,
        fs=rec.get_sampling_frequency(),
        total_duration_s=rec.get_total_duration(),
    )
    if time_chunks:
        if len(_globals.REC_CHUNKS) > 0:
            raise ValueError(
                "Cannot combine frame-based 'rec_chunks' with time-based "
                "'start_time_s'/'end_time_s'/'rec_chunks_s'. Use one or the "
                "other."
            )
        _globals.REC_CHUNKS = time_chunks

    if _globals.FIRST_N_MINS is not None:
        end_frame = _globals.FIRST_N_MINS * 60 * rec.get_sampling_frequency()
        if end_frame > rec.get_num_samples():
            print(
                f"'first_n_mins' is set to {_globals.FIRST_N_MINS}, but recording is only {rec.get_total_duration() / 60:.2f} min long"
            )
            print(
                f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
            )
        else:
            print(f"Only analyzing the first {_globals.FIRST_N_MINS} min of recording")
            rec = rec.frame_slice(start_frame=0, end_frame=end_frame)
    else:
        print(
            f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
        )

    if len(_globals.REC_CHUNKS) > 0:
        print(f"Using {len(_globals.REC_CHUNKS)} chunks of the recording")
        rec_chunks = []
        for c, (start_frame, end_frame) in enumerate(_globals.REC_CHUNKS):
            print(f"Chunk {c}: {start_frame} to {end_frame} frame")
            chunk = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
            rec_chunks.append(chunk)
        rec = si_segmentutils.concatenate_recordings(rec_chunks)
    else:
        print(f"Using entire recording")

    if _globals.MEA_Y_MAX is not None:
        print(
            f"Flipping y-coordinates of channel locations. MEA height: {_globals.MEA_Y_MAX}"
        )
        probes_all = []
        for probe in rec.get_probes():
            y_cords = probe._contact_positions[:, 1]

            if _globals.MEA_Y_MAX is None:
                y_cords_flipped = y_cords
            elif _globals.MEA_Y_MAX == -1:
                y_cords_flipped = max(y_cords) - y_cords
            else:
                y_cords_flipped = _globals.MEA_Y_MAX - y_cords

            probe._contact_positions[np.arange(y_cords_flipped.size), 1] = (
                y_cords_flipped
            )
            probes_all.append(probe)
        rec = rec.set_probes(probes_all)

    stopwatch.log_time("Done loading recording.")

    return rec


def load_single_recording(rec_path: Any) -> BaseRecording:
    """Load one recording file and return a scaled, bandpass-filtered recording.

    Supports Maxwell ``.h5`` files, NWB ``.nwb`` files, and pre-loaded
    SpikeInterface ``BaseRecording`` objects. The recording is scaled to
    µV (using ``GAIN_TO_UV`` / ``OFFSET_TO_UV`` or the recording's own
    gains) and bandpass-filtered between ``FREQ_MIN`` and ``FREQ_MAX``.

    Parameters:
        rec_path (str, Path, or BaseRecording): Path to a ``.h5`` or
            ``.nwb`` file, or an already-loaded ``BaseRecording``.

    Returns:
        rec (BaseRecording): Scaled and bandpass-filtered recording.
    """
    if isinstance(rec_path, BaseRecording):
        rec = rec_path
    elif str(rec_path).endswith(".h5"):
        maxwell_kwargs = {}
        if _globals.STREAM_ID is not None:
            maxwell_kwargs["stream_id"] = _globals.STREAM_ID
        rec = MaxwellRecordingExtractor(rec_path, **maxwell_kwargs)
        test_file = h5py.File(rec_path)
        if "sig" not in test_file:  # Test if hdf5_plugin_path is needed
            try:
                test_file["/data_store/data0000/groups/routed/raw"][0, 0]
            except OSError as exception:
                test_file.close()
                print("*" * 10)
                print("""This MaxWell Biosystems file format is based on HDF5.
The internal compression requires a custom plugin.
Please visit this page and install the missing decompression libraries:
https://share.mxwbio.com/d/4742248b2e674a85be97/

Setup options (choose one):
    1. Pass hdf5_plugin_path='/path/to/plugin/' to sort_with_kilosort2().
    2. Set os.environ['HDF5_PLUGIN_PATH'] BEFORE importing this module.
    3. Follow the Maxwell instructions at the link above.
""")
                print("*" * 10)
                raise (exception)
        test_file.close()
        # Reconcile declared vs. routed channels. MaxOne recordings report
        # 1024 readout channels but get_traces() returns the full 1024-wide
        # array regardless of routing; slicing by the extractor's own
        # channel_ids forces the width to match get_num_channels(). No-op
        # when all channels are routed (MaxTwo).
        rec = rec.select_channels(rec.get_channel_ids())
    elif str(rec_path).endswith(".nwb"):
        rec = NwbRecordingExtractor(rec_path)
    else:
        raise ValueError(
            f"Recording {rec_path} is not in .h5 or .nwb format.\n"
            f"Load it with SpikeInterface and pass the BaseRecording object "
            f"instead of the file path. See "
            f"https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html"
        )

    if rec.get_num_segments() != 1:
        raise ValueError(
            f"Recording has {rec.get_num_segments()} segments — expected 1. "
            "Divide the recording into separate single-segment recordings."
        )

    if _globals.GAIN_TO_UV is not None:
        gain = _globals.GAIN_TO_UV
    elif rec.get_channel_gains() is not None:
        gain = rec.get_channel_gains()
    else:
        print("Recording does not have channel gains to uV")
        gain = 1.0

    if _globals.OFFSET_TO_UV is not None:
        offset = _globals.OFFSET_TO_UV
    elif rec.get_channel_offsets() is not None:
        offset = rec.get_channel_offsets()
    else:
        print("Recording does not have channel offsets to uV")
        offset = 0.0

    print(
        f"Scaling recording to uV with gain {np.median(np.array(gain))} and offset {np.median(np.array(offset))}"
    )
    print(f"Converting recording dtype from {rec.get_dtype()} to float32")

    rec = ScaleRecording(rec, gain=gain, offset=offset, dtype="float32")

    rec = bandpass_filter(rec, freq_min=_globals.FREQ_MIN, freq_max=_globals.FREQ_MAX)

    return rec


def concatenate_recordings(rec_path: Path) -> BaseRecording:
    """Load and concatenate all recordings in a directory.

    Scans *rec_path* for ``.raw.h5`` and ``.nwb`` files, loads each via
    ``load_single_recording``, and concatenates them into a single
    multi-segment recording. Updates the global ``REC_CHUNKS`` with the
    frame boundaries of each constituent recording.

    Parameters:
        rec_path (Path): Directory containing recording files.

    Returns:
        rec (BaseRecording): The concatenated recording.

    Notes:
        Before concatenation, all recordings are validated for
        compatibility:

        - **Channel count** and **sampling frequency** must match
          across all files — a ``ValueError`` is raised otherwise.
        - **Channel IDs** and **channel locations** are compared
          against the first file.  Mismatches produce a warning but
          do not block concatenation, since the user may intentionally
          combine recordings with different routing configurations.
          However, differing electrode layouts will likely produce
          unreliable sorting results.
    """
    print("Concatenating recordings")
    recordings = []

    new_rec_chunks = []
    start_frame = 0

    recording_names = natsorted(
        [
            p.name
            for p in rec_path.iterdir()
            if p.name.endswith(".raw.h5") or p.name.endswith(".nwb")
        ]
    )
    for rec_name in recording_names:
        rec_file = [p for p in rec_path.iterdir() if p.name == rec_name][0]
        rec = load_single_recording(rec_file)
        recordings.append(rec)
        print(
            f"{rec_name}: DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
            f"NUM. CHANNELS: {rec.get_num_channels()}"
        )

        end_frame = start_frame + rec.get_total_samples()
        new_rec_chunks.append((start_frame, end_frame))
        start_frame = end_frame

    # Validate compatibility before concatenation
    if len(recordings) > 1:
        ref = recordings[0]
        ref_name = recording_names[0]
        ref_n_ch = ref.get_num_channels()
        ref_fs = ref.get_sampling_frequency()
        ref_ids = list(ref.get_channel_ids())
        ref_locs = ref.get_channel_locations()

        for i, (rec_i, name_i) in enumerate(
            zip(recordings[1:], recording_names[1:]), start=1
        ):
            # Hard error: channel count or sampling frequency mismatch
            n_ch = rec_i.get_num_channels()
            if n_ch != ref_n_ch:
                raise ValueError(
                    f"Cannot concatenate: {name_i} has {n_ch} channels "
                    f"but {ref_name} has {ref_n_ch}."
                )
            fs = rec_i.get_sampling_frequency()
            if fs != ref_fs:
                raise ValueError(
                    f"Cannot concatenate: {name_i} has sampling frequency "
                    f"{fs} Hz but {ref_name} has {ref_fs} Hz."
                )

            # Warning: channel IDs differ
            ids_i = list(rec_i.get_channel_ids())
            if ids_i != ref_ids:
                warnings.warn(
                    f"{name_i} has different channel IDs than {ref_name}. "
                    "Concatenation will proceed but results may be unreliable "
                    "if the electrode configurations differ.",
                    stacklevel=2,
                )

            # Warning: channel locations differ
            locs_i = rec_i.get_channel_locations()
            if not np.array_equal(ref_locs, locs_i):
                warnings.warn(
                    f"{name_i} has different channel locations than "
                    f"{ref_name}. This likely means different electrode "
                    "configurations — concatenation will proceed but "
                    "sorting results may be unreliable.",
                    stacklevel=2,
                )

    if len(recordings) == 1:
        rec = recordings[0]
    else:
        rec = si_segmentutils.concatenate_recordings(recordings)
        if len(_globals.REC_CHUNKS) == 0:
            _globals.REC_CHUNKS = new_rec_chunks

    print(f"Done concatenating {len(recordings)} recordings")
    print(f"Total duration: {rec.get_total_duration()}s")

    # Store file names globally so downstream code (build_spikedata) can
    # include them in metadata for epoch splitting.
    _globals._REC_CHUNK_NAMES = recording_names

    return rec


def extract_waveforms(
    recording_path: Any,
    recording: BaseRecording,
    sorting: Any,
    root_folder: Path,
    initial_folder: Path,
    rng: Any = None,
    **job_kwargs: Any,
) -> Any:
    """
    Extracts waveform on paired Recording-Sorting objects.
    Waveforms are persistent on disk and cached in memory.

    Parameters
    ----------
    recording_path: Path
        The path of the raw recording
    recording: Recording
        The recording object
    sorting: Sorting
        The sorting object
    root_folder: Path
        The root folder of waveforms
    initial_folder: Path
        Folder representing units before curation

    Returns
    -------
    we: WaveformExtractor
        The WaveformExtractor object that represents the waveforms
    """

    print_stage("EXTRACTING WAVEFORMS")
    stopwatch = Stopwatch()

    if (
        not _globals.REEXTRACT_WAVEFORMS and (root_folder / "waveforms").is_dir()
    ):  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(
            recording, sorting, root_folder, initial_folder, rng=rng
        )
        stopwatch.log_time("Done extracting waveforms.")
    else:  # Create new waveform extractor
        we = WaveformExtractor.create_initial(
            recording_path, recording, sorting, root_folder, initial_folder, rng=rng
        )
        if _globals.STREAMING_WAVEFORMS:
            # Streaming path: per-unit waveforms + templates in one pass.
            # Bounded peak RAM (one unit's buffer at a time); avoids the
            # 39 GB pre-allocated per-unit memmap pile that the parallel
            # path creates for high-unit-count sorts on dense MEAs.
            print("Streaming waveform extraction (per-unit, low RAM)")
            we.run_extract_waveforms_streaming()
            stopwatch.log_time("Done extracting waveforms (streaming).")
            # Templates already populated by the streaming pass.
        else:
            we.run_extract_waveforms(**job_kwargs)
            stopwatch.log_time("Done extracting waveforms.")
            we.compute_templates(
                modes=("average", "std"), n_jobs=job_kwargs.get("n_jobs", 1)
            )
    return we
