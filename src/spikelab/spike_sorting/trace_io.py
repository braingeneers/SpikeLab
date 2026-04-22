"""Trace saving utilities for downstream detection model training."""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
from tqdm import tqdm

from spikeinterface.core import BaseRecording
from spikeinterface.extractors.extractor_classes import MaxwellRecordingExtractor


def save_traces(
    recording: Any,
    inter_path: Union[str, Path],
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    num_processes: Optional[int] = None,
    dtype: str = "float16",
    verbose: bool = True,
) -> None:
    """Save scaled voltage traces to a ``.npy`` file for fast downstream access.

    Dispatches to a Maxwell-optimised path (direct HDF5 reads via ``h5py``)
    or a generic SpikeInterface path depending on the recording type.

    Parameters:
        recording: File path to a recording or a SpikeInterface
            ``BaseRecording`` object.
        inter_path (str or Path): Directory for intermediate files.
            Created if it does not exist.
        start_ms (float): Start time in milliseconds (default 0).
        end_ms (float or None): End time in milliseconds. When *None*,
            the full recording is used.
        num_processes (int or None): Number of parallel workers. Defaults
            to half the available CPU cores.
        dtype (str): NumPy dtype for the saved traces (default
            ``'float16'``).
        verbose (bool): Print progress messages.

    Returns:
        scaled_traces_path (Path): Path to the saved ``.npy`` file.
    """
    from .recording_io import load_recording

    if verbose:
        print("Saving traces:")
    recording = load_recording(recording)

    if num_processes is None:
        num_processes = max(1, os.cpu_count() // 2)

    inter_path = Path(inter_path)
    inter_path.mkdir(exist_ok=True, parents=True)
    scaled_traces_path = inter_path / "scaled_traces.npy"
    if isinstance(recording, MaxwellRecordingExtractor):
        # Use h5py instead of spikeinterface to save Maxwell recording traces since h5py is much faster
        save_traces_mea(
            recording._kwargs["file_path"],
            scaled_traces_path,
            start_ms=start_ms,
            end_ms=end_ms,
            num_processes=num_processes,
            dtype=dtype,
            verbose=verbose,
        )
    else:
        save_traces_si(
            recording,
            scaled_traces_path,
            start_ms=start_ms,
            end_ms=end_ms,
            num_processes=num_processes,
            dtype=dtype,
            verbose=verbose,
        )
    return scaled_traces_path


def save_traces_si(
    recording: BaseRecording,
    scaled_traces_path: Union[str, Path],
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    num_processes: int = 16,
    dtype: str = "float16",
    verbose: bool = True,
) -> None:
    """Save scaled traces from a SpikeInterface recording to a ``.npy`` file.

    Each channel is extracted in parallel and written into a pre-allocated
    memory-mapped array of shape ``(num_channels, num_frames)``.

    Parameters:
        recording (BaseRecording): SpikeInterface recording object.
        scaled_traces_path (str or Path): Output ``.npy`` file path.
        start_ms (float): Start time in milliseconds (default 0).
        end_ms (float or None): End time in milliseconds. When *None*,
            the full recording is used.
        num_processes (int): Number of parallel workers (default 16).
        dtype (str): NumPy dtype for the saved traces (default
            ``'float16'``).
        verbose (bool): Print progress messages.
    """

    samp_freq = recording.get_sampling_frequency() / 1000  # kHz
    num_elecs = recording.get_num_channels()

    start_frame = round(start_ms * samp_freq)

    if end_ms is None:
        end_frame = recording.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

    if verbose:
        print("Allocating disk space for traces ...")
    traces = np.zeros((num_elecs, end_frame - start_frame), dtype=dtype)
    np.save(scaled_traces_path, traces)
    del traces

    if verbose:
        print("Extracting traces")

    from multiprocessing import Pool, Manager

    with Manager() as manager:
        config = manager.Namespace()
        config.recording = recording
        tasks = [
            (config, start_frame, end_frame, channel_idx, scaled_traces_path, dtype)
            for channel_idx in range(num_elecs)
        ]
        with Pool(processes=num_processes) as pool:
            imap = pool.imap_unordered(_save_traces_si, tasks)
            if verbose:
                imap = tqdm(imap, total=len(tasks))
            for _ in imap:
                pass


def _save_traces_si(task: tuple) -> None:
    """Worker function for ``save_traces_si``.

    Extracts traces for a single channel and writes them into the
    pre-allocated ``.npy`` file via memory-mapped access.

    Parameters:
        task (tuple): ``(config, start_frame, end_frame, channel_idx,
            save_path, dtype)`` packed by ``save_traces_si``.
    """
    config, start_frame, end_frame, channel_idx, save_path, dtype = task
    recording = config.recording
    traces = (
        recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[recording.get_channel_ids()[channel_idx]],
            return_scaled=recording.has_scaleable_traces(),
        )
        .flatten()
        .astype(dtype)
    )
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[channel_idx] = traces


def save_traces_mea(
    rec_path: Union[str, Path],
    save_path: Union[str, Path],
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    samp_freq: float = 20,  # kHz
    default_gain: float = 1,
    chunk_size: int = 100000,
    num_processes: int = 2,
    dtype: str = "float16",
    verbose: bool = True,
) -> None:
    """Save scaled traces from a Maxwell MEA recording to a ``.npy`` file.

    Reads the HDF5 file directly with ``h5py`` instead of SpikeInterface's
    ``get_traces()``, which is significantly slower on Maxwell recordings.
    Traces are extracted in parallel chunks and written into a pre-allocated
    memory-mapped array.

    Parameters:
        rec_path (str or Path): Path to the Maxwell ``.h5`` recording file.
        save_path (str or Path): Output ``.npy`` file path.
        start_ms (float): Start time in milliseconds (default 0).
        end_ms (float or None): End time in milliseconds. When *None*,
            the full recording is used.
        samp_freq (float): Sampling frequency in kHz (default 20).
        default_gain (float): Fallback gain factor when the recording does
            not report channel gains (default 1).
        chunk_size (int): Number of frames per processing chunk
            (default 100000).
        num_processes (int): Number of parallel workers (default 2).
        dtype (str): NumPy dtype for the saved traces (default
            ``'float16'``).
        verbose (bool): Print progress messages.
    """

    rec_h5 = h5py.File(rec_path, "r")
    rec_si = MaxwellRecordingExtractor(rec_path)

    start_frame = round(start_ms * samp_freq)

    if end_ms is None:
        end_frame = rec_si.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

    try:
        if "sig" in rec_h5:  # Old file format
            chan_ind = [int(chan_id) for chan_id in rec_si.get_channel_ids()]
            get_traces = _get_traces_mea_old
        else:
            # Check that h5py matches rec_si
            raw_shape = rec_h5["recordings"]["rec0000"]["well000"]["groups"]["routed"][
                "raw"
            ].shape
            expected_shape = (rec_si.get_num_channels(), rec_si.get_total_samples())
            if raw_shape != expected_shape:
                raise ValueError(
                    f"HDF5 raw data shape {raw_shape} does not match "
                    f"SpikeInterface shape {expected_shape}."
                )
            chan_ind = list(range(rec_si.get_num_channels()))
            get_traces = _get_traces_mea_new
    finally:
        rec_h5.close()
    if rec_si.has_scaleable_traces():
        gain = rec_si.get_channel_gains()
    else:
        gain = np.full_like(chan_ind, default_gain, dtype="float16")
        if verbose:
            print(f"Recording does not have channel gains. Setting gain to {gain}")
    gain = gain[:, None]

    if verbose:
        print("Allocating memory for traces ...")
    traces = np.zeros((len(chan_ind), end_frame - start_frame), dtype=dtype)
    np.save(save_path, traces)
    del traces

    if verbose:
        print("Extracting traces ...")
    tasks = [
        (
            rec_path,
            save_path,
            start_frame,
            chan_ind,
            chunk_start,
            chunk_size,
            gain,
            dtype,
            get_traces,
        )
        for chunk_start in range(start_frame, end_frame, chunk_size)
    ]

    with mp.Pool(processes=num_processes) as pool:
        imap = pool.imap_unordered(_save_traces_mea, tasks)
        if verbose:
            imap = tqdm(imap, total=len(tasks))
        for _ in imap:
            pass


def _get_traces_mea_old(rec_path: Union[str, Path]) -> Any:
    """Return the raw signal dataset from an old-format Maxwell HDF5 file.

    Parameters:
        rec_path (str or Path): Path to the Maxwell ``.h5`` file.

    Returns:
        sig (h5py.Dataset): The ``'sig'`` dataset.
    """
    return h5py.File(rec_path, "r")["sig"]


def _get_traces_mea_new(rec_path: Union[str, Path]) -> Any:
    """Return the raw signal dataset from a new-format Maxwell HDF5 file.

    Parameters:
        rec_path (str or Path): Path to the Maxwell ``.h5`` file.

    Returns:
        raw (h5py.Dataset): The ``recordings/rec0000/well000/groups/routed/raw``
            dataset.
    """
    return h5py.File(rec_path, "r")["recordings"]["rec0000"]["well000"]["groups"][
        "routed"
    ]["raw"]


def _save_traces_mea(task: tuple) -> None:
    """Worker function for ``save_traces_mea``.

    Reads one chunk of frames from the HDF5 file, scales by gain, and
    writes the result into the pre-allocated ``.npy`` file via
    memory-mapped access.

    Parameters:
        task (tuple): ``(rec_path, save_path, start_frame, chan_ind,
            chunk_start, chunk_size, gain, dtype, get_traces)`` packed
            by ``save_traces_mea``.
    """
    (
        rec_path,
        save_path,
        start_frame,
        chan_ind,
        chunk_start,
        chunk_size,
        gain,
        dtype,
        get_traces,
    ) = task
    sig = get_traces(rec_path)
    traces = sig[chan_ind, chunk_start : chunk_start + chunk_size].astype(dtype) * gain
    saved_traces = np.load(save_path, mmap_mode="r+")
    saved_traces[
        :, chunk_start - start_frame : chunk_start - start_frame + traces.shape[1]
    ] = traces  # using traces.shape[1] in case chunk_start is within chunk_size of the end of the file (does not raise index error)
