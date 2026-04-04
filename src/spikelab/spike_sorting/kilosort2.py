"""
Kilosort2 spike-sorting pipeline for Maxwell Biosystems and NWB recordings.

The single public entry point is :func:`sort_with_kilosort2`, which runs the
full pipeline — loading, bandpass filtering, binary-file conversion, MATLAB
execution, waveform extraction, and two-stage quality curation — and returns
the curated units as a :class:`~spikelab.spikedata.SpikeData` object.

All internal classes and helper functions are private to this module.

Much of this code is adapted from SpikeInterface.
"""

import datetime
import h5py
import json
import os
import shutil
import signal
import subprocess
import sys
import warnings
import tempfile
import time
import traceback
from math import ceil
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.io import savemat
from tqdm import tqdm

import spikeinterface.core.segmentutils as si_segmentutils
from spikeinterface.core import BaseRecording
from spikeinterface.extractors.extractor_classes import (
    BinaryRecordingExtractor,
    MaxwellRecordingExtractor,
    NwbRecordingExtractor,
)
from spikeinterface.core import write_binary_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.preprocessing.preprocessing_classes import ScaleRecording
from spikeinterface.sorters import run_sorter

__all__ = ["sort_with_kilosort2"]

# Module-level global set by sort_with_kilosort2 and consumed by load_single_recording.
STREAM_ID = None


from .sorting_utils import (  # noqa: E402
    print_stage,
    Stopwatch,
    Tee,
    create_folder,
    delete_folder,
    get_paths,
)


# region Save traces
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

    rec_h5 = h5py.File(rec_path)
    rec_si = MaxwellRecordingExtractor(rec_path)

    start_frame = round(start_ms * samp_freq)

    if end_ms is None:
        end_frame = rec_si.get_total_samples()
    else:
        end_frame = round(end_ms * samp_freq)

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


# endregion


# region Kilosort
class RunKilosort:
    """Kilosort2 MATLAB sorter interface.

    Manages the full Kilosort2 execution lifecycle: locating the MATLAB
    installation, generating MATLAB scripts and channel maps, launching the
    sorter as a subprocess, and collecting the results as a
    ``KilosortSortingExtractor``.

    The constructor validates the Kilosort2 path (from the module-level
    ``KILOSORT_PATH`` global or the ``KILOSORT_PATH`` environment variable),
    checks that the expected MATLAB entry-point script exists, and formats
    the Kilosort2 parameter dict.

    Attributes:
        path (str): Absolute path to the Kilosort2 MATLAB source tree.
    """

    def __init__(self):
        # Set paths
        self.path = self.set_kilosort_path(KILOSORT_PATH)

        # Check if kilosort is installed
        if not self.check_if_installed():
            raise Exception(f"Kilosort2 is not installed.")

        # Make sure parameters are formatted correctly
        RunKilosort.format_params()

    # Run kilosort
    def run(self, recording, recording_dat_path, output_folder):
        # STEP 1) Creates kilosort and recording files needed to run kilosort
        self.setup_recording_files(recording, recording_dat_path, output_folder)

        # STEP 2) Actually run kilosort
        self.start_sorting(output_folder, raise_error=True, verbose=True)

        # STEP 3) Return results of Kilosort as Python object for auto curation
        return RunKilosort.get_result_from_folder(output_folder)

    def setup_recording_files(self, recording, recording_dat_path, output_folder):
        # Prepare electrode positions for this group (only one group, the split is done in spikeinterface's basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError(
                "3D 'location' are not supported. Set 2D locations instead"
            )

        # region Make substitutions in txt files to set kilosort parameters
        # region Config text
        kilosort2_master_txt = """try
            % prepare for kilosort execution
            addpath(genpath('{kilosort2_path}'));

            % set file path
            fpath = '{output_folder}';

            % add npy-matlab functions (copied in the output folder)
            addpath(genpath(fpath));

            % create channel map file
            run(fullfile('{channel_path}'));

            % Run the configuration file, it builds the structure of options (ops)
            run(fullfile('{config_path}'))

            ops.trange = [0 Inf]; % time range to sort

            % preprocess data to create temp_wh.dat
            rez = preprocessDataSub(ops);

            % time-reordering as a function of drift
            rez = clusterSingleBatches(rez);

            % main tracking and template matching algorithm
            rez = learnAndSolve8b(rez);

            % final merges
            rez = find_merges(rez, 1);

            % final splits by SVD
            rez = splitAllClusters(rez, 1);

            % final splits by amplitudes
            rez = splitAllClusters(rez, 0);

            % decide on cutoff
            rez = set_cutoff(rez);

            fprintf('found %d good units \\n', sum(rez.good>0))

            fprintf('Saving results to Phy  \\n')
            rezToPhy(rez, fullfile(fpath));
        catch
            fprintf('----------------------------------------');
            fprintf(lasterr());
            settings  % https://www.mathworks.com/matlabcentral/answers/1566246-got-error-using-exit-in-nodesktop-mode
            quit(1);
        end
        settings  % https://www.mathworks.com/matlabcentral/answers/1566246-got-error-using-exit-in-nodesktop-mode
        quit(0);"""
        kilosort2_config_txt = """ops.NchanTOT            = {nchan};           % total number of channels (omit if already in chanMap file)
        ops.Nchan               = {nchan};           % number of active channels (omit if already in chanMap file)
        ops.fs                  = {sample_rate};     % sampling rate

        ops.datatype            = 'dat';  % binary ('dat', 'bin') or 'openEphys'
        ops.fbinary             = fullfile('{dat_file}'); % will be created for 'openEphys'
        ops.fproc               = fullfile(fpath, 'temp_wh.dat'); % residual from RAM of preprocessed data
        ops.root                = fpath; % 'openEphys' only: where raw files are
        % define the channel map as a filename (string) or simply an array
        ops.chanMap             = fullfile('chanMap.mat'); % make this file using createChannelMapFile.m

        % frequency for high pass filtering (150)
        ops.fshigh = {freq_min};

        % minimum firing rate on a "good" channel (0 to skip)
        ops.minfr_goodchannels = {minfr_goodchannels};

        % threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops.Th = {projection_threshold};

        % how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops.lam = 10;

        % splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
        ops.AUCsplit = 0.9;

        % minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        ops.minFR = {minFR};

        % number of samples to average over (annealed from first to second value)
        ops.momentum = [20 400];

        % spatial constant in um for computing residual variance of spike
        ops.sigmaMask = {sigmaMask};

        % threshold crossings for pre-clustering (in PCA projection space)
        ops.ThPre = {preclust_threshold};
        %% danger, changing these settings can lead to fatal errors
        % options for determining PCs
        ops.spkTh           = -{kilo_thresh};      % spike threshold in standard deviations (-6)
        ops.reorder         = 1;       % whether to reorder batches for drift correction.
        ops.nskip           = 25;  % how many batches to skip for determining spike PCs

        ops.CAR             = {use_car}; % perform CAR

        ops.GPU                 = 1; % has to be 1, no CPU version yet, sorry
        % ops.Nfilt             = 1024; % max number of clusters
        ops.nfilt_factor        = {nfilt_factor}; % max number of clusters per good channel (even temporary ones) 4
        ops.ntbuff              = {ntbuff};    % samples of symmetrical buffer for whitening and spike detection 64
        ops.NT                  = {NT}; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).  64*1024 + ops.ntbuff
        ops.whiteningRange      = 32; % number of channels to use for whitening each channel
        ops.nSkipCov            = 25; % compute whitening matrix from every N-th batch
        ops.scaleproc           = 200;   % int16 scaling of whitened data
        ops.nPCs                = {nPCs}; % how many PCs to project the spikes into
        ops.useRAM              = 0; % not yet available

        %%"""
        kilosort2_channelmap_txt = """%  create a channel map file

        Nchannels = {nchan}; % number of channels
        connected = true(Nchannels, 1);
        chanMap   = 1:Nchannels;
        chanMap0ind = chanMap - 1;

        xcoords = {xcoords};
        ycoords = {ycoords};
        kcoords   = {kcoords};

        fs = {sample_rate}; % sampling frequency
        save(fullfile('chanMap.mat'), ...
            'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')"""
        # endregion
        kilosort2_master_txt = kilosort2_master_txt.format(
            kilosort2_path=str(Path(self.path).absolute()),
            output_folder=str(output_folder.absolute()),
            channel_path=str((output_folder / "kilosort2_channelmap.m").absolute()),
            config_path=str((output_folder / "kilosort2_config.m").absolute()),
        )

        kilosort2_config_txt = kilosort2_config_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str(recording_dat_path.absolute()),
            projection_threshold=KILOSORT_PARAMS["projection_threshold"],
            preclust_threshold=KILOSORT_PARAMS["preclust_threshold"],
            minfr_goodchannels=KILOSORT_PARAMS["minfr_goodchannels"],
            minFR=KILOSORT_PARAMS["minFR"],
            freq_min=KILOSORT_PARAMS["freq_min"],
            sigmaMask=KILOSORT_PARAMS["sigmaMask"],
            kilo_thresh=KILOSORT_PARAMS["detect_threshold"],
            use_car=KILOSORT_PARAMS["car"],
            nPCs=int(KILOSORT_PARAMS["nPCs"]),
            ntbuff=int(KILOSORT_PARAMS["ntbuff"]),
            nfilt_factor=int(KILOSORT_PARAMS["nfilt_factor"]),
            NT=int(KILOSORT_PARAMS["NT"]),
        )

        kilosort2_channelmap_txt = kilosort2_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=[p[0] for p in positions],
            ycoords=[p[1] for p in positions],
            kcoords=groups,
        )
        # endregion

        # Create config files
        for fname, txt in zip(
            ["kilosort2_master.m", "kilosort2_config.m", "kilosort2_channelmap.m"],
            [kilosort2_master_txt, kilosort2_config_txt, kilosort2_channelmap_txt],
        ):
            with (output_folder / fname).open("w") as f:
                f.write(txt)

        # Matlab (for reading and writing numpy) scripts texts
        writeNPY_text = """% NPY-MATLAB writeNPY function. Copied from https://github.com/kwikteam/npy-matlab

function writeNPY(var, filename)
% function writeNPY(var, filename)
%
% Only writes little endian, fortran (column-major) ordering; only writes
% with NPY version number 1.0.
%
% Always outputs a shape according to matlab's convention, e.g. (10, 1)
% rather than (10,).

shape = size(var);
dataType = class(var);

header = constructNPYheader(dataType, shape);

fid = fopen(filename, 'w');
fwrite(fid, header, 'uint8');
fwrite(fid, var, dataType);
fclose(fid);


end"""
        constructNPYheader_text = """% NPY-MATLAB constructNPYheader function. Copied from https://github.com/kwikteam/npy-matlab


function header = constructNPYheader(dataType, shape, varargin)

	if ~isempty(varargin)
		fortranOrder = varargin{1}; % must be true/false
		littleEndian = varargin{2}; % must be true/false
	else
		fortranOrder = true;
		littleEndian = true;
	end

    dtypesMatlab = {'uint8','uint16','uint32','uint64','int8','int16','int32','int64','single','double', 'logical'};
    dtypesNPY = {'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'b1'};

    magicString = uint8([147 78 85 77 80 89]); %x93NUMPY

    majorVersion = uint8(1);
    minorVersion = uint8(0);

    % build the dict specifying data type, array order, endianness, and
    % shape
    dictString = '{''descr'': ''\';

    if littleEndian
        dictString = [dictString '<'];
    else
        dictString = [dictString '>'];
    end

    dictString = [dictString dtypesNPY{strcmp(dtypesMatlab,dataType)} ''', '];

    dictString = [dictString '''fortran_order'': '];

    if fortranOrder
        dictString = [dictString 'True, '];
    else
        dictString = [dictString 'False, '];
    end

    dictString = [dictString '''shape'': ('];

%     if length(shape)==1 && shape==1
%
%     else
%         for s = 1:length(shape)
%             if s==length(shape) && shape(s)==1
%
%             else
%                 dictString = [dictString num2str(shape(s))];
%                 if length(shape)>1 && s+1==length(shape) && shape(s+1)==1
%                     dictString = [dictString ','];
%                 elseif length(shape)>1 && s<length(shape)
%                     dictString = [dictString ', '];
%                 end
%             end
%         end
%         if length(shape)==1
%             dictString = [dictString ','];
%         end
%     end

    for s = 1:length(shape)
        dictString = [dictString num2str(shape(s))];
        if s<length(shape)
            dictString = [dictString ', '];
        end
    end

    dictString = [dictString '), '];

    dictString = [dictString '}'];

    totalHeaderLength = length(dictString)+10; % 10 is length of magicString, version, and headerLength

    headerLengthPadded = ceil(double(totalHeaderLength+1)/16)*16; % the whole thing should be a multiple of 16
                                                                  % I add 1 to the length in order to allow for the newline character

	% format specification is that headerlen is little endian. I believe it comes out so using this command...
    headerLength = typecast(int16(headerLengthPadded-10), 'uint8');

    zeroPad = zeros(1,headerLengthPadded-totalHeaderLength, 'uint8')+uint8(32); % +32 so they are spaces
    zeroPad(end) = uint8(10); % newline character

    header = uint8([magicString majorVersion minorVersion headerLength dictString zeroPad]);

end"""

        # Create matlab scripts
        for fname, txt in zip(
            ["writeNPY.m", "constructNPYheader.m"],
            [writeNPY_text, constructNPYheader_text],
        ):
            with (output_folder / fname).open("w") as f:
                f.write(txt)

    def start_sorting(self, output_folder, raise_error, verbose):
        output_folder = Path(output_folder)

        t0 = time.perf_counter()
        try:
            self.execute_kilosort_file(output_folder, verbose)
            t1 = time.perf_counter()
            run_time = float(t1 - t0)
            has_error = False
        except Exception as err:
            has_error = True
            run_time = None

        # Kilosort has a log file dir to shellscript launcher
        runtime_trace_path = output_folder / "kilosort2.log"
        runtime_trace = []
        if runtime_trace_path.is_file():
            with open(runtime_trace_path, "r") as fp:
                line = fp.readline()
                while line:
                    runtime_trace.append(line.strip())
                    line = fp.readline()

        if verbose:
            if has_error:
                print("Error running kilosort2")
            else:
                print(f"kilosort2 run time: {run_time:0.2f}s")

        if has_error and raise_error:
            raise Exception(
                f"You can inspect the runtime trace in {output_folder}/kilosort2.log"
            )

        return run_time

    @staticmethod
    def execute_kilosort_file(output_folder, verbose):
        print("Running kilosort file")

        if "win" in sys.platform and sys.platform != "darwin":
            shell_cmd = f"""cd "{output_folder}"
            matlab -nosplash -wait -log -r kilosort2_master
            """
        else:
            shell_cmd = f"""
                        #!/bin/bash
                        cd "{output_folder}"
                        matlab -nosplash -nodisplay -log -r kilosort2_master
                    """
        shell_script = ShellScript(
            shell_cmd,
            script_path=output_folder / "run_kilosort2",
            log_path=output_folder / "kilosort2.log",
            verbose=verbose,
        )
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception("kilosort2 returned a non-zero exit code")

    def check_if_installed(self):
        if (Path(self.path) / "master_kilosort.m").is_file() or (
            Path(self.path) / "main_kilosort.m"
        ).is_file():
            return True
        else:
            return False

    @staticmethod
    def set_kilosort_path(kilosort_path):
        if kilosort_path is None:
            if "KILOSORT_PATH" not in os.environ:
                raise ValueError(
                    "Because environment variable KILOSORT_PATH is not defined, you must set kilosort_path='/path/to/kilosort2' in the call to sort_with_kilosort2"
                )
            return os.environ["KILOSORT_PATH"]

        path = str(Path(kilosort_path).absolute())

        try:
            print(
                "Setting KILOSORT_PATH environment variable for subprocess calls to:",
                path,
            )
            os.environ["KILOSORT_PATH"] = path
        except Exception as e:
            print("Could not set KILOSORT_PATH environment variable:", e)

        return path

    @staticmethod
    def format_params():
        if KILOSORT_PARAMS["NT"] is None:
            KILOSORT_PARAMS["NT"] = (
                64 * 1024 + KILOSORT_PARAMS["ntbuff"]
            )  # https://github.com/MouseLand/Kilosort/issues/380
        else:
            KILOSORT_PARAMS["NT"] = (
                KILOSORT_PARAMS["NT"] // 32 * 32
            )  # make sure is multiple of 32

        if KILOSORT_PARAMS["car"]:
            KILOSORT_PARAMS["car"] = 1
        else:
            KILOSORT_PARAMS["car"] = 0

    @classmethod
    def get_result_from_folder(cls, output_folder):
        return KilosortSortingExtractor(folder_path=output_folder)


class KilosortSortingExtractor:
    """
    Represents data from Phy and Kilosort output folder as Python object

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py which stores data about the raw recording)
    exclude_cluster_groups: list or str (optional)
        Cluster groups to exclude (e.g. "noise" or ["noise", "mua"])
    """

    def __init__(self, folder_path, exclude_cluster_groups=None):
        # Folder containing the numpy results of Kilosort
        phy_folder = Path(folder_path)
        self.folder = phy_folder.absolute()

        self.spike_times = np.atleast_1d(
            np.load(str(phy_folder / "spike_times.npy")).astype(int).flatten()
        )
        self.spike_clusters = np.atleast_1d(
            np.load(str(phy_folder / "spike_clusters.npy")).flatten()
        )

        # The unit_ids with at least 1 spike
        unit_ids_with_spike = set(self.spike_clusters)

        params = Utils.read_python(str(phy_folder / "params.py"))
        self.sampling_frequency = params["sample_rate"]

        # Load properties from tsv/csv files
        all_property_files = [
            p for p in phy_folder.iterdir() if p.suffix in [".csv", ".tsv"]
        ]

        cluster_info = None
        for file in all_property_files:
            if file.suffix == ".tsv":
                delimeter = "\t"
            else:
                delimeter = ","
            new_property = pd.read_csv(file, delimiter=delimeter)
            if cluster_info is None:
                cluster_info = new_property
            else:
                if new_property.columns[-1] not in cluster_info.columns:
                    # cluster_KSLabel.tsv and cluster_group.tsv are identical and have the same columns
                    # This prevents the same column data being added twice
                    cluster_info = pd.merge(cluster_info, new_property, on="cluster_id")

        # In case no tsv/csv files are found populate cluster info with minimal info
        if cluster_info is None:
            unit_ids_with_spike_list = list(unit_ids_with_spike)
            cluster_info = pd.DataFrame({"cluster_id": unit_ids_with_spike_list})
            cluster_info["group"] = ["unsorted"] * len(unit_ids_with_spike_list)

        # If pandas column for the unit_ids uses different name
        if "cluster_id" not in cluster_info.columns:
            if "id" not in cluster_info.columns:
                raise ValueError(
                    "Couldn't find cluster IDs in the TSV file. Expected a "
                    f"'cluster_id' or 'id' column, found: {list(cluster_info.columns)}"
                )
            cluster_info["cluster_id"] = cluster_info["id"]
            del cluster_info["id"]

        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(
                    f"group != '{exclude_cluster_groups}'"
                )
            elif isinstance(exclude_cluster_groups, list):
                if len(exclude_cluster_groups) > 0:
                    for exclude_group in exclude_cluster_groups:
                        cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if KILOSORT_PARAMS["keep_good_only"] and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        all_unit_ids = cluster_info["cluster_id"].values
        self.unit_ids = []
        # Exclude units with 0 spikes
        for unit_id in all_unit_ids:
            if unit_id in unit_ids_with_spike:
                self.unit_ids.append(int(unit_id))

    @staticmethod
    def get_num_segments():
        # Sorting should always have 1 segment
        return 1

    def get_unit_spike_train(
        self,
        unit_id,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ):
        spike_times = self.spike_times[self.spike_clusters == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]

        return np.atleast_1d(spike_times.copy().squeeze())

    def get_templates_all(self):
        # Returns Kilosort2's outputted templates as mmap np.array
        return np.load(str(self.folder / "templates.npy"), mmap_mode="r")

    def get_channel_map(self):
        # Returns Kilosort2's channel map as mmap np.array
        return np.load(str(self.folder / "channel_map.npy"), mmap_mode="r").squeeze()

    def get_chans_max(self):
        """
        Get the max channel of each unit based on Kilosort2's template
        and whether to use (min/argmin or max/argmax) for computing peak values

        Returns
        -------
        All are np.arrays that follow np.array[unit_id] = value
        In other words, the np.arrays contain data for ALL units (even units with 0 spikes)

        use_pos_peak
            0 = Use negative peak
            1 = Use positive peak
        chans_max_kilosort
            The channel with the highest amplitude for each unit based on kilosort's selected channels
            that were used during spike sorting (considered not "bad channels")
        chans_max
            The channel with the highest amplitude for each unit converted from kilosort's channels
            to channels in the recording (with all channels)
        """

        templates_all = self.get_templates_all()

        chans_neg_peaks_values = np.min(templates_all, axis=1)
        chans_neg_peaks_indices = chans_neg_peaks_values.argmin(axis=1)
        chans_neg_peaks_values = np.min(chans_neg_peaks_values, axis=1)

        chans_pos_peaks_values = np.max(templates_all, axis=1)
        chans_pos_peaks_indices = chans_pos_peaks_values.argmax(axis=1)
        chans_pos_peaks_values = np.max(chans_pos_peaks_values, axis=1)

        use_pos_peak = chans_pos_peaks_values >= POS_PEAK_THRESH * np.abs(
            chans_neg_peaks_values
        )
        chans_max_kilosort = np.where(
            use_pos_peak, chans_pos_peaks_indices, chans_neg_peaks_indices
        )
        chans_max_all = self.get_channel_map()[chans_max_kilosort]

        return use_pos_peak, chans_max_kilosort, chans_max_all

    def get_templates_half_windows_sizes(
        self, chans_max_kilosort, window_size_scale=0.75
    ):
        """
        Get the half window sizes that will be used to recenter the spike times on the peak

        Parameters
        ----------
        chans_max_kilosort: np.array
            np.array with shape (n_templates,) giving the max channel of each template using
            Kilosort's channel map
        window_size_scale: float
            Value to scale the window size for finding the peak
                Smaller = smaller window, less risk of picking wrong peak, higher risk of picking not the peak value of the peak

        Returns
        -------

        """
        # Get the half window sizes that will be used to recenter the spike times on the peak
        templates_all = self.get_templates_all()[
            np.arange(chans_max_kilosort.size), :, chans_max_kilosort
        ]
        n_templates, n_samples = templates_all.shape
        template_mid = n_samples // 2
        half_windows_sizes = []
        for i in range(n_templates):
            template = templates_all[i, :]
            # Find where the template amplitude drops below 1% of peak
            # before the midpoint.  Works for both KS2 (zero-padded) and
            # KS4 (dense, non-zero edges) templates.
            peak_amp = np.abs(template).max()
            threshold = peak_amp * 0.01
            small_indices = np.flatnonzero(np.abs(template[:template_mid]) < threshold)
            if small_indices.size > 0:
                size = template_mid - small_indices[-1]
            else:
                size = template_mid
            half_windows_sizes.append(int(size * window_size_scale))

        return half_windows_sizes

    def ms_to_samples(self, ms: float) -> int:
        return round(ms * self.sampling_frequency / 1000.0)


class ShellScript:
    """Shell script runner for launching MATLAB from Python.

    Writes a shell script to a temporary or specified path, executes it
    as a subprocess, and optionally captures output to a log file. Used
    to run Kilosort2's MATLAB entry-point scripts.

    Parameters:
        script (str): Shell script contents (leading indentation is
            automatically stripped).
        script_path (str, Path, or None): Where to write the script
            file. When *None*, a temporary file is created.
        log_path (str, Path, or None): Path for capturing stdout/stderr.
            When *None*, output is not saved to disk.
        keep_temp_files (bool): Keep the script file after execution
            (default False).
        verbose (bool): Print the script contents before running
            (default False).
    """

    PathType = Union[str, Path]

    def __init__(
        self,
        script: str,
        script_path: Optional[PathType] = None,
        log_path: Optional[PathType] = None,
        keep_temp_files: bool = False,
        verbose: bool = False,
    ):
        lines = script.splitlines()
        lines = self._remove_initial_blank_lines(lines)
        if len(lines) > 0:
            num_initial_spaces = self._get_num_initial_spaces(lines[0])
            for ii, line in enumerate(lines):
                if len(line.strip()) > 0:
                    n = self._get_num_initial_spaces(line)
                    if n < num_initial_spaces:
                        print(script)
                        raise Exception(
                            "Problem in script. First line must not be indented relative to others"
                        )
                    lines[ii] = lines[ii][num_initial_spaces:]
        self._script = "\n".join(lines)
        self._script_path = script_path
        self._log_path = log_path
        self._keep_temp_files = keep_temp_files
        self._process: Optional[subprocess.Popen] = None
        self._files_to_remove: List[str] = []
        self._dirs_to_remove: List[str] = []
        self._start_time: Optional[float] = None
        self._verbose = verbose

    def __del__(self):
        self.cleanup()

    def substitute(self, old: str, new: Any) -> None:
        self._script = self._script.replace(old, "{}".format(new))

    def write(self, script_path: Optional[str] = None) -> None:
        if script_path is None:
            script_path = self._script_path
        if script_path is None:
            raise Exception("Cannot write script. No path specified")
        with open(script_path, "w") as f:
            f.write(self._script)
        os.chmod(script_path, 0o744)

    def start(self) -> None:
        if self._script_path is not None:
            script_path = Path(self._script_path)
            if script_path.suffix == "":
                if "win" in sys.platform and sys.platform != "darwin":
                    script_path = script_path.parent / (script_path.name + ".bat")
                else:
                    script_path = script_path.parent / (script_path.name + ".sh")
        else:
            tempdir = Path(tempfile.mkdtemp(prefix="tmp_shellscript"))
            if "win" in sys.platform and sys.platform != "darwin":
                script_path = tempdir / "script.bat"
            else:
                script_path = tempdir / "script.sh"
            self._dirs_to_remove.append(tempdir)

        if self._log_path is None:
            script_log_path = script_path.parent / "spike_sorters_log.txt"
        else:
            script_log_path = Path(self._log_path)
            if script_path.suffix == "":
                script_log_path = script_log_path.parent / (
                    script_log_path.name + ".txt"
                )

        self.write(script_path)
        cmd = str(script_path)
        print("RUNNING SHELL SCRIPT: " + cmd)
        self._start_time = time.time()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        with open(script_log_path, "w+") as script_log_file:
            for line in self._process.stdout:
                script_log_file.write(line)
                if (
                    self._verbose
                ):  # Print onto console depending on the verbose property passed on from the sorter class
                    print(line)

    def wait(self, timeout=None) -> Optional[int]:
        if not self.isRunning():
            return self.returnCode()
        if self._process is None:
            raise RuntimeError(
                "ShellScript process is None — start() was not called or "
                "the process has already been cleaned up."
            )
        try:
            retcode = self._process.wait(timeout=timeout)
            return retcode
        except:
            return None

    def cleanup(self) -> None:
        if self._keep_temp_files:
            return
        for dirpath in self._dirs_to_remove:
            ShellScript._rmdir_with_retries(str(dirpath), num_retries=5)

    def stop(self) -> None:
        if not self.isRunning():
            return
        if self._process is None:
            raise RuntimeError(
                "ShellScript process is None — start() was not called or "
                "the process has already been cleaned up."
            )

        signals = [signal.SIGINT] * 10 + [signal.SIGTERM] * 10 + [signal.SIGKILL] * 10

        for signal0 in signals:
            self._process.send_signal(signal0)
            try:
                self._process.wait(timeout=0.02)
                return
            except:
                pass

    def kill(self) -> None:
        if not self.isRunning():
            return

        if self._process is None:
            raise RuntimeError(
                "ShellScript process is None — start() was not called or "
                "the process has already been cleaned up."
            )
        self._process.send_signal(signal.SIGKILL)
        try:
            self._process.wait(timeout=1)
        except:
            print("WARNING: unable to kill shell script.")
            pass

    def stopWithSignal(self, sig, timeout) -> bool:
        if not self.isRunning():
            return True

        if self._process is None:
            raise RuntimeError(
                "ShellScript process is None — start() was not called or "
                "the process has already been cleaned up."
            )
        self._process.send_signal(sig)
        try:
            self._process.wait(timeout=timeout)
            return True
        except:
            return False

    def elapsedTimeSinceStart(self) -> Optional[float]:
        if self._start_time is None:
            return None

        return time.time() - self._start_time

    def isRunning(self) -> bool:
        if not self._process:
            return False
        retcode = self._process.poll()
        if retcode is None:
            return True
        return False

    def isFinished(self) -> bool:
        if not self._process:
            return False
        return not self.isRunning()

    def returnCode(self) -> Optional[int]:
        if not self.isFinished():
            raise Exception("Cannot get return code before process is finished.")
        if self._process is None:
            raise RuntimeError(
                "ShellScript process is None — start() was not called or "
                "the process has already been cleaned up."
            )
        return self._process.returncode

    def scriptPath(self) -> Optional[str]:
        return self._script_path

    def _remove_initial_blank_lines(self, lines: List[str]) -> List[str]:
        ii = 0
        while ii < len(lines) and len(lines[ii].strip()) == 0:
            ii = ii + 1
        return lines[ii:]

    def _get_num_initial_spaces(self, line: str) -> int:
        ii = 0
        while ii < len(line) and line[ii] == " ":
            ii = ii + 1
        return ii

    @staticmethod
    def _rmdir_with_retries(dirname, num_retries, delay_between_tries=1):
        for retry_num in range(1, num_retries + 1):
            if not os.path.exists(dirname):
                return
            try:
                shutil.rmtree(dirname)
                break
            except:
                if retry_num < num_retries:
                    print("Retrying to remove directory: {}".format(dirname))
                    time.sleep(delay_between_tries)
                else:
                    raise Exception(
                        "Unable to remove directory after {} tries: {}".format(
                            num_retries, dirname
                        )
                    )


# endregion


# region Extract Waveforms
class WaveformExtractor:
    """Per-unit waveform storage, template computation, and curation helper.

    Extracts spike waveforms from a recording aligned to each unit's
    spike times, computes average and standard-deviation templates, and
    supports saving/loading curated subsets of units to disk.

    Parameters:
        recording (BaseRecording): SpikeInterface recording object.
        sorting (KilosortSortingExtractor): Sorting result containing
            unit IDs and spike trains.
        root_folder (Path): Root folder for all waveform data (contains
            ``extraction_parameters.json``).
        folder (Path): Sub-folder for this instance's unit ID list
            (e.g. initial, first-curation, second-curation).
    """

    # region Initialize
    def __init__(self, recording, sorting, root_folder, folder):
        with open(root_folder / "extraction_parameters.json", "r") as f:
            parameters = json.load(f)

        self.recording = recording
        self.sampling_frequency = parameters["sampling_frequency"]

        self.sorting = sorting
        self.root_folder = root_folder
        self.folder = Path(folder)
        create_folder(self.folder)

        # Cache in memory
        self._waveforms = {}
        self.template_cache = {}

        # Set Parameters
        self.nbefore = self.ms_to_samples(
            parameters["ms_before"]
        )  # Number of samples before waveform peak to include
        self.nafter = (
            self.ms_to_samples(parameters["ms_after"]) + 1
        )  # Number of samples after waveform peak to include (+1 since Python slicing is [inlusive, exclusive))
        self.nsamples = (
            self.nbefore + self.nafter
        )  # Total number of samples in waveform
        self.peak_ind = parameters["peak_ind"]

        # Extract waveforms as µV when the recording supports scaling
        if recording.has_scaleable_traces():
            self.return_scaled = True
            self.dtype = "float32"
        else:
            self.return_scaled = False
            self.dtype = parameters["dtype"]

        self.chans_max_folder = root_folder / "channels_max"
        self.use_pos_peak = None
        self.chans_max_kilosort = None
        self.chans_max_all = None

    @classmethod
    def create_initial(
        cls, recording_path, recording, sorting, root_folder, initial_folder
    ):
        # Create root waveform folder and data
        root_folder = Path(root_folder)
        create_folder(root_folder / "waveforms")

        # Use float32 when the recording supports µV scaling
        if recording.has_scaleable_traces():
            waveform_dtype = "float32"
        else:
            waveform_dtype = str(recording.get_dtype())

        parameters = {
            "recording_path": str(recording_path.absolute()),
            "sampling_frequency": recording.get_sampling_frequency(),
            "ms_before": WAVEFORMS_MS_BEFORE,
            "ms_after": WAVEFORMS_MS_AFTER,
            "peak_ind": sorting.ms_to_samples(WAVEFORMS_MS_BEFORE),
            "pos_peak_thresh": POS_PEAK_THRESH,
            "max_waveforms_per_unit": MAX_WAVEFORMS_PER_UNIT,
            "dtype": waveform_dtype,
            "n_jobs": N_JOBS,
            "total_memory": TOTAL_MEMORY,
        }
        with open(root_folder / "extraction_parameters.json", "w") as f:
            json.dump(parameters, f)

        we = cls(recording, sorting, root_folder, initial_folder)

        # Get template window sizes for computing location of negative peak during waveform extraction
        (
            we.use_pos_peak,
            we.chans_max_kilosort,
            we.chans_max_all,
        ) = we.sorting.get_chans_max()
        create_folder(we.chans_max_folder)
        for save_file, save_data in zip(
            ("use_pos_peak.npy", "chans_max_kilosort.npy", "chans_max_all.npy"),
            (we.use_pos_peak, we.chans_max_kilosort, we.chans_max_all),
        ):
            np.save(we.chans_max_folder / save_file, save_data)

        # Save unit data
        np.save(str(initial_folder / "unit_ids.npy"), sorting.unit_ids)
        np.save(str(initial_folder / "spike_times.npy"), sorting.spike_times)
        np.save(str(initial_folder / "spike_clusters.npy"), sorting.spike_clusters)

        return we

    @classmethod
    def load_from_folder(
        cls,
        recording,
        sorting,
        root_folder,
        folder,
        use_pos_peak=None,
        chans_max_kilosort=None,
        chans_max_all=None,
    ):
        # Load waveform data from folder
        we = cls(recording, sorting, root_folder, folder)

        _possible_template_modes = ("average", "std", "median")
        for mode in _possible_template_modes:
            # Load cached templates
            template_file = we.root_folder / f"templates/templates_{mode}.npy"
            if template_file.is_file():
                we.template_cache[mode] = np.load(template_file, mmap_mode="r")

        if use_pos_peak is None:
            we.use_pos_peak = np.load(
                we.chans_max_folder / "use_pos_peak.npy", mmap_mode="r"
            )
            we.chans_max_kilosort = np.load(
                we.chans_max_folder / "chans_max_kilosort.npy", mmap_mode="r"
            )
            we.chans_max_all = np.load(
                we.chans_max_folder / "chans_max_all.npy", mmap_mode="r"
            )
        else:
            we.use_pos_peak = use_pos_peak
            we.chans_max_kilosort = chans_max_kilosort
            we.chans_max_all = chans_max_all

        we.load_units()
        return we

    def ms_to_samples(self, ms: float) -> int:
        return int(ms * self.sampling_frequency / 1000.0)

    # endregion

    # region Extract waveforms
    def run_extract_waveforms(self, **job_kwargs: Any) -> None:
        self.templates_half_windows_sizes = (
            self.sorting.get_templates_half_windows_sizes(self.chans_max_kilosort)
        )

        num_chans = self.recording.get_num_channels()
        job_kwargs["n_jobs"] = Utils.ensure_n_jobs(
            self.recording, job_kwargs.get("n_jobs", None)
        )

        selected_spikes = self.sample_spikes()

        # Get spike times
        selected_spike_times = {}
        for unit_id in self.sorting.unit_ids:
            selected_spike_times[unit_id] = []
            for segment_index in range(self.sorting.get_num_segments()):
                spike_times = self.sorting.get_unit_spike_train(
                    unit_id=unit_id, segment_index=segment_index
                )
                sel = selected_spikes[unit_id][segment_index]
                spike_times_sel = spike_times[sel]

                selected_spike_times[unit_id].append(spike_times_sel)

        # Prepare memmap for waveforms
        print("Preparing memory maps for waveforms")
        wfs_memmap = {}
        for unit_id in self.sorting.unit_ids:
            file_path = self.root_folder / "waveforms" / f"waveforms_{unit_id}.npy"
            n_spikes = np.sum([e.size for e in selected_spike_times[unit_id]])
            shape = (n_spikes, self.nsamples, num_chans)
            wfs = np.zeros(shape, self.dtype)
            np.save(str(file_path), wfs)
            wfs_memmap[unit_id] = file_path

        # Run extract waveforms
        func = WaveformExtractor._waveform_extractor_chunk
        init_func = WaveformExtractor._init_worker_waveform_extractor

        init_args = (
            self.recording,
            self.sorting,
            self,
            wfs_memmap,
            selected_spikes,
            selected_spike_times,
            self.nbefore,
            self.nafter,
            self.return_scaled,
        )
        processor = ChunkRecordingExecutor(
            self.recording,
            func,
            init_func,
            init_args,
            job_name="extract waveforms",
            handle_returns=True,
            **job_kwargs,
        )
        spike_times_centered_dicts = processor.run()

        # Copy original kilosort spike times
        shutil.copyfile(
            self.sorting.folder / "spike_times.npy",
            self.sorting.folder / "spike_times_kilosort.npy",
        )

        # Center spike times
        spike_times = self.sorting.spike_times
        spike_time_to_ind = {}
        for i, st in enumerate(spike_times):
            spike_time_to_ind[st] = i

        for st_dict in spike_times_centered_dicts:
            for st, st_cen in st_dict.items():
                spike_times[spike_time_to_ind[st]] = st_cen
        np.save(self.sorting.folder / "spike_times.npy", spike_times)

    def sample_spikes(self) -> dict:
        """
        Uniform random selection of spikes per unit and save to .npy

        self.samples_spikes just calls self.random_spikes_uniformly and saves data to .npy files

        Returns
        -------
        Dictionary of {unit_id, [selected_spike_times]}
        """

        print("Sampling spikes for each unit")
        selected_spikes = self.select_random_spikes_uniformly()

        # Store in 2 columns (spike_index, segment_index) in a .npy file
        # NOT NECESSARY BUT COULD BE USEFUL FOR DEBUGGING
        print("Saving sampled spikes in .npy format")
        for unit_id in self.sorting.unit_ids:
            n = np.sum([e.size for e in selected_spikes[unit_id]])
            sampled_index = np.zeros(
                n, dtype=[("spike_index", "int64"), ("segment_index", "int64")]
            )
            pos = 0
            for segment_index in range(self.sorting.get_num_segments()):
                inds = selected_spikes[unit_id][segment_index]
                sampled_index[pos : pos + inds.size]["spike_index"] = inds
                sampled_index[pos : pos + inds.size]["segment_index"] = segment_index
                pos += inds.size

            sampled_index_file = (
                self.root_folder / "waveforms" / f"sampled_index_{unit_id}.npy"
            )
            np.save(str(sampled_index_file), sampled_index)

        return selected_spikes

    def select_random_spikes_uniformly(self) -> dict:
        """
        Uniform random selection of spikes per unit.

        More complicated than necessary because it is designed to handle multi-segment data
        Must keep complications since ChunkRecordingExecutor expects multi-segment data

        :return:
        Dictionary of {unit_id, [selected_spike_times]}
        """
        sorting = self.sorting
        unit_ids = sorting.unit_ids
        num_seg = sorting.get_num_segments()

        selected_spikes = {}
        for unit_id in unit_ids:
            # spike per segment
            n_per_segment = [
                sorting.get_unit_spike_train(unit_id, segment_index=i).size
                for i in range(num_seg)
            ]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            total = np.sum(n_per_segment)
            if MAX_WAVEFORMS_PER_UNIT is not None:
                if total > MAX_WAVEFORMS_PER_UNIT:
                    global_inds = np.random.choice(
                        total, size=MAX_WAVEFORMS_PER_UNIT, replace=False
                    )
                    global_inds = np.sort(global_inds)
                else:
                    global_inds = np.arange(total)
            else:
                global_inds = np.arange(total)
            sel_spikes = []
            for segment_index in range(num_seg):
                in_segment = (global_inds >= cum_sum[segment_index]) & (
                    global_inds < cum_sum[segment_index + 1]
                )
                inds = global_inds[in_segment] - cum_sum[segment_index]

                if MAX_WAVEFORMS_PER_UNIT is not None:
                    # clean border when sub selection
                    if self.nafter is None:
                        raise RuntimeError(
                            "nafter is not set — waveform extraction parameters "
                            "were not initialized."
                        )
                    spike_times = sorting.get_unit_spike_train(
                        unit_id=unit_id, segment_index=segment_index
                    )
                    sampled_spike_times = spike_times[inds]
                    num_samples = self.recording.get_num_samples(
                        segment_index=segment_index
                    )
                    mask = (sampled_spike_times >= self.nbefore) & (
                        sampled_spike_times < (num_samples - self.nafter)
                    )
                    inds = inds[mask]

                sel_spikes.append(inds)
            selected_spikes[unit_id] = sel_spikes
        return selected_spikes

    @staticmethod
    def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx["recording"]
        sorting = worker_ctx["sorting"]

        waveform_extractor = worker_ctx["waveform_extractor"]
        templates_half_windows_sizes = waveform_extractor.templates_half_windows_sizes
        use_pos_peak = waveform_extractor.use_pos_peak
        chans_max_all = waveform_extractor.chans_max_all

        wfs_memmap_files = worker_ctx["wfs_memmap_files"]
        selected_spikes = worker_ctx["selected_spikes"]
        selected_spike_times = worker_ctx["selected_spike_times"]
        nbefore = worker_ctx["nbefore"]
        nafter = worker_ctx["nafter"]
        return_scaled = worker_ctx["return_scaled"]
        unit_cum_sum = worker_ctx["unit_cum_sum"]

        seg_size = recording.get_num_samples(segment_index=segment_index)

        to_extract = {}
        for unit_id in sorting.unit_ids:
            spike_times = selected_spike_times[unit_id][segment_index]
            i0 = np.searchsorted(spike_times, start_frame)
            i1 = np.searchsorted(spike_times, end_frame)
            if i0 != i1:
                # protect from spikes on border :  spike_time<0 or spike_time>seg_size
                # useful only when max_spikes_per_unit is not None
                # waveform will not be extracted and a zeros will be left in the memmap file
                template_half_window_size = templates_half_windows_sizes[unit_id]
                before_buffer = max(nbefore, template_half_window_size)
                after_buffer = max(nafter, template_half_window_size)
                while (spike_times[i0] - before_buffer) < 0 and (i0 != i1):
                    i0 = i0 + 1
                while (spike_times[i1 - 1] + after_buffer) > seg_size and (i0 != i1):
                    i1 = i1 - 1

            if i0 != i1:
                to_extract[unit_id] = i0, i1, spike_times[i0:i1]

        spike_times_centered = {}
        if len(to_extract) > 0:
            start = min(
                st[0] - nbefore - templates_half_windows_sizes[uid]
                for uid, (_, _, st) in to_extract.items()
            )  # Get the minimum time frame from recording needed for extracting waveform from the minimum spike time - nbefore
            end = max(
                st[-1] + nafter + templates_half_windows_sizes[uid]
                for uid, (_, _, st) in to_extract.items()
            )
            start = int(max(0, start))
            end = int(min(end, recording.get_num_samples()))
            # load trace in memory
            traces = recording.get_traces(
                start_frame=start,
                end_frame=end,
                segment_index=segment_index,
                return_scaled=return_scaled,
            )
            max_trace_ind = traces.shape[0] - 1
            for unit_id, (i0, i1, local_spike_times) in to_extract.items():
                wfs = np.load(wfs_memmap_files[unit_id], mmap_mode="r+")
                half_window_size = templates_half_windows_sizes[unit_id]
                chan_max = chans_max_all[unit_id]
                for i in range(local_spike_times.size):
                    st = int(local_spike_times[i])  # spike time
                    st_trace = (
                        st - start
                    )  # Convert the spike time defined by all the samples in recording to only samples in "traces"

                    peak_window_left = max(st_trace - half_window_size, 0)
                    peak_window_right = min(
                        st_trace + half_window_size + 1, max_trace_ind + 1
                    )
                    traces_peak_window = traces[
                        peak_window_left:peak_window_right, chan_max
                    ]
                    if traces_peak_window.size == 0:
                        # Spike at chunk boundary — skip recentering
                        spike_times_centered[st] = st
                        continue
                    if use_pos_peak[unit_id]:
                        peak_value = np.max(traces_peak_window)
                    else:
                        peak_value = np.min(traces_peak_window)
                    peak_indices = np.flatnonzero(traces_peak_window == peak_value)
                    st_offset = (
                        peak_indices[peak_indices.size // 2]
                        - traces_peak_window.size // 2
                    )
                    st_trace += st_offset

                    spike_times_centered[st] = st + st_offset

                    pos = (
                        unit_cum_sum[unit_id][segment_index] + i0 + i
                    )  # Index for waveform along 0th axis in .npy waveforms file
                    wf = traces[
                        st_trace - nbefore : st_trace + nafter, :
                    ]  # Python slices with [start, end), so waveform is in format (nbefore + spike_location + nafter-1, n_channels)
                    wfs[pos, :, :] = wf
        return spike_times_centered

    @staticmethod
    def _init_worker_waveform_extractor(
        recording,
        sorting,
        waveform_extractor,
        wfs_memmap,
        selected_spikes,
        selected_spike_times,
        nbefore,
        nafter,
        return_scaled,
    ):
        # create a local dict per worker
        worker_ctx = {}
        worker_ctx["recording"] = recording
        worker_ctx["sorting"] = sorting
        worker_ctx["waveform_extractor"] = waveform_extractor

        worker_ctx["wfs_memmap_files"] = wfs_memmap
        worker_ctx["selected_spikes"] = selected_spikes
        worker_ctx["selected_spike_times"] = selected_spike_times
        worker_ctx["nbefore"] = nbefore
        worker_ctx["nafter"] = nafter
        worker_ctx["return_scaled"] = return_scaled

        num_seg = sorting.get_num_segments()
        unit_cum_sum = {}
        for unit_id in sorting.unit_ids:
            # spike per segment
            n_per_segment = [selected_spikes[unit_id][i].size for i in range(num_seg)]
            cum_sum = [0] + np.cumsum(n_per_segment).tolist()
            unit_cum_sum[unit_id] = cum_sum
        worker_ctx["unit_cum_sum"] = unit_cum_sum

        return worker_ctx

    # endregion

    # region Get waveforms and templates
    def get_waveforms(
        self,
        unit_id: int,
        with_index: bool = False,
        cache: bool = False,
        memmap: bool = True,
    ) -> Any:  # SpikeInterface has cache=True by default
        """
        Return waveforms for the specified unit id.

        Parameters
        ----------
        unit_id: int or str
            Unit id to retrieve waveforms for
        with_index: bool
            If True, spike indices of extracted waveforms are returned (default False)
        cache: bool
            If True, waveforms are cached to the self.waveforms dictionary (default False)
        memmap: bool
            If True, waveforms are loaded as memmap objects.
            If False, waveforms are loaded as np.array objects (default True)

        Returns
        -------
        wfs: np.array
            The returned waveform (num_spikes, num_samples, num_channels)
            num_samples = nbefore + 1 (for value at peak) + nafter
        indices: np.array
            If 'with_index' is True, the spike indices corresponding to the waveforms extracted
        """
        wfs = self._waveforms.get(unit_id, None)
        if wfs is None:
            waveform_file = self.root_folder / "waveforms" / f"waveforms_{unit_id}.npy"
            if not waveform_file.is_file():
                raise Exception(
                    "Waveforms not extracted yet: "
                    "please set 'REEXTRACT_WAVEFORMS' to True"
                )
            if memmap:
                wfs = np.load(waveform_file, mmap_mode="r")
            else:
                wfs = np.load(waveform_file)
            if cache:
                self._waveforms[unit_id] = wfs

        if with_index:
            sampled_index = self.get_sampled_indices(unit_id)
            return wfs, sampled_index
        else:
            return wfs

    def get_sampled_indices(self, unit_id: int) -> list:
        """
        Return sampled spike indices of extracted waveforms
        (which waveforms correspond to which spikes if "max_spikes_per_unit" is not None)

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve indices for

        Returns
        -------
        sampled_indices: np.array
            The sampled indices with shape (n_waveforms,)
        """

        sampled_index_file = (
            self.root_folder / "waveforms" / f"sampled_index_{unit_id}.npy"
        )
        sampled_index = np.load(str(sampled_index_file))

        # When this function was written, the sampled_index .npy files also included segment index of spikes
        # This disregards segment index since there should only be 1 segment
        sampled_index_without_segment_index = []
        for index in sampled_index:
            sampled_index_without_segment_index.append(index[0])
        return sampled_index_without_segment_index

    def get_computed_template(self, unit_id: int, mode: str) -> np.ndarray:
        """
        Return template (average waveform).

        Parameters
        ----------
        unit_id: int
            Unit id to retrieve waveforms for
        mode: str
            'average' (default), 'median' , 'std'(standard deviation)
        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        _possible_template_modes = {"average", "std", "median"}
        if mode not in _possible_template_modes:
            raise ValueError(
                f"mode must be one of {_possible_template_modes}, got '{mode}'"
            )

        if mode in self.template_cache:
            # already in the global cache
            template = self.template_cache[mode][unit_id, :, :]
            return template

        # compute from waveforms
        wfs = self.get_waveforms(unit_id)
        if mode == "median":
            template = np.median(wfs, axis=0)
        elif mode == "average":
            template = np.average(wfs, axis=0)
        elif mode == "std":
            template = np.std(wfs, axis=0)
        return template

    def compute_templates(
        self, modes=("average", "std"), unit_ids=None, folder=None, n_jobs=1
    ):
        """
        Compute all template for different "modes":
          * average
          * std
          * median

        The results are cached in memory as 3d ndarray (nunits, nsamples, nchans)
        and also saved as npy file in the folder to avoid recomputation each time.

        Parameters
        ----------
        modes: tuple
            Template modes to compute (average, std, median)
        unit_ids: None or List
            Unit ids to compute templates for
            If None-> unit ids are taken from self.sorting.unit_ids
        folder: None or Path
            Folder to save templates to
            If None-> use self.folder
        n_jobs: int
            Number of threads for parallel template computation.
            Default 1 (sequential). Values > 1 use a thread pool
            which speeds up I/O-bound waveform loading from disk.
        """
        print_stage("COMPUTING TEMPLATES")
        print("Template modes: " + ", ".join(modes))
        stopwatch = Stopwatch()

        if unit_ids is None:
            unit_ids = self.sorting.unit_ids
        if folder is None:
            folder = self.root_folder / "templates"

        num_chans = self.recording.get_num_channels()

        for mode in modes:
            # With max(unit_ids)+1 instead of len(unit_ids), the template of unit_id can be retrieved by template[unit_id]
            # Instead of first converting unit_id to an index
            templates = np.zeros(
                (max(unit_ids) + 1, self.nsamples, num_chans), dtype=self.dtype
            )
            self.template_cache[mode] = templates

        def _compute_unit_template(unit_id):
            """Load waveforms and compute templates for a single unit."""
            wfs = self.get_waveforms(unit_id, cache=False)
            for mode in modes:
                if mode == "median":
                    arr = np.median(wfs, axis=0)
                elif mode == "average":
                    arr = np.average(wfs, axis=0)
                elif mode == "std":
                    arr = np.std(wfs, axis=0)
                else:
                    raise ValueError("mode must in median/average/std")
                self.template_cache[mode][unit_id, :, :] = arr

        n_units = len(unit_ids)
        n_workers = min(n_jobs, n_units) if n_jobs > 1 else 1
        print(f"Computing templates for {n_units} units (n_jobs={n_workers})")

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_compute_unit_template, uid): uid for uid in unit_ids
                }
                for _ in tqdm(as_completed(futures), total=n_units, desc="Templates"):
                    pass
        else:
            for unit_id in tqdm(unit_ids):
                _compute_unit_template(unit_id)

        create_folder(folder)
        print("Saving templates to .npy")
        for mode in modes:
            templates = self.template_cache[mode]
            template_file = folder / f"templates_{mode}.npy"
            np.save(str(template_file), templates)
        stopwatch.log_time("Done computing and saving templates.")

    def load_units(self) -> None:
        self.sorting.unit_ids = np.load(str(self.folder / "unit_ids.npy")).tolist()
        self.sorting.spike_times = np.load(str(self.folder / "spike_times.npy"))
        self.sorting.spike_clusters = np.load(str(self.folder / "spike_clusters.npy"))

    def get_curation_history(self) -> Optional[dict]:
        path = self.folder / "curation_history.json"
        if path.exists():
            with open(self.folder / "curation_history.json", "r") as f:
                return json.load(f)
        else:
            return None

    # endregion

    # region Format files
    def save_curated_units(
        self, unit_ids, waveforms_root_folder, curated_folder, curation_history
    ):
        """
        Filters units by storing curated unit ids in a new folder.

        Parameters
        ----------
        unit_ids: list
            Contains which unit ids are curated
        waveforms_root_folder: Path
            The root of all waveforms
        curated_folder: Path
            The new folder where curated unit ids are saved
        curation_history: dict
            Contains curation history to be saved

        Return
        ------
        we :  WaveformExtractor
            The newly create waveform extractor with the selected units
        """
        print_stage("SAVING CURATED UNITS")
        stopwatch = Stopwatch()
        print(f"Saving {len(unit_ids)} curated units to new folder")
        create_folder(curated_folder)

        # Save data about unit ids
        spike_times_og = self.sorting.spike_times
        spike_clusters_og = self.sorting.spike_clusters
        unit_ids_set = set(unit_ids)
        selected_indices = [
            i for i, c in enumerate(spike_clusters_og) if c in unit_ids_set
        ]
        spike_times = spike_times_og[selected_indices]
        spike_clusters = spike_clusters_og[selected_indices]

        np.save(str(curated_folder / "unit_ids.npy"), unit_ids)
        np.save(str(curated_folder / "spike_times.npy"), spike_times)
        np.save(str(curated_folder / "spike_clusters.npy"), spike_clusters)

        # Save curation history
        with open(curated_folder / "curation_history.json", "w") as f:
            json.dump(curation_history, f)

        we = WaveformExtractor.load_from_folder(
            self.recording,
            self.sorting,
            waveforms_root_folder,
            curated_folder,
            self.use_pos_peak,
            self.chans_max_kilosort,
            self.chans_max_all,
        )
        stopwatch.log_time("Done saving curated units.")

        return we

    # endregion


class ChunkRecordingExecutor:
    """
    Used to extract waveforms from recording

    Core class for parallel processing to run a "function" over chunks on a recording.

    It supports running a function:
        * in loop with chunk processing (low RAM usage)
        * at once if chunk_size is None (high RAM usage)
        * in parallel with ProcessPoolExecutor (higher speed)

    The initializer ('init_func') allows to set a global context to avoid heavy serialization
    (for examples, see implementation in `core.WaveformExtractor`).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording to be processed
    func: function
        Function that runs on each chunk
    init_func: function
        Initializer function to set the global context (accessible by 'func')
    init_args: tuple
        Arguments for init_func
    verbose: bool
        If True, output is verbose
    progress_bar: bool
        If True, a progress bar is printed to monitor the progress of the process
    handle_returns: bool
        If True, the function can return values
    n_jobs: int
        Number of jobs to be used (default 1). Use -1 to use as many jobs as number of cores
    total_memory: str
        Total memory (RAM) to use (e.g. "1G", "500M")
    chunk_memory: str
        Memory per chunk (RAM) to use (e.g. "1G", "500M")
    chunk_size: int or None
        Size of each chunk in number of samples. If 'TOTAL_MEMORY' or 'CHUNK_MEMORY' are used, it is ignored.
    job_name: str
        Job name

    Returns
    -------
    res: list
        If 'handle_returns' is True, the results for each chunk process
    """

    def __init__(
        self,
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=False,
        handle_returns=False,
        n_jobs=1,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        job_name="",
    ):
        self.recording = recording
        self.func = func
        self.init_func = init_func
        self.init_args = init_args

        self.verbose = verbose
        self.progress_bar = progress_bar

        self.handle_returns = handle_returns

        self.n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
        self.chunk_size = Utils.ensure_chunk_size(
            recording,
            total_memory=total_memory,
            chunk_size=chunk_size,
            chunk_memory=chunk_memory,
            n_jobs=self.n_jobs,
        )
        self.job_name = job_name

        if verbose:
            print(
                self.job_name,
                "with",
                "n_jobs",
                self.n_jobs,
                " chunk_size",
                self.chunk_size,
            )

    def run(self):
        """
        Runs the defined jobs.
        """
        all_chunks = ChunkRecordingExecutor.divide_recording_into_chunks(
            self.recording, self.chunk_size
        )

        if self.handle_returns:
            returns = []
        else:
            returns = None

        import sys

        if self.n_jobs != 1 and not (sys.version_info >= (3, 8)):
            self.n_jobs = 1

        if self.n_jobs == 1:
            if self.progress_bar:
                all_chunks = tqdm(all_chunks, ascii=True, desc=self.job_name)

            worker_ctx = self.init_func(*self.init_args)
            for segment_index, frame_start, frame_stop in all_chunks:
                res = self.func(segment_index, frame_start, frame_stop, worker_ctx)
                if self.handle_returns:
                    returns.append(res)
        else:
            n_jobs = min(self.n_jobs, len(all_chunks))

            ######## Do you want to limit the number of threads per process?
            ######## It has to be done to speed up numpy a lot if multicores
            ######## Otherwise, np.dot will be slow. How to do that, up to you
            ######## This is just a suggestion, but here it adds a dependency

            # parallel
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=ChunkRecordingExecutor.worker_initializer,
                initargs=(self.func, self.init_func, self.init_args),
            ) as executor:
                results = executor.map(
                    ChunkRecordingExecutor.function_wrapper, all_chunks
                )

                if self.progress_bar:
                    results = tqdm(results, desc=self.job_name, total=len(all_chunks))

                if self.handle_returns:  # Should be false
                    for res in results:
                        returns.append(res)
                else:
                    for res in results:
                        pass

        return returns

    @staticmethod
    def function_wrapper(args):
        segment_index, start_frame, end_frame = args
        global _func
        global _worker_ctx
        return _func(segment_index, start_frame, end_frame, _worker_ctx)

    @staticmethod
    def divide_recording_into_chunks(recording, chunk_size):
        all_chunks = []
        for segment_index in range(recording.get_num_segments()):
            num_frames = recording.get_num_samples(segment_index)
            chunks = ChunkRecordingExecutor.divide_segment_into_chunks(
                num_frames, chunk_size
            )
            all_chunks.extend(
                [
                    (segment_index, frame_start, frame_stop)
                    for frame_start, frame_stop in chunks
                ]
            )
        return all_chunks

    @staticmethod
    def divide_segment_into_chunks(num_frames, chunk_size):
        if chunk_size is None:
            chunks = [(0, num_frames)]
        else:
            n = num_frames // chunk_size

            frame_starts = np.arange(n) * chunk_size
            frame_stops = frame_starts + chunk_size

            frame_starts = frame_starts.tolist()
            frame_stops = frame_stops.tolist()

            if (num_frames % chunk_size) > 0:
                frame_starts.append(n * chunk_size)
                frame_stops.append(num_frames)

            chunks = list(zip(frame_starts, frame_stops))

        return chunks

    @staticmethod
    def worker_initializer(func, init_func, init_args):
        global _worker_ctx
        _worker_ctx = init_func(*init_args)
        global _func
        _func = func


global _worker_ctx
global _func
# ProcessPoolExecutor: using stdlib concurrent.futures instead of vendored copy
from concurrent.futures import ProcessPoolExecutor  # noqa: F401

# endregion


# NOTE: The Curation class, curate_first(), curate_second(), and
# curate() orchestrator have been removed.  All curation logic now
# lives in spikelab.spikedata.curation and is invoked via
# sd.curate() / _curate_spikedata() in process_recording().
# region Utilities
class Utils:
    """Utility helpers adapted from SpikeInterface.

    Provides static methods for parsing Kilosort2 Python parameter
    files, clamping worker counts to OS limits, and computing chunk
    sizes for parallel waveform extraction.
    """

    @staticmethod
    def read_python(path):
        """Parses python scripts in a dictionary

        Parameters
        ----------
        path: str or Path
            Path to file to parse

        Returns
        -------
        metadata:
            dictionary containing parsed file

        """
        from six import exec_
        import re

        path = Path(path).absolute()
        if not path.is_file():
            raise FileNotFoundError(f"Kilosort2 parameter file not found: {path}")
        with path.open("r") as f:
            contents = f.read()
        contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
        metadata = {}
        exec_(contents, {}, metadata)
        metadata = {k.lower(): v for (k, v) in metadata.items()}
        return metadata

    @staticmethod
    def ensure_n_jobs(recording, n_jobs=1):
        # Ensures that the number of jobs specified is possible by the operating system

        import joblib

        if n_jobs == -1:
            n_jobs = joblib.cpu_count()
        elif n_jobs == 0:
            n_jobs = 1
        elif n_jobs is None:
            n_jobs = 1

        version = sys.version_info

        if (n_jobs != 1) and not (version.major >= 3 and version.minor >= 7):
            print(f"Python {sys.version} does not support parallel processing")
            n_jobs = 1

        return n_jobs

    @staticmethod
    def ensure_chunk_size(
        recording,
        total_memory=None,
        chunk_size=None,
        chunk_memory=None,
        n_jobs=1,
        **other_kwargs,
    ):
        """
        'chunk_size' is the traces.shape[0] for each worker.

        Flexible chunk_size setter with 3 ways:
            * "chunk_size": is the length in sample for each chunk independently of channel count and dtype.
            * "chunk_memory": total memory per chunk per worker
            * "total_memory": total memory over all workers.

        If chunk_size/chunk_memory/total_memory are all None then there is no chunk computing
        and the full trace is retrieved at once.

        Parameters
        ----------
        chunk_size: int or None
            size for one chunk per job
        chunk_memory: str or None
            must endswith 'k', 'M' or 'G'
        total_memory: str or None
            must endswith 'k', 'M' or 'G'
        """

        if chunk_size is not None:
            # manual setting
            chunk_size = int(chunk_size)
        elif chunk_memory is not None:
            if total_memory is not None:
                raise ValueError(
                    "Cannot specify both 'chunk_memory' and 'total_memory'. "
                    "Provide only one."
                )
            # set by memory per worker size
            chunk_memory = Utils._mem_to_int(chunk_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(chunk_memory / (num_channels * n_bytes))
        if total_memory is not None:
            # clip by total memory size
            n_jobs = Utils.ensure_n_jobs(recording, n_jobs=n_jobs)
            total_memory = Utils._mem_to_int(total_memory)
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            num_channels = recording.get_num_channels()
            chunk_size = int(total_memory / (num_channels * n_bytes * n_jobs))
        else:
            if n_jobs == 1:
                # not chunk computing
                chunk_size = None
            else:
                raise ValueError(
                    "For N_JOBS >1 you must specify TOTAL_MEMORY or chunk_size or CHUNK_MEMORY"
                )

        return chunk_size

    @staticmethod
    def _mem_to_int(mem):
        # Converts specified memory (e.g. 4G) to integer number
        _exponents = {"k": 1e3, "M": 1e6, "G": 1e9}

        suffix = mem[-1]
        if suffix not in _exponents:
            raise ValueError(
                f"Invalid memory suffix '{suffix}' in '{mem}'. "
                f"Expected one of: {list(_exponents.keys())} (e.g. '4G', '500M')"
            )
        mem = int(float(mem[:-1]) * _exponents[suffix])
        return mem


# endregion


# region spikesort_matlab.py
# Stopwatch is imported from sorting_utils at the top of this file.


class Compiler:
    """Aggregates sorting results from one or more SpikeData objects for export.

    Reads unit metadata from ``neuron_attributes`` and writes combined
    ``.npz``, ``.mat``, and figure outputs.
    """

    def __init__(self):
        self.create_figures = CREATE_FIGURES
        self.create_std_scatter_plot = (
            CURATE_SECOND and SPIKES_MIN_SECOND is not None and STD_NORM_MAX is not None
        )
        self.compile_to_mat = COMPILE_TO_MAT
        self.compile_to_npz = COMPILE_TO_NPZ
        self.save_electrodes = SAVE_ELECTRODES

        self.recs_cache = []

    def add_recording(
        self, rec_name: str, sd: Any, curation_history: Optional[dict] = None
    ) -> None:
        """Queue a recording for compilation.

        Parameters:
            rec_name (str): Short name for the recording.
            sd (SpikeData): Curated SpikeData with enriched
                ``neuron_attributes``.
            curation_history (dict or None): Curation history dict
                from ``build_curation_history``.
        """
        self.recs_cache.append((rec_name, sd, curation_history))

    def save_results(self, folder: Union[str, Path]) -> None:
        """Compile and save results from all queued recordings.

        Parameters:
            folder (Path or str): Output directory.
        """
        create_folder(folder)
        folder = Path(folder)

        # ------------------------------------------------------------------
        # Collect all units from all recordings
        # ------------------------------------------------------------------
        all_units = []  # list of (attrs_dict, is_curated, rec_name)
        rec_metadata = {}  # rec_name -> {fs, locations, n_samples}

        # Figure data
        bar_rec_names = []
        bar_n_total = []
        bar_n_selected = []
        scatter_n_spikes = {}
        scatter_std_norms = {}
        fig_fs_Hz = None

        for rec_name, sd, curation_history in self.recs_cache:
            print(f"Adding recording: {rec_name}")

            fs_Hz = sd.metadata.get("fs_Hz", 30000.0)
            rec_metadata[rec_name] = {
                "fs": fs_Hz,
                "locations": sd.metadata.get("channel_locations"),
                "n_samples": sd.metadata.get("n_samples", 0),
            }
            if fig_fs_Hz is None:
                fig_fs_Hz = fs_Hz

            # All units are curated (sd is already curated)
            curated_ids = set()
            if sd.neuron_attributes is not None:
                for attrs in sd.neuron_attributes:
                    curated_ids.add(int(attrs.get("unit_id", -1)))

            for i in range(sd.N):
                attrs = sd.neuron_attributes[i] if sd.neuron_attributes else {}
                all_units.append((attrs, True, rec_name))

            # Figure data
            if self.create_figures:
                n_total = len(curated_ids)
                if curation_history is not None:
                    n_total = len(curation_history.get("initial", curated_ids))
                bar_rec_names.append(rec_name)
                bar_n_total.append(n_total)
                bar_n_selected.append(sd.N)

                if self.create_std_scatter_plot and curation_history is not None:
                    scatter_n_spikes[rec_name] = curation_history.get(
                        "metrics", {}
                    ).get("spike_count", {})
                    scatter_std_norms[rec_name] = curation_history.get(
                        "metrics", {}
                    ).get("std_norm", {})

        # ------------------------------------------------------------------
        # Sort units by amplitude within polarity groups
        # ------------------------------------------------------------------
        neg_units = [
            (a, c, r) for a, c, r in all_units if not a.get("has_pos_peak", False)
        ]
        pos_units = [(a, c, r) for a, c, r in all_units if a.get("has_pos_peak", False)]

        # Sort by amplitude descending
        neg_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)
        pos_units.sort(key=lambda x: float(x[0].get("amplitude", 0)), reverse=True)

        # ------------------------------------------------------------------
        # Build compile_dict and save waveforms/figures
        # ------------------------------------------------------------------
        compile_dict = None
        if self.compile_to_mat or self.compile_to_npz:
            if len(rec_metadata) == 1:
                rec = list(rec_metadata.keys())[0]
                meta = rec_metadata[rec]
                compile_dict = {
                    "units": [],
                    "locations": meta["locations"],
                    "fs": meta["fs"],
                }

        if COMPILE_WAVEFORMS:
            create_folder(folder / "negative_peaks")
            create_folder(folder / "positive_peaks")

        fig_templates = []
        fig_peak_indices = []
        fig_is_curated = []
        fig_has_pos_peak = []

        sorted_index = 0
        for group_label, units_group in [
            ("negative", neg_units),
            ("positive", pos_units),
        ]:
            has_pos = group_label == "positive"
            print(
                f"\nIterating through {len(units_group)} units with "
                f"{group_label} peaks"
            )
            for attrs, is_curated, rec_name in tqdm(units_group):
                if is_curated:
                    if compile_dict is not None:
                        spike_train_samples = attrs.get("spike_train_samples")
                        if SAVE_DL_DATA:
                            unit_dict = {
                                "unit_id": attrs.get("unit_id"),
                                "spike_train": spike_train_samples,
                                "x_max": attrs.get("x"),
                                "y_max": attrs.get("y"),
                                "template": attrs.get("template_windowed"),
                                "sorted_index": sorted_index,
                                "max_channel_si": attrs.get("channel"),
                                "max_channel_id": attrs.get("channel_id"),
                                "peak_sign": group_label,
                                "peak_ind": attrs.get("peak_inds"),
                                "amplitudes": attrs.get("amplitudes"),
                                "std_norms": attrs.get("std_norms_all"),
                            }
                        else:
                            unit_dict = {
                                "unit_id": attrs.get("unit_id"),
                                "spike_train": spike_train_samples,
                                "x_max": attrs.get("x"),
                                "y_max": attrs.get("y"),
                                "template": attrs.get("template_windowed"),
                            }
                        if self.save_electrodes:
                            unit_dict["electrode"] = attrs.get("electrode")
                        compile_dict["units"].append(unit_dict)

                    if COMPILE_WAVEFORMS:
                        wf_path = attrs.get("_waveforms_path")
                        wf_window = attrs.get("_waveforms_window")
                        if wf_path is not None:
                            waveforms = np.load(wf_path, mmap_mode="r")
                            if wf_window is not None:
                                waveforms = waveforms[:, wf_window[0] : wf_window[1], :]
                            wf_folder = (
                                folder / "positive_peaks"
                                if has_pos
                                else folder / "negative_peaks"
                            )
                            np.save(
                                wf_folder / f"waveforms_{sorted_index}.npy",
                                np.array(waveforms),
                            )

                    sorted_index += 1

                if self.create_figures:
                    fig_templates.append(attrs.get("template", np.array([])))
                    fig_peak_indices.append(attrs.get("template_peak_ind", 0))
                    fig_is_curated.append(is_curated)
                    fig_has_pos_peak.append(has_pos)

        if compile_dict is not None:
            if self.compile_to_mat:
                savemat(folder / "sorted.mat", compile_dict)
                print("Compiled results to .mat")
            if self.compile_to_npz:
                np.savez(folder / "sorted.npz", **compile_dict)
                print("Compiled results to .npz")

        if self.create_figures:
            from .figures import plot_curation_bar, plot_std_scatter, plot_templates

            figures_path = folder / "figures"
            print("\nSaving figures")
            create_folder(figures_path)

            plot_curation_bar(
                bar_rec_names,
                bar_n_total,
                bar_n_selected,
                total_label=BAR_TOTAL_LABEL,
                selected_label=BAR_SELECTED_LABEL,
                x_label=BAR_X_LABEL,
                y_label=BAR_Y_LABEL,
                label_rotation=BAR_LABEL_ROTATION,
                save_path=str(figures_path / "curation_bar_plot.png"),
            )
            print("Curation bar plot has been saved")

            if self.create_std_scatter_plot and scatter_n_spikes:
                plot_std_scatter(
                    scatter_n_spikes,
                    scatter_std_norms,
                    spikes_thresh=SPIKES_MIN_SECOND,
                    std_thresh=STD_NORM_MAX,
                    colors=SCATTER_RECORDING_COLORS[:],
                    alpha=SCATTER_RECORDING_ALPHA,
                    x_label=SCATTER_X_LABEL,
                    y_label=SCATTER_Y_LABEL,
                    x_max_buffer=SCATTER_X_MAX_BUFFER,
                    y_max_buffer=SCATTER_Y_MAX_BUFFER,
                    save_path=str(figures_path / "std_scatter_plot.png"),
                )
                print("Std scatter plot has been saved")

            if fig_templates and fig_fs_Hz is not None:
                plot_templates(
                    fig_templates,
                    fig_peak_indices,
                    fig_fs_Hz,
                    fig_is_curated,
                    fig_has_pos_peak,
                    templates_per_column=ALL_TEMPLATES_PER_COLUMN,
                    y_spacing=ALL_TEMPLATES_Y_SPACING,
                    y_lim_buffer=ALL_TEMPLATES_Y_LIM_BUFFER,
                    color_curated=ALL_TEMPLATES_COLOR_CURATED,
                    color_failed=ALL_TEMPLATES_COLOR_FAILED,
                    window_ms_before=ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK,
                    window_ms_after=ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK,
                    line_ms_before=ALL_TEMPLATES_LINE_MS_BEFORE_PEAK,
                    line_ms_after=ALL_TEMPLATES_LINE_MS_AFTER_PEAK,
                    x_label=ALL_TEMPLATES_X_LABEL,
                    save_path=str(figures_path / "all_templates_plot.png"),
                )
                print("All templates plot has been saved")


# create_folder and delete_folder are imported from sorting_utils.


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
    if FIRST_N_MINS is not None:
        end_frame = FIRST_N_MINS * 60 * rec.get_sampling_frequency()
        if end_frame > rec.get_num_samples():
            print(
                f"'first_n_mins' is set to {FIRST_N_MINS}, but recording is only {rec.get_total_duration() / 60:.2f} min long"
            )
            print(
                f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
            )
        else:
            print(f"Only analyzing the first {FIRST_N_MINS} min of recording")
            rec = rec.frame_slice(start_frame=0, end_frame=end_frame)
    else:
        print(
            f"Using entire duration of recording: {rec.get_total_duration() / 60:.2f}min"
        )

    if len(REC_CHUNKS) > 0:
        print(f"Using {len(REC_CHUNKS)} chunks of the recording")
        rec_chunks = []
        for c, (start_frame, end_frame) in enumerate(REC_CHUNKS):
            print(f"Chunk {c}: {start_frame} to {end_frame} frame")
            chunk = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
            rec_chunks.append(chunk)
        rec = si_segmentutils.concatenate_recordings(rec_chunks)
    else:
        print(f"Using entire recording")

    if MEA_Y_MAX is not None:
        print(f"Flipping y-coordinates of channel locations. MEA height: {MEA_Y_MAX}")
        probes_all = []
        for probe in rec.get_probes():
            y_cords = probe._contact_positions[:, 1]

            if MEA_Y_MAX is None:
                y_cords_flipped = y_cords
            elif MEA_Y_MAX == -1:
                y_cords_flipped = max(y_cords) - y_cords
            else:
                y_cords_flipped = MEA_Y_MAX - y_cords

            probe._contact_positions[np.arange(y_cords_flipped.size), 1] = (
                y_cords_flipped
            )
            probes_all.append(probe)
        rec = rec.set_probes(probes_all)

    stopwatch.log_time("Done loading recording.")

    return rec


def _get_noise_levels(
    recording: Any,
    return_scaled: bool = True,
    num_chunks: int = 20,
    chunk_size: int = 10000,
    seed: int = 0,
) -> np.ndarray:
    """Estimate per-channel noise using MAD on random recording chunks.

    Parameters:
        recording: SpikeInterface BaseRecording.
        return_scaled (bool): Use scaled traces.
        num_chunks (int): Number of random chunks to sample.
        chunk_size (int): Samples per chunk.
        seed (int): Random seed.

    Returns:
        noise_levels (np.ndarray): Per-channel noise, shape ``(channels,)``.
    """
    length = recording.get_num_samples()
    rng = np.random.RandomState(seed=seed)
    starts = rng.randint(0, length - chunk_size, size=num_chunks)
    chunks = []
    for s in starts:
        chunks.append(
            recording.get_traces(
                start_frame=s,
                end_frame=s + chunk_size,
                return_scaled=return_scaled,
            )
        )
    data = np.concatenate(chunks, axis=0)
    med = np.median(data, axis=0, keepdims=True)
    return np.median(np.abs(data - med), axis=0) / 0.6745


def _waveform_extractor_to_spikedata(
    w_e: Any, rec_path: Any, rec_chunks: Optional[list] = None
) -> Any:
    """Convert a WaveformExtractor to a SpikeData with rich neuron attributes.

    Extracts spike trains, full waveform templates, channel locations,
    SNR, normalized STD, polarity, and all per-unit metadata needed by
    the Compiler.  The resulting SpikeData does **not** carry
    ``raw_data`` (to avoid duplicating large voltage traces).

    When *rec_chunks* is provided (list of ``(start_frame, end_frame)``
    tuples from concatenated recordings), per-epoch average waveform
    templates are computed and stored as ``epoch_templates``.

    Parameters
    ----------
    w_e : WaveformExtractor
        Waveform extractor (curated or uncurated).
    rec_path : str or Path
        Original recording file path, stored as source metadata.
    rec_chunks : list of (int, int) or None
        Frame boundaries for each concatenated recording epoch.
        When None or empty, ``epoch_templates`` is not stored.

    Returns
    -------
    SpikeData
        Spike trains in milliseconds with per-unit attributes:
        ``unit_id``, ``channel``, ``channel_id``, ``x``, ``y``,
        ``electrode``, ``template``, ``template_full``,
        ``template_peak_ind``, ``amplitude``, ``amplitudes``,
        ``peak_inds``, ``std_norms_all``, ``has_pos_peak``,
        ``snr``, ``std_norm``, and optionally ``epoch_templates``.
    """
    from spikelab.spikedata import SpikeData

    sorting = w_e.sorting
    fs_Hz = float(w_e.sampling_frequency)
    rec_locations = w_e.recording.get_channel_locations()
    channel_ids = w_e.recording.get_channel_ids()

    # Electrode IDs (optional)
    try:
        electrode_ids = w_e.recording.get_property("electrode")
    except Exception:
        electrode_ids = None
    if electrode_ids is None:
        electrode_ids = channel_ids

    # Noise levels for SNR
    noise_levels = _get_noise_levels(w_e.recording, getattr(w_e, "return_scaled", True))

    # Polarity flags
    use_pos_peak = w_e.use_pos_peak

    # Template windowing for compile_dict
    nbefore_compiled = w_e.ms_to_samples(COMPILED_WAVEFORMS_MS_BEFORE)
    nafter_compiled = w_e.ms_to_samples(COMPILED_WAVEFORMS_MS_AFTER) + 1

    has_epochs = rec_chunks is not None and len(rec_chunks) > 1

    trains = []
    neuron_attributes = []
    for uid in sorting.unit_ids:
        spike_samples = sorting.get_unit_spike_train(uid)
        spike_times_ms = np.sort(spike_samples.astype(float) / fs_Hz * 1000.0)
        trains.append(spike_times_ms)

        # Channel with largest amplitude
        chan_max = int(w_e.chans_max_all[uid])
        x, y = rec_locations[chan_max]

        # Full template (all channels)
        template_mean = w_e.get_computed_template(unit_id=uid, mode="average")
        template_std = w_e.get_computed_template(unit_id=uid, mode="std")
        peak_ind_full = w_e.peak_ind

        # When SCALE_COMPILED_WAVEFORMS is False, convert µV templates
        # back to raw ADC counts.  Waveforms are now extracted as µV by
        # default (return_scaled=True), so this inverts the scaling.
        if not SCALE_COMPILED_WAVEFORMS and w_e.return_scaled:
            gain = w_e.recording.get_channel_gains()
            offset = w_e.recording.get_channel_offsets()
            template_mean = ((template_mean - offset) / gain).astype(
                w_e.recording.get_dtype()
            )
            template_std = ((template_std - offset) / gain).astype(
                w_e.recording.get_dtype()
            )

        # Windowed template (for compile_dict)
        template_windowed = template_mean[
            peak_ind_full - nbefore_compiled : peak_ind_full + nafter_compiled, :
        ]

        # Per-channel amplitudes and peak indices (from windowed template)
        template_abs = np.abs(template_windowed)
        peak_inds = np.argmax(template_abs, axis=0)
        amplitudes = template_abs[peak_inds, range(peak_inds.size)]
        amplitude_max = float(amplitudes[chan_max])

        # SNR on max channel
        noise = float(noise_levels[chan_max]) if chan_max < len(noise_levels) else 1.0
        snr = float(amplitude_max / noise) if noise > 0 else 0.0

        # Normalized STD per channel
        peak_ind_buffer = peak_ind_full - nbefore_compiled
        if STD_AT_PEAK:
            stds = template_std[peak_ind_buffer + peak_inds, range(peak_inds.size)]
        else:
            nb = w_e.ms_to_samples(STD_OVER_WINDOW_MS_BEFORE)
            na = w_e.ms_to_samples(STD_OVER_WINDOW_MS_AFTER) + 1
            stds = np.mean(
                template_std[
                    peak_ind_buffer + peak_inds - nb : peak_ind_buffer + peak_inds + na,
                    range(peak_inds.size),
                ],
                axis=0,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            std_norms_all = np.where(amplitudes > 0, stds / amplitudes, np.inf)
        std_norm = float(std_norms_all[chan_max])

        # Spike train in samples (for compilation)
        spike_train_samples = spike_samples.copy()

        attrs = {
            "unit_id": int(uid),
            "channel": chan_max,
            "channel_id": channel_ids[chan_max],
            "x": float(x),
            "y": float(y),
            "electrode": electrode_ids[chan_max],
            "template": template_mean[:, chan_max].copy(),
            "template_full": template_mean.copy(),
            "template_windowed": template_windowed.copy(),
            "template_peak_ind": int(peak_ind_full),
            "amplitude": amplitude_max,
            "amplitudes": amplitudes.copy(),
            "peak_inds": peak_inds.copy(),
            "std_norms_all": std_norms_all.copy(),
            "has_pos_peak": bool(use_pos_peak[uid]),
            "snr": snr,
            "std_norm": std_norm,
            "spike_train_samples": spike_train_samples,
        }

        # Per-epoch templates
        if has_epochs:
            wfs, sampled_indices = w_e.get_waveforms(uid, with_index=True)
            all_spike_samples = sorting.get_unit_spike_train(uid)
            epoch_templates = []
            for start_frame, end_frame in rec_chunks:
                epoch_mask = np.array(
                    [
                        start_frame <= all_spike_samples[idx] < end_frame
                        for idx in sampled_indices
                    ]
                )
                if np.any(epoch_mask):
                    epoch_wfs = wfs[epoch_mask]
                    epoch_avg = np.mean(epoch_wfs, axis=0)
                    epoch_templates.append(epoch_avg[:, chan_max].copy())
                else:
                    epoch_templates.append(np.zeros_like(template_mean[:, chan_max]))
            attrs["epoch_templates"] = epoch_templates

        # Waveforms path (for COMPILE_WAVEFORMS — loaded on demand)
        wf_file = w_e.root_folder / "waveforms" / f"waveforms_{uid}.npy"
        if wf_file.exists():
            attrs["_waveforms_path"] = str(wf_file)
            attrs["_waveforms_window"] = (
                int(peak_ind_full - nbefore_compiled),
                int(peak_ind_full + nafter_compiled),
            )

        neuron_attributes.append(attrs)

    metadata = {
        "source_file": str(rec_path),
        "source_format": "Kilosort2",
        "fs_Hz": fs_Hz,
        "channel_locations": rec_locations.copy(),
        "n_samples": int(w_e.recording.get_num_samples()),
    }
    if has_epochs:
        metadata["rec_chunks_frames"] = list(rec_chunks)
        metadata["rec_chunks_ms"] = [
            (s / fs_Hz * 1000.0, e / fs_Hz * 1000.0) for s, e in rec_chunks
        ]
        metadata["rec_chunk_names"] = (
            list(_REC_CHUNK_NAMES) if _REC_CHUNK_NAMES else None
        )

    return SpikeData(trains, metadata=metadata, neuron_attributes=neuron_attributes)


def _curate_spikedata(
    sd: Any,
    curation_folder: Union[str, Path],
    recurate: bool = False,
    **curate_kwargs: Any,
) -> Tuple[Any, dict]:
    """Curate a SpikeData with disk caching for the sorting pipeline.

    If cached results exist and *recurate* is False, loads the cached
    unit IDs and returns a subset of *sd*.  Otherwise runs
    ``sd.curate()``, saves the results (``unit_ids.npy`` and
    ``curation_history.json``) to *curation_folder*, and returns the
    curated SpikeData.

    Parameters
    ----------
    sd : SpikeData
        Uncurated (or partially curated) SpikeData.
    curation_folder : str or Path
        Directory for cached curation artefacts.
    recurate : bool
        If True, re-run curation even when cached results exist.
    **curate_kwargs
        Keyword arguments forwarded to ``sd.curate()`` (e.g.
        ``min_spikes``, ``min_rate_hz``, ``min_snr``, etc.).

    Returns
    -------
    sd_curated : SpikeData
        SpikeData containing only units that passed all criteria.
    history : dict
        Serializable curation history dict.
    """
    import json
    from spikelab.spikedata import SpikeData
    from spikelab.spikedata.curation import build_curation_history

    curation_folder = Path(curation_folder)
    unit_ids_path = curation_folder / "unit_ids.npy"
    history_path = curation_folder / "curation_history.json"

    # Check cache
    if not recurate and unit_ids_path.exists() and history_path.exists():
        cached_ids = set(int(x) for x in np.load(str(unit_ids_path)))
        passing = [
            i
            for i in range(sd.N)
            if sd.neuron_attributes is not None
            and int(sd.neuron_attributes[i].get("unit_id", i)) in cached_ids
        ]
        sd_curated = sd.subset(passing)
        with open(history_path, "r") as f:
            history = json.load(f)
        return sd_curated, history

    # Run curation
    sd_curated, results = sd.curate(**curate_kwargs)
    history = build_curation_history(sd, sd_curated, results, parameters=curate_kwargs)

    # Save to disk
    curation_folder.mkdir(parents=True, exist_ok=True)
    np.save(str(unit_ids_path), np.array(history["curated_final"]))
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    return sd_curated, history


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
        if STREAM_ID is not None:
            maxwell_kwargs["stream_id"] = STREAM_ID
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

    if GAIN_TO_UV is not None:
        gain = GAIN_TO_UV
    elif rec.get_channel_gains() is not None:
        gain = rec.get_channel_gains()
    else:
        print("Recording does not have channel gains to uV")
        gain = 1.0

    if OFFSET_TO_UV is not None:
        offset = OFFSET_TO_UV
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

    rec = bandpass_filter(rec, freq_min=FREQ_MIN, freq_max=FREQ_MAX)

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
        global REC_CHUNKS
        if len(REC_CHUNKS) == 0:
            REC_CHUNKS = new_rec_chunks

    print(f"Done concatenating {len(recordings)} recordings")
    print(f"Total duration: {rec.get_total_duration()}s")

    # Store file names globally so _waveform_extractor_to_spikedata can
    # include them in metadata for downstream epoch splitting.
    global _REC_CHUNK_NAMES
    _REC_CHUNK_NAMES = recording_names

    return rec


def get_paths(rec_path: Any, inter_path: Any, results_path: Any) -> tuple:
    """Resolve and prepare all directory paths for one recording run.

    Derives paths for the binary ``.dat`` file, Kilosort2 output,
    waveforms, curation stages, and final results. Deletes stale
    intermediate folders when ``RECOMPUTE_*`` flags are set, and creates
    the intermediate directory.

    Parameters:
        rec_path (str or Path): Path to the recording file.
        inter_path (str or Path): Root intermediate directory.
        results_path (str or Path): Root results directory.

    Returns:
        tuple: ``(rec_path, inter_path, recording_dat_path,
            output_folder, waveforms_root_folder,
            curation_initial_folder, curation_first_folder,
            curation_second_folder, results_path)`` as ``Path`` objects.
    """
    print_stage("PROCESSING RECORDING")
    print(f"Recording path: {rec_path}")
    print(f"Intermediate results path: {inter_path}")
    print(f"Compiled results path: {results_path}")

    rec_path = Path(rec_path)
    rec_name = rec_path.name.split(".")[0]

    inter_path = Path(inter_path)

    recording_dat_path = inter_path / (rec_name + "_scaled_filtered.dat")
    output_folder = inter_path / "kilosort2_results"
    waveforms_root_folder = inter_path / "waveforms"
    curation_folder = inter_path / "curation"
    curation_initial_folder = curation_folder / "initial"
    curation_first_folder = curation_folder / "first"
    curation_second_folder = curation_folder / "second"
    results_path = Path(results_path)

    if results_path == inter_path:
        results_path /= "results"

    delete_folders = []
    if RECOMPUTE_RECORDING:
        delete_folders.extend(
            (recording_dat_path, output_folder, waveforms_root_folder, curation_folder)
        )
    if RECOMPUTE_SORTING:
        delete_folders.extend((output_folder, waveforms_root_folder))
    if REEXTRACT_WAVEFORMS:
        delete_folders.append(waveforms_root_folder)
        delete_folders.append(curation_folder)
    if RECURATE_FIRST:
        delete_folders.append(curation_first_folder)
        delete_folders.append(curation_second_folder)
    if RECURATE_SECOND:
        delete_folders.append(curation_second_folder)
    for folder in delete_folders:
        delete_folder(folder)

    if len(delete_folders) > 0:
        global RECOMPILE_ALL_RECORDINGS
        RECOMPILE_ALL_RECORDINGS = True

    create_folder(inter_path)
    return (
        rec_path,
        inter_path,
        recording_dat_path,
        output_folder,
        waveforms_root_folder,
        curation_initial_folder,
        curation_first_folder,
        curation_second_folder,
        results_path,
    )


def write_recording(
    recording_filtered: BaseRecording, recording_dat_path: Path, verbose: bool = True
) -> None:
    """Convert a filtered recording to the binary ``.dat`` format for Kilosort2.

    Writes an ``int16`` binary file using SpikeInterface's
    ``BinaryRecordingExtractor``. Skips writing if the file already exists.

    Parameters:
        recording_filtered (BaseRecording): Scaled and bandpass-filtered
            SpikeInterface recording.
        recording_dat_path (Path): Destination ``.dat`` file path.
        verbose (bool): Print progress messages and show progress bar.
    """
    stopwatch = Stopwatch(start_msg="CONVERTING RECORDING", use_print_stage=True)
    if USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION:
        job_kwargs = {
            "progress_bar": verbose,
            "verbose": verbose,
            "n_jobs": N_JOBS,
            "total_memory": TOTAL_MEMORY,
        }
    else:
        job_kwargs = {
            "progress_bar": verbose,
            "verbose": False,
            "n_jobs": 1,
            "total_memory": "100G",
        }
        print("Converting entire recording at once with 1 job")

    print(f"Kilosort2's .dat path: {recording_dat_path}")
    if not recording_dat_path.exists():
        # dtype has to be 'int16' (that's what Kilosort2 expects--but can change in config)
        print("Converting raw Maxwell recording to .dat format for Kilosort2")
        BinaryRecordingExtractor.write_recording(
            recording_filtered,
            file_paths=recording_dat_path,
            dtype="int16",
            **job_kwargs,
        )
    else:
        print(f"Using existing .dat as recording file for Kilosort2")

    stopwatch.log_time("Done converting recording.")


def _spike_sort_docker(recording: BaseRecording, output_folder: Path) -> Any:
    """Run Kilosort2 inside a Docker container via SpikeInterface.

    Uses the ``spikeinterface/kilosort2-compiled-base`` image which bundles a
    compiled MATLAB Runtime — no MATLAB license or local installation required.
    Requires Docker with NVIDIA GPU support (``--gpus all``).

    The recording is first written to a binary ``.dat`` file on the host so
    that the Docker container does not need vendor-specific HDF5 plugins
    (e.g. Maxwell compression).  A lightweight
    ``BinaryRecordingExtractor`` pointing at the ``.dat`` is then passed
    to ``run_sorter``.

    Parameters:
        recording (BaseRecording): Scaled and filtered SpikeInterface recording.
        output_folder (Path): Directory for Kilosort2 output files.

    Returns:
        sorting (KilosortSortingExtractor): The sorting result loaded from the
            Docker output folder.
    """
    from .docker_utils import get_docker_image

    # Pre-convert recording to int16 binary on the host so that:
    # 1. The container doesn't need vendor-specific HDF5 plugins (e.g. Maxwell)
    # 2. SI's kilosortbase._setup_recording skips the redundant copy (it checks
    #    binary_compatible_with(dtype="int16")) — saves ~22 GB of disk I/O
    # Write to a sibling directory so it's not inside the sorter folder
    # (which run_sorter may delete/recreate).
    dat_dir = output_folder.parent / (output_folder.name + "_binary")
    dat_dir.mkdir(exist_ok=True, parents=True)
    dat_path = dat_dir / "recording.dat"
    if not dat_path.exists():
        print("Writing binary recording for Docker container...")
        write_binary_recording(recording, file_paths=[str(dat_path)], dtype="int16")
    else:
        print(f"Reusing existing binary recording at {dat_path}")

    bin_recording = BinaryRecordingExtractor(
        file_paths=[str(dat_path)],
        sampling_frequency=recording.get_sampling_frequency(),
        num_channels=recording.get_num_channels(),
        dtype="int16",
    )
    bin_recording.set_channel_locations(recording.get_channel_locations())

    # Map KILOSORT_PARAMS to SpikeInterface's run_sorter kwargs.
    si_params = {k: v for k, v in KILOSORT_PARAMS.items()}

    print("Running Kilosort2 via Docker container")

    # Inject MW_CUDA_FORWARD_COMPATIBILITY=1 into the Docker container so
    # that the compiled MATLAB Runtime supports newer GPU architectures
    # (e.g. RTX 5090 / compute capability 12.0).
    from spikeinterface.sorters.container_tools import ContainerClient

    _orig_init = ContainerClient.__init__

    def _patched_init(self, mode, container_image, volumes, py_user_base, extra_kwargs):
        if mode == "docker":
            extra_kwargs.setdefault("environment", {})
            extra_kwargs["environment"]["MW_CUDA_FORWARD_COMPATIBILITY"] = "1"
            # Cap container memory to 80% of system RAM to prevent OOM crashes
            import os

            try:
                total_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                extra_kwargs["mem_limit"] = int(total_mem * 0.8)
            except (ValueError, OSError):
                pass
        _orig_init(self, mode, container_image, volumes, py_user_base, extra_kwargs)

    ContainerClient.__init__ = _patched_init
    try:
        si_sorting = run_sorter(
            sorter_name="kilosort2",
            recording=bin_recording,
            folder=str(output_folder),
            docker_image=get_docker_image("kilosort2"),
            verbose=True,
            raise_error=True,
            remove_existing_folder=True,
            with_output=False,  # We load results ourselves via KilosortSortingExtractor
            installation_mode="no-install",
            **si_params,
        )
    finally:
        ContainerClient.__init__ = _orig_init

    # Keep the pre-converted binary for potential reuse (recompute_recording=False).
    # It will be cleaned up with the rest of the intermediates if delete_inter=True.
    if dat_path.exists():
        print(
            f"Keeping pre-converted binary for reuse ({dat_path.stat().st_size / 1e9:.1f} GB)"
        )

    # SI places sorter output in a subfolder; locate the Phy output files
    sorter_output = output_folder / "sorter_output"
    if not (sorter_output / "spike_times.npy").exists():
        # Fallback: some SI versions put output directly in the folder
        sorter_output = output_folder

    return KilosortSortingExtractor(folder_path=sorter_output)


def spike_sort(
    rec_cache: BaseRecording,
    rec_path: Any,
    recording_dat_path: Path,
    output_folder: Path,
) -> Any:
    """Run Kilosort2 on a single recording and return the sorting result.

    Converts the recording to ``.dat`` format (if needed), launches
    Kilosort2 via MATLAB (or Docker when ``USE_DOCKER`` is True), and
    returns the detected units as a ``KilosortSortingExtractor``. Skips
    re-sorting when ``RECOMPUTE_SORTING`` is False and results already exist.

    Parameters:
        rec_cache (BaseRecording): Scaled and filtered recording.
        rec_path (str or Path): Original recording path (for logging).
        recording_dat_path (Path): Path to the binary ``.dat`` file.
        output_folder (Path): Kilosort2 output directory.

    Returns:
        sorting (KilosortSortingExtractor or Exception): The sorting
            result, or the caught exception if sorting failed.
    """
    print_stage("SPIKE SORTING")
    stopwatch = Stopwatch()

    try:
        if not RECOMPUTE_SORTING and (output_folder / "spike_times.npy").exists():
            print("Loading Kilosort2's sorting results")
            sorting = KilosortSortingExtractor(folder_path=output_folder)
        elif USE_DOCKER:
            # Docker: SpikeInterface handles .dat conversion internally
            create_folder(output_folder)
            sorting = _spike_sort_docker(rec_cache, output_folder)
        else:
            # Local MATLAB
            kilosort = RunKilosort()
            try:
                write_recording(rec_cache, recording_dat_path, verbose=True)
            except Exception as e:
                print(
                    f"Could not convert recording because of {e}.\nMoving on to next recording"
                )
                return e

            create_folder(output_folder)
            sorting = kilosort.run(
                recording=rec_cache,
                recording_dat_path=recording_dat_path,
                output_folder=output_folder,
            )

    except Exception as e:
        print(f"Kilosort2 failed on recording {rec_path}\n{e}")
        print("Moving on to next recording")
        return e

    stopwatch.log_time("Done sorting.")
    print(f"Kilosort detected {len(sorting.unit_ids)} units")
    return sorting


def extract_waveforms(
    recording_path: Any,
    recording: BaseRecording,
    sorting: Any,
    root_folder: Path,
    initial_folder: Path,
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
        not REEXTRACT_WAVEFORMS and (root_folder / "waveforms").is_dir()
    ):  # Load saved waveform extractor
        print("Loading waveforms from folder")
        we = WaveformExtractor.load_from_folder(
            recording, sorting, root_folder, initial_folder
        )
        stopwatch.log_time("Done extracting waveforms.")
    else:  # Create new waveform extractor
        we = WaveformExtractor.create_initial(
            recording_path, recording, sorting, root_folder, initial_folder
        )
        we.run_extract_waveforms(**job_kwargs)
        stopwatch.log_time("Done extracting waveforms.")

        we.compute_templates(
            modes=("average", "std"), n_jobs=job_kwargs.get("n_jobs", 1)
        )
    return we


def process_recording(
    rec_name: str,
    rec_path: Any,
    inter_path: Any,
    results_path: Any,
    rec_loaded: Any = None,
) -> Any:
    """Run the full sorting pipeline on a single recording.

    Orchestrates path setup, recording loading, spike sorting, waveform
    extraction, SpikeData-based curation, result compilation, and
    optional trace saving for downstream models.

    Parameters:
        rec_name (str): Short name for the recording (used in logging
            and result filenames).
        rec_path (str or Path): Path to the recording file.
        inter_path (str or Path): Root intermediate directory.
        results_path (str or Path): Root results directory.
        rec_loaded (BaseRecording or None): Pre-loaded recording object.
            When provided, used instead of loading from *rec_path*.

    Returns:
        result (tuple or Exception): ``(sd_raw, sd_curated)`` on success
            when ``SAVE_RAW_PKL`` is True, otherwise just ``sd_curated``.
            Returns the caught exception if any stage failed.
    """
    create_folder(inter_path)
    with Tee(Path(inter_path) / OUT_FILE, "a"):
        stopwatch = Stopwatch()

        # Get Paths
        (
            rec_path,
            inter_path,
            recording_dat_path,
            output_folder,
            waveforms_root_folder,
            curation_initial_folder,
            curation_first_folder,
            curation_second_folder,
            results_path,
        ) = get_paths(rec_path, inter_path, results_path)

        # Save a copy of the script
        if SAVE_SCRIPT:
            print_stage("SAVING SCRIPT")
            copy_script(inter_path)

        # Load Recording
        try:
            recording_filtered = load_recording(
                rec_path if rec_loaded is None else rec_loaded
            )
        except Exception as e:
            print(f"Could not open the recording file because of {e}")
            print("Moving on to next recording")
            return e

        # Spike sorting
        sorting = spike_sort(
            rec_cache=recording_filtered,
            rec_path=rec_path,
            recording_dat_path=recording_dat_path,
            output_folder=output_folder,
        )
        if isinstance(sorting, BaseException):  # Could not sort recording
            return sorting

        # Extract waveforms
        w_e_raw = extract_waveforms(
            rec_path,
            recording_filtered,
            sorting,
            waveforms_root_folder,
            curation_initial_folder,
            n_jobs=N_JOBS,
            total_memory=TOTAL_MEMORY,
            progress_bar=True,
        )

        # Convert to SpikeData with enriched neuron_attributes
        # (SNR, std_norm, channel locations, templates)
        sd = _waveform_extractor_to_spikedata(
            w_e_raw, rec_path, rec_chunks=REC_CHUNKS or None
        )

        # Curate via SpikeData methods with disk caching
        curate_kwargs = {}
        if CURATE_FIRST:
            if FR_MIN is not None:
                curate_kwargs["min_rate_hz"] = FR_MIN
            if ISI_VIOL_MAX is not None:
                curate_kwargs["isi_max"] = ISI_VIOL_MAX
                curate_kwargs["isi_threshold_ms"] = 1.5
                curate_kwargs["isi_method"] = ISI_VIOLATION_METHOD
            if SNR_MIN is not None:
                curate_kwargs["min_snr"] = SNR_MIN
            if SPIKES_MIN_FIRST is not None:
                curate_kwargs["min_spikes"] = SPIKES_MIN_FIRST
        if CURATE_SECOND:
            # Use the stricter spike count if second-stage is enabled
            if SPIKES_MIN_SECOND is not None:
                curate_kwargs["min_spikes"] = SPIKES_MIN_SECOND
            if STD_NORM_MAX is not None:
                curate_kwargs["max_std_norm"] = STD_NORM_MAX

        # Determine which SpikeData to curate on: the full concatenated
        # one (default) or a single epoch's data.
        has_epochs = bool(sd.metadata.get("rec_chunks_ms"))
        if CURATION_EPOCH is not None and has_epochs:
            epoch_sds = sd.split_epochs()
            if CURATION_EPOCH < 0 or CURATION_EPOCH >= len(epoch_sds):
                raise ValueError(
                    f"curation_epoch={CURATION_EPOCH} is out of range "
                    f"(recording has {len(epoch_sds)} epochs, 0-indexed)."
                )
            sd_for_curation = epoch_sds[CURATION_EPOCH]
            print(
                f"Curating based on epoch {CURATION_EPOCH} "
                f"({sd_for_curation.metadata.get('source_file', '')})"
            )
        else:
            sd_for_curation = sd

        sd_epoch_curated, curation_history = _curate_spikedata(
            sd_for_curation,
            curation_folder=curation_first_folder,
            recurate=RECURATE_FIRST or RECURATE_SECOND,
            **curate_kwargs,
        )

        # When curating on a single epoch, apply the passing unit IDs
        # back to the full concatenated SpikeData.
        if sd_for_curation is not sd:
            passing_ids = set()
            if sd_epoch_curated.neuron_attributes is not None:
                for attrs in sd_epoch_curated.neuron_attributes:
                    uid = attrs.get("unit_id")
                    if uid is not None:
                        passing_ids.add(int(uid))
            passing_indices = [
                i
                for i in range(sd.N)
                if sd.neuron_attributes is not None
                and int(sd.neuron_attributes[i].get("unit_id", -1)) in passing_ids
            ]
            sd_curated = sd.subset(passing_indices)
        else:
            sd_curated = sd_epoch_curated

        n_before = sd.N
        n_after = sd_curated.N
        print(
            f"Curation: {n_before} -> {n_after} units "
            f"({n_before - n_after} removed)"
        )

        # Compile results using SpikeData
        compile_results(rec_name, rec_path, results_path, sd_curated, curation_history)

        # Save scaled traces for training detection model
        if SAVE_DL_DATA:
            save_stopwatch = Stopwatch("SAVING TRACES FOR DETECTION MODEL")
            save_traces(rec_path if rec_loaded is None else rec_loaded, results_path)
            save_stopwatch.log_time()

        print_stage(f"DONE WITH RECORDING")
        print(f"Recording: {rec_path}")
        stopwatch.log_time("Total")

        if SAVE_RAW_PKL:
            return sd, sd_curated
        return sd_curated


def copy_script(path: Path) -> None:
    """Save a timestamped copy of this module to the given directory.

    Parameters:
        path (Path): Destination directory.
    """
    copied_script_name = (
        time.strftime("%y%m%d_%H%M%S") + "_" + os.path.basename(__file__)
    )
    copied_path = (path / copied_script_name).absolute()
    shutil.copyfile(__file__, copied_path)
    print(f"Saved a copy of script to {copied_path}")


def compile_results(
    rec_name: str,
    rec_path: Any,
    results_path: Any,
    sd: Any,
    curation_history: Optional[dict] = None,
) -> None:
    """Compile and export sorting results for a single recording.

    Saves spike times, electrode information, and optionally ``.npz`` /
    ``.mat`` files via a ``Compiler`` instance. When the recording was
    built from multiple chunks (``REC_CHUNKS``), each chunk is compiled
    separately into its own sub-folder using ``split_epochs``.

    Parameters:
        rec_name (str): Short name for the recording.
        rec_path (str or Path): Original recording file path.
        results_path (Path): Output directory for compiled results.
        sd (SpikeData): Curated SpikeData with enriched neuron_attributes.
        curation_history (dict or None): Curation history dict.
    """
    compile_stopwatch = Stopwatch("COMPILING RESULTS")
    print(f"For recording: {rec_path}")
    if COMPILE_SINGLE_RECORDING:
        if (
            not (results_path / "parameters.json").exists()
            or RECOMPILE_SINGLE_RECORDING
        ):
            print(f"Saving to path: {results_path}")
            if len(REC_CHUNKS) > 1:
                epoch_sds = sd.split_epochs()
                for c, sd_chunk in enumerate(epoch_sds):
                    print(f"Compiling chunk {c}")
                    compiler = Compiler()
                    compiler.add_recording(rec_name, sd_chunk, curation_history)
                    compiler.save_results(results_path / f"chunk{c}")
            else:
                compiler = Compiler()
                compiler.add_recording(rec_name, sd, curation_history)
                compiler.save_results(results_path)
                compile_stopwatch.log_time("Done compiling results.")
        else:
            print(
                "Skipping compiling results because 'recompile_single_recording' is set to False and already compiled"
            )
    else:
        print(
            f"Skipping compiling results because 'compile_single_recording' is set to False"
        )


# Tee is imported from sorting_utils at the top of this file.
