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
import warnings
import tempfile
import time
import traceback
from math import ceil
from pathlib import Path
from types import MethodType
from typing import Any, List, Optional, Union

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


def print_stage(text):
    """Print a centered banner message framed by ``=`` lines.

    Parameters:
        text: Message to display. Converted to ``str`` if not already.
    """
    text = str(text)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    num_chars = 70
    char = "="
    indent = int((num_chars - len(text)) / 2)

    print("\n" + num_chars * char)
    print(indent * " " + text)
    print(f"  [{timestamp}]".center(num_chars))
    print(num_chars * char)


# region Save traces
def save_traces(
    recording,
    inter_path,
    start_ms=0,
    end_ms=None,
    num_processes=None,
    dtype="float16",
    verbose=True,
):
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
    scaled_traces_path,
    start_ms=0,
    end_ms=None,
    num_processes=16,
    dtype="float16",
    verbose=True,
):
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


def _save_traces_si(task):
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
    rec_path,
    save_path,
    start_ms=0,
    end_ms=None,
    samp_freq=20,  # kHz
    default_gain=1,
    chunk_size=100000,
    num_processes=2,
    dtype="float16",
    verbose=True,
):
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


def _get_traces_mea_old(rec_path):
    """Return the raw signal dataset from an old-format Maxwell HDF5 file.

    Parameters:
        rec_path (str or Path): Path to the Maxwell ``.h5`` file.

    Returns:
        sig (h5py.Dataset): The ``'sig'`` dataset.
    """
    return h5py.File(rec_path, "r")["sig"]


def _get_traces_mea_new(rec_path):
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


def _save_traces_mea(task):
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
            size = (
                template_mid
                - np.flatnonzero(np.isclose(template[:template_mid], 0))[-1]
            )
            half_windows_sizes.append(int(size * window_size_scale))

        return half_windows_sizes

    def ms_to_samples(self, ms):
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

        parameters = {
            "recording_path": str(recording_path.absolute()),
            "sampling_frequency": recording.get_sampling_frequency(),
            "ms_before": WAVEFORMS_MS_BEFORE,
            "ms_after": WAVEFORMS_MS_AFTER,
            "peak_ind": sorting.ms_to_samples(WAVEFORMS_MS_BEFORE),
            "pos_peak_thresh": POS_PEAK_THRESH,
            "max_waveforms_per_unit": MAX_WAVEFORMS_PER_UNIT,
            "dtype": str(recording.get_dtype()),
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

    def ms_to_samples(self, ms):
        return int(ms * self.sampling_frequency / 1000.0)

    # endregion

    # region Extract waveforms
    def run_extract_waveforms(self, **job_kwargs):
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

    def sample_spikes(self):
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

    def select_random_spikes_uniformly(self):
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
                    peak_window_right = min(st_trace + half_window_size, max_trace_ind)
                    peak_window_size = peak_window_right - peak_window_left + 1
                    traces_peak_window = traces[
                        peak_window_left:peak_window_right, chan_max
                    ]
                    if use_pos_peak[unit_id]:
                        peak_value = np.max(traces_peak_window)
                    else:
                        peak_value = np.min(traces_peak_window)
                    peak_indices = np.flatnonzero(traces_peak_window == peak_value)
                    st_offset = (
                        peak_indices[peak_indices.size // 2] - peak_window_size // 2
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
        self, unit_id, with_index=False, cache=False, memmap=True
    ):  # SpikeInterface has cache=True by default
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

    def get_sampled_indices(self, unit_id):
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

    def get_computed_template(self, unit_id, mode):
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

    def compute_templates(self, modes=("average", "std"), unit_ids=None, folder=None):
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

        print(f"Computing templates for {len(unit_ids)} units")
        for unit_id in tqdm(unit_ids):
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

        create_folder(folder)
        print("Saving templates to .npy")
        for mode in modes:
            templates = self.template_cache[mode]
            template_file = folder / f"templates_{mode}.npy"
            np.save(str(template_file), templates)
        stopwatch.log_time("Done computing and saving templates.")

    def load_units(self):
        self.sorting.unit_ids = np.load(str(self.folder / "unit_ids.npy")).tolist()
        self.sorting.spike_times = np.load(str(self.folder / "spike_times.npy"))
        self.sorting.spike_clusters = np.load(str(self.folder / "spike_clusters.npy"))

    def get_curation_history(self):
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
# region ProcessPoolExecutor
# Used for parallel processing in ChunkRecordingExecutor
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The following diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     |        |     | 4, result |    |         |
|          |     | ...        |     |        |     | 3, except |    |         |
+----------+     +------------+     +--------+     +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""

__author__ = "Brian Quinlan (brian@sweetapp.com)"

import atexit
import os
from concurrent.futures import _base
import queue
from queue import Full
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
import traceback

# Workers are created as daemon threads and processes. This is done to allow the
# interpreter to exit when there are still idle processes in a
# ProcessPoolExecutor's process pool (i.e. shutdown() was not called). However,
# allowing workers to die with the interpreter has two undesirable properties:
#   - The workers would still be running during interpreter shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads/processes finish.

_threads_wakeups = weakref.WeakKeyDictionary()
_global_shutdown = False


class _ThreadWakeup:
    def __init__(self):
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        self._writer.close()
        self._reader.close()

    def wakeup(self):
        self._writer.send_bytes(b"")

    def clear(self):
        while self._reader.poll():
            self._reader.recv_bytes()


def _python_exit():
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeups.items())
    for _, thread_wakeup in items:
        thread_wakeup.wakeup()
    for t, _ in items:
        t.join()


# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1

# On Windows, WaitForMultipleObjects is used to wait for processes to finish.
# It can wait on, at most, 63 objects. There is an overhead of two objects:
# - the result queue reader
# - the thread wakeup reader
_MAX_WINDOWS_WORKERS = 63 - 2


# Hack to embed stringification of remote traceback in local traceback


class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class _ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = "".join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)


def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc


class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _ResultItem(object):
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result


class _CallItem(object):
    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(self, max_size=0, *, ctx, pending_work_items):
        self.pending_work_items = pending_work_items
        super().__init__(max_size, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format("".join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            # work_item can be None if another process terminated. In this case,
            # the queue_manager_thread fails all work_items with BrokenProcessPool
            if work_item is not None:
                work_item.future.set_exception(e)
        else:
            super()._on_queue_feeder_error(e, obj)


def _get_chunks(*iterables, chunksize):
    """Iterates over zip()ed iterables in chunks."""
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result, exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(_ResultItem(work_id, exception=exc))


def _process_worker(call_queue, result_queue, initializer, initargs):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item


def _add_call_item_to_queue(pending_work_items, work_ids, call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(
                    _CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs),
                    block=True,
                )
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(
    executor_reference,
    processes,
    pending_work_items,
    work_ids_queue,
    call_queue,
    result_queue,
    thread_wakeup,
):
    """Manages the communication between this process and the worker processes.

    This function is run in a local thread.

    Args:
        executor_reference: A weakref.ref to the ProcessPoolExecutor that owns
            this thread. Used to determine if the ProcessPoolExecutor has been
            garbage collected and that this function can exit.
        process: A list of the ctx.Process instances used as
            workers.
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids_queue: A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        call_queue: A ctx.Queue that will be filled with _CallItems
            derived from _WorkItems for processing by the process workers.
        result_queue: A ctx.SimpleQueue of _ResultItems generated by the
            process workers.
        thread_wakeup: A _ThreadWakeup to allow waking up the
            queue_manager_thread from the main Thread and avoid deadlocks
            caused by permanently locked queues.
    """
    executor = None

    def shutting_down():
        return _global_shutdown or executor is None or executor._shutdown_thread

    def shutdown_worker():
        # This is an upper bound on the number of children alive.
        n_children_alive = sum(p.is_alive() for p in processes.values())
        n_children_to_stop = n_children_alive
        n_sentinels_sent = 0
        # Send the right number of sentinels, to make sure all children are
        # properly terminated.
        while n_sentinels_sent < n_children_to_stop and n_children_alive > 0:
            for i in range(n_children_to_stop - n_sentinels_sent):
                try:
                    call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except Full:
                    break
            n_children_alive = sum(p.is_alive() for p in processes.values())

        # Release the queue's resources as soon as possible.
        call_queue.close()
        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on Mac OS X.
        for p in processes.values():
            p.join()

    result_reader = result_queue._reader
    wakeup_reader = thread_wakeup._reader
    readers = [result_reader, wakeup_reader]

    while True:
        _add_call_item_to_queue(pending_work_items, work_ids_queue, call_queue)

        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        worker_sentinels = [p.sentinel for p in processes.values()]
        ready = mp.connection.wait(readers + worker_sentinels)

        cause = None
        is_broken = True
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                is_broken = False
            except BaseException as e:
                cause = traceback.format_exception(type(e), e, e.__traceback__)

        elif wakeup_reader in ready:
            is_broken = False
            result_item = None
        thread_wakeup.clear()
        if is_broken:
            # Mark the process pool broken so that submits fail right now.
            executor = executor_reference()
            if executor is not None:
                executor._broken = (
                    "A child process terminated "
                    "abruptly, the process pool is not "
                    "usable anymore"
                )
                executor._shutdown_thread = True
                executor = None
            bpe = BrokenProcessPool(
                "A process in the process pool was "
                "terminated abruptly while the future was "
                "running or pending."
            )
            if cause is not None:
                bpe.__cause__ = _RemoteTraceback(f"\n'''\n{''.join(cause)}'''")
            # All futures in flight must be marked failed
            for work_id, work_item in pending_work_items.items():
                work_item.future.set_exception(bpe)
                # Delete references to object. See issue16284
                del work_item
            pending_work_items.clear()
            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            for p in processes.values():
                p.terminate()
            shutdown_worker()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert shutting_down()
            p = processes.pop(result_item)
            p.join()
            if not processes:
                shutdown_worker()
                return
        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
            # Delete reference to result_item
            del result_item

        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            try:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown_thread = True
                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not pending_work_items:
                    shutdown_worker()
                    return
            except Full:
                # This is not a problem: we will eventually be woken up (in
                # result_queue.get()) and be able to send a sentinel again.
                pass
        executor = None


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # indetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = (
        "system provides too few semaphores (%d"
        " available, 256 necessary)" % nsems_max
    )
    raise NotImplementedError(_system_limited)


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


class BrokenProcessPool(_base.BrokenExecutor):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


class ProcessPoolExecutor(_base.Executor):
    def __init__(
        self, max_workers=None, mp_context=None, initializer=None, initargs=()
    ):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
            mp_context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: A callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = os.cpu_count() or 1
            if sys.platform == "win32":
                self._max_workers = min(_MAX_WINDOWS_WORKERS, self._max_workers)
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            elif sys.platform == "win32" and max_workers > _MAX_WINDOWS_WORKERS:
                raise ValueError(f"max_workers must be <= {_MAX_WINDOWS_WORKERS}")

            self._max_workers = max_workers

        if mp_context is None:
            mp_context = mp.get_context()
        self._mp_context = mp_context

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        self._initializer = initializer
        self._initargs = initargs

        # Management thread
        self._queue_management_thread = None

        # Map of pids to processes
        self._processes = {}

        # Shutdown is a two-step process.
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}

        # Create communication channels for the executor
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        queue_size = self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(
            max_size=queue_size,
            ctx=self._mp_context,
            pending_work_items=self._pending_work_items,
        )
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True
        self._result_queue = mp_context.SimpleQueue()
        self._work_ids = queue.Queue()

        # _ThreadWakeup is a communication channel used to interrupt the wait
        # of the main loop of queue_manager_thread from another thread (e.g.
        # when calling executor.submit or executor.shutdown). We do not use the
        # _result_queue to send the wakeup signal to the queue_manager_thread
        # as it could result in a deadlock if a worker process dies with the
        # _result_queue write lock still acquired.
        self._queue_management_thread_wakeup = _ThreadWakeup()

    def _start_queue_management_thread(self):
        if self._queue_management_thread is None:
            # When the executor gets garbarge collected, the weakref callback
            # will wake up the queue management thread so that it can terminate
            # if there is no pending work item.
            def weakref_cb(_, thread_wakeup=self._queue_management_thread_wakeup):
                mp.util.debug(
                    "Executor collected: triggering callback for" " QueueManager wakeup"
                )
                thread_wakeup.wakeup()

            # Start the processes so that their sentinels are known.
            self._adjust_process_count()
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._processes,
                    self._pending_work_items,
                    self._work_ids,
                    self._call_queue,
                    self._result_queue,
                    self._queue_management_thread_wakeup,
                ),
                name="QueueManagerThread",
            )
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()
            _threads_wakeups[self._queue_management_thread] = (
                self._queue_management_thread_wakeup
            )

    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            p = self._mp_context.Process(
                target=_process_worker,
                args=(
                    self._call_queue,
                    self._result_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            p.start()
            self._processes[p.pid] = p

    def submit(*args, **kwargs):
        if len(args) >= 2:
            self, fn, *args = args
        elif not args:
            raise TypeError(
                "descriptor 'submit' of 'ProcessPoolExecutor' object "
                "needs an argument"
            )
        elif "fn" in kwargs:
            fn = kwargs.pop("fn")
            self, *args = args
            import warnings

            warnings.warn(
                "Passing 'fn' as keyword argument is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            raise TypeError(
                "submit expected at least 1 positional argument, "
                "got %d" % (len(args) - 1)
            )

        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool(self._broken)
            if self._shutdown_thread:
                raise RuntimeError("cannot schedule new futures after shutdown")
            if _global_shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after " "interpreter shutdown"
                )

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()

            self._start_queue_management_thread()
            return f

    # submit.__text_signature__ = _base.Executor.submit.__text_signature__
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(
            partial(_process_chunk, fn),
            _get_chunks(*iterables, chunksize=chunksize),
            timeout=timeout,
        )
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown_thread = True
        if self._queue_management_thread:
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()
            if wait:
                self._queue_management_thread.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._queue_management_thread = None
        if self._call_queue is not None:
            self._call_queue.close()
            if wait:
                self._call_queue.join_thread()
            self._call_queue = None
        self._result_queue = None
        self._processes = None

        if self._queue_management_thread_wakeup:
            self._queue_management_thread_wakeup.close()
            self._queue_management_thread_wakeup = None

    shutdown.__doc__ = _base.Executor.shutdown.__doc__


atexit.register(_python_exit)


# endregion
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
class Stopwatch:
    """Simple wall-clock timer for logging pipeline stage durations.

    Parameters:
        start_msg (str or None): Optional message printed when the timer
            starts. When *None*, nothing is printed on construction.
        use_print_stage (bool): If True (default), format *start_msg*
            with the ``print_stage`` banner; otherwise use plain ``print``.
    """

    def __init__(self, start_msg=None, use_print_stage=True):
        if start_msg is not None:
            if use_print_stage:
                print_stage(start_msg)
            else:
                print(start_msg)

        self._time_start = time.time()

    def log_time(self, text=None):
        if text is None:
            print(f"Time: {time.time() - self._time_start:.2f}s")
        else:
            print(f"{text} Time: {time.time() - self._time_start:.2f}s")


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

    def add_recording(self, rec_name, sd, curation_history=None):
        """Queue a recording for compilation.

        Parameters:
            rec_name (str): Short name for the recording.
            sd (SpikeData): Curated SpikeData with enriched
                ``neuron_attributes``.
            curation_history (dict or None): Curation history dict
                from ``build_curation_history``.
        """
        self.recs_cache.append((rec_name, sd, curation_history))

    def save_results(self, folder):
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


def create_folder(folder, parents=True):
    """Create a directory if it does not already exist.

    Parameters:
        folder (str or Path): Directory path to create.
        parents (bool): Create parent directories as needed (default True).
    """
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=parents)
        print(f"Created folder: {folder}")


def delete_folder(folder):
    """Delete a file or directory tree if it exists.

    Parameters:
        folder (str or Path): Path to the file or directory to delete.
    """
    folder = Path(folder)
    if folder.exists():
        if folder.is_dir():
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        else:
            folder.unlink()
            print(f"Deleted file: {folder}")


def load_recording(rec_path):
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
    recording, return_scaled=True, num_chunks=20, chunk_size=10000, seed=0
):
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


def _waveform_extractor_to_spikedata(w_e, rec_path, rec_chunks=None):
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

        # Optionally un-scale templates
        if not SCALE_COMPILED_WAVEFORMS and w_e.recording.has_scaleable_traces():
            gain = w_e.recording.get_channel_gains()
            offset = w_e.recording.get_channel_offsets()
            template_mean = ((template_mean - offset) / gain).astype("float32")
            template_std = ((template_std - offset) / gain).astype("float32")

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


def _curate_spikedata(sd, curation_folder, recurate=False, **curate_kwargs):
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


def load_single_recording(rec_path):
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


def concatenate_recordings(rec_path):
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


def get_paths(rec_path, inter_path, results_path):
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


def write_recording(recording_filtered, recording_dat_path, verbose=True):
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


def _spike_sort_docker(recording, output_folder):
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
            docker_image="spikeinterface/kilosort2-compiled-base:py310-si0.104",
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


def spike_sort(rec_cache, rec_path, recording_dat_path, output_folder):
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
    recording_path, recording, sorting, root_folder, initial_folder, **job_kwargs
):
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

        we.compute_templates(modes=("average", "std"))
    return we


def process_recording(rec_name, rec_path, inter_path, results_path, rec_loaded=None):
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


def copy_script(path):
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


def compile_results(rec_name, rec_path, results_path, sd, curation_history=None):
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


class Tee:
    """Context manager that mirrors ``stdout`` to a log file.

    While the context is active, every ``print`` call writes to both the
    original ``stdout`` and the specified file. On exit, ``stdout`` is
    restored and the file is closed. Exceptions raised inside the
    context are written to the log before re-raising.

    Parameters:
        file_path (str or Path): Path to the log file.
        file_mode (str): File open mode (e.g. ``'w'`` or ``'a'``).
    """

    def __init__(self, file_path, file_mode):
        _file = open(file_path, file_mode)
        _file.stdout = sys.stdout
        _file.file_write = _file.write
        _file.write = MethodType(Tee._write, _file)
        self._file = _file

    def __enter__(self):
        sys.stdout = self._file
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._file.write = self._file.file_write
            print("Traceback (most recent call last):")
            traceback.print_tb(exc_tb, file=self._file)
            print(f"{exc_type}: {exc_val}")
        sys.stdout = self._file.stdout
        self._file.close()

    @staticmethod
    def _write(self, s):
        self.file_write(s)
        if s != "\n" and s != " ":  # Prevents random whitespaces
            print(s, file=self.stdout)


def sort_with_kilosort2(
    recording_files: list,
    intermediate_folders=None,
    results_folders=None,
    compiled_results_folder=None,
    out_file="sort_with_kilosort2.out",
    kilosort_path=None,  # "/path/to/Kilosort2"
    stream_id=None,  # stream_id for MaxwellRecordingExtractor (multi-stream .h5 files)
    hdf5_plugin_path=None,  # path to custom HDF5 decompression plugin (Maxwell new-format .h5)
    kilosort_params=None,
    use_docker=False,
    recompute_recording=False,
    recompute_sorting=False,
    reextract_waveforms=False,
    recurate_first=False,
    recurate_second=False,
    recompile_single_recording=False,
    recompile_all_recordings=False,
    delete_inter=True,
    save_script=False,
    n_jobs=8,
    total_memory="16G",
    use_parallel_processing_for_raw_conversion=True,
    first_n_mins=None,
    mea_y_max=None,  # 2100
    gain_to_uv=None,
    offset_to_uv=None,
    rec_chunks=None,
    freq_min=300,
    freq_max=6000,
    waveforms_ms_before=2,
    waveforms_ms_after=2,
    pos_peak_thresh=2,
    max_waveforms_per_unit=300,
    curate_first=True,
    curate_second=True,
    curation_epoch=None,
    fr_min=0.05,
    isi_viol_max=1,
    isi_violation_method="percent",
    snr_min=5,
    spikes_min_first=30,
    spikes_min_second=50,
    std_norm_max=1,
    std_at_peak=True,
    std_over_window_ms_before=0.5,
    std_over_window_ms_after=1.5,
    save_electrodes=True,
    save_spike_times=True,
    save_raw_pkl=False,
    save_dl_data=False,
    compile_single_recording=True,
    compile_to_mat=False,
    compile_to_npz=True,
    compile_waveforms=False,
    compile_all_recordings=False,
    compiled_waveforms_ms_before=2,
    compiled_waveforms_ms_after=2,
    scale_compiled_waveforms=True,
    create_figures=False,
    figures_dpi=None,
    figures_font_size=12,
    bar_x_label="Recording",
    bar_y_label="Number of Units",
    bar_label_rotation=0,
    bar_total_label="First Curation",
    bar_selected_label="Selected Curation",
    scatter_std_max_units_per_recording=None,
    scatter_recording_colors=None,
    scatter_recording_alpha=1,
    scatter_x_label="Number of Spikes",
    scatter_y_label="avg. STD / amplitude",
    scatter_x_max_buffer=300,
    scatter_y_max_buffer=0.2,
    all_templates_color_curated="#000000",
    all_templates_color_failed="#FF0000",
    all_templates_per_column=50,
    all_templates_y_spacing=50,
    all_templates_y_lim_buffer=10,
    all_templates_window_ms_before_peak=5,
    all_templates_window_ms_after_peak=5,
    all_templates_line_ms_before_peak=1,
    all_templates_line_ms_after_peak=4,
    all_templates_x_label="Time Rel. to Peak (ms)",
):
    """
    Run Kilosort2 spike sorting on multiple recordings, using various processing options.

    Args:
        recording_files (list): List of file paths to raw recordings.
        intermediate_folders (list, optional): List of folders for intermediate results (one per
            recording). When ``None`` (default), a timestamped folder is created next to each
            recording file, ensuring a fresh run every time. When an explicit path is provided,
            existing intermediate files are reused according to the ``recompute_*`` flags.
        results_folders (list, optional): Output folders for compiled results. Defaults to None.
        compiled_results_folder (str, optional): Folder for final compiled results. Defaults to None.
        out_file (str, optional): Name of the .out file for stdout logs. Defaults to "sort_with_kilosort2.out".

        kilosort_path (str, optional): Path to Kilosort2 installation. Defaults to None.
        stream_id (str, optional): Stream identifier passed to ``MaxwellRecordingExtractor``
            when loading ``.h5`` files that contain multiple recording streams. If ``None``
            the extractor uses its default stream. Defaults to None.
        hdf5_plugin_path (str or None): Path to a folder containing the
            custom HDF5 decompression plugin required by new-format Maxwell
            ``.h5`` files.  When provided, ``os.environ['HDF5_PLUGIN_PATH']``
            is set automatically before any file is opened, removing the
            need for manual environment variable configuration.  Defaults
            to None (no change to the environment).

        kilosort_params (dict, optional): Kilosort2 configuration parameters. Defaults to preset values.

        use_docker (bool, optional): Run Kilosort2 inside a Docker container
            (``spikeinterface/kilosort2-compiled-base``) instead of a local MATLAB
            installation. Requires Docker with NVIDIA GPU support. When True,
            ``kilosort_path`` is not required. Defaults to False.

        recompute_recording (bool, optional): Whether to recompute the reformatted ``.dat`` file
            and delete all downstream results (sorting, waveforms, curation). Defaults to False.
            Only takes effect when ``intermediate_folders`` is provided — when ``None``, a fresh
            timestamped folder is used and all files are created from scratch regardless of this
            flag. Setting this to True forces a full recomputation of the entire pipeline.
        recompute_sorting (bool, optional): Whether to rerun Kilosort2 if the saved files already exist. Defaults to False.
        reextract_waveforms (bool, optional): Whether to reextract waveforms if the saved files already exist. Defaults to False.
        recurate_first (bool, optional): Whether to rerun first curation if the saved files already exist. Defaults to False.
        recurate_second (bool, optional): Whether to rerun second curation if the saved files already exist. Defaults to False.
        recompile_single_recording (bool, optional): Whether to recompile single recording results if the file already exists. Defaults to False.
        recompile_all_recordings (bool, optional): Whether to recompile results of all recordings if the file already exists. Defaults to False.

        delete_inter (bool, optional): Whether to delete intermediate results after sorting. Defaults to True.

        save_script (bool, optional): Whether to save a copy of this script with results. Defaults to False.

        n_jobs (int, optional): Number of CPU threads for parallel processing. Defaults to 8.
        total_memory (str, optional): Maximum RAM for parallel processing. Defaults to "16G".
        use_parallel_processing_for_raw_conversion (bool, optional): Whether to use parallel processing during raw data conversion. Defaults to True.

        first_n_mins (float, optional): Number of minutes to process from the start of each recording. Defaults to None.
        mea_y_max (int, optional): Maximum height (μm) for flipping MEA y-coordinates. Defaults to None.
        gain_to_uv (float, optional): Gain factor for converting to microvolts (μV). Defaults to None.
        offset_to_uv (float, optional): Offset for converting to μV. Defaults to None.
        rec_chunks (list, optional): Recording chunks to process, defined as frame ranges. Defaults to empty list.

        freq_min (int, optional): Minimum frequency for bandpass filter (Hz). Defaults to 300.
        freq_max (int, optional): Maximum frequency for bandpass filter (Hz). Defaults to 6000.

        waveforms_ms_before (float, optional): Time window before waveform peak to extract (ms). Defaults to 2.
        waveforms_ms_after (float, optional): Time window after waveform peak to extract (ms). Defaults to 2.
        pos_peak_thresh (float, optional): Threshold ratio between positive and negative peaks to use positive for centering. Defaults to 2.
        max_waveforms_per_unit (int, optional): Maximum number of waveforms per unit for extraction. Defaults to 300.

        curate_first (bool, optional): Whether to curate units based on first-stage criteria (e.g., firing rate, ISI). Defaults to True.
        curate_second (bool, optional): Whether to curate units based on second-stage criteria (e.g., consistency). Defaults to True.
        curation_epoch (int or None): When set and the recording was
            concatenated from multiple files, curation criteria are
            evaluated on this epoch's data only (0-indexed). The resulting
            unit selection is then applied to all epochs. When None
            (default), curation is based on all epochs combined.

        fr_min (float, optional): Minimum firing rate threshold for first curation (Hz). Defaults to 0.05.
        isi_viol_max (float, optional): Maximum inter-spike interval violations allowed for first curation.
            Interpreted as a percentage when ``isi_violation_method="percent"`` or as a ratio when
            ``isi_violation_method="hill"``. Defaults to 1.
        isi_violation_method (str, optional): Method for computing ISI violations. ``"percent"`` (default)
            computes violation count / total spikes * 100. ``"hill"`` computes the violation rate ratio
            from Hill et al. (2011) J Neurosci 31: 8699-8705.
        snr_min (float, optional): Minimum signal-to-noise ratio threshold for first curation. Defaults to 5.
        spikes_min_first (int, optional): Minimum spikes per unit for first curation. Defaults to 30.

        spikes_min_second (int, optional): Minimum spikes per unit for second curation. Defaults to 50.
        std_norm_max (float, optional): Maximum normalized waveform standard deviation for second curation. Defaults to 1.
        std_at_peak (bool, optional): Whether to use standard deviation at the waveform peak. Defaults to True.
        std_over_window_ms_before (float, optional): Time window before peak for average standard deviation calculation. Defaults to 0.5.
        std_over_window_ms_after (float, optional): Time window after peak for average standard deviation calculation. Defaults to 1.5.

        save_electrodes (bool, optional): Whether to save electrode information in the result files. Defaults to True.
        save_spike_times (bool, optional): Whether to save spike times in result files. Defaults to True.
        save_raw_pkl (bool, optional): Save a pickle of the raw (pre-curation)
            SpikeData to ``sorted_spikedata.pkl`` in the results folder.
            Defaults to False.
        save_dl_data (bool, optional): Whether to save additional data for machine learning applications. Defaults to True.

        compile_single_recording (bool, optional): Whether to compile the results of a single recording. Defaults to True.
        compile_to_mat (bool, optional): Whether to export results in MATLAB format. Defaults to False.
        compile_to_npz (bool, optional): Whether to export results in NumPy (.npz) format. Defaults to True.
        compile_waveforms (bool, optional): Whether to export waveforms with the compiled results. Defaults to False.
        compile_all_recordings (bool, optional): Whether to compile all recordings' results. Defaults to False.
        compiled_waveforms_ms_before (float, optional): Time window before waveform peak for saving in compiled results (ms). Defaults to 2.
        compiled_waveforms_ms_after (float, optional): Time window after waveform peak for saving in compiled results (ms). Defaults to 2.
        scale_compiled_waveforms (bool, optional): Whether to scale compiled waveforms to μV. Defaults to True.

        create_figures (bool, optional): Whether to generate summary figures for the results. Defaults to False.
        figures_dpi (int, optional): DPI for generated figures. Defaults to None.
        figures_font_size (int, optional): Font size for generated figures. Defaults to 12.

        bar_x_label (str, optional): X-axis label for the bar plot figure. Defaults to "Recording".
        bar_y_label (str, optional): Y-axis label for the bar plot figure. Defaults to "Number of Units".
        bar_label_rotation (int, optional): Rotation angle for bar labels (degrees). Defaults to 0.
        bar_total_label (str, optional): Label for total number of units in the bar plot. Defaults to "First Curation".
        bar_selected_label (str, optional): Label for selected units in the bar plot. Defaults to "Selected Curation".

        scatter_std_max_units_per_recording (int, optional): Max units per recording for scatter plot. Defaults to None.
        scatter_recording_colors (list, optional): Colors for each recording in the scatter plot. Defaults to None.
        scatter_recording_alpha (float, optional): Alpha transparency for scatter plot points. Defaults to 1.
        scatter_x_label (str, optional): X-axis label for scatter plot. Defaults to "Number of Spikes".
        scatter_y_label (str, optional): Y-axis label for scatter plot. Defaults to "avg. STD / amplitude".
        scatter_x_max_buffer (int, optional): Maximum buffer for x-axis in scatter plot. Defaults to 300.
        scatter_y_max_buffer (float, optional): Maximum buffer for y-axis in scatter plot. Defaults to 0.2.

        all_templates_color_curated (str, optional): Color for curated templates in waveform plots. Defaults to "#000000".
        all_templates_color_failed (str, optional): Color for failed templates in waveform plots. Defaults to "#FF0000".
        all_templates_per_column (int, optional): Number of templates per column in waveform plots. Defaults to 50.
        all_templates_y_spacing (int, optional): Y-spacing between templates in waveform plots. Defaults to 50.
        all_templates_y_lim_buffer (int, optional): Buffer for y-axis limits in waveform plots. Defaults to 10.
        all_templates_window_ms_before_peak (float, optional): Time window before peak for template plots (ms). Defaults to 5.
        all_templates_window_ms_after_peak (float, optional): Time window after peak for template plots (ms). Defaults to 5.
        all_templates_line_ms_before_peak (float, optional): Time window before peak for vertical line in template plots (ms). Defaults to 1.
        all_templates_line_ms_after_peak (float, optional): Time window after peak for vertical line in template plots (ms). Defaults to 4.
        all_templates_x_label (str, optional): X-axis label for template plots. Defaults to "Time Rel. to Peak (ms)".

    Returns:
        list[SpikeData]: One :class:`~spikelab.spikedata.SpikeData` per successfully
            sorted recording. Failed recordings are silently skipped so the
            list may be shorter than *recording_files*. Spike times are in
            **milliseconds**. Each object's ``neuron_attributes`` contains
            per-unit metadata (``unit_id``, ``channel``, ``x``, ``y``,
            ``template``, ``amplitude``, ``snr``, ``std_norm``), and
            ``metadata`` includes ``source_file``,
            ``source_format="Kilosort2"``, and ``fs_Hz``.

    Notes:
        **Concatenated recordings.** When an entry in *recording_files* is
        a directory, all ``.raw.h5`` / ``.nwb`` files inside it are
        concatenated into a single recording before sorting. Kilosort2
        sorts the concatenated recording as one continuous session, and
        curation is applied to the combined result. After curation, the
        concatenated SpikeData is automatically split back into one
        SpikeData per original file via
        :meth:`~spikelab.spikedata.SpikeData.split_epochs`:

        - Each epoch SpikeData contains only the spikes that fell within
          that file's time range, shifted to start at t = 0.
        - ``neuron_attributes["template"]`` on each epoch is the average
          waveform computed from that epoch's spikes only (not the
          global average).
        - ``metadata["source_file"]`` is set to the original filename.
        - ``metadata["epoch_index"]`` indicates the position within the
          concatenation.
        - Unit IDs, channel assignments, curation metrics (``snr``,
          ``std_norm``) are shared across epochs because sorting and
          curation operate on the full concatenated recording.

        The concatenated SpikeData is **not** returned — only the
        per-file splits are included in the output.  The return list
        therefore contains one SpikeData per *original file*, not one
        per *recording_files entry*.  For example, if
        ``recording_files=["dir_with_3_files/", "single_file.h5"]``,
        the directory contains ``a.h5``, ``b.h5``, ``c.h5``, and the
        return list will be ``[sd_a, sd_b, sd_c, sd_single]`` — four
        SpikeData objects, three from the directory and one from the
        standalone file.

        **Concatenation compatibility checks.** Before concatenation,
        recordings in a directory are validated: channel count and
        sampling frequency must match (raises ``ValueError``).
        Mismatched channel IDs or channel locations produce warnings
        but do not block concatenation — this allows intentional
        combination of differently-routed recordings, though results
        will likely be unreliable if the electrode layout differs.
    """

    _default_kilosort_params = {
        "detect_threshold": 6,
        "projection_threshold": [10, 4],
        "preclust_threshold": 8,
        "car": True,
        "minFR": 0.1,
        "minfr_goodchannels": 0.1,
        "freq_min": 150,
        "sigmaMask": 30,
        "nPCs": 3,
        "ntbuff": 64,
        "nfilt_factor": 4,
        "NT": None,
        "keep_good_only": False,
    }
    kilosort_params = {**_default_kilosort_params, **(kilosort_params or {})}
    rec_chunks = list(rec_chunks or [])
    scatter_recording_colors = list(
        scatter_recording_colors
        or [
            "#f74343",  # red
            "#fccd56",  # yellow
            "#74fc56",  # green
            "#56fcf6",  # light blue
            "#1e1efa",  # dark blue
            "#fa1ed2",  # pink
        ]
    )

    if intermediate_folders is None:
        cur_datetime = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        intermediate_folders = [
            Path(rec).parent / f"inter_kilosort2_{cur_datetime}"
            for rec in recording_files
        ]
    if results_folders is None:
        results_folders = [
            Path(rec).parent / "sorted_kilosort2" for rec in recording_files
        ]
    if compiled_results_folder is None:
        compiled_results_folder = (
            "None"  # Set this to a string to prevent error later on
        )
        if compile_all_recordings:
            raise ValueError(
                "'compile_all_recordings' is set to True, so you must specify where the results will be stored with 'compiled_results_folder'"
            )
    global RECORDING_FILES, INTERMEDIATE_FOLDERS, OUT_FILE, RESULTS_FOLDERS
    global KILOSORT_PATH, COMPILED_RESULTS_FOLDER, KILOSORT_PARAMS, STREAM_ID, USE_DOCKER
    global RECOMPUTE_RECORDING, RECOMPUTE_SORTING, REEXTRACT_WAVEFORMS, RECURATE_FIRST, RECURATE_SECOND
    global RECOMPILE_SINGLE_RECORDING, RECOMPILE_ALL_RECORDINGS, SAVE_SCRIPT, N_JOBS, TOTAL_MEMORY
    global USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION, FIRST_N_MINS, MEA_Y_MAX, GAIN_TO_UV, OFFSET_TO_UV, REC_CHUNKS
    global FREQ_MIN, FREQ_MAX, WAVEFORMS_MS_BEFORE, WAVEFORMS_MS_AFTER, POS_PEAK_THRESH, MAX_WAVEFORMS_PER_UNIT
    global CURATE_FIRST, CURATE_SECOND, FR_MIN, ISI_VIOL_MAX, ISI_VIOLATION_METHOD, SNR_MIN, SPIKES_MIN_FIRST, SPIKES_MIN_SECOND
    global STD_NORM_MAX, STD_AT_PEAK, STD_OVER_WINDOW_MS_BEFORE, STD_OVER_WINDOW_MS_AFTER, SAVE_ELECTRODES
    global SAVE_SPIKE_TIMES, SAVE_DL_DATA, COMPILE_SINGLE_RECORDING, COMPILE_TO_MAT, COMPILE_TO_NPZ
    global COMPILE_WAVEFORMS, COMPILE_ALL_RECORDINGS, COMPILED_WAVEFORMS_MS_BEFORE, COMPILED_WAVEFORMS_MS_AFTER
    global SCALE_COMPILED_WAVEFORMS, CREATE_FIGURES, FIGURES_DPI, FIGURES_FONT_SIZE, BAR_X_LABEL, BAR_Y_LABEL
    global BAR_LABEL_ROTATION, BAR_TOTAL_LABEL, BAR_SELECTED_LABEL, SCATTER_STD_MAX_UNITS_PER_RECORDING
    global SCATTER_RECORDING_COLORS, SCATTER_RECORDING_ALPHA, SCATTER_X_LABEL, SCATTER_Y_LABEL
    global SCATTER_X_MAX_BUFFER, SCATTER_Y_MAX_BUFFER, ALL_TEMPLATES_COLOR_CURATED, ALL_TEMPLATES_COLOR_FAILED
    global ALL_TEMPLATES_PER_COLUMN, ALL_TEMPLATES_Y_SPACING, ALL_TEMPLATES_Y_LIM_BUFFER
    global ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK, ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK, ALL_TEMPLATES_LINE_MS_BEFORE_PEAK
    global ALL_TEMPLATES_LINE_MS_AFTER_PEAK, ALL_TEMPLATES_X_LABEL

    RECORDING_FILES = recording_files
    INTERMEDIATE_FOLDERS = intermediate_folders
    OUT_FILE = out_file
    RESULTS_FOLDERS = results_folders
    KILOSORT_PATH = kilosort_path
    COMPILED_RESULTS_FOLDER = compiled_results_folder
    KILOSORT_PARAMS = kilosort_params
    STREAM_ID = stream_id
    if hdf5_plugin_path is not None:
        os.environ["HDF5_PLUGIN_PATH"] = str(hdf5_plugin_path)
    USE_DOCKER = use_docker
    RECOMPUTE_RECORDING = recompute_recording
    RECOMPUTE_SORTING = recompute_sorting
    REEXTRACT_WAVEFORMS = reextract_waveforms
    RECURATE_FIRST = recurate_first
    RECURATE_SECOND = recurate_second
    RECOMPILE_SINGLE_RECORDING = recompile_single_recording
    RECOMPILE_ALL_RECORDINGS = recompile_all_recordings
    SAVE_SCRIPT = save_script
    N_JOBS = n_jobs
    TOTAL_MEMORY = total_memory
    USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION = (
        use_parallel_processing_for_raw_conversion
    )
    FIRST_N_MINS = first_n_mins
    MEA_Y_MAX = mea_y_max
    GAIN_TO_UV = gain_to_uv
    OFFSET_TO_UV = offset_to_uv
    REC_CHUNKS = rec_chunks
    global _REC_CHUNK_NAMES
    _REC_CHUNK_NAMES = []
    FREQ_MIN = freq_min
    FREQ_MAX = freq_max
    WAVEFORMS_MS_BEFORE = waveforms_ms_before
    WAVEFORMS_MS_AFTER = waveforms_ms_after
    POS_PEAK_THRESH = pos_peak_thresh
    MAX_WAVEFORMS_PER_UNIT = max_waveforms_per_unit
    CURATE_FIRST = curate_first
    CURATE_SECOND = curate_second
    global CURATION_EPOCH
    CURATION_EPOCH = curation_epoch
    FR_MIN = fr_min
    ISI_VIOL_MAX = isi_viol_max
    ISI_VIOLATION_METHOD = isi_violation_method
    SNR_MIN = snr_min
    SPIKES_MIN_FIRST = spikes_min_first
    SPIKES_MIN_SECOND = spikes_min_second
    STD_NORM_MAX = std_norm_max
    STD_AT_PEAK = std_at_peak
    STD_OVER_WINDOW_MS_BEFORE = std_over_window_ms_before
    STD_OVER_WINDOW_MS_AFTER = std_over_window_ms_after
    SAVE_ELECTRODES = save_electrodes
    SAVE_SPIKE_TIMES = save_spike_times
    global SAVE_RAW_PKL
    SAVE_RAW_PKL = save_raw_pkl
    SAVE_DL_DATA = save_dl_data
    COMPILE_SINGLE_RECORDING = compile_single_recording
    COMPILE_TO_MAT = compile_to_mat
    COMPILE_TO_NPZ = compile_to_npz
    COMPILE_WAVEFORMS = compile_waveforms
    COMPILE_ALL_RECORDINGS = compile_all_recordings
    COMPILED_WAVEFORMS_MS_BEFORE = compiled_waveforms_ms_before
    COMPILED_WAVEFORMS_MS_AFTER = compiled_waveforms_ms_after
    SCALE_COMPILED_WAVEFORMS = scale_compiled_waveforms
    CREATE_FIGURES = create_figures
    FIGURES_DPI = figures_dpi
    FIGURES_FONT_SIZE = figures_font_size
    BAR_X_LABEL = bar_x_label
    BAR_Y_LABEL = bar_y_label
    BAR_LABEL_ROTATION = bar_label_rotation
    BAR_TOTAL_LABEL = bar_total_label
    BAR_SELECTED_LABEL = bar_selected_label
    SCATTER_STD_MAX_UNITS_PER_RECORDING = scatter_std_max_units_per_recording
    SCATTER_RECORDING_COLORS = scatter_recording_colors
    SCATTER_RECORDING_ALPHA = scatter_recording_alpha
    SCATTER_X_LABEL = scatter_x_label
    SCATTER_Y_LABEL = scatter_y_label
    SCATTER_X_MAX_BUFFER = scatter_x_max_buffer
    SCATTER_Y_MAX_BUFFER = scatter_y_max_buffer
    ALL_TEMPLATES_COLOR_CURATED = all_templates_color_curated
    ALL_TEMPLATES_COLOR_FAILED = all_templates_color_failed
    ALL_TEMPLATES_PER_COLUMN = all_templates_per_column
    ALL_TEMPLATES_Y_SPACING = all_templates_y_spacing
    ALL_TEMPLATES_Y_LIM_BUFFER = all_templates_y_lim_buffer
    ALL_TEMPLATES_WINDOW_MS_BEFORE_PEAK = all_templates_window_ms_before_peak
    ALL_TEMPLATES_WINDOW_MS_AFTER_PEAK = all_templates_window_ms_after_peak
    ALL_TEMPLATES_LINE_MS_BEFORE_PEAK = all_templates_line_ms_before_peak
    ALL_TEMPLATES_LINE_MS_AFTER_PEAK = all_templates_line_ms_after_peak
    ALL_TEMPLATES_X_LABEL = all_templates_x_label

    # Set seed for reproducibility
    np.random.seed(1)

    if CREATE_FIGURES:
        if FIGURES_DPI is not None:
            mpl.rcParams["figure.dpi"] = FIGURES_DPI
        if FIGURES_FONT_SIZE is not None:
            mpl.rcParams["font.size"] = FIGURES_FONT_SIZE

    compiled_results_folder = Path(COMPILED_RESULTS_FOLDER)
    if COMPILE_ALL_RECORDINGS:
        if not compiled_results_folder.exists() or RECOMPILE_ALL_RECORDINGS:
            all_recs_compiler = Compiler()
            create_folder(compiled_results_folder)
        else:
            all_recs_compiler = "Skipping compiling results from all recordings because 'RECOMPILE_ALL_RECORDINGS' is set to False and already compiled"
    else:
        all_recs_compiler = "Skipping compiling results from all recordings because 'COMPILE_ALL_RECORDINGS' is set to False"

    if not (len(RECORDING_FILES) == len(INTERMEDIATE_FOLDERS) == len(RESULTS_FOLDERS)):
        raise ValueError(
            f"recording_files ({len(RECORDING_FILES)}), "
            f"intermediate_folders ({len(INTERMEDIATE_FOLDERS)}), and "
            f"results_folders ({len(RESULTS_FOLDERS)}) must all have "
            "the same length."
        )

    if CREATE_FIGURES and COMPILE_ALL_RECORDINGS:
        if len(SCATTER_RECORDING_COLORS) < len(RECORDING_FILES):
            raise ValueError(
                f"scatter_recording_colors has {len(SCATTER_RECORDING_COLORS)} "
                f"entries but there are {len(RECORDING_FILES)} recordings. "
                "Provide at least as many colors as recordings when "
                "compile_all_recordings is True."
            )

    spikedata_results = []
    for rec_path, inter_path, results_path in zip(
        RECORDING_FILES, INTERMEDIATE_FOLDERS, RESULTS_FOLDERS
    ):
        if isinstance(rec_path, BaseRecording):
            rec_loaded = rec_path
            if "file_path" in rec_loaded._kwargs:
                rec_path = rec_loaded._kwargs["file_path"]
            else:
                rec_path = rec_loaded._kwargs["file_paths"][0]
        else:
            rec_loaded = None
        rec_name = str(rec_path).split(r"/")[-1].split(".")[0]
        result = process_recording(
            rec_name, rec_path, inter_path, results_path, rec_loaded=rec_loaded
        )

        if isinstance(result, BaseException):
            continue

        if SAVE_RAW_PKL:
            sd_raw, sd_curated = result
        else:
            sd_curated = result

        # Save SpikeData as pickle
        import pickle

        results_path = Path(results_path)

        if SAVE_RAW_PKL:
            raw_pkl = results_path / "sorted_spikedata.pkl"
            with open(raw_pkl, "wb") as f:
                pickle.dump(sd_raw, f)
            print(f"Saved {sd_raw.N} raw units to {raw_pkl}")

        curated_pkl = results_path / "sorted_spikedata_curated.pkl"
        with open(curated_pkl, "wb") as f:
            pickle.dump(sd_curated, f)
        print(f"Saved {sd_curated.N} curated units to {curated_pkl}")

        # If the recording was concatenated from multiple files, split
        # back into per-epoch SpikeData objects with per-epoch templates.
        if sd_curated.metadata.get("rec_chunks_ms"):
            epoch_sds = sd_curated.split_epochs()
            spikedata_results.extend(epoch_sds)
        else:
            spikedata_results.append(sd_curated)

        if (
            not compiled_results_folder.exists() and delete_inter
        ):  # If not compiling all recording results
            shutil.rmtree(inter_path)

        if type(all_recs_compiler) == Compiler:
            all_recs_compiler.add_recording(rec_name, sd_curated)

    if compiled_results_folder.exists():
        with Tee(compiled_results_folder / "log.out", "w"):
            stopwatch = Stopwatch("COMPILING DATA FROM ALL RECORDINGS")
            if isinstance(all_recs_compiler, Compiler):
                all_recs_compiler.save_results(compiled_results_folder)
                print_stage("DONE COMPILING DATA FROM ALL RECORDINGS")
                stopwatch.log_time()
            else:
                print(all_recs_compiler)
        if delete_inter:
            for inter_path in INTERMEDIATE_FOLDERS:
                shutil.rmtree(inter_path)

    return spikedata_results


def sort_maxtwo_multiwell(recording, stream_ids, **kwargs):
    """Sort a MaxTwo multi-well recording across multiple stream IDs.

    Calls ``sort_with_kilosort2`` once per stream ID, routing each
    stream to its own intermediate and results folders. All other
    parameters are forwarded unchanged.

    Parameters:
        recording (str or Path): Path to a single MaxTwo ``.raw.h5``
            file or a directory of ``.raw.h5`` files. When a directory
            is given, all files inside it are concatenated per stream
            (same behavior as ``sort_with_kilosort2`` with a directory
            entry in ``recording_files``).
        stream_ids (list of str): Stream identifiers to sort, e.g.
            ``["well000", "well001", "well002"]``. Each stream is
            sorted independently with its own Kilosort2 run.
        **kwargs: All remaining keyword arguments are forwarded to
            ``sort_with_kilosort2``. The following are handled
            specially:

            - ``intermediate_folders`` — if not provided, auto-generated
              with a ``_<stream_id>`` suffix per stream. If provided,
              must be None (auto-generated per stream) or omitted.
            - ``results_folders`` — if not provided, auto-generated
              with a ``_<stream_id>`` suffix per stream.
            - ``stream_id`` — **must not be provided**; it is set
              automatically per iteration.

    Returns:
        results (dict): ``{stream_id: list[SpikeData]}``. Each value
            is the list of SpikeData objects returned by
            ``sort_with_kilosort2`` for that stream. For a single file,
            each list contains one SpikeData. For a directory of N
            files, each list contains N SpikeData objects (one per
            original file, after epoch splitting).

    Notes:
        - Each stream is sorted as a completely independent run.
          Kilosort2 sees only the data from that stream.
        - Intermediate and results folders are kept separate by
          appending the stream ID as a suffix (e.g.
          ``sorted_kilosort2_well000/``).
        - The ``hdf5_plugin_path`` parameter (if needed for Maxwell
          new-format files) should be passed via ``**kwargs`` and is
          forwarded to each ``sort_with_kilosort2`` call.
        - ``compile_all_recordings`` is not supported across streams.
          Set it to False (the default) or compile manually after
          sorting.
        - When *recording* is a directory of files, each file is
          concatenated per stream and sorted together. Before
          concatenation, channel count and sampling frequency are
          validated (raises ``ValueError`` on mismatch). Mismatched
          channel IDs or locations produce warnings but do not block
          concatenation.

    Examples:
        Sort a single multi-well file::

            from spikelab.spike_sorting import sort_maxtwo_multiwell

            results = sort_maxtwo_multiwell(
                recording="multiwell.raw.h5",
                stream_ids=["well000", "well001", "well002"],
                kilosort_path="/opt/Kilosort2",
            )
            sd_well0 = results["well000"][0]  # SpikeData for well000

        Sort a directory of files per well (files are concatenated
        within each well, then split back into per-file SpikeData)::

            results = sort_maxtwo_multiwell(
                recording="recordings_dir/",
                stream_ids=["well000", "well001"],
                kilosort_path="/opt/Kilosort2",
            )
            # results["well000"] has one SpikeData per file in the dir
    """
    if "stream_id" in kwargs:
        raise ValueError(
            "Do not pass 'stream_id' to sort_maxtwo_multiwell — it is "
            "set automatically for each stream. Pass stream IDs via the "
            "'stream_ids' parameter instead."
        )

    recording = Path(recording)

    # Ensure intermediate/results folders are auto-generated per stream
    if kwargs.get("intermediate_folders") is not None:
        raise ValueError(
            "'intermediate_folders' cannot be specified for "
            "sort_maxtwo_multiwell — folders are auto-generated per "
            "stream ID to avoid collisions."
        )
    if kwargs.get("results_folders") is not None:
        raise ValueError(
            "'results_folders' cannot be specified for "
            "sort_maxtwo_multiwell — folders are auto-generated per "
            "stream ID to avoid collisions."
        )

    # Validate that all requested stream IDs exist in the recording(s)
    h5_files = []
    if recording.is_dir():
        from natsort import natsorted

        h5_files = [
            recording / name
            for name in natsorted(
                p.name for p in recording.iterdir() if p.name.endswith(".raw.h5")
            )
        ]
    elif str(recording).endswith(".h5"):
        h5_files = [recording]

    if h5_files:
        # Check the first file for available streams (all files in a
        # MaxTwo experiment share the same well layout)
        _, available_ids = MaxwellRecordingExtractor.get_streams(str(h5_files[0]))
        missing = [sid for sid in stream_ids if sid not in available_ids]
        if missing:
            raise ValueError(
                f"Stream ID(s) {missing} not found in {h5_files[0].name}. "
                f"Available streams: {available_ids}"
            )

    results = {}
    for sid in stream_ids:
        print_stage(f"SORTING STREAM: {sid}")

        # Build per-stream folder paths
        if recording.is_dir():
            base = recording
        else:
            base = recording.parent

        cur_datetime = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        inter = [base / f"inter_kilosort2_{sid}_{cur_datetime}"]
        res = [base / f"sorted_kilosort2_{sid}"]

        stream_results = sort_with_kilosort2(
            recording_files=[str(recording)],
            intermediate_folders=[str(p) for p in inter],
            results_folders=[str(p) for p in res],
            stream_id=sid,
            **kwargs,
        )
        results[sid] = stream_results

    return results
