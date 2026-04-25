"""Kilosort2-specific MATLAB runner and Docker sorting entry points."""

import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from math import ceil
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Union

import numpy as np
from natsort import natsorted
from scipy.io import savemat

from spikeinterface.core import BaseRecording, write_binary_recording
from spikeinterface.extractors.extractor_classes import BinaryRecordingExtractor
from spikeinterface.sorters import run_sorter

from . import _globals
from ._classifier import classify_ks2_failure
from ._exceptions import InsufficientActivityError, SpikeSortingClassifiedError
from .docker_utils import get_docker_image
from .sorting_extractor import KilosortSortingExtractor
from .sorting_utils import Stopwatch, create_folder, print_stage


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
        self.path = self.set_kilosort_path(_globals.KILOSORT_PATH)

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
            projection_threshold=_globals.KILOSORT_PARAMS["projection_threshold"],
            preclust_threshold=_globals.KILOSORT_PARAMS["preclust_threshold"],
            minfr_goodchannels=_globals.KILOSORT_PARAMS["minfr_goodchannels"],
            minFR=_globals.KILOSORT_PARAMS["minFR"],
            freq_min=_globals.KILOSORT_PARAMS["freq_min"],
            sigmaMask=_globals.KILOSORT_PARAMS["sigmaMask"],
            kilo_thresh=_globals.KILOSORT_PARAMS["detect_threshold"],
            use_car=_globals.KILOSORT_PARAMS["car"],
            nPCs=int(_globals.KILOSORT_PARAMS["nPCs"]),
            ntbuff=int(_globals.KILOSORT_PARAMS["ntbuff"]),
            nfilt_factor=int(_globals.KILOSORT_PARAMS["nfilt_factor"]),
            NT=int(_globals.KILOSORT_PARAMS["NT"]),
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
        caught_exception: Optional[BaseException] = None
        try:
            self.execute_kilosort_file(output_folder, verbose)
            t1 = time.perf_counter()
            run_time = float(t1 - t0)
            has_error = False
        except Exception as err:
            has_error = True
            run_time = None
            caught_exception = err

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
            classified = classify_ks2_failure(
                output_folder, caught_exception or RuntimeError("unknown")
            )
            if classified is not None:
                raise classified from caught_exception
            raise Exception(
                f"You can inspect the runtime trace in {output_folder}/kilosort2.log"
            ) from caught_exception

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
        if _globals.KILOSORT_PARAMS["NT"] is None:
            _globals.KILOSORT_PARAMS["NT"] = (
                64 * 1024 + _globals.KILOSORT_PARAMS["ntbuff"]
            )  # https://github.com/MouseLand/Kilosort/issues/380
        else:
            _globals.KILOSORT_PARAMS["NT"] = (
                _globals.KILOSORT_PARAMS["NT"] // 32 * 32
            )  # make sure is multiple of 32

        if _globals.KILOSORT_PARAMS["car"]:
            _globals.KILOSORT_PARAMS["car"] = 1
        else:
            _globals.KILOSORT_PARAMS["car"] = 0

    @classmethod
    def get_result_from_folder(cls, output_folder):
        return KilosortSortingExtractor(
            folder_path=output_folder,
            keep_good_only=bool(
                _globals.KILOSORT_PARAMS
                and _globals.KILOSORT_PARAMS.get("keep_good_only")
            ),
            pos_peak_thresh=_globals.POS_PEAK_THRESH,
        )


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
        except subprocess.TimeoutExpired:
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
            except subprocess.TimeoutExpired:
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
        except subprocess.TimeoutExpired:
            print("WARNING: unable to kill shell script.")

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
        except subprocess.TimeoutExpired:
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
            except OSError:
                if retry_num < num_retries:
                    print("Retrying to remove directory: {}".format(dirname))
                    time.sleep(delay_between_tries)
                else:
                    raise Exception(
                        "Unable to remove directory after {} tries: {}".format(
                            num_retries, dirname
                        )
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
    if _globals.USE_PARALLEL_PROCESSING_FOR_RAW_CONVERSION:
        job_kwargs = {
            "progress_bar": verbose,
            "verbose": verbose,
            "n_jobs": _globals.N_JOBS,
            "total_memory": _globals.TOTAL_MEMORY,
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
    si_params = {k: v for k, v in _globals.KILOSORT_PARAMS.items()}

    print("Running Kilosort2 via Docker container")

    # Inject MW_CUDA_FORWARD_COMPATIBILITY=1 into the Docker container so
    # that the compiled MATLAB Runtime supports newer GPU architectures
    # (e.g. RTX 5090 / compute capability 12.0), and cap container memory
    # to 80% of system RAM to prevent OOM crashes.
    from .docker_utils import patched_container_client

    with patched_container_client(
        extra_env={"MW_CUDA_FORWARD_COMPATIBILITY": "1"},
        mem_limit_frac=0.8,
    ):
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
        except Exception as err:
            classified = classify_ks2_failure(Path(output_folder), err)
            if classified is not None:
                raise classified from err
            raise

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

    return KilosortSortingExtractor(
        folder_path=sorter_output,
        keep_good_only=bool(
            _globals.KILOSORT_PARAMS and _globals.KILOSORT_PARAMS.get("keep_good_only")
        ),
        pos_peak_thresh=_globals.POS_PEAK_THRESH,
    )


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
        if (
            not _globals.RECOMPUTE_SORTING
            and (output_folder / "spike_times.npy").exists()
        ):
            print("Loading Kilosort2's sorting results")
            sorting = KilosortSortingExtractor(
                folder_path=output_folder,
                keep_good_only=bool(
                    _globals.KILOSORT_PARAMS
                    and _globals.KILOSORT_PARAMS.get("keep_good_only")
                ),
                pos_peak_thresh=_globals.POS_PEAK_THRESH,
            )
        elif _globals.USE_DOCKER:
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

    except SpikeSortingClassifiedError:
        # Classified failures (biology / environment / resource) propagate
        # so callers can implement per-category retry / skip / stop policies
        # without inspecting a returned sentinel.
        raise
    except Exception as e:
        print(f"Kilosort2 failed on recording {rec_path}\n{e}")
        print("Moving on to next recording")
        return e

    stopwatch.log_time("Done sorting.")
    print(f"Kilosort detected {len(sorting.unit_ids)} units")
    return sorting
