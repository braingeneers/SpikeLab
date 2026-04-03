"""Shared utility classes and functions for the spike sorting pipeline.

These are used by both ``kilosort2.py`` and ``pipeline.py`` and live
in this separate module to avoid circular imports.
"""

import datetime
import os
import shutil
import sys
import time
import warnings
from pathlib import Path


def print_stage(text):
    """Print a centered banner message framed by ``=`` lines.

    Parameters:
        text: Message to display (converted to string if not already).
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

    def __init__(self, file_path, file_mode="a"):
        self.file_path = Path(file_path)
        self.file_mode = file_mode
        self.log_file = None
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        self.log_file = open(self.file_path, self.file_mode)
        sys.stdout = self._TeeWriter(self._original_stdout, self.log_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.log_file.write(f"\nException: {exc_val}\n")
        sys.stdout = self._original_stdout
        self.log_file.close()
        return False

    class _TeeWriter:
        def __init__(self, *writers):
            self.writers = writers

        def write(self, data):
            for w in self.writers:
                w.write(data)

        def flush(self):
            for w in self.writers:
                w.flush()


def create_folder(folder, parents=True):
    """Create a directory if it does not already exist.

    Parameters:
        folder (str or Path): Directory path to create.
        parents (bool): Create parent directories as needed.
    """
    Path(folder).mkdir(parents=parents, exist_ok=True)


def delete_folder(folder):
    """Delete a directory tree if it exists.

    Parameters:
        folder (str or Path): Directory path to delete.
    """
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)


def get_paths(rec_path, inter_path, results_path, execution_config=None):
    """Resolve and prepare all directory paths for one recording run.

    Derives paths for the binary ``.dat`` file, sorter output,
    waveforms, curation stages, and final results.  Optionally deletes
    stale intermediate folders based on ``execution_config`` recompute
    flags.

    Parameters:
        rec_path (str or Path): Path to the recording file.
        inter_path (str or Path): Root intermediate directory.
        results_path (str or Path): Root results directory.
        execution_config (ExecutionConfig or None): When provided, its
            ``recompute_*`` flags control which intermediate folders
            are deleted before running.

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

    # Delete stale intermediate folders based on recompute flags
    if execution_config is not None:
        exe = execution_config
        delete_folders = []
        if exe.recompute_recording:
            delete_folders.extend(
                (
                    recording_dat_path,
                    output_folder,
                    waveforms_root_folder,
                    curation_folder,
                )
            )
        if exe.recompute_sorting:
            delete_folders.extend((output_folder, waveforms_root_folder))
        if exe.reextract_waveforms:
            delete_folders.append(waveforms_root_folder)
            delete_folders.append(curation_folder)
        if exe.recurate_first:
            delete_folders.append(curation_first_folder)
            delete_folders.append(curation_second_folder)
        if exe.recurate_second:
            delete_folders.append(curation_second_folder)
        for folder in delete_folders:
            delete_folder(folder)

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
