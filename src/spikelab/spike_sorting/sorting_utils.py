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
from typing import Any, Optional, Tuple, Union


def get_system_ram_bytes() -> Optional[int]:
    """Return total system physical RAM in bytes, or None if unavailable.

    Tries POSIX ``os.sysconf`` first, then ``psutil`` if installed,
    then the Windows ``GlobalMemoryStatusEx`` API via ctypes.

    Returns:
        ram_bytes (int or None): Total physical RAM in bytes, or None
            if no detection method succeeds.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass

    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            pass

    return None


def print_stage(text: Any) -> None:
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

    def __init__(
        self, start_msg: Optional[str] = None, use_print_stage: bool = True
    ) -> None:
        if start_msg is not None:
            if use_print_stage:
                print_stage(start_msg)
            else:
                print(start_msg)

        self._time_start = time.time()

    def log_time(self, text: Optional[str] = None) -> None:
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

    def __init__(self, file_path: Union[str, Path], file_mode: str = "a") -> None:
        from types import MethodType

        _file = open(file_path, file_mode)
        _file.stdout = sys.stdout
        _file.file_write = _file.write
        _file.write = MethodType(Tee._write, _file)
        self._file = _file

    def __enter__(self) -> Any:
        sys.stdout = self._file
        return self._file

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        import traceback

        if exc_type:
            self._file.write = self._file.file_write
            print("Traceback (most recent call last):")
            traceback.print_tb(exc_tb, file=self._file)
            print(f"{exc_type}: {exc_val}")
        sys.stdout = self._file.stdout
        self._file.close()

    @staticmethod
    def _write(self, s: str) -> None:
        self.file_write(s)
        if s != "\n" and s != " ":
            print(s, file=self.stdout)


def create_folder(folder: Union[str, Path], parents: bool = True) -> None:
    """Create a directory if it does not already exist.

    Parameters:
        folder (str or Path): Directory path to create.
        parents (bool): Create parent directories as needed (default True).
    """
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=parents)
        print(f"Created folder: {folder}")


def delete_folder(folder: Union[str, Path]) -> None:
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


def get_paths(
    rec_path: Any, inter_path: Any, results_path: Any, execution_config: Any = None
) -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
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
