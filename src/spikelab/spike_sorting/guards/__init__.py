"""System-crash safeguards for the spike-sorting pipeline.

This subpackage contains the pre-loop and live guards that protect the
host workstation from being taken down by a sort. Major pieces:

Live watchdogs (daemon threads that poll a resource and abort on
threshold breach):
* :class:`HostMemoryWatchdog` — host RAM percentage.
* :class:`GpuMemoryWatchdog` — GPU VRAM + thermal + throttle reasons.
* :class:`DiskUsageWatchdog` — free disk on the intermediate volume.
* :class:`IOStallWatchdog` — read+write byte counter inactivity.
* :class:`LogInactivityWatchdog` — sorter log file inactivity.

Pre-loop checks:
* :func:`run_preflight` — free disk, RAM, VRAM, HDF5 plugin path,
  WSL2 config, RT-Sort projected disk, recording inputs, sorter
  dependency probes (KS2 / KS4 / Docker / RT-Sort), GPU device index,
  recording sample-rate window.

Lifecycle helpers:
* :func:`acquire_sort_lock` — concurrent-sort prevention.
* :func:`windows_job_object_cap` — kernel-level Windows memory cap.
* :func:`linux_cgroup_v2_memory_cap` — kernel-level Linux memory cap.
* :func:`prevent_system_sleep` — Windows / macOS / Linux sleep
  inhibitors.
* :func:`cleanup_temp_files` — sorter temp-file sweep on clean exit.
* :func:`append_audit_event` — JSONL events log.
* :func:`find_tripped_global_watchdog` — classified-error router for
  the per-recording catch site.

All associated exception types live in
:mod:`spikelab.spike_sorting._exceptions` so the full classified-error
hierarchy stays in one place.
"""

from ._audit import append_audit_event
from ._cgroup_cap import linux_cgroup_v2_memory_cap
from ._disk_watchdog import DiskExhaustionReport, DiskUsageWatchdog
from ._io_stall import IOStallWatchdog, get_active_io_stall_watchdog
from ._job_object import windows_job_object_cap
from ._power_state import prevent_system_sleep
from ._sort_lock import acquire_sort_lock
from ._tempfile_cleanup import cleanup_temp_files
from ._gpu_watchdog import (
    GpuMemoryWatchdog,
    capture_gpu_snapshot,
    get_active_gpu_watchdog,
    read_gpu_memory,
    resolve_active_device,
)
from ._inactivity import (
    LogInactivityWatchdog,
    compute_inactivity_timeout_s,
    get_active_inactivity_timeout_s,
    get_active_log_path,
    make_in_process_kill_callback,
    set_active_inactivity_timeout_s,
    set_active_log_path,
)
from ._preflight import (
    PreflightFinding,
    estimate_rt_sort_intermediate_gb,
    report_findings,
    run_preflight,
)
from ._watchdog import HostMemoryWatchdog, get_active_watchdog


def find_tripped_global_watchdog():
    """Return the first tripped global-scope watchdog, or ``None``.

    Walks the watchdogs whose lifetime spans the whole sort
    (host memory, GPU memory + thermal) in priority order and
    returns the first one whose ``tripped()`` is True. The caller
    typically uses the return value to convert a
    :class:`KeyboardInterrupt` (delivered by the watchdog's
    ``_thread.interrupt_main`` call) into the corresponding
    classified error via ``make_error()``.

    Per-recording watchdogs (disk, I/O stall, log inactivity) are
    not consulted here — those are handled at narrower catch
    sites where the local watchdog reference is in scope.

    Returns:
        watchdog: A tripped watchdog instance, or ``None`` when no
            global-scope watchdog has fired.
    """
    host = get_active_watchdog()
    if host is not None and host.tripped():
        return host
    gpu = get_active_gpu_watchdog()
    if gpu is not None and gpu.tripped():
        return gpu
    io = get_active_io_stall_watchdog()
    if io is not None and io.tripped():
        return io
    return None


__all__ = [
    "HostMemoryWatchdog",
    "get_active_watchdog",
    "LogInactivityWatchdog",
    "compute_inactivity_timeout_s",
    "get_active_log_path",
    "set_active_log_path",
    "get_active_inactivity_timeout_s",
    "set_active_inactivity_timeout_s",
    "make_in_process_kill_callback",
    "DiskUsageWatchdog",
    "DiskExhaustionReport",
    "acquire_sort_lock",
    "windows_job_object_cap",
    "linux_cgroup_v2_memory_cap",
    "append_audit_event",
    "IOStallWatchdog",
    "get_active_io_stall_watchdog",
    "cleanup_temp_files",
    "prevent_system_sleep",
    "GpuMemoryWatchdog",
    "get_active_gpu_watchdog",
    "capture_gpu_snapshot",
    "read_gpu_memory",
    "resolve_active_device",
    "find_tripped_global_watchdog",
    "PreflightFinding",
    "run_preflight",
    "report_findings",
    "estimate_rt_sort_intermediate_gb",
]
