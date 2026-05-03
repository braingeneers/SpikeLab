"""System-crash safeguards for the spike-sorting pipeline.

This subpackage contains the pre-loop and live guards that protect the
host workstation from being taken down by a sort. The two main pieces:

* :class:`HostMemoryWatchdog` — daemon-thread monitor that polls
  system memory and aborts the run (and any registered subprocesses)
  before the OS starts thrashing.
* :func:`run_preflight` — pre-loop checks for free disk, available
  RAM, free GPU memory, and HDF5 plugin path validity.

The associated exception type
(:class:`spikelab.spike_sorting._exceptions.HostMemoryWatchdogError`)
is intentionally kept in the spike-sorting top-level
``_exceptions`` module so the full classified-error hierarchy stays in
one place.
"""

from ._audit import append_audit_event
from ._disk_watchdog import DiskExhaustionReport, DiskUsageWatchdog
from ._io_stall import IOStallWatchdog
from ._job_object import windows_job_object_cap
from ._power_state import prevent_system_sleep
from ._sort_lock import acquire_sort_lock
from ._tempfile_cleanup import cleanup_temp_files
from ._gpu_watchdog import (
    GpuMemoryWatchdog,
    capture_gpu_snapshot,
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
    "append_audit_event",
    "IOStallWatchdog",
    "cleanup_temp_files",
    "prevent_system_sleep",
    "GpuMemoryWatchdog",
    "capture_gpu_snapshot",
    "read_gpu_memory",
    "resolve_active_device",
    "PreflightFinding",
    "run_preflight",
    "report_findings",
    "estimate_rt_sort_intermediate_gb",
]
