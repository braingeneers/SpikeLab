"""Windows Job Object memory cap for the sort process.

On Linux, ``_bounded_host_memory`` uses ``RLIMIT_DATA`` to install a
kernel-enforced cap on anonymous heap allocations. On Windows the
equivalent is a `Job Object`_ assigned to the current process — the
kernel monitors and terminates the process if its committed memory
exceeds the configured ``ProcessMemoryLimit``. This is sharper and
more reliable than the userspace :class:`HostMemoryWatchdog`,
which depends on a daemon thread and ``_thread.interrupt_main``.

.. _Job Object: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects

The cap is a complement to (not a replacement for) the watchdog:

* The Job Object cap is the **hard kernel limit** — when the
  process tries to commit beyond it, the OS terminates immediately.
* The watchdog still provides early **warn** signals at 85% and a
  graceful **abort** at 92%, with diagnostic capture before the
  hard limit is hit.

Best-effort: the cap is silently a no-op on non-Windows hosts and
when ``pywin32`` is missing. On Windows with pywin32 installed it
is the most reliable way to bound a Python sort process to a
fraction of host RAM.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Iterator, Optional


def _get_total_ram_bytes() -> Optional[int]:
    """Return total host RAM in bytes via ``get_system_ram_bytes``."""
    try:
        from ..sorting_utils import get_system_ram_bytes

        return get_system_ram_bytes()
    except Exception:
        return None


@contextlib.contextmanager
def windows_job_object_cap(frac: float = 0.8) -> Iterator[bool]:
    """Bound the current Python process to *frac* of host RAM via a Job Object.

    On Windows, creates an anonymous Job Object, attaches the
    current process to it, and sets ``ProcessMemoryLimit`` to
    ``frac * total_host_ram``. The Windows kernel terminates the
    process if its committed memory (private bytes) exceeds the
    cap.

    On non-Windows or when ``pywin32`` is missing, the context
    manager is a no-op and yields ``False``.

    Parameters:
        frac (float): Fraction of total host RAM to cap the process
            at. Defaults to ``0.8`` (80%).

    Yields:
        active (bool): ``True`` when the cap was successfully
            installed; ``False`` otherwise. Useful for telling
            the user whether the kernel-enforced limit is in
            effect.

    Notes:
        - Job Objects are inherited by child processes spawned
          from this one, so a Kilosort2 MATLAB child or RT-Sort's
          multiprocessing workers will share the cap.
        - The Job Object is destroyed when the last handle is
          closed (on context exit or process death), unbinding the
          process from the cap automatically.
    """
    if sys.platform != "win32":
        yield False
        return

    ram_bytes = _get_total_ram_bytes()
    if ram_bytes is None or ram_bytes <= 0:
        print("[job object] could not detect host RAM; cap not enforced.")
        yield False
        return

    cap_bytes = int(ram_bytes * float(frac))

    try:
        import win32job  # noqa: WPS433
        import win32api  # noqa: WPS433
        import win32con  # noqa: WPS433
    except ImportError:
        print(
            "[job object] pywin32 not installed; Windows Job Object cap "
            "is unavailable. Install pywin32 to enable the kernel-"
            "enforced memory cap (the userspace watchdog still works)."
        )
        yield False
        return

    try:
        # Create an anonymous Job Object and assign the current
        # process to it.
        job = win32job.CreateJobObject(None, "")
        process = win32api.GetCurrentProcess()
        win32job.AssignProcessToJobObject(job, process)

        # Set ProcessMemoryLimit. The basic+extended limit info
        # struct accepts both a per-process and a per-job cap; we
        # use per-process so the cap follows the original Python
        # process even if it spawns children.
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"][
            "LimitFlags"
        ] |= win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info["ProcessMemoryLimit"] = cap_bytes
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info
        )
        print(
            f"[job object] active: cap = {cap_bytes / 1e9:.1f} GB "
            f"(= {frac * 100:.0f}% of {ram_bytes / 1e9:.1f} GB host RAM)."
        )
    except Exception as exc:
        print(f"[job object] failed to install cap: {exc!r}")
        yield False
        return

    try:
        yield True
    finally:
        # Releasing the Job Object handle destroys the object and
        # detaches the process. Best-effort cleanup; failures are
        # cosmetic since process exit cleans up regardless.
        try:
            win32api.CloseHandle(job)
        except Exception:
            pass
