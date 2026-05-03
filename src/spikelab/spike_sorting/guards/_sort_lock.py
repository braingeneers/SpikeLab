"""Per-recording lock to prevent concurrent sorts on the same folder.

Two concurrent calls to ``sort_recording`` against the same
intermediate folder will corrupt each other's KS2 ``.dat`` file,
RT-Sort scaled traces, curation cache, and waveform extraction
output — usually silently, sometimes catastrophically.
:func:`acquire_sort_lock` prevents this by atomically creating a
lock file at the start of the sort and detecting a pre-existing
lock at entry.

The lock file is JSON with the holding process's PID, hostname,
and start timestamp. On entry: if the lock exists and the recorded
PID is alive on the current host, the new sort aborts with a clear
:class:`spikelab.spike_sorting._exceptions.ConcurrentSortError`. If
the PID is dead (sort crashed), the lock is reclaimed and the new
sort proceeds. If the hostname differs (multi-host sharing — rare
for spike sorting), the lock is treated as alive to be conservative.

Atomic creation uses ``os.O_EXCL`` so the check-then-write pattern
is race-free even when two sorts start simultaneously.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .._exceptions import ConcurrentSortError

_LOCK_FILENAME = ".spikelab_sort.lock"


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process on this host.

    Tries ``psutil.pid_exists`` first (most reliable); falls back to
    POSIX ``os.kill(pid, 0)`` and Windows ``OpenProcess`` via
    ``ctypes`` as a last resort. Returns ``True`` (conservative)
    when neither method works — better to refuse a sort that might
    race than to clobber a live one.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:
        pass

    if hasattr(os, "kill"):
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it.
            return True
        except OSError:
            return True

    # Windows without psutil — assume alive to be conservative.
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle == 0:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return True


def _read_lock_info(lock_path: Path) -> Optional[dict]:
    """Read and parse a lock file, returning ``None`` on any error."""
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@contextlib.contextmanager
def acquire_sort_lock(folder: Path) -> Iterator[Path]:
    """Acquire an exclusive sort lock on *folder*; yield the lock path.

    Atomically creates ``<folder>/.spikelab_sort.lock`` containing
    the holding PID, hostname, and ISO start timestamp. If the lock
    already exists:

    * If the recorded PID is alive on this host, raises
      :class:`ConcurrentSortError`.
    * If the recorded hostname differs from this host, raises
      :class:`ConcurrentSortError` (we cannot verify the holder).
    * If the recorded PID is dead on this host (stale lock from a
      crashed sort), the lock is reclaimed and the new sort
      proceeds.
    * If the lock file is unparseable, raises
      :class:`ConcurrentSortError` and instructs the user to remove
      it manually.

    On normal exit (successful or failing sort), the lock file is
    deleted. ``os._exit`` paths leave the lock behind, which is
    correctly reclaimed on the next sort because the holding PID
    will no longer be alive.

    Parameters:
        folder (Path): The folder to lock — typically the
            per-recording intermediate folder.

    Yields:
        lock_path (Path): The path to the lock file.

    Raises:
        ConcurrentSortError: When another sort is already running
            against *folder*, or when the lock file is unparseable
            and a stale-lock reclaim is unsafe.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / _LOCK_FILENAME
    this_host = socket.gethostname()

    fd: Optional[int] = None
    while True:
        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            info = _read_lock_info(lock_path)
            if info is None:
                raise ConcurrentSortError(
                    f"Lock file at {lock_path} is unparseable. Remove "
                    "it manually if you are sure no sort is running.",
                    lock_path=lock_path,
                )
            holder_pid = int(info.get("pid", -1)) if info.get("pid") else -1
            holder_host = str(info.get("hostname", "")) or None
            started_at = info.get("started_at")
            same_host = holder_host == this_host

            if not same_host:
                raise ConcurrentSortError(
                    f"Lock file at {lock_path} held by PID {holder_pid} on "
                    f"host {holder_host!r} (this host: {this_host!r}). "
                    "Cannot verify liveness across hosts. Remove the lock "
                    "manually if you are sure the holder is dead.",
                    lock_path=lock_path,
                    holder_pid=holder_pid,
                    holder_hostname=holder_host,
                    started_at=started_at,
                )

            if _pid_alive(holder_pid):
                raise ConcurrentSortError(
                    f"Another sort is already running on {folder} "
                    f"(PID {holder_pid}, started {started_at}). Wait for "
                    "it to finish, or use a different intermediate "
                    "folder.",
                    lock_path=lock_path,
                    holder_pid=holder_pid,
                    holder_hostname=holder_host,
                    started_at=started_at,
                )

            # Stale lock: previous sort crashed. Reclaim.
            print(
                f"[sort lock] reclaiming stale lock at {lock_path} "
                f"(holder PID {holder_pid} no longer alive)."
            )
            try:
                lock_path.unlink()
            except OSError as exc:
                raise ConcurrentSortError(
                    f"Could not remove stale lock at {lock_path}: {exc!r}",
                    lock_path=lock_path,
                ) from exc
            # Loop and retry the create — another fresh sort may
            # race with us at this point and we want one of the two
            # to win cleanly.
            continue
        else:
            break

    assert fd is not None
    try:
        payload = {
            "pid": os.getpid(),
            "hostname": this_host,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
    finally:
        os.close(fd)

    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except OSError:
            # Best-effort: a watchdog os._exit may have prevented
            # cleanup. The next sort will reclaim via stale-lock
            # detection.
            pass
