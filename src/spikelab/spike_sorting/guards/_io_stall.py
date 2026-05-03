"""Disk I/O stall watchdog.

Network-mounted recordings (S3-fuse, NFS, SMB) can hang at the
kernel level while still accepting file handles. The :class:`IOStallWatchdog`
polls ``psutil.disk_io_counters()`` for the volume holding the
intermediate folder; when read+write byte counters stop changing
for the configured tolerance window, it trips just like the other
watchdogs.

This complements but does not replace the inactivity watchdog —
the inactivity watchdog tracks log-file mtime, which catches
sorters that go silent. The I/O stall watchdog catches sorters
that keep printing while waiting for hung kernel I/O.

Detection requires ``psutil``. On platforms or filesystems where
``disk_io_counters(perdisk=True)`` does not expose the relevant
device, the watchdog reports as disabled and yields a no-op.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .._exceptions import IOStallError


def _resolve_device_for_path(path: Path) -> Optional[str]:
    """Return the device (e.g. ``"sda1"`` / ``"C:"``) for *path*.

    Best-effort: ``psutil.disk_partitions`` to find the longest
    mountpoint prefix. Returns ``None`` when no match.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:
        return None
    best: Optional[Tuple[int, str]] = None
    target = str(Path(path).resolve()).lower()
    for part in partitions:
        mp = str(Path(part.mountpoint).resolve()).lower()
        if not mp:
            continue
        if target == mp or target.startswith(
            mp.rstrip("/\\") + ("/" if "/" in mp else "\\")
        ):
            length = len(mp)
            if best is None or length > best[0]:
                best = (length, part.device)
    if best is None:
        return None
    # Strip path prefix so it matches disk_io_counters keys.
    dev = best[1]
    # Linux: ``/dev/sda1`` → ``sda1``. Windows: ``C:\`` → ``C:``.
    return dev.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rstrip(":\\") + (
        ":" if dev.endswith(":\\") or dev.endswith(":") else ""
    )


def _read_io_bytes(device: str) -> Optional[int]:
    """Return read+write byte total for *device* via psutil.

    Returns ``None`` when the device cannot be found in
    ``disk_io_counters(perdisk=True)``.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        counters = psutil.disk_io_counters(perdisk=True)
    except Exception:
        return None
    if counters is None:
        return None
    info = counters.get(device)
    if info is None:
        # Try without trailing colon (Windows) or with sda partition
        # stripped (Linux: ``sda1`` may aggregate under ``sda``).
        for key in (
            device.rstrip(":"),
            device + ":",
        ):
            if key in counters:
                info = counters[key]
                break
    if info is None:
        return None
    try:
        return int(info.read_bytes) + int(info.write_bytes)
    except Exception:
        return None


class IOStallWatchdog:
    """Daemon-thread watchdog that aborts the sort on disk I/O stalls.

    Use as a context manager around the per-recording sort. Polls
    ``read_bytes + write_bytes`` for the volume holding *folder*
    every ``poll_interval_s``. If the counter does not change for
    ``stall_s`` seconds, builds an :class:`IOStallError`,
    terminates registered subprocesses, runs kill callbacks, and
    raises into the main thread via ``_thread.interrupt_main``.

    Parameters:
        folder (Path): A path on the volume to monitor (typically
            the per-recording intermediate folder).
        stall_s (float): Inactivity tolerance for the byte
            counter, in seconds. Defaults to ``300`` (5 min) —
            long enough to span normal write bursts and quiet
            stretches, short enough to flag genuinely hung mounts.
        poll_interval_s (float): Seconds between polls. Defaults
            to ``10.0``.
        warn_repeat_s (float): Minimum seconds between repeated
            warnings.
        kill_grace_s (float): Seconds between ``terminate()`` and
            ``kill()`` for registered subprocesses.

    Notes:
        - Disabled (no-op) when ``psutil`` is missing, when no
          device can be resolved for *folder*, or when ``stall_s``
          is non-positive.
    """

    def __init__(
        self,
        folder: Path,
        *,
        stall_s: float = 300.0,
        poll_interval_s: float = 10.0,
        warn_repeat_s: float = 60.0,
        kill_grace_s: float = 5.0,
    ) -> None:
        if poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be positive, got {poll_interval_s}."
            )
        self.folder = Path(folder)
        self.stall_s = float(stall_s)
        self.poll_interval_s = float(poll_interval_s)
        self.warn_repeat_s = float(warn_repeat_s)
        self.kill_grace_s = float(kill_grace_s)

        self._subprocesses: List[Tuple[object, float]] = []
        self._kill_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped = False
        self._stall_at_trip: Optional[float] = None
        self._device: Optional[str] = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Trip-state queries
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True once the watchdog has fired its abort path."""
        return self._tripped

    def device(self) -> Optional[str]:
        """Return the resolved device identifier (e.g. "sda1")."""
        return self._device

    def make_error(self, message: Optional[str] = None) -> IOStallError:
        """Build an :class:`IOStallError` from the trip state."""
        if message is None:
            stall = self._stall_at_trip
            stall_str = f"{stall:.1f}" if stall is not None else "?"
            message = (
                f"I/O on device {self._device!r} stalled for {stall_str}s "
                f"(tolerance: {self.stall_s:.1f}s). Likely a hung "
                "network mount or unresponsive storage. Sort aborted."
            )
        return IOStallError(
            message,
            device=self._device,
            stall_s=self.stall_s,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_kill_callback(self, callback: Callable[[], None]) -> None:
        """Track a zero-arg callable to invoke on watchdog abort."""
        with self._lock:
            self._kill_callbacks.append(callback)

    def unregister_kill_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._kill_callbacks = [
                c for c in self._kill_callbacks if c is not callback
            ]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "IOStallWatchdog":
        if self.stall_s <= 0:
            self._enabled = False
            return self
        device = _resolve_device_for_path(self.folder)
        if device is None:
            self._enabled = False
            return self
        # Probe once to confirm we can read counters for the device.
        if _read_io_bytes(device) is None:
            self._enabled = False
            return self
        self._device = device
        self._enabled = True
        print(
            f"[io stall watchdog] active: device={device} "
            f"folder={self.folder} stall_s={self.stall_s:.1f} "
            f"poll={self.poll_interval_s:.1f}s"
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"IOStallWatchdog[{device}]",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s + 1.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Polling loop: warn, then trip, then exit."""
        if self._stop_event.wait(self.poll_interval_s):
            return
        last_bytes = _read_io_bytes(self._device or "")
        last_change_t = time.time()
        last_warn_t = 0.0
        while not self._stop_event.is_set():
            current = _read_io_bytes(self._device or "")
            now = time.time()
            if current is None:
                self._stop_event.wait(self.poll_interval_s)
                continue
            if last_bytes is None:
                last_bytes = current
                last_change_t = now
            elif current != last_bytes:
                last_bytes = current
                last_change_t = now
            stalled_for = now - last_change_t
            if stalled_for >= self.stall_s:
                self._on_trip(stalled_for)
                return
            if (
                stalled_for >= self.stall_s * 0.5
                and now - last_warn_t >= self.warn_repeat_s
            ):
                last_warn_t = now
                self._maybe_warn(stalled_for)
            self._stop_event.wait(self.poll_interval_s)

    def _maybe_warn(self, stalled_for: float) -> None:
        print(
            f"[io stall watchdog] WARNING: device {self._device!r} "
            f"idle for {stalled_for:.1f}s (will abort at "
            f"{self.stall_s:.1f}s)."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="io_stall",
                event="warn",
                device=self._device,
                stalled_for_s=stalled_for,
                tolerance_s=self.stall_s,
            )
        except Exception:
            pass

    def _on_trip(self, stalled_for: float) -> None:
        self._tripped = True
        self._stall_at_trip = stalled_for
        print(
            f"[io stall watchdog] TRIP: device {self._device!r} stalled "
            f"for {stalled_for:.1f}s (>= {self.stall_s:.1f}s). "
            "Aborting sort."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="io_stall",
                event="abort",
                device=self._device,
                stalled_for_s=stalled_for,
                tolerance_s=self.stall_s,
            )
        except Exception:
            pass
        with self._lock:
            callbacks = list(self._kill_callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                print(
                    f"[io stall watchdog] kill_callback raised: {exc!r}; " "continuing."
                )
        try:
            import _thread as _t

            _t.interrupt_main()
        except Exception as exc:
            print(f"[io stall watchdog] failed to interrupt main: {exc}")
