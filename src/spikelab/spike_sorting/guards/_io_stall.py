"""Disk I/O stall watchdog.

Network-mounted recordings (S3-fuse, NFS, SMB) can hang at the
kernel level while still accepting file handles. The :class:`IOStallWatchdog`
detects stalls by polling read+write byte counters at one of two
scopes:

* **Device mode** (``folder=...``): polls
  ``psutil.disk_io_counters(perdisk=True)`` for the volume holding
  the intermediate folder. Catches kernel-wide I/O hangs that
  freeze the entire device, but background activity from any
  other process on the same disk can mask a stall in the sort
  process specifically.
* **Process mode** (``pids=[...]``): polls
  ``psutil.Process(pid).io_counters()`` summed across the
  registered processes (and their descendants by default).
  Immune to ambient I/O on the same device — a stalled sort
  trips the watchdog even when other processes keep the disk
  busy. Use this when you want to detect hangs *specifically* in
  the sort process tree.

Either *folder* or *pids* (or both) must be provided. When both
are given, process mode wins.

This watchdog complements but does not replace the inactivity
watchdog — the inactivity watchdog tracks log-file mtime, which
catches sorters that go silent. The I/O stall watchdog catches
sorters that keep printing while waiting for hung kernel I/O.

Detection requires ``psutil``. On platforms or filesystems where
``disk_io_counters(perdisk=True)`` does not expose the relevant
device, device mode reports as disabled and yields a no-op;
process mode is unaffected because per-process I/O counters are
read from ``/proc/<pid>/io`` (Linux) or the equivalent on other
platforms.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .._exceptions import IOStallError
from ._audit import append_audit_event

_logger = logging.getLogger(__name__)

_active_io_stall_watchdog: contextvars.ContextVar[Optional["IOStallWatchdog"]] = (
    contextvars.ContextVar("active_io_stall_watchdog", default=None)
)


def get_active_io_stall_watchdog() -> Optional["IOStallWatchdog"]:
    """Return the I/O stall watchdog active for the current context, or None.

    Mirror of :func:`._watchdog.get_active_watchdog`. Lets the
    per-recording :class:`KeyboardInterrupt` catch site discover a
    tripped I/O stall watchdog and convert the interrupt into the
    appropriate :class:`IOStallError` rather than letting the raw
    interrupt bubble up.

    Returns:
        watchdog (IOStallWatchdog or None): The active instance, or
            ``None`` when no I/O stall watchdog is currently
            running.
    """
    return _active_io_stall_watchdog.get()


def _resolve_device_for_path(path: Path) -> Optional[str]:
    """Return the device (e.g. ``"sda1"`` / ``"C:"``) for *path*.

    Best-effort: ``psutil.disk_partitions`` to find the longest
    mountpoint prefix. Returns ``None`` when no match.
    """
    try:
        import psutil
    except ImportError:
        from .._psutil_warn import warn_psutil_missing_once

        warn_psutil_missing_once(_logger, "I/O stall watchdog")
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
    dev = best[1]
    # Map ``part.device`` to the key shape psutil's
    # ``disk_io_counters(perdisk=True)`` uses on each platform.
    # Windows: ``part.device`` is ``"C:\\"`` and the perdisk keys are
    # things like ``"C:"`` or ``"PhysicalDrive0"``. Strip the trailing
    # ``\\`` so the colon-suffixed drive form matches.
    # POSIX: ``part.device`` is ``"/dev/sda1"`` and the perdisk keys
    # are the basename (``"sda1"``).
    if sys.platform == "win32":
        return dev.rstrip("\\") if dev.endswith(":\\") else dev
    return dev.rsplit("/", 1)[-1]


def _read_io_bytes_for_pids(
    pids: Sequence[int],
    *,
    include_descendants: bool = True,
) -> Tuple[Optional[int], int]:
    """Sum read+write bytes across *pids* (and optionally descendants).

    Used by :class:`IOStallWatchdog` in process mode. Returns a
    ``(total_bytes, alive_count)`` pair:

    * ``total_bytes`` is the sum of ``read_bytes + write_bytes``
      across every reachable process. Returns ``None`` when none
      of the registered PIDs are alive (all dead → counter
      unreadable, watchdog goes blind rather than tripping on a
      vanished sort).
    * ``alive_count`` is the number of registered PIDs that
      were observed alive during this scan (excluding descendants).
      Useful for telemetry but not for the trip decision.

    Per-process ``io_counters()`` accumulates bytes for the
    process's lifetime — so the value only ever grows or stays
    flat, which is what the stall detector wants. ``AccessDenied``
    on individual processes is silently skipped (a single
    permission-denied PID should not blind the whole watchdog).

    Notes:
        - On Linux, the underlying counter is ``/proc/<pid>/io``
          (read_bytes / write_bytes). These count syscall I/O —
          NOT page-cache hits — so a sort that's reading from RAM
          cache will look idle. That matches the watchdog's
          intent: the watchdog cares about *device-bound* I/O
          progress, and a fully cached read is not device-bound.
        - Children that spawn between the parent enumeration and
          their ``io_counters()`` call may be missed in this scan
          but picked up in the next. Bounded blindness window =
          ``poll_interval_s``.
    """
    try:
        import psutil
    except ImportError:
        from .._psutil_warn import warn_psutil_missing_once

        warn_psutil_missing_once(_logger, "I/O stall watchdog")
        return None, 0

    total = 0
    alive = 0
    for pid in list(pids):
        try:
            proc = psutil.Process(pid)
            io = proc.io_counters()
            total += int(io.read_bytes) + int(io.write_bytes)
            alive += 1
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            # The PID is alive but we can't read its counters.
            # Treat as alive so the watchdog stays armed; the
            # process's I/O contribution is invisible to us, which
            # is a known limitation worth a debug log.
            alive += 1
            _logger.debug(
                "io_counters denied for pid=%d; skipping its "
                "contribution this poll.",
                pid,
            )
            continue
        except Exception as exc:
            # Defensive: an unexpected psutil failure on one PID
            # should not bring down the whole watchdog.
            _logger.debug("unexpected io_counters error for pid=%d: %r", pid, exc)
            continue

        if not include_descendants:
            continue
        try:
            children = proc.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        for child in children:
            try:
                cio = child.io_counters()
                total += int(cio.read_bytes) + int(cio.write_bytes)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
            except Exception as exc:
                _logger.debug(
                    "unexpected io_counters error for child pid=%d: " "%r",
                    child.pid,
                    exc,
                )
                continue
    if alive == 0:
        return None, 0
    return total, alive


def _read_io_bytes(device: str) -> Optional[int]:
    """Return read+write byte total for *device* via psutil.

    Returns ``None`` when the device cannot be found in
    ``disk_io_counters(perdisk=True)``.

    The fallback lookup handles Windows / POSIX device-name shape
    differences. ``psutil.disk_partitions()`` and
    ``disk_io_counters(perdisk=True)`` use slightly different
    conventions for the same device on Windows, so we try two
    normalisations after the direct lookup misses.

    Examples:
        Windows direct match::

            device = "C:"
            counters = {"C:": <iostat>, ...}
            # Direct lookup hits; no fallback needed.

        Windows fallback via ``device + ":"``::

            device = "C"  # caller resolved it without the colon
            counters = {"C:": <iostat>, ...}
            # Direct lookup misses; fallback ``"C" + ":"`` matches.

        Windows fallback via ``rstrip(":")``::

            device = "C:"  # caller has the colon
            counters = {"C": <iostat>, ...}
            # Direct lookup misses; fallback ``"C:".rstrip(":")``
            # → ``"C"`` matches.

        POSIX direct match::

            device = "sda1"
            counters = {"sda1": <iostat>, "sda": <iostat>, ...}
            # Direct lookup hits the partition entry.
    """
    try:
        import psutil
    except ImportError:
        from .._psutil_warn import warn_psutil_missing_once

        warn_psutil_missing_once(_logger, "I/O stall watchdog")
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
    """Daemon-thread watchdog that aborts the sort on I/O stalls.

    Use as a context manager around the per-recording sort.
    Operates in one of two modes (chosen at construction):

    * **Device mode** — pass *folder*: polls
      ``read_bytes + write_bytes`` for the volume holding the
      folder every ``poll_interval_s``. Catches kernel-wide I/O
      hangs but is sensitive to ambient I/O on the same disk.
    * **Process mode** — pass *pids*: polls
      ``psutil.Process(pid).io_counters()`` summed across the
      registered PIDs (and their descendants by default). Detects
      stalls in the sort process tree specifically; immune to
      ambient I/O from unrelated processes on the same device.

    Either *folder* or *pids* (or both) must be provided. When
    both are given, process mode is used. Additional PIDs can be
    registered after construction via :meth:`register_pid` —
    useful for catching e.g. a Docker container PID after the
    container actually starts.

    On stall, the watchdog builds an :class:`IOStallError`,
    terminates registered subprocesses, runs kill callbacks, and
    raises into the main thread via ``_thread.interrupt_main``.

    Parameters:
        folder (Path or None): A path on the volume to monitor
            (typically the per-recording intermediate folder).
            Provide for device-mode monitoring. ``None`` to skip
            device monitoring entirely.
        pids (Sequence[int] or None): Process IDs to monitor in
            process mode. Defaults to ``None`` (device mode). The
            watchdog sums I/O bytes across these processes and
            (if ``include_descendants``) their entire descendant
            trees on every poll.
        include_descendants (bool): When in process mode, recurse
            into each registered PID's children on every poll so
            subprocesses spawned by the sort (e.g. spikeinterface
            workers, KS2 MATLAB child) are accounted for. Defaults
            to ``True``. Set ``False`` if you want to detect a
            stall in *only* the registered PIDs without their
            descendants — rare; mostly useful for debugging.
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
        - Process mode requires ``psutil``. Device mode is also
          disabled when ``psutil`` is missing or when no device
          can be resolved for *folder*. To skip the I/O-stall
          check intentionally, omit any ``register_kill_callback``
          calls — the watchdog still polls but has nothing to
          abort.
        - Unlike :class:`HostMemoryWatchdog`, this watchdog does
          not accept subprocess registrations — only kill
          callbacks. A Docker-backed sort whose container is
          registered with the host watchdog will not have its
          container killed when the I/O stall watchdog trips.
        - Docker container processes are visible to the host's
          ``psutil`` but are NOT children of the orchestrating
          Python process — Docker daemon is the parent. To
          monitor a Docker-backed sort in process mode, register
          the container's main PID explicitly via
          :meth:`register_pid` once it's known
          (``docker inspect --format '{{.State.Pid}}' <id>``).
    """

    def __init__(
        self,
        folder: Optional[Path] = None,
        *,
        pids: Optional[Sequence[int]] = None,
        include_descendants: bool = True,
        stall_s: float = 300.0,
        poll_interval_s: float = 10.0,
        warn_repeat_s: float = 60.0,
        kill_grace_s: float = 5.0,
    ) -> None:
        if folder is None and not pids:
            raise ValueError(
                "IOStallWatchdog requires either a folder (device mode) "
                "or pids (process mode) to monitor."
            )
        if np.isnan(stall_s) or stall_s <= 0.0:
            raise ValueError(f"stall_s must be positive, got {stall_s}.")
        if np.isnan(poll_interval_s) or poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be positive, got {poll_interval_s}."
            )
        if np.isnan(kill_grace_s) or kill_grace_s < 0.0:
            raise ValueError(f"kill_grace_s must be non-negative, got {kill_grace_s}.")
        self.folder = Path(folder) if folder is not None else None
        # Sanity-check pids early so a typo lands at construction
        # rather than when the polling thread starts.
        cleaned_pids: List[int] = []
        for pid in pids or []:
            pid_int = int(pid)
            if pid_int <= 0:
                raise ValueError(f"PIDs must be positive integers, got {pid!r}.")
            cleaned_pids.append(pid_int)
        self._pids: List[int] = cleaned_pids
        self.include_descendants = bool(include_descendants)
        # Process mode is implied by a non-empty pids list at
        # construction. Adding pids later via register_pid does
        # not retroactively switch from device to process mode —
        # the mode is chosen once at __enter__.
        self._mode: str = "process" if cleaned_pids else "device"
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
        self._token: Optional[contextvars.Token] = None
        # Set True when the trip cascade ran but
        # ``_thread.interrupt_main`` raised — see
        # :meth:`interrupt_delivery_failed`.
        self._interrupt_main_failed = False

    # ------------------------------------------------------------------
    # Trip-state queries
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True once the watchdog has fired its abort path."""
        return self._tripped

    def interrupt_delivery_failed(self) -> bool:
        """Return True if the trip fired but ``_thread.interrupt_main`` raised.

        When True, host I/O protection ran successfully (kill
        callbacks invoked) but the main thread did not receive a
        ``KeyboardInterrupt``. The pipeline's catch site checks this
        to reclassify a downstream exception.

        Returns:
            failed (bool): True only when the watchdog tripped and
                the interrupt delivery raised.
        """
        return self._interrupt_main_failed

    def device(self) -> Optional[str]:
        """Return the resolved device identifier (e.g. "sda1")."""
        return self._device

    def mode(self) -> str:
        """Return the active polling mode: ``"device"`` or ``"process"``."""
        return self._mode

    def pids(self) -> List[int]:
        """Snapshot of the currently registered PIDs (process mode)."""
        with self._lock:
            return list(self._pids)

    def make_error(self, message: Optional[str] = None) -> IOStallError:
        """Build an :class:`IOStallError` from the trip state."""
        if message is None:
            stall = self._stall_at_trip
            stall_str = f"{stall:.1f}" if stall is not None else "?"
            if self._mode == "process":
                pid_str = ",".join(str(p) for p in self.pids()) or "?"
                message = (
                    f"Sort process tree (pids={pid_str}) stalled for "
                    f"{stall_str}s (tolerance: {self.stall_s:.1f}s). "
                    "The process(es) issued no read/write syscalls in "
                    "this window — likely an internal deadlock, a hung "
                    "kernel I/O wait, or a CUDA / sorter binary hang."
                )
            else:
                message = (
                    f"I/O on device {self._device!r} stalled for "
                    f"{stall_str}s (tolerance: {self.stall_s:.1f}s). "
                    "Likely a hung network mount or unresponsive "
                    "storage. Sort aborted."
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

    def register_pid(self, pid: int) -> None:
        """Add a PID to the process-mode poll set.

        Useful for tracking processes that don't exist yet at
        watchdog construction — e.g. registering the Docker
        container's main PID once the container has actually
        started, or registering a sorter subprocess after
        ``Popen`` returns.

        No-op when called in device mode (the watchdog isn't
        polling per-PID counters there). The PID is added
        atomically; the next poll picks it up.

        Parameters:
            pid (int): The PID to monitor. Must be a positive
                integer.

        Raises:
            ValueError: If *pid* is not a positive integer.
        """
        pid_int = int(pid)
        if pid_int <= 0:
            raise ValueError(f"pid must be a positive integer, got {pid!r}.")
        if self._mode != "process":
            _logger.debug(
                "register_pid(%d) called on a device-mode watchdog " "— no-op.",
                pid_int,
            )
            return
        with self._lock:
            if pid_int not in self._pids:
                self._pids.append(pid_int)
                _logger.info(
                    "now tracking pid=%d (total %d pid(s))",
                    pid_int,
                    len(self._pids),
                )

    def unregister_pid(self, pid: int) -> None:
        """Remove a PID from the process-mode poll set.

        No-op when *pid* is not currently registered or when
        called in device mode.
        """
        pid_int = int(pid)
        with self._lock:
            self._pids = [p for p in self._pids if p != pid_int]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "IOStallWatchdog":
        # Reject double-``__enter__``. ``self._token`` is a single
        # attribute; a second ``__enter__`` without an intervening
        # ``__exit__`` overwrites the first token reference and
        # leaks the original active-watchdog publication. Symmetric
        # with the guard added to HostMemoryWatchdog and
        # GpuMemoryWatchdog so all three watchdogs fail loudly on
        # reentry rather than silently corrupting ContextVar state.
        if self._token is not None:
            raise RuntimeError(
                "IOStallWatchdog is not reentrant: __enter__ was "
                "called while the watchdog is still active. Exit the "
                "existing context manager before entering a new one."
            )

        if self._mode == "process":
            # Probe once to confirm we can read at least one PID's
            # counters. If none of the registered PIDs are alive
            # at entry, disable rather than running a watchdog
            # that's perpetually blind.
            initial, alive = _read_io_bytes_for_pids(
                self._pids, include_descendants=self.include_descendants
            )
            if initial is None:
                _logger.warning(
                    "no live registered PIDs at entry (initial: %s) "
                    "— disabled. Process-mode IOStallWatchdog needs "
                    "at least one alive PID to be useful.",
                    self._pids,
                )
                self._enabled = False
                return self
            self._enabled = True
            thread_name = f"IOStallWatchdog[pids={self._pids}]"
            _logger.info(
                "active: mode=process pids=%s descendants=%s "
                "stall_s=%.1f poll=%.1fs (initial bytes=%d, "
                "alive_pids=%d)",
                self._pids,
                self.include_descendants,
                self.stall_s,
                self.poll_interval_s,
                initial,
                alive,
            )
        else:
            assert self.folder is not None
            device = _resolve_device_for_path(self.folder)
            if device is None:
                _logger.warning(
                    "could not resolve a block device for %s (psutil "
                    "missing or no matching mountpoint) — disabled. "
                    "The log inactivity watchdog still covers most "
                    "stall cases.",
                    self.folder,
                )
                self._enabled = False
                return self
            # Probe once to confirm we can read counters for the device.
            if _read_io_bytes(device) is None:
                _logger.warning(
                    "device %r is not exposed by "
                    "psutil.disk_io_counters(perdisk=True) — disabled. "
                    "Common on Linux NVMe setups where only the parent "
                    "disk is reported; consider monitoring at the parent "
                    "device instead.",
                    device,
                )
                self._enabled = False
                return self
            self._device = device
            self._enabled = True
            thread_name = f"IOStallWatchdog[{device}]"
            _logger.info(
                "active: mode=device device=%s folder=%s " "stall_s=%.1f poll=%.1fs",
                device,
                self.folder,
                self.stall_s,
                self.poll_interval_s,
            )
        # Publish the active watchdog so the per-recording
        # ``KeyboardInterrupt`` catch site can convert a
        # ``_thread.interrupt_main`` from this watchdog into a
        # classified ``IOStallError`` rather than letting it
        # bubble up raw.
        self._token = _active_io_stall_watchdog.set(self)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s + 1.0)
            self._thread = None
        if self._token is not None:
            try:
                _active_io_stall_watchdog.reset(self._token)
            except (LookupError, ValueError, RuntimeError):
                # Another context modified the var between set/reset,
                # or the token was already consumed (Python 3.10+
                # raises RuntimeError on re-used tokens).
                pass
            self._token = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_bytes(self) -> Optional[int]:
        """Read the current byte counter from the active source.

        In process mode, sums over the registered PIDs (snapshotted
        under the lock so a concurrent ``register_pid`` doesn't
        race with the read). In device mode, falls back to the
        cached device.
        """
        if self._mode == "process":
            with self._lock:
                pids_snapshot = list(self._pids)
            total, _alive = _read_io_bytes_for_pids(
                pids_snapshot,
                include_descendants=self.include_descendants,
            )
            return total
        return _read_io_bytes(self._device or "")

    def _poll_loop(self) -> None:
        """Polling loop: warn, then trip, then exit."""
        if self._stop_event.wait(self.poll_interval_s):
            return
        last_bytes = self._read_bytes()
        last_change_t = time.time()
        last_warn_t = 0.0
        blind_started_t: Optional[float] = None
        blind_warned = False
        while not self._stop_event.is_set():
            current = self._read_bytes()
            now = time.time()
            if current is None:
                # Counters unreadable this poll. Two semantics to preserve:
                #
                # 1. ``last_change_t`` is NOT reset. Resetting it (the
                #    original behaviour) silently masked any true stall
                #    that happened to coincide with even a brief psutil
                #    hiccup — the watchdog went blind precisely when
                #    something was wrong. The rare false-positive case
                #    (counters coincidentally landing on the same value
                #    at the start and end of a blind interval) is far
                #    less common and far less harmful than missing a
                #    real stall.
                #
                # 2. Sustained blindness is itself a trip condition.
                #    After ``stall_s`` of unreadable counters we emit a
                #    one-shot warning (existing behaviour); after
                #    ``2 * stall_s`` we trip via ``_on_trip_blind`` so
                #    the sort is killed rather than running forever
                #    with a silently disabled watchdog. The 2× factor
                #    gives one warn cycle of grace where an operator
                #    monitoring logs can investigate before the kill.
                if blind_started_t is None:
                    blind_started_t = now
                else:
                    blind_for = now - blind_started_t
                    if not blind_warned and blind_for >= self.stall_s:
                        self._warn_blind(blind_for)
                        blind_warned = True
                    if blind_warned and blind_for >= 2 * self.stall_s:
                        self._on_trip_blind(blind_for)
                        return
                self._stop_event.wait(self.poll_interval_s)
                continue
            # Successful read clears the blindness tracker so a later
            # episode is reported afresh.
            blind_started_t = None
            blind_warned = False
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

    def _scope_label(self) -> str:
        """Human-readable label of the polling scope for log lines."""
        if self._mode == "process":
            return f"sort process tree (pids={self._pids})"
        return f"device {self._device!r}"

    def _maybe_warn(self, stalled_for: float) -> None:
        _logger.warning(
            "%s idle for %.1fs (will abort at %.1fs).",
            self._scope_label(),
            stalled_for,
            self.stall_s,
        )
        append_audit_event(
            watchdog="io_stall",
            event="warn",
            mode=self._mode,
            device=self._device,
            pids=list(self._pids) if self._mode == "process" else None,
            stalled_for_s=stalled_for,
            tolerance_s=self.stall_s,
        )

    def _warn_blind(self, blind_for: float) -> None:
        _logger.warning(
            "I/O counter for %s unreadable for %.1fs — watchdog is "
            "blind to stalls until counters become readable again. "
            "Other watchdogs (log inactivity, host memory) still apply.",
            self._scope_label(),
            blind_for,
        )
        append_audit_event(
            watchdog="io_stall",
            event="blind_warn",
            mode=self._mode,
            device=self._device,
            pids=list(self._pids) if self._mode == "process" else None,
            blind_for_s=blind_for,
            tolerance_s=self.stall_s,
        )

    def _on_trip(self, stalled_for: float) -> None:
        self._tripped = True
        self._stall_at_trip = stalled_for
        _logger.error(
            "TRIP: %s stalled for %.1fs (>= %.1fs). Aborting sort.",
            self._scope_label(),
            stalled_for,
            self.stall_s,
        )
        append_audit_event(
            watchdog="io_stall",
            event="abort",
            mode=self._mode,
            device=self._device,
            pids=list(self._pids) if self._mode == "process" else None,
            stalled_for_s=stalled_for,
            tolerance_s=self.stall_s,
        )
        with self._lock:
            callbacks = list(self._kill_callbacks)
        for cb in callbacks:
            try:
                cb()
            except (SystemExit, KeyboardInterrupt):
                # An in-process kill callback delivers KeyboardInterrupt
                # via _thread.interrupt_main(); SystemExit signals
                # operator-requested abort. Both must propagate.
                raise
            except Exception as exc:
                _logger.error("kill_callback raised: %r; continuing.", exc)
        # If __exit__ ran while we were mid-cascade (callbacks can
        # take several seconds), the with-block has already torn
        # down. Sending interrupt_main() now would land a phantom
        # KeyboardInterrupt in whatever code is running next — the
        # next sort, an exception handler, or the interactive
        # prompt. Skip it.
        if self._stop_event.is_set():
            _logger.info("suppressing interrupt_main: watchdog is already exiting.")
            return
        try:
            import _thread as _t

            _t.interrupt_main()
        except Exception as exc:
            self._interrupt_main_failed = True
            _logger.error("failed to interrupt main: %s", exc)
            append_audit_event(
                watchdog="io_stall",
                event="interrupt_delivery_failed",
                device=self._device,
                error=repr(exc),
            )

    def _on_trip_blind(self, blind_for: float) -> None:
        """Trip when sustained blindness prevents verifying I/O is moving.

        Mirrors :meth:`_on_trip` but with a distinct log and audit-event
        semantic: we have not observed a stall, we have observed that
        we are unable to determine whether one is occurring. The abort
        cascade (kill callbacks + ``interrupt_main``) is identical so a
        blind trip cleans up the same way as an observed trip. Downstream
        post-mortems can grep ``event="abort_blind"`` to attribute
        incidents to a watchdog-blind cause rather than a real stall.
        """
        self._tripped = True
        self._stall_at_trip = blind_for
        _logger.error(
            "TRIP: %s I/O counter unreadable for %.1fs (>= %.1fs). "
            "Aborting sort because watchdog cannot verify progress.",
            self._scope_label(),
            blind_for,
            2 * self.stall_s,
        )
        append_audit_event(
            watchdog="io_stall",
            event="abort_blind",
            mode=self._mode,
            device=self._device,
            pids=list(self._pids) if self._mode == "process" else None,
            blind_for_s=blind_for,
            tolerance_s=2 * self.stall_s,
        )
        with self._lock:
            callbacks = list(self._kill_callbacks)
        for cb in callbacks:
            try:
                cb()
            except (SystemExit, KeyboardInterrupt):
                # An in-process kill callback delivers KeyboardInterrupt
                # via _thread.interrupt_main(); SystemExit signals
                # operator-requested abort. Both must propagate.
                raise
            except Exception as exc:
                _logger.error("kill_callback raised: %r; continuing.", exc)
        # If __exit__ ran while we were mid-cascade (callbacks can
        # take several seconds), the with-block has already torn
        # down. Sending interrupt_main() now would land a phantom
        # KeyboardInterrupt in whatever code is running next — the
        # next sort, an exception handler, or the interactive
        # prompt. Skip it.
        if self._stop_event.is_set():
            _logger.info("suppressing interrupt_main: watchdog is already exiting.")
            return
        try:
            import _thread as _t

            _t.interrupt_main()
        except Exception as exc:
            self._interrupt_main_failed = True
            _logger.error("failed to interrupt main: %s", exc)
            append_audit_event(
                watchdog="io_stall",
                event="interrupt_delivery_failed",
                device=self._device,
                error=repr(exc),
            )
