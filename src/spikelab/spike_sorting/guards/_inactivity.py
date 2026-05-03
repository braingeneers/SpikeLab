"""Sorter-log inactivity watchdog.

Some sorter failure modes hang the subprocess without crashing it
outright: a CUDA kernel deadlock in Kilosort2's mex code, a stuck
MATLAB JVM after a Java exception, a Docker container that lost its
GPU. The host-memory watchdog can catch the rare host-RAM blowup but
not these silent stalls — the subprocess just sits there forever
holding its GPU and the parent ``shell_script.wait()`` blocks
indefinitely.

:class:`LogInactivityWatchdog` is a daemon-thread context manager
that polls the sorter's log file and trips when neither the file's
mtime *nor* its byte size has advanced for the configured tolerance.
When the sort is making progress — KS2 prints per-batch lines, KS4
prints per-stage banners, RT-Sort writes per-chunk diagnostics — at
least one of the two signals keeps moving and the watchdog never
fires. When the sort hangs, both signals stay flat and the watchdog
terminates the registered subprocess.

Tracking size as well as mtime avoids two false-positive failure
modes: NTFS lazy-mtime updates (Windows can defer mtime stamping
on long-held file handles), and ``relatime``-style filesystems with
delayed metadata flushes. In both cases the file content keeps
growing while mtime appears static.

The tolerance scales with recording duration so a long sort that
takes minutes between log writes doesn't get killed by a watchdog
sized for a 5-minute test recording. The scaling formula is:

    timeout_s = clamp(base_s + per_min_s * recording_duration_min, max_s)

with all four parameters configurable via ``ExecutionConfig``.
"""

from __future__ import annotations

import _thread
import contextlib
import contextvars
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .._exceptions import SorterTimeoutError

# ---------------------------------------------------------------------------
# Active-log-path discovery
# ---------------------------------------------------------------------------
#
# In-process sorters (KS4 host, RT-Sort) have no subprocess to monitor; the
# inactivity watchdog instead watches the per-recording ``Tee`` log file, the
# same artefact ``sort_recording`` already creates. Backends look the path
# up via :func:`get_active_log_path`, so they don't need a new parameter on
# every ``sort()`` signature.

_active_log_path: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "active_tee_log_path", default=None
)


def get_active_log_path() -> Optional[Path]:
    """Return the per-recording log path active in this context, or None.

    Used by in-process sorter backends to install a
    :class:`LogInactivityWatchdog` watching the same Tee-mirrored log
    that ``sort_recording`` is writing to.

    Returns:
        log_path (Path or None): The per-recording log file path, or
            ``None`` when no ``sort_recording`` context is active.
    """
    return _active_log_path.get()


@contextlib.contextmanager
def set_active_log_path(log_path: Path):
    """Publish *log_path* via the active-log-path ContextVar for the duration.

    Parameters:
        log_path (Path): The per-recording Tee log file path. Set by
            ``sort_recording`` immediately after opening the ``Tee``.
    """
    token = _active_log_path.set(Path(log_path))
    try:
        yield
    finally:
        _active_log_path.reset(token)


# Tolerance value to use for any opportunistic inactivity watchdog
# spawned for the active recording (e.g. by ``patched_container_client``
# for Docker-backed sorts). Backends compute this from the recording
# duration via ``_resolve_inactivity_timeout_s`` and publish it before
# diving into sorter-specific code.
_active_inactivity_timeout_s: contextvars.ContextVar[Optional[float]] = (
    contextvars.ContextVar("active_inactivity_timeout_s", default=None)
)


def get_active_inactivity_timeout_s() -> Optional[float]:
    """Return the inactivity tolerance (s) active in this context, or None.

    Returns:
        timeout_s (float or None): Resolved inactivity tolerance for
            the current sort, or ``None`` when no backend has
            published one.
    """
    return _active_inactivity_timeout_s.get()


@contextlib.contextmanager
def set_active_inactivity_timeout_s(seconds: Optional[float]):
    """Publish an inactivity tolerance via the ContextVar for the duration.

    Parameters:
        seconds (float or None): Tolerance in seconds, or ``None``
            to leave the value unset for nested code.
    """
    token = _active_inactivity_timeout_s.set(seconds)
    try:
        yield
    finally:
        _active_inactivity_timeout_s.reset(token)


# ---------------------------------------------------------------------------
# Built-in in-process kill callback
# ---------------------------------------------------------------------------


def make_in_process_kill_callback(
    *,
    interrupt_grace_s: float = 10.0,
    sorter: str = "in_process",
) -> Callable[[], None]:
    """Build a kill callback for in-process sorters (KS4 host, RT-Sort).

    The callback is meant to be passed to
    :class:`LogInactivityWatchdog`. When the watchdog trips, it calls:

    1. :func:`_thread.interrupt_main` — queues a ``KeyboardInterrupt``
       in the main thread. If Python is responsive (i.e. not stuck
       deep in a single C extension call), the interrupt fires at the
       next bytecode boundary and the backend's ``try/except`` can
       recover cleanly.
    2. :func:`os._exit` after ``interrupt_grace_s`` seconds — nuclear
       fallback for hangs that interrupt cannot reach (a stuck CUDA
       kernel, a numba ``@njit(parallel=True)`` deadlock). This kills
       the whole Python process with no further cleanup, but is the
       only way to free the OS resources held by the hung sort.

    Parameters:
        interrupt_grace_s (float): Seconds to wait between the
            ``interrupt_main`` and the ``os._exit`` fallback. Defaults
            to ``10.0`` — short enough to keep the workstation
            responsive, long enough that a Python-level recovery can
            complete an in-flight pickle write.
        sorter (str): Short identifier used only in the diagnostic
            print before ``os._exit`` fires.

    Returns:
        callback (Callable[[], None]): Zero-argument function suitable
            for ``LogInactivityWatchdog(kill_callback=...)``.

    Notes:
        - ``os._exit`` skips ``finally`` blocks, ``atexit`` handlers,
          and ``__exit__``. Files mid-write may be left in an
          inconsistent state. The watchdog only escalates after the
          interrupt has had a chance to recover, so well-behaved
          Python paths will exit cleanly.
        - Returning the callback (rather than installing the kill
          inline) lets each backend customise the grace period without
          forking the watchdog.
    """

    def _callback() -> None:
        try:
            _thread.interrupt_main()
        except Exception as exc:
            print(
                f"[inactivity watchdog] interrupt_main failed for " f"{sorter}: {exc}"
            )
        # Sleep on a fresh thread is fine here: the watchdog already
        # runs on its own daemon thread, so sleeping does not block
        # anyone.
        time.sleep(float(interrupt_grace_s))
        print(
            f"[inactivity watchdog] {sorter} did not respond to "
            f"interrupt_main within {interrupt_grace_s:.1f}s — escalating "
            "to os._exit(1) to free OS resources. Per-recording pickles "
            "already on disk are unaffected."
        )
        os._exit(1)

    return _callback


def compute_inactivity_timeout_s(
    *,
    recording_duration_min: float,
    base_s: float = 600.0,
    per_min_s: float = 30.0,
    max_s: Optional[float] = 7200.0,
) -> float:
    """Compute a recording-size-aware inactivity tolerance.

    Parameters:
        recording_duration_min (float): Recording length in minutes.
            Negative or NaN values are clamped to zero.
        base_s (float): Minimum tolerance applied even for tiny
            recordings. Defaults to 600 (10 min).
        per_min_s (float): Extra seconds of tolerance per minute of
            recording. Defaults to 30.
        max_s (float or None): Hard cap on the tolerance. ``None``
            means no cap. Defaults to 7200 (2 h).

    Returns:
        timeout_s (float): Resolved inactivity tolerance in seconds.
    """
    duration = max(0.0, float(recording_duration_min or 0.0))
    timeout = float(base_s) + float(per_min_s) * duration
    if max_s is not None:
        timeout = min(timeout, float(max_s))
    return timeout


class LogInactivityWatchdog:
    """Daemon watchdog that kills a subprocess on sorter-log inactivity.

    Use as a context manager around the call that waits for the
    sorter subprocess. While the context is active a daemon thread
    polls ``log_path`` (via ``os.stat().st_mtime``) every
    ``poll_interval_s``. If the file's mtime has not advanced for
    ``inactivity_s`` seconds the watchdog terminates the registered
    subprocess and records the trip; the wait then returns and the
    runner can detect the kill via ``tripped()`` and raise
    :class:`SorterTimeoutError`.

    Parameters:
        log_path (Path): Path to the sorter's log file. The file does
            not need to exist when the watchdog starts — it's polled
            for first appearance, and the watchdog is forgiving about
            "no log yet" until the file shows up. The pre-existing
            mtime (from a previous run, if any) is recorded at start
            so an old stale log doesn't trip immediately.
        popen (subprocess.Popen or None): Subprocess handle to
            terminate on trip. Pass ``None`` when the sort runs
            in-process — see ``kill_callback`` instead.
        inactivity_s (float): Inactivity tolerance in seconds. Use
            :func:`compute_inactivity_timeout_s` to derive a sensible
            value from recording duration.
        sorter (str): Short identifier of the sorter (used for
            logging and the resulting :class:`SorterTimeoutError`).
        poll_interval_s (float): Seconds between mtime polls.
            Defaults to ``5.0``.
        kill_grace_s (float): Seconds between ``terminate()`` and
            ``kill()`` if the subprocess does not exit. Defaults to
            ``5.0``.
        kill_callback (Callable[[], None] or None): Optional callback
            invoked after the subprocess termination step. Used by
            in-process backends (KS4 host, RT-Sort) to install a
            two-stage kill: ``_thread.interrupt_main`` first, then
            ``os._exit`` if Python is unresponsive. See
            :func:`make_in_process_kill_callback`.

    Notes:
        - When ``inactivity_s`` is ``None`` / non-positive, OR when
          neither ``popen`` nor ``kill_callback`` is provided, the
          watchdog is a no-op context manager. This makes it safe to
          drop in unconditionally.
        - The watchdog only trips once. After trip, the polling
          thread exits.
    """

    def __init__(
        self,
        log_path: Path,
        popen: Optional[subprocess.Popen],
        inactivity_s: Optional[float],
        *,
        sorter: str,
        poll_interval_s: float = 5.0,
        kill_grace_s: float = 5.0,
        kill_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.popen = popen
        self.inactivity_s = (
            None if inactivity_s is None or inactivity_s <= 0 else float(inactivity_s)
        )
        self.sorter = sorter
        self.poll_interval_s = float(poll_interval_s)
        self.kill_grace_s = float(kill_grace_s)
        self.kill_callback = kill_callback

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped = False
        self._last_seen_mtime: Optional[float] = None
        self._last_seen_size: Optional[int] = None
        self._inactivity_at_trip: Optional[float] = None
        # Disabled when there is no timeout to enforce, or when there
        # is no kill target at all (neither a subprocess nor a
        # callback). The first three combinations make the watchdog
        # a no-op.
        has_kill_target = (self.popen is not None) or (self.kill_callback is not None)
        self._enabled = self.inactivity_s is not None and has_kill_target

    # ------------------------------------------------------------------
    # Trip-state queries
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True once the watchdog has fired its terminate path."""
        return self._tripped

    def make_error(self, message: Optional[str] = None) -> SorterTimeoutError:
        """Build a :class:`SorterTimeoutError` from the trip state.

        Parameters:
            message (str or None): Override the default message.

        Returns:
            err (SorterTimeoutError): Exception ready to raise.
        """
        if message is None:
            seen = self._inactivity_at_trip
            seen_str = f"{seen:.1f}" if seen is not None else "?"
            message = (
                f"{self.sorter} produced no log output for {seen_str}s "
                f"(tolerance: {self.inactivity_s:.1f}s). Subprocess "
                "terminated; sort considered hung."
            )
        return SorterTimeoutError(
            message,
            sorter=self.sorter,
            inactivity_s=self.inactivity_s,
            log_path=self.log_path,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LogInactivityWatchdog":
        if not self._enabled:
            return self
        # Capture the pre-existing mtime + size so a stale log from
        # a previous run does not register as a fresh trip.
        signals = self._read_signals()
        if signals is not None:
            self._last_seen_mtime, self._last_seen_size = signals
        else:
            self._last_seen_mtime = None
            self._last_seen_size = None
        print(
            f"[inactivity watchdog] active: sorter={self.sorter} "
            f"tolerance={self.inactivity_s:.1f}s "
            f"poll={self.poll_interval_s:.1f}s "
            f"log={self.log_path}"
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"LogInactivityWatchdog[{self.sorter}]",
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

    def _read_signals(self) -> Optional[Tuple[float, int]]:
        """Return ``(mtime, size)`` for the log file, or None if absent."""
        try:
            st = os.stat(self.log_path)
            return float(st.st_mtime), int(st.st_size)
        except (OSError, FileNotFoundError):
            return None

    def _poll_loop(self) -> None:
        """Polling loop: track mtime + size, trip on inactivity, exit on stop."""
        # Defer the first measurement so __enter__ has time to return.
        if self._stop_event.wait(self.poll_interval_s):
            return

        # Time of the most recent progress signal (mtime change or
        # size change) observed by the watchdog. Initialised to
        # watchdog start so a file that never appears still trips
        # after the configured tolerance.
        last_progress_t = time.time()
        seen_any = self._last_seen_mtime is not None

        while not self._stop_event.is_set():
            signals = self._read_signals()
            now = time.time()

            if signals is not None:
                cur_mtime, cur_size = signals
                if not seen_any:
                    # File just appeared.
                    seen_any = True
                    self._last_seen_mtime = cur_mtime
                    self._last_seen_size = cur_size
                    last_progress_t = now
                elif (
                    cur_mtime != self._last_seen_mtime
                    or cur_size != self._last_seen_size
                ):
                    # Either signal advanced — reset the inactivity clock.
                    self._last_seen_mtime = cur_mtime
                    self._last_seen_size = cur_size
                    last_progress_t = now

            inactivity = now - last_progress_t
            if inactivity >= self.inactivity_s:
                self._on_trip(inactivity)
                return

            self._stop_event.wait(self.poll_interval_s)

    def _on_trip(self, inactivity_s: float) -> None:
        """Record trip state, terminate any subprocess, then run the callback."""
        self._tripped = True
        self._inactivity_at_trip = inactivity_s
        print(
            f"[inactivity watchdog] TRIP: {self.sorter} log idle for "
            f"{inactivity_s:.1f}s (tolerance: {self.inactivity_s:.1f}s)."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="inactivity",
                event="abort",
                log_path=self.log_path,
                sorter=self.sorter,
                inactivity_s=inactivity_s,
                tolerance_s=self.inactivity_s,
            )
        except Exception:
            pass

        if self.popen is not None:
            try:
                if self.popen.poll() is None:
                    self.popen.terminate()
            except Exception as exc:
                print(
                    f"[inactivity watchdog] terminate() failed for pid="
                    f"{getattr(self.popen, 'pid', '?')}: {exc}"
                )
            time.sleep(self.kill_grace_s)
            try:
                if self.popen.poll() is None:
                    self.popen.kill()
                    print(
                        f"[inactivity watchdog] killed pid="
                        f"{getattr(self.popen, 'pid', '?')} (terminate ignored)."
                    )
            except Exception as exc:
                print(
                    f"[inactivity watchdog] kill() failed for pid="
                    f"{getattr(self.popen, 'pid', '?')}: {exc}"
                )

        if self.kill_callback is not None:
            try:
                self.kill_callback()
            except SystemExit:
                # ``os._exit`` does not raise SystemExit, but if a
                # custom callback uses ``sys.exit`` we let it propagate
                # so the caller observes the requested exit code.
                raise
            except Exception as exc:
                print(
                    f"[inactivity watchdog] kill_callback raised: {exc!r}; "
                    "continuing."
                )
