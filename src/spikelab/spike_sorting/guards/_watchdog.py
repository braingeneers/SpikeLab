"""Live host-memory watchdog for spike-sorting runs.

The watchdog is a daemon thread that polls
``psutil.virtual_memory().percent`` at a configurable cadence. When the
system memory percentage crosses a *warning* threshold the watchdog
prints a rate-limited notice; when it crosses an *abort* threshold it:

1. Terminates every registered subprocess (e.g. the Kilosort2 MATLAB
   child) so they release their RAM promptly.
2. Calls :func:`_thread.interrupt_main` to inject a
   ``KeyboardInterrupt`` into the main Python thread at the next
   bytecode boundary.

The pipeline catches the resulting interrupt and re-raises it as
:class:`spikelab.spike_sorting._exceptions.HostMemoryWatchdogError`,
which is a :class:`ResourceSortFailure` subclass — so callers can apply
retry/skip policies uniformly with other resource failures.

Discovery
---------
The watchdog publishes itself via a :class:`contextvars.ContextVar` on
``__enter__`` and clears it on ``__exit__``. Backends that spawn child
processes (e.g. the Kilosort2 MATLAB runner) call
:func:`get_active_watchdog` to find the live instance and register
their ``subprocess.Popen`` handle with it. This avoids threading a
watchdog parameter through every backend signature.

Platform notes
--------------
The detection step is fully platform-agnostic — ``psutil`` reads
system-wide pressure on Linux, macOS, and Windows alike. The reaction
step has known limits:

* ``_thread.interrupt_main`` only fires at Python bytecode boundaries.
  A long-running C extension (a single multi-GB ``np.concatenate``,
  a numba ``@njit(parallel=True)`` kernel, a PyTorch CUDA kernel) will
  not see the interrupt until it returns. The watchdog still emits its
  warning, and the abort takes effect as soon as control returns to
  the interpreter.
* Subprocess termination uses :meth:`subprocess.Popen.terminate` then
  :meth:`subprocess.Popen.kill` after a grace period. On Windows the
  MATLAB JVM occasionally ignores the initial terminate; the kill
  fallback handles that.
* If ``psutil`` is not installed the watchdog degrades to a no-op
  context manager so the pipeline still runs.
"""

from __future__ import annotations

import _thread
import contextvars
import logging
import math
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

from .._exceptions import HostMemoryWatchdogError
from ._audit import append_audit_event

_logger = logging.getLogger(__name__)

_active_watchdog: contextvars.ContextVar[Optional["HostMemoryWatchdog"]] = (
    contextvars.ContextVar("active_host_memory_watchdog", default=None)
)


def get_active_watchdog() -> Optional["HostMemoryWatchdog"]:
    """Return the watchdog active for the current context, or None.

    Backends that spawn child processes (Kilosort2 MATLAB, Docker
    containers, etc.) call this to find the live watchdog and register
    their ``Popen`` handle so the watchdog can terminate the child on
    abort.

    Returns:
        watchdog (HostMemoryWatchdog or None): The active instance, or
            ``None`` when no watchdog is currently running.
    """
    return _active_watchdog.get()


class HostMemoryWatchdog:
    """Daemon-thread watchdog that aborts the sort on host RAM pressure.

    Use as a context manager. While the context is active a daemon
    thread polls system memory; on abort it terminates registered
    subprocesses and injects a ``KeyboardInterrupt`` into the main
    thread.

    Parameters:
        warn_pct (float): System memory percentage at which the
            watchdog prints a (rate-limited) warning. Defaults to
            ``85.0``.
        abort_pct (float): System memory percentage at which the
            watchdog terminates registered subprocesses and aborts
            the main thread. Defaults to ``92.0``.
        poll_interval_s (float): Seconds between polls. Defaults to
            ``2.0``.
        warn_repeat_s (float): Minimum seconds between repeated
            warnings at the same level. Defaults to ``30.0``.
        kill_grace_s (float): Default seconds between
            ``terminate()`` and ``kill()`` for registered
            subprocesses. Per-subprocess overrides are accepted in
            :meth:`register_subprocess`. Defaults to ``5.0``.

    Notes:
        - Degrades to a no-op when ``psutil`` is missing.
        - Safe to nest: the inner context is the active one for the
          duration of its body, and the outer context resumes on exit.
    """

    def __init__(
        self,
        warn_pct: float = 85.0,
        abort_pct: float = 92.0,
        poll_interval_s: float = 2.0,
        warn_repeat_s: float = 30.0,
        kill_grace_s: float = 5.0,
    ) -> None:
        if not 0.0 < warn_pct < abort_pct <= 100.0:
            raise ValueError(
                f"warn_pct ({warn_pct}) and abort_pct ({abort_pct}) must "
                f"satisfy 0 < warn_pct < abort_pct <= 100."
            )
        if poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be positive, got {poll_interval_s}."
            )
        if kill_grace_s < 0.0:
            raise ValueError(f"kill_grace_s must be non-negative, got {kill_grace_s}.")

        self.warn_pct = float(warn_pct)
        self.abort_pct = float(abort_pct)
        self.poll_interval_s = float(poll_interval_s)
        self.warn_repeat_s = float(warn_repeat_s)
        self.kill_grace_s = float(kill_grace_s)

        self._subprocesses: List[Tuple[subprocess.Popen, float]] = []
        self._kill_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._token: Optional[contextvars.Token] = None
        self._tripped = False
        self._percent_at_trip: Optional[float] = None
        self._last_warn_t = 0.0
        self._psutil = None
        self._enabled = False
        # Captured at ``__enter__`` time on the main thread because
        # ContextVars do not propagate to the polling thread.
        self._snapshot_log_path = None
        # Set True when the trip cascade ran but
        # ``_thread.interrupt_main`` raised — the main thread did not
        # receive the KeyboardInterrupt and will surface a downstream
        # error instead of the classified watchdog error. Catch sites
        # check this via :meth:`interrupt_delivery_failed` to
        # reclassify the downstream exception.
        self._interrupt_main_failed = False

    # ------------------------------------------------------------------
    # Subprocess registration (called by backends)
    # ------------------------------------------------------------------

    def register_subprocess(
        self,
        popen: subprocess.Popen,
        *,
        kill_grace_s: Optional[float] = None,
    ) -> None:
        """Track a subprocess for termination on watchdog abort.

        Parameters:
            popen (subprocess.Popen): The child process handle. The
                watchdog calls ``terminate()`` first, then ``kill()``
                after ``kill_grace_s`` seconds if the process is still
                alive.
            kill_grace_s (float or None): Override the default grace
                period for this subprocess. ``None`` uses the
                watchdog's ``kill_grace_s``.
        """
        grace = self.kill_grace_s if kill_grace_s is None else float(kill_grace_s)
        with self._lock:
            self._subprocesses.append((popen, grace))

    def unregister_subprocess(self, popen: subprocess.Popen) -> None:
        """Stop tracking a previously registered subprocess.

        Parameters:
            popen (subprocess.Popen): Handle previously passed to
                :meth:`register_subprocess`. No-op if not registered.
        """
        with self._lock:
            self._subprocesses = [
                (p, g) for (p, g) in self._subprocesses if p is not popen
            ]

    def register_kill_callback(self, callback: Callable[[], None]) -> None:
        """Track a zero-arg callable to invoke on watchdog abort.

        Used for kill targets that are not ``subprocess.Popen``
        objects — Docker containers, kubernetes pods, custom
        cleanup hooks. The callback runs after any registered
        subprocesses have been terminated. Exceptions raised by a
        callback are logged but do not prevent other callbacks from
        running.

        Parameters:
            callback (Callable[[], None]): Zero-arg function. Should
                be idempotent and tolerate being called on an
                already-stopped target — the watchdog cannot tell
                whether the kill target is still alive.

        Notes:
            - To allow the kill target to be garbage-collected even
              while registered, build the callback with a weakref to
              the target rather than capturing it directly. See
              ``docker_utils.patched_container_client`` for the
              container-kill pattern.
        """
        with self._lock:
            self._kill_callbacks.append(callback)

    def unregister_kill_callback(self, callback: Callable[[], None]) -> None:
        """Stop tracking a previously registered kill callback.

        Parameters:
            callback (Callable[[], None]): Callable previously passed
                to :meth:`register_kill_callback`. No-op if not
                registered. Identity comparison is used.
        """
        with self._lock:
            self._kill_callbacks = [
                c for c in self._kill_callbacks if c is not callback
            ]

    # ------------------------------------------------------------------
    # Trip state (read by the pipeline catch site)
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True if the watchdog has fired its abort path."""
        return self._tripped

    def interrupt_delivery_failed(self) -> bool:
        """Return True if the trip fired but ``_thread.interrupt_main`` raised.

        When True, host protection ran successfully (subprocesses
        terminated, kill callbacks invoked) but the main thread did
        not receive a ``KeyboardInterrupt``. The pipeline's catch
        site checks this to reclassify a downstream
        ``BrokenPipeError`` / ``RuntimeError`` (caused by the now-dead
        subprocess) as the appropriate watchdog error.

        Returns:
            failed (bool): True only when the watchdog tripped and
                the interrupt delivery raised.
        """
        return self._interrupt_main_failed

    def percent_at_trip(self) -> Optional[float]:
        """Return the memory percent at the trip moment, or None."""
        return self._percent_at_trip

    def make_error(self, message: Optional[str] = None) -> HostMemoryWatchdogError:
        """Build a :class:`HostMemoryWatchdogError` from the trip state.

        Parameters:
            message (str or None): Override the default message.

        Returns:
            err (HostMemoryWatchdogError): Exception ready to raise.
        """
        if message is None:
            pct = (
                f"{self._percent_at_trip:.1f}"
                if self._percent_at_trip is not None
                else "?"
            )
            message = (
                f"Host RAM watchdog tripped at {pct}% "
                f"(abort threshold: {self.abort_pct:.1f}%). "
                "Subprocesses terminated; current recording aborted."
            )
        return HostMemoryWatchdogError(
            message,
            percent_at_trip=self._percent_at_trip,
            abort_pct=self.abort_pct,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HostMemoryWatchdog":
        # Reject double-``__enter__``. ``self._token`` is a single
        # attribute, so a second ``__enter__`` without an intervening
        # ``__exit__`` would overwrite the first token reference and
        # leak the original active-watchdog publication after teardown
        # (only the second token's reset would run). The class is
        # not designed to be reentrant; surface the misuse rather
        # than silently corrupting the ContextVar state.
        if self._token is not None:
            raise RuntimeError(
                "HostMemoryWatchdog is not reentrant: __enter__ was "
                "called while the watchdog is still active. Exit the "
                "existing context manager before entering a new one."
            )

        # Capture the active per-recording log path on the main
        # thread; the daemon polling thread cannot read the
        # ContextVar reliably.
        try:
            from ._inactivity import get_active_log_path

            self._snapshot_log_path = get_active_log_path()
        except Exception:
            self._snapshot_log_path = None

        try:
            import psutil

            self._psutil = psutil
            self._enabled = True
        except ImportError:
            _logger.warning(
                "psutil not installed — watchdog disabled. Install "
                "psutil to enable host RAM monitoring."
            )
            self._enabled = False
            self._token = _active_watchdog.set(self)
            return self

        _logger.info(
            "active: warn=%.1f%% abort=%.1f%% poll=%.1fs",
            self.warn_pct,
            self.abort_pct,
            self.poll_interval_s,
        )
        self._token = _active_watchdog.set(self)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="HostMemoryWatchdog",
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
                _active_watchdog.reset(self._token)
            except (LookupError, ValueError, RuntimeError):
                # Another context modified the var between set/reset,
                # or the token was already consumed (Python 3.10+
                # raises RuntimeError on re-used tokens). Matches the
                # symmetric guards in GpuMemoryWatchdog and
                # IOStallWatchdog so a degenerate teardown does not
                # leak the active-watchdog publication.
                pass
            self._token = None
        with self._lock:
            self._subprocesses.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Polling loop: warn, then trip, then exit."""
        # Defer the first measurement by one poll interval so
        # ``__enter__`` always returns and the protected body starts
        # executing before any trip can fire. Without this delay a
        # watchdog spawned in an already-stressed environment could
        # land its KeyboardInterrupt inside ``Thread.start`` itself,
        # which leaves the with-block in a half-entered state.
        if self._stop_event.wait(self.poll_interval_s):
            return
        blind_threshold_s = 5.0 * self.warn_repeat_s
        blind_started_t: Optional[float] = None
        blind_warned = False
        while not self._stop_event.is_set():
            now = time.time()
            try:
                pct = float(self._psutil.virtual_memory().percent)
            except Exception:
                # psutil on some platforms can transiently fail; skip
                # this tick rather than tearing down the watchdog.
                # Track how long the readings have been unavailable so
                # a sustained psutil failure surfaces a one-time
                # warning instead of silently disabling the abort path.
                if blind_started_t is None:
                    blind_started_t = now
                elif not blind_warned and now - blind_started_t >= blind_threshold_s:
                    self._warn_blind(now - blind_started_t)
                    blind_warned = True
                self._stop_event.wait(self.poll_interval_s)
                continue

            # NaN comparisons are always False, so a NaN reading would
            # silently disable the watchdog. Skip the tick rather than
            # treating it as either a healthy or unhealthy reading.
            if math.isnan(pct):
                if blind_started_t is None:
                    blind_started_t = now
                elif not blind_warned and now - blind_started_t >= blind_threshold_s:
                    self._warn_blind(now - blind_started_t)
                    blind_warned = True
                self._stop_event.wait(self.poll_interval_s)
                continue

            # Successful reading — clear the blindness tracker so a
            # later episode is reported afresh.
            blind_started_t = None
            blind_warned = False

            if pct >= self.abort_pct:
                self._on_abort(pct)
                return

            if pct >= self.warn_pct:
                self._maybe_warn(pct)

            self._stop_event.wait(self.poll_interval_s)

    def _maybe_warn(self, pct: float) -> None:
        """Print a warning if enough time has passed since the last one."""
        now = time.time()
        if now - self._last_warn_t < self.warn_repeat_s:
            return
        self._last_warn_t = now
        _logger.warning(
            "system memory at %.1f%% (warn=%.1f%% / abort=%.1f%%). "
            "Free memory or expect an abort if pressure keeps climbing.",
            pct,
            self.warn_pct,
            self.abort_pct,
        )
        append_audit_event(
            watchdog="host_memory",
            event="warn",
            log_path=self._snapshot_log_path,
            used_pct=pct,
            warn_pct=self.warn_pct,
            abort_pct=self.abort_pct,
        )

    def _warn_blind(self, blind_for: float) -> None:
        _logger.warning(
            "psutil.virtual_memory() unreadable for %.1fs — watchdog "
            "is blind to RAM-pressure aborts until readings recover.",
            blind_for,
        )
        append_audit_event(
            watchdog="host_memory",
            event="blind_warn",
            source="virtual_memory",
            log_path=self._snapshot_log_path,
            blind_for_s=blind_for,
        )

    def _on_abort(self, pct: float) -> None:
        """Terminate registered subprocesses and interrupt the main thread."""
        self._tripped = True
        self._percent_at_trip = pct
        _logger.error(
            "ABORT: system memory at %.1f%% (>= %.1f%%). Terminating "
            "subprocesses and raising into main thread.",
            pct,
            self.abort_pct,
        )
        append_audit_event(
            watchdog="host_memory",
            event="abort",
            log_path=self._snapshot_log_path,
            used_pct=pct,
            abort_pct=self.abort_pct,
        )
        # Best-effort GPU snapshot for postmortem analysis. Useful
        # even on host-RAM trips since RT-Sort / KS4 often hold
        # significant GPU state alongside their host buffers.
        try:
            from ._gpu_watchdog import _try_capture_snapshot_to_results

            _try_capture_snapshot_to_results(
                self._snapshot_log_path,
                f"Host memory watchdog trip — system at {pct:.1f}%",
            )
        except Exception:
            pass
        self._terminate_registered()
        self._run_kill_callbacks()
        # If __exit__ ran while we were mid-cascade (terminate +
        # grace + callbacks can take several seconds), the with-block
        # has already torn down. Sending interrupt_main() now would
        # land a phantom KeyboardInterrupt in whatever code is running
        # next — the next sort, an exception handler, or the
        # interactive prompt. Skip it.
        if self._stop_event.is_set():
            _logger.info("suppressing interrupt_main: watchdog is already exiting.")
            return
        try:
            _thread.interrupt_main()
        except Exception as exc:
            self._interrupt_main_failed = True
            _logger.error("failed to interrupt main: %s", exc)
            append_audit_event(
                watchdog="host_memory",
                event="interrupt_delivery_failed",
                log_path=self._snapshot_log_path,
                error=repr(exc),
            )

    def _run_kill_callbacks(self) -> None:
        """Invoke every registered kill callback; isolate failures."""
        with self._lock:
            callbacks = list(self._kill_callbacks)
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                _logger.error("kill_callback raised: %r; continuing.", exc)

    def _terminate_registered(self) -> None:
        """Best-effort terminate-then-kill of every registered subprocess."""
        with self._lock:
            entries = list(self._subprocesses)

        # Terminate every still-alive process first; gives them all the
        # full grace period to exit cleanly in parallel.
        for popen, _grace in entries:
            try:
                if popen.poll() is None:
                    popen.terminate()
            except Exception as exc:
                _logger.error(
                    "terminate() failed for pid=%s: %s",
                    getattr(popen, "pid", "?"),
                    exc,
                )

        if entries:
            grace = max(g for _, g in entries)
            time.sleep(grace)

        for popen, _grace in entries:
            try:
                if popen.poll() is None:
                    popen.kill()
                    _logger.warning(
                        "killed pid=%s (terminate ignored).",
                        getattr(popen, "pid", "?"),
                    )
            except Exception as exc:
                _logger.error(
                    "kill() failed for pid=%s: %s",
                    getattr(popen, "pid", "?"),
                    exc,
                )
