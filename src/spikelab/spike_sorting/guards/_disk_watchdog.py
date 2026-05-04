"""Live disk-usage watchdog for spike-sorting runs.

RT-Sort writes large intermediate ``.npy`` files (scaled traces,
model traces, model outputs) during ``detect_sequences``; on a
multi-channel multi-hour recording this can climb past 100 GB.
Kilosort2 writes a binary ``.dat`` recording. If the volume holding
the intermediate folder fills mid-sort, the sorter typically hangs
on the next write or fails with an opaque OS error.

:class:`DiskUsageWatchdog` is a daemon-thread context manager that
polls ``shutil.disk_usage(folder).free`` and trips when free space
drops below the configured abort threshold. On trip it builds a
:class:`DiskExhaustionReport` (free space, projected need, top
existing consumers, suggested actions) and either kills a registered
subprocess (KS2 MATLAB / Docker container) or invokes a kill
callback (in-process KS4 / RT-Sort).

The on-trip report is the primary user-facing artefact: it carries
the information an operator needs to free space without guessing
which folder to clean up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .._exceptions import DiskExhaustionError


@dataclass
class DiskExhaustionReport:
    """Diagnostic payload built when the disk watchdog trips.

    Parameters:
        folder (str): The folder whose free space crossed the
            abort threshold.
        free_gb_at_trip (float): Free disk space (GB) at the trip
            moment.
        abort_threshold_gb (float): Configured abort threshold (GB).
        projected_need_gb (float or None): Sorter-specific projected
            on-disk footprint in GB when known (e.g. RT-Sort's
            ``estimate_rt_sort_intermediate_gb`` value).
        bytes_consumed_during_sort (float): Bytes consumed inside
            ``folder`` since the watchdog started — i.e. how much
            this sort has written. Useful for distinguishing "I
            started near full and crossed the line" vs "I wrote
            everything".
        top_consumers (list[tuple[str, float]]): Up to 10 largest
            files inside ``folder`` (depth-bounded ``os.walk``) as
            ``(path, gb)`` tuples, sorted descending. Helps the
            operator identify what to clean up.
        suggested_actions (list[str]): Free-form text hints. The
            watchdog seeds these from the trip context; callers can
            extend.
    """

    folder: str
    free_gb_at_trip: float
    abort_threshold_gb: float
    projected_need_gb: Optional[float] = None
    bytes_consumed_during_sort: float = 0.0
    top_consumers: List[Tuple[str, float]] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict representation of the report."""
        return {
            "folder": self.folder,
            "free_gb_at_trip": self.free_gb_at_trip,
            "abort_threshold_gb": self.abort_threshold_gb,
            "projected_need_gb": self.projected_need_gb,
            "bytes_consumed_during_sort": self.bytes_consumed_during_sort,
            "top_consumers": [
                {"path": p, "size_gb": gb} for p, gb in self.top_consumers
            ],
            "suggested_actions": list(self.suggested_actions),
        }


_GB = 1024**3


def _disk_free_gb(folder: Path) -> Optional[float]:
    """Return free disk space at ``folder`` in GB, or None on failure."""
    p = Path(folder)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / _GB
    except OSError:
        return None


def _folder_size_bytes(folder: Path) -> float:
    """Return the total size (bytes) of files under *folder*.

    Best-effort: errors traversing the tree are swallowed so a
    partial result is returned rather than failing the watchdog.
    """
    total = 0
    folder = Path(folder)
    if not folder.exists():
        return 0.0
    try:
        for dirpath, _dirs, files in os.walk(folder, onerror=lambda _e: None):
            for name in files:
                p = Path(dirpath) / name
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
    except Exception:
        pass
    return float(total)


def _top_consumers(
    folder: Path, *, limit: int = 10, max_depth: int = 4
) -> List[Tuple[str, float]]:
    """Return the *limit* largest files under *folder* as (path, gb).

    Walks at most *max_depth* directories deep to keep the cost
    bounded on very deep trees. Errors are swallowed; the partial
    list is returned rather than raising.
    """
    folder = Path(folder)
    if not folder.exists():
        return []
    base_depth = len(folder.parts)
    candidates: List[Tuple[str, float]] = []
    try:
        for dirpath, _dirs, files in os.walk(folder, onerror=lambda _e: None):
            depth = len(Path(dirpath).parts) - base_depth
            if depth > max_depth:
                _dirs[:] = []  # prune deeper traversal
                continue
            for name in files:
                p = Path(dirpath) / name
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                candidates.append((str(p), float(size)))
    except Exception:
        pass
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [(p, sz / _GB) for p, sz in candidates[:limit]]


class DiskUsageWatchdog:
    """Daemon watchdog that aborts the sort on low free disk space.

    Use as a context manager around the per-recording sort. While
    active, a daemon thread polls free space on *folder* every
    ``poll_interval_s`` seconds. Crossing ``warn_free_gb`` prints a
    rate-limited warning; crossing ``abort_free_gb`` builds a
    :class:`DiskExhaustionReport`, terminates any registered
    subprocess, and runs an optional kill callback (mirroring the
    in-process kill path used by ``LogInactivityWatchdog``).

    Parameters:
        folder (Path): The folder to monitor (typically the
            per-recording intermediate folder).
        warn_free_gb (float): Free-disk threshold at which to print
            a warning. Defaults to ``5.0``.
        abort_free_gb (float): Free-disk threshold at which to abort
            the sort. Defaults to ``1.0``.
        poll_interval_s (float): Seconds between polls. Defaults to
            ``10.0``.
        warn_repeat_s (float): Minimum seconds between repeated
            warnings. Defaults to ``30.0``.
        sorter (str): Short identifier used in diagnostic prints and
            in the resulting :class:`DiskExhaustionError`.
        projected_need_gb (float or None): Optional sorter-specific
            disk projection; included verbatim in the trip report
            when present.
        popen (subprocess.Popen or None): Subprocess to terminate
            on trip (e.g. KS2 MATLAB child).
        kill_callback (Callable[[], None] or None): Optional zero-arg
            callable invoked on trip — used by in-process sorters to
            install a two-stage interrupt-then-os._exit fallback.
        kill_grace_s (float): Seconds between ``terminate()`` and
            ``kill()`` on a registered subprocess.

    Notes:
        - The watchdog only trips once. After trip the polling thread
          exits.
        - Disabled (no-op) when ``abort_free_gb`` is non-positive or
          when neither a popen nor a kill_callback is provided.
    """

    def __init__(
        self,
        folder: Path,
        *,
        warn_free_gb: float = 5.0,
        abort_free_gb: float = 1.0,
        poll_interval_s: float = 10.0,
        warn_repeat_s: float = 30.0,
        sorter: str = "sort",
        projected_need_gb: Optional[float] = None,
        popen: Optional[subprocess.Popen] = None,
        kill_callback: Optional[Callable[[], None]] = None,
        kill_grace_s: float = 5.0,
    ) -> None:
        if warn_free_gb <= abort_free_gb:
            raise ValueError(
                f"warn_free_gb ({warn_free_gb}) must be greater than "
                f"abort_free_gb ({abort_free_gb})."
            )
        if poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be positive, got {poll_interval_s}."
            )

        self.folder = Path(folder)
        self.warn_free_gb = float(warn_free_gb)
        self.abort_free_gb = float(abort_free_gb)
        self.poll_interval_s = float(poll_interval_s)
        self.warn_repeat_s = float(warn_repeat_s)
        self.sorter = sorter
        self.projected_need_gb = (
            float(projected_need_gb) if projected_need_gb is not None else None
        )
        self.popen = popen
        self.kill_callback = kill_callback
        self.kill_grace_s = float(kill_grace_s)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped = False
        self._last_warn_t = 0.0
        self._free_at_trip: Optional[float] = None
        self._initial_folder_size: Optional[float] = None
        self._initial_top_consumers: List[Tuple[str, float]] = []
        self._report: Optional[DiskExhaustionReport] = None
        has_kill_target = (self.popen is not None) or (self.kill_callback is not None)
        self._enabled = self.abort_free_gb > 0 and has_kill_target

    # ------------------------------------------------------------------
    # Trip-state queries
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True once the watchdog has fired its abort path."""
        return self._tripped

    def report(self) -> Optional[DiskExhaustionReport]:
        """Return the :class:`DiskExhaustionReport` if the watchdog tripped."""
        return self._report

    def make_error(self, message: Optional[str] = None) -> DiskExhaustionError:
        """Build a :class:`DiskExhaustionError` from the trip state.

        Parameters:
            message (str or None): Override the default message.

        Returns:
            err (DiskExhaustionError): Exception ready to raise.
        """
        if message is None:
            free = self._free_at_trip
            free_str = f"{free:.2f}" if free is not None else "?"
            message = (
                f"Free disk on {self.folder} dropped to {free_str} GB "
                f"(<= {self.abort_free_gb:.2f} GB abort threshold) "
                "during sort. Aborted."
            )
        return DiskExhaustionError(
            message,
            folder=self.folder,
            free_gb_at_trip=self._free_at_trip,
            abort_threshold_gb=self.abort_free_gb,
            report=self._report,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DiskUsageWatchdog":
        if not self._enabled:
            return self
        # Snapshot the baseline folder size so we can later report
        # how much THIS sort consumed (vs starting near-full).
        self._initial_folder_size = _folder_size_bytes(self.folder)
        # Snapshot the initial top consumers so we have a fallback
        # ready if the trip-time walk on a near-full disk is too
        # slow (millions-of-files trees can stall the os.walk).
        try:
            self._initial_top_consumers = _top_consumers(self.folder)
        except Exception:
            self._initial_top_consumers = []
        print(
            f"[disk watchdog] active: folder={self.folder} "
            f"warn<={self.warn_free_gb:.1f}GB "
            f"abort<={self.abort_free_gb:.1f}GB "
            f"poll={self.poll_interval_s:.1f}s"
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"DiskUsageWatchdog[{self.folder.name}]",
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
        # Defer the first measurement so __enter__ has time to return.
        if self._stop_event.wait(self.poll_interval_s):
            return
        while not self._stop_event.is_set():
            free_gb = _disk_free_gb(self.folder)
            if free_gb is None:
                self._stop_event.wait(self.poll_interval_s)
                continue

            if free_gb <= self.abort_free_gb:
                self._on_trip(free_gb)
                return
            if free_gb <= self.warn_free_gb:
                self._maybe_warn(free_gb)
            self._stop_event.wait(self.poll_interval_s)

    def _maybe_warn(self, free_gb: float) -> None:
        """Print a warning if enough time has passed since the last one."""
        now = time.time()
        if now - self._last_warn_t < self.warn_repeat_s:
            return
        self._last_warn_t = now
        print(
            f"[disk watchdog] WARNING: free disk on {self.folder} = "
            f"{free_gb:.2f} GB (warn<={self.warn_free_gb:.1f} / "
            f"abort<={self.abort_free_gb:.1f}). Free space soon or "
            "the sort will be aborted."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="disk",
                event="warn",
                folder=str(self.folder),
                free_gb=free_gb,
                warn_free_gb=self.warn_free_gb,
                abort_free_gb=self.abort_free_gb,
            )
        except Exception:
            pass

    def _on_trip(self, free_gb: float) -> None:
        """Build the report, terminate any subprocess, then run the callback."""
        self._tripped = True
        self._free_at_trip = free_gb
        print(
            f"[disk watchdog] TRIP: free disk on {self.folder} = "
            f"{free_gb:.2f} GB (<= {self.abort_free_gb:.2f} GB)."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="disk",
                event="abort",
                folder=str(self.folder),
                free_gb=free_gb,
                abort_free_gb=self.abort_free_gb,
            )
        except Exception:
            pass

        # Build the report on the watchdog thread before killing
        # anything — once the kill_callback fires (os._exit) we lose
        # the chance.
        self._report = self._build_report(free_gb)
        if self._report.top_consumers:
            top_path, top_gb = self._report.top_consumers[0]
            top_summary = f"{top_path} ({round(top_gb, 2)} GB)"
        else:
            top_summary = "(none found)"
        print(f"[disk watchdog] report: top consumer = {top_summary}")

        if self.popen is not None:
            try:
                if self.popen.poll() is None:
                    self.popen.terminate()
            except Exception as exc:
                print(
                    f"[disk watchdog] terminate() failed for pid="
                    f"{getattr(self.popen, 'pid', '?')}: {exc}"
                )
            time.sleep(self.kill_grace_s)
            try:
                if self.popen.poll() is None:
                    self.popen.kill()
                    print(
                        f"[disk watchdog] killed pid="
                        f"{getattr(self.popen, 'pid', '?')} (terminate ignored)."
                    )
            except Exception as exc:
                print(
                    f"[disk watchdog] kill() failed for pid="
                    f"{getattr(self.popen, 'pid', '?')}: {exc}"
                )

        if self.kill_callback is not None:
            try:
                self.kill_callback()
            except (SystemExit, KeyboardInterrupt):
                # KeyboardInterrupt is exactly what an in-process kill
                # callback delivers via _thread.interrupt_main(); never
                # swallow either kind of intentional interrupt.
                raise
            except Exception as exc:
                print(f"[disk watchdog] kill_callback raised: {exc!r}; continuing.")

    def _top_consumers_with_timeout(
        self, timeout_s: float
    ) -> Optional[List[Tuple[str, float]]]:
        """Run :func:`_top_consumers` on a worker thread with a timeout.

        Returns the fresh result when the walk completes inside
        *timeout_s*, ``None`` when the walk is still running. The
        caller is expected to fall back to the entry-time snapshot
        in the ``None`` case so a hung os.walk does not block the
        kill path on a stalled filesystem.
        """
        result: List[Optional[List[Tuple[str, float]]]] = [None]

        def _worker() -> None:
            try:
                result[0] = _top_consumers(self.folder)
            except Exception:
                result[0] = []

        t = threading.Thread(
            target=_worker,
            name=f"DiskUsageWatchdog[{self.folder.name}]:walk",
            daemon=True,
        )
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            return None
        return result[0]

    def _build_report(self, free_gb: float) -> DiskExhaustionReport:
        """Snapshot folder state into a :class:`DiskExhaustionReport`."""
        try:
            current_size_bytes = _folder_size_bytes(self.folder)
        except Exception:
            current_size_bytes = 0.0
        baseline = self._initial_folder_size or 0.0
        consumed = max(0.0, current_size_bytes - baseline)

        # Bounded fresh walk, with the entry-time snapshot as the
        # fallback when the filesystem is too slow to enumerate.
        top = self._top_consumers_with_timeout(timeout_s=5.0)
        if top is None:
            top = list(self._initial_top_consumers)
            if top:
                print(
                    "[disk watchdog] live top-consumer walk timed out; "
                    "falling back to entry-time snapshot."
                )

        suggestions: List[str] = []
        if self.projected_need_gb is not None and self.projected_need_gb > free_gb:
            shortfall = self.projected_need_gb - free_gb
            suggestions.append(
                f"Sort projects ~{self.projected_need_gb:.1f} GB intermediate "
                f"need, exceeding free disk by ~{shortfall:.1f} GB. Free at "
                "least that much before retrying."
            )
        else:
            suggestions.append(
                f"Free at least {self.warn_free_gb:.1f} GB on the volume "
                "holding the intermediate folder before retrying."
            )
        if top:
            largest_path, largest_gb = top[0]
            suggestions.append(
                f"Largest existing file in {self.folder}: {largest_path} "
                f"({largest_gb:.2f} GB). Inspect before deleting."
            )
        suggestions.append(
            "Consider pointing intermediate_folders at a larger volume, "
            "or shorten the recording window via "
            "RTSortConfig.recording_window_ms / first_n_mins."
        )

        return DiskExhaustionReport(
            folder=str(self.folder),
            free_gb_at_trip=float(free_gb),
            abort_threshold_gb=float(self.abort_free_gb),
            projected_need_gb=self.projected_need_gb,
            bytes_consumed_during_sort=consumed,
            top_consumers=top,
            suggested_actions=suggestions,
        )
