"""Live GPU memory watchdog for spike-sorting runs.

Symmetric to :class:`HostMemoryWatchdog` but watches GPU VRAM via
``pynvml`` (or ``nvidia-smi`` as a fallback). Trips when the
device-in-use crosses the configured percentage thresholds; on trip
it terminates registered subprocesses, runs registered kill
callbacks, and raises a
:class:`spikelab.spike_sorting._exceptions.GpuMemoryWatchdogError`
into the main thread via ``_thread.interrupt_main``.

The watchdog narrows its measurement to the device the sort is using
(KS4 ``torch_device``, RT-Sort ``device``, KS2-Docker default
``cuda:0``) so unrelated GPUs running other workloads are ignored.

Detection priority:

1. ``pynvml`` (already an optional spikelab dep) — fastest, exact
   API for free/used/total memory per device.
2. ``nvidia-smi`` parse — fallback when ``pynvml`` is missing.
3. No-op when neither is available — the watchdog reports as
   disabled rather than raising.
"""

from __future__ import annotations

import _thread
import contextvars
import re
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

from .._exceptions import GpuMemoryWatchdogError


def _resolve_device_index(device: Optional[str]) -> int:
    """Return the integer device index for a torch-style device string.

    Accepts ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, integer-like
    strings, and ``None`` (interpreted as device 0). Falls back to 0
    on parse failure rather than raising — the watchdog is
    best-effort.

    Parameters:
        device (str or None): Torch-style device identifier.

    Returns:
        index (int): Device index (>= 0).
    """
    if device is None:
        return 0
    s = str(device).strip().lower()
    if s in ("", "cuda"):
        return 0
    if ":" in s:
        try:
            return max(0, int(s.split(":", 1)[1]))
        except ValueError:
            return 0
    if s.isdigit():
        return int(s)
    return 0


def _read_gpu_memory_pynvml(device_index: int) -> Optional[Tuple[float, float]]:
    """Return ``(used_pct, total_gb)`` for *device_index* via pynvml.

    Returns ``None`` when pynvml is missing or the read fails.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return None
        total = float(info.total)
        used = float(info.used)
        if total <= 0:
            return None
        return used / total * 100.0, total / (1024**3)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _read_gpu_memory_nvidia_smi(
    device_index: int,
) -> Optional[Tuple[float, float]]:
    """Return ``(used_pct, total_gb)`` via parsing ``nvidia-smi``.

    Returns ``None`` when nvidia-smi is unavailable or the device
    index is out of range.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            used_mib = float(parts[1])
            total_mib = float(parts[2])
        except ValueError:
            continue
        if idx != device_index or total_mib <= 0:
            continue
        return used_mib / total_mib * 100.0, total_mib / 1024.0
    return None


def capture_gpu_snapshot(output_path, *, header: str = "") -> Optional[str]:
    """Write a GPU diagnostic snapshot to disk for postmortem analysis.

    Captures the current ``nvidia-smi`` output and (if PyTorch is
    available with CUDA) ``torch.cuda.memory_summary`` for every
    visible device. The result is a plain-text file the operator can
    inspect to determine which process owned the GPU memory or what
    PyTorch's allocator thought it had reserved.

    Best-effort: failures during capture are recorded in the file
    rather than raising.

    Parameters:
        output_path (path-like): Destination file path. Parent
            directories are created if missing.
        header (str): Optional banner prepended to the file (e.g.
            "Host memory watchdog trip at 93.2%").

    Returns:
        path (str or None): The string path on success, ``None`` on
            failure.
    """
    import datetime as _dt
    from pathlib import Path as _Path

    target = _Path(output_path)
    lines: List[str] = []
    if header:
        lines.append(header)
        lines.append("=" * len(header))
        lines.append("")
    lines.append(f"Captured: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # nvidia-smi
    lines.append("-- nvidia-smi --")
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            timeout=10,
        )
        lines.append(out.rstrip())
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        lines.append(f"(nvidia-smi unavailable: {exc!r})")
    lines.append("")

    # torch memory summary
    lines.append("-- torch.cuda.memory_summary --")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                lines.append(f"\nDevice {i}:")
                try:
                    lines.append(torch.cuda.memory_summary(device=i, abbreviated=True))
                except Exception as exc:
                    lines.append(f"(memory_summary failed: {exc!r})")
        else:
            lines.append("(torch.cuda.is_available() = False)")
    except ImportError:
        lines.append("(torch not installed)")
    except Exception as exc:
        lines.append(f"(torch.cuda probe failed: {exc!r})")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8")
        return str(target)
    except Exception as exc:
        print(f"[gpu snapshot] failed to write {target}: {exc!r}")
        return None


def read_gpu_memory(
    device_index: int,
) -> Optional[Tuple[float, float]]:
    """Return ``(used_pct, total_gb)`` for *device_index*, or ``None``.

    Tries ``pynvml`` first, then ``nvidia-smi``. Returns ``None``
    when neither source can produce a reading (e.g. no NVIDIA driver,
    or the index is out of range).

    Parameters:
        device_index (int): Zero-based GPU index.

    Returns:
        info (tuple[float, float] or None): ``(used_pct, total_gb)``
            on success.
    """
    info = _read_gpu_memory_pynvml(device_index)
    if info is not None:
        return info
    return _read_gpu_memory_nvidia_smi(device_index)


def _try_capture_snapshot_to_results(log_path, header: str) -> None:
    """Write a GPU snapshot to the per-recording results folder.

    Used by watchdog abort paths to leave a postmortem artefact at
    ``<results_folder>/gpu_snapshot_at_trip.txt``. The watchdog must
    pass the log path captured at ``__enter__`` time on the main
    thread — the watchdog's polling thread cannot reliably look up
    the ``get_active_log_path`` ContextVar because Python does not
    propagate ContextVars across thread boundaries.

    Best-effort: failures (None log_path, write failure, etc.) are
    silent so a snapshot bug never breaks the surrounding watchdog.

    Parameters:
        log_path (Path or None): Per-recording log path; the
            results folder is its parent. ``None`` short-circuits.
        header (str): Banner to prepend to the snapshot file.
    """
    if log_path is None:
        return
    try:
        from pathlib import Path as _Path

        results_folder = _Path(log_path).parent
        target = results_folder / "gpu_snapshot_at_trip.txt"
        capture_gpu_snapshot(target, header=header)
    except Exception as exc:
        print(f"[gpu snapshot] failed to capture on trip: {exc!r}")


def resolve_active_device(config) -> int:
    """Pick the GPU device index implied by the sorter config.

    The watchdog measures only this device so unrelated GPUs running
    other workloads are ignored.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        index (int): Device index to monitor (defaults to 0).
    """
    sorter_name = getattr(config.sorter, "sorter_name", "").lower()
    if sorter_name == "rt_sort":
        return _resolve_device_index(getattr(config.rt_sort, "device", None))
    if sorter_name == "kilosort4":
        params = getattr(config.sorter, "sorter_params", None) or {}
        return _resolve_device_index(params.get("torch_device"))
    return 0


class GpuMemoryWatchdog:
    """Daemon-thread watchdog that aborts on GPU VRAM pressure.

    Use as a context manager around the per-recording sort. The
    watchdog polls the device's used-memory percentage; crossing
    ``warn_pct`` prints a rate-limited warning, and crossing
    ``abort_pct`` builds a :class:`GpuMemoryWatchdogError`,
    terminates registered subprocesses, runs kill callbacks, and
    raises into the main thread.

    Parameters:
        device_index (int): GPU index to monitor. Use
            :func:`resolve_active_device` to pick from the config.
        warn_pct (float): Used-memory percentage at which to warn.
            Defaults to ``85.0``.
        abort_pct (float): Used-memory percentage at which to abort.
            Defaults to ``95.0``.
        poll_interval_s (float): Seconds between polls. Defaults to
            ``2.0`` — GPU OOM is sharp, faster polling than host
            RAM is appropriate.
        warn_repeat_s (float): Minimum seconds between repeated
            warnings. Defaults to ``30.0``.
        kill_grace_s (float): Seconds between ``terminate()`` and
            ``kill()`` on registered subprocesses.

    Notes:
        - Disabled (no-op context manager) when no usable GPU info
          source is available, or when ``warn_pct >= abort_pct``
          would be invalid.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        warn_pct: float = 85.0,
        abort_pct: float = 95.0,
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
        self.device_index = int(device_index)
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
        self._tripped = False
        self._used_pct_at_trip: Optional[float] = None
        self._last_warn_t = 0.0
        self._enabled = False
        # Captured at ``__enter__`` time on the main thread because
        # ContextVars do not propagate to the polling thread.
        self._snapshot_log_path = None

    # ------------------------------------------------------------------
    # Trip-state queries
    # ------------------------------------------------------------------

    def tripped(self) -> bool:
        """Return True once the watchdog has fired its abort path."""
        return self._tripped

    def used_pct_at_trip(self) -> Optional[float]:
        """Return the used-memory percent at the trip moment, or None."""
        return self._used_pct_at_trip

    def make_error(self, message: Optional[str] = None) -> GpuMemoryWatchdogError:
        """Build a :class:`GpuMemoryWatchdogError` from the trip state.

        Parameters:
            message (str or None): Override the default message.

        Returns:
            err (GpuMemoryWatchdogError): Exception ready to raise.
        """
        if message is None:
            pct = (
                f"{self._used_pct_at_trip:.1f}"
                if self._used_pct_at_trip is not None
                else "?"
            )
            message = (
                f"GPU watchdog tripped: device {self.device_index} used "
                f"{pct}% (abort threshold {self.abort_pct:.1f}%)."
            )
        return GpuMemoryWatchdogError(
            message,
            device_index=self.device_index,
            used_pct_at_trip=self._used_pct_at_trip,
            abort_pct=self.abort_pct,
        )

    # ------------------------------------------------------------------
    # Registration (subprocesses + kill callbacks)
    # ------------------------------------------------------------------

    def register_subprocess(
        self,
        popen: subprocess.Popen,
        *,
        kill_grace_s: Optional[float] = None,
    ) -> None:
        """Track a subprocess for termination on watchdog abort."""
        grace = self.kill_grace_s if kill_grace_s is None else float(kill_grace_s)
        with self._lock:
            self._subprocesses.append((popen, grace))

    def unregister_subprocess(self, popen: subprocess.Popen) -> None:
        """Stop tracking a previously registered subprocess."""
        with self._lock:
            self._subprocesses = [
                (p, g) for (p, g) in self._subprocesses if p is not popen
            ]

    def register_kill_callback(self, callback: Callable[[], None]) -> None:
        """Track a zero-arg callable to invoke on watchdog abort."""
        with self._lock:
            self._kill_callbacks.append(callback)

    def unregister_kill_callback(self, callback: Callable[[], None]) -> None:
        """Stop tracking a previously registered kill callback."""
        with self._lock:
            self._kill_callbacks = [
                c for c in self._kill_callbacks if c is not callback
            ]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GpuMemoryWatchdog":
        # Capture the active per-recording log path on the main
        # thread; the daemon polling thread cannot read the
        # ContextVar reliably.
        try:
            from ._inactivity import get_active_log_path

            self._snapshot_log_path = get_active_log_path()
        except Exception:
            self._snapshot_log_path = None

        # Probe once before starting the thread so we can disable
        # cleanly when no GPU info source is available.
        info = read_gpu_memory(self.device_index)
        if info is None:
            print(
                f"[gpu memory watchdog] no GPU info available for "
                f"device {self.device_index} (no pynvml, no nvidia-smi). "
                "Disabled."
            )
            self._enabled = False
            return self
        self._enabled = True
        used_pct, total_gb = info
        print(
            f"[gpu memory watchdog] active: device={self.device_index} "
            f"({total_gb:.1f} GB) start={used_pct:.1f}% "
            f"warn>={self.warn_pct:.1f}% abort>={self.abort_pct:.1f}% "
            f"poll={self.poll_interval_s:.1f}s"
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"GpuMemoryWatchdog[{self.device_index}]",
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
        # Defer the first poll so __enter__ has time to return.
        if self._stop_event.wait(self.poll_interval_s):
            return
        while not self._stop_event.is_set():
            info = read_gpu_memory(self.device_index)
            if info is None:
                self._stop_event.wait(self.poll_interval_s)
                continue
            used_pct, _total_gb = info
            if used_pct >= self.abort_pct:
                self._on_abort(used_pct)
                return
            if used_pct >= self.warn_pct:
                self._maybe_warn(used_pct)
            self._stop_event.wait(self.poll_interval_s)

    def _maybe_warn(self, used_pct: float) -> None:
        """Print a warning if enough time has passed since the last one."""
        now = time.time()
        if now - self._last_warn_t < self.warn_repeat_s:
            return
        self._last_warn_t = now
        print(
            f"[gpu memory watchdog] WARNING: device {self.device_index} "
            f"VRAM at {used_pct:.1f}% (warn={self.warn_pct:.1f}% / "
            f"abort={self.abort_pct:.1f}%)."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="gpu_memory",
                event="warn",
                log_path=self._snapshot_log_path,
                device_index=self.device_index,
                used_pct=used_pct,
                warn_pct=self.warn_pct,
                abort_pct=self.abort_pct,
            )
        except Exception:
            pass

    def _on_abort(self, used_pct: float) -> None:
        """Record trip, terminate subprocesses, run callbacks, interrupt main."""
        self._tripped = True
        self._used_pct_at_trip = used_pct
        print(
            f"[gpu memory watchdog] ABORT: device {self.device_index} "
            f"VRAM at {used_pct:.1f}% (>= {self.abort_pct:.1f}%). "
            "Terminating subprocesses and raising into main thread."
        )
        try:
            from ._audit import append_audit_event

            append_audit_event(
                watchdog="gpu_memory",
                event="abort",
                log_path=self._snapshot_log_path,
                device_index=self.device_index,
                used_pct=used_pct,
                abort_pct=self.abort_pct,
            )
        except Exception:
            pass
        _try_capture_snapshot_to_results(
            self._snapshot_log_path,
            f"GPU memory watchdog trip — device {self.device_index} at "
            f"{used_pct:.1f}%",
        )
        with self._lock:
            entries = list(self._subprocesses)
            callbacks = list(self._kill_callbacks)
        for popen, _grace in entries:
            try:
                if popen.poll() is None:
                    popen.terminate()
            except Exception as exc:
                print(
                    f"[gpu memory watchdog] terminate() failed for pid="
                    f"{getattr(popen, 'pid', '?')}: {exc}"
                )
        if entries:
            time.sleep(max((g for _, g in entries), default=self.kill_grace_s))
        for popen, _grace in entries:
            try:
                if popen.poll() is None:
                    popen.kill()
            except Exception as exc:
                print(
                    f"[gpu memory watchdog] kill() failed for pid="
                    f"{getattr(popen, 'pid', '?')}: {exc}"
                )
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                print(
                    f"[gpu memory watchdog] kill_callback raised: {exc!r}; "
                    "continuing."
                )
        try:
            _thread.interrupt_main()
        except Exception as exc:
            print(f"[gpu memory watchdog] failed to interrupt main: {exc}")
