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
import logging
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .._exceptions import GpuMemoryWatchdogError, GpuThermalWatchdogError
from ._audit import append_audit_event

_logger = logging.getLogger(__name__)

# Throttle reason bits we surface as warnings (kernel docs:
# https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html).
# We deliberately ignore Idle/AppClocks because those reflect benign
# OS scheduling rather than hardware distress.
_THROTTLE_REASON_LABELS = (
    (0x4, "SW power cap"),
    (0x8, "HW slowdown"),
    (0x20, "SW thermal slowdown"),
    (0x40, "HW thermal slowdown"),
    (0x80, "HW power brake"),
)


_active_gpu_watchdog: contextvars.ContextVar[Optional["GpuMemoryWatchdog"]] = (
    contextvars.ContextVar("active_gpu_memory_watchdog", default=None)
)


def get_active_gpu_watchdog() -> Optional["GpuMemoryWatchdog"]:
    """Return the GPU watchdog active for the current context, or None.

    Mirror of :func:`._watchdog.get_active_watchdog` for the GPU
    watchdog. Lets the per-recording :class:`KeyboardInterrupt`
    catch site discover a tripped GPU watchdog and convert the
    interrupt into the appropriate classified error.

    Returns:
        watchdog (GpuMemoryWatchdog or None): The active instance,
            or ``None`` when no GPU watchdog is currently running.
    """
    return _active_gpu_watchdog.get()


class _PynvmlSession:
    """Long-lived pynvml handle for one device.

    Initialises pynvml once, caches the per-device handle, and reads
    memory / temperature / throttle reasons via the cached handle.
    Replaces the per-call ``nvmlInit/nvmlShutdown`` pattern, which
    serialised every poll and added measurable overhead at the
    default 2-second cadence.

    Best-effort: per-method failures return ``None`` rather than
    raising. ``shutdown()`` is idempotent and safe to call on a
    session that never initialised.
    """

    def __init__(self, device_index: int) -> None:
        self.device_index = int(device_index)
        self._pynvml = None
        self._handle = None

    def start(self) -> bool:
        """Initialise pynvml and resolve the device handle.

        Returns:
            ok (bool): ``True`` when pynvml is importable and the
                handle resolves; ``False`` otherwise.
        """
        try:
            import pynvml
        except ImportError:
            return False
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return False
        self._pynvml = pynvml
        self._handle = handle
        return True

    def read_memory(self) -> Optional[Tuple[float, float]]:
        """Return ``(used_pct, total_gb)`` or ``None`` on failure."""
        if self._handle is None or self._pynvml is None:
            return None
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception:
            return None
        total = float(info.total)
        if total <= 0:
            return None
        return float(info.used) / total * 100.0, total / (1024**3)

    def read_temperature_c(self) -> Optional[float]:
        """Return device temperature in degrees Celsius, or ``None``."""
        if self._handle is None or self._pynvml is None:
            return None
        try:
            # Sensor 0 is NVML_TEMPERATURE_GPU on every supported device.
            return float(self._pynvml.nvmlDeviceGetTemperature(self._handle, 0))
        except Exception:
            return None

    def read_throttle_reasons(self) -> Optional[int]:
        """Return the active throttle-reasons bitmask, or ``None``."""
        if self._handle is None or self._pynvml is None:
            return None
        try:
            return int(
                self._pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle)
            )
        except Exception:
            return None

    def shutdown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._pynvml = None
        self._handle = None


def _format_throttle_reasons(mask: int) -> str:
    """Render an active throttle-reasons mask as a comma-separated string."""
    parts = [label for bit, label in _THROTTLE_REASON_LABELS if mask & bit]
    return ", ".join(parts)


def _resolve_device_index(device: Optional[str]) -> int:
    """Return the integer device index for a torch-style device string.

    Accepts ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, integer-like
    strings, and ``None`` (interpreted as device 0). Falls back to 0
    on parse failure rather than raising — the watchdog is
    best-effort — but emits a warning so the silent fallback is
    visible in logs. A user who meant ``cuda:1`` and typo'd
    ``cuda;1`` would otherwise have their GPU watchdog quietly
    watching the wrong device.

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
            _logger.warning(
                "GPU watchdog: could not parse device index from %r; "
                "falling back to device 0.",
                device,
            )
            return 0
    if s.isdigit():
        return int(s)
    _logger.warning(
        "GPU watchdog: unrecognised device string %r; " "falling back to device 0.",
        device,
    )
    return 0


def _read_gpu_memory_pynvml(device_index: int) -> Optional[Tuple[float, float]]:
    """Return ``(used_pct, total_gb)`` for *device_index* via pynvml.

    Reuses :class:`_PynvmlSession` so the init / shutdown lifecycle
    is owned by a single class rather than duplicated across the
    per-call helper. Returns ``None`` when pynvml is missing or the
    read fails.
    """
    session = _PynvmlSession(device_index)
    if not session.start():
        return None
    try:
        return session.read_memory()
    finally:
        session.shutdown()


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

    # nvidia-smi. Force the C locale so column labels and units in
    # the human-readable output are predictable English text — the
    # snapshot is grepped through after the fact, and operators on
    # non-English hosts (de_DE, ja_JP, etc.) would otherwise have to
    # decode localised labels.
    import os as _os

    smi_env = {**_os.environ, "LANG": "C", "LC_ALL": "C"}
    lines.append("-- nvidia-smi --")
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            timeout=10,
            env=smi_env,
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
        _logger.error("snapshot failed to write %s: %r", target, exc)
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
        _logger.error("snapshot failed to capture on trip: %r", exc)


def _resolve_rt_sort_device(config) -> int:
    """Resolver for the RT-Sort sorter."""
    return _resolve_device_index(getattr(config.rt_sort, "device", None))


def _resolve_kilosort4_device(config) -> int:
    """Resolver for the Kilosort4 host sorter."""
    params = getattr(config.sorter, "sorter_params", None) or {}
    return _resolve_device_index(params.get("torch_device"))


# Registry mapping sorter name (lowercase) to a callable that takes
# the config and returns the integer device index to monitor.
# Sorters absent from this registry default to device 0 — the
# convention KS2-Docker and other CUDA-using sorters that don't
# expose a configurable device follow.
#
# Adding a new GPU-using sorter: write a ``_resolve_<sorter>_device``
# function that pulls the device index from wherever the sorter
# config holds it, then add an entry here. No other call site needs
# updating.
_DEVICE_RESOLVERS: Dict[str, Callable[[Any], int]] = {
    "rt_sort": _resolve_rt_sort_device,
    "kilosort4": _resolve_kilosort4_device,
}


def resolve_active_device(config) -> int:
    """Pick the GPU device index implied by the sorter config.

    The watchdog measures only this device so unrelated GPUs running
    other workloads are ignored.

    Parameters:
        config (SortingPipelineConfig): Pipeline configuration.

    Returns:
        index (int): Device index to monitor (defaults to 0 for any
            sorter not registered in :data:`_DEVICE_RESOLVERS`).
    """
    sorter_name = getattr(config.sorter, "sorter_name", "").lower()
    resolver = _DEVICE_RESOLVERS.get(sorter_name)
    if resolver is None:
        return 0
    return resolver(config)


class GpuMemoryWatchdog:
    """Daemon-thread watchdog that aborts on GPU VRAM or thermal pressure.

    Use as a context manager around the per-recording sort. Each
    poll inspects three signals:

    * **VRAM usage** — crossing ``warn_pct`` prints a rate-limited
      warning; crossing ``abort_pct`` builds a
      :class:`GpuMemoryWatchdogError`, terminates registered
      subprocesses, runs kill callbacks, and raises into the main
      thread.
    * **Device temperature** — crossing ``warn_temp_c`` prints a
      rate-limited warning; crossing ``abort_temp_c`` aborts with a
      :class:`GpuThermalWatchdogError`. Sustained operation above
      the GPU's thermal junction limit risks driver-level throttling
      that silently degrades sort output.
    * **Active throttle reasons** — when the device reports SW/HW
      power-cap or thermal slowdown, prints a rate-limited warning
      (no abort: the device is already protecting itself).

    Parameters:
        device_index (int): GPU index to monitor. Use
            :func:`resolve_active_device` to pick from the config.
        warn_pct (float): Used-memory percentage at which to warn.
            Defaults to ``85.0``.
        abort_pct (float): Used-memory percentage at which to abort.
            Defaults to ``95.0``.
        poll_interval_s (float): Seconds between polls. Defaults to
            ``2.0``.
        warn_repeat_s (float): Minimum seconds between repeated
            warnings. Defaults to ``30.0``.
        kill_grace_s (float): Seconds between ``terminate()`` and
            ``kill()`` on registered subprocesses.
        warn_temp_c (float or None): Temperature in degrees Celsius
            at which to warn. ``None`` disables the warn-stage temp
            check. Defaults to ``85.0``.
        abort_temp_c (float or None): Temperature at which to abort.
            ``None`` disables thermal aborts. Defaults to ``92.0``.
        monitor_throttle_reasons (bool): When True, surface NVML
            throttle reasons (SW power cap, HW thermal slowdown,
            HW power brake) as rate-limited warnings. Defaults to
            ``True``.

    Notes:
        - Thermal monitoring requires ``pynvml``; the
          ``nvidia-smi``-only fallback path used by
          :func:`read_gpu_memory` does not surface temperature.
          When pynvml is missing, thermal/throttle checks silently
          degrade while VRAM monitoring continues via nvidia-smi.
        - Disabled (no-op context manager) when no usable GPU info
          source is available.
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
        warn_temp_c: Optional[float] = 85.0,
        abort_temp_c: Optional[float] = 92.0,
        monitor_throttle_reasons: bool = True,
    ) -> None:
        if not 0.0 < warn_pct < abort_pct <= 100.0:
            raise ValueError(
                f"warn_pct ({warn_pct}) and abort_pct ({abort_pct}) must "
                f"satisfy 0 < warn_pct < abort_pct <= 100."
            )
        if np.isnan(poll_interval_s) or poll_interval_s <= 0.0:
            raise ValueError(
                f"poll_interval_s must be positive, got {poll_interval_s}."
            )
        if np.isnan(kill_grace_s) or kill_grace_s < 0.0:
            raise ValueError(f"kill_grace_s must be non-negative, got {kill_grace_s}.")
        if (
            warn_temp_c is not None
            and abort_temp_c is not None
            and not 0.0 < warn_temp_c < abort_temp_c
        ):
            raise ValueError(
                f"warn_temp_c ({warn_temp_c}) and abort_temp_c "
                f"({abort_temp_c}) must satisfy 0 < warn_temp_c < "
                "abort_temp_c."
            )
        self.device_index = int(device_index)
        self.warn_pct = float(warn_pct)
        self.abort_pct = float(abort_pct)
        self.poll_interval_s = float(poll_interval_s)
        self.warn_repeat_s = float(warn_repeat_s)
        self.kill_grace_s = float(kill_grace_s)
        self.warn_temp_c = float(warn_temp_c) if warn_temp_c is not None else None
        self.abort_temp_c = float(abort_temp_c) if abort_temp_c is not None else None
        self.monitor_throttle_reasons = bool(monitor_throttle_reasons)

        self._subprocesses: List[Tuple[subprocess.Popen, float]] = []
        self._kill_callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tripped = False
        self._tripped_kind: Optional[str] = None
        self._used_pct_at_trip: Optional[float] = None
        self._temp_c_at_trip: Optional[float] = None
        self._last_warn_t = 0.0
        self._last_temp_warn_t = 0.0
        self._last_throttle_warn_t = 0.0
        self._enabled = False
        self._session: Optional[_PynvmlSession] = None
        self._token: Optional[contextvars.Token] = None
        # Captured at ``__enter__`` time on the main thread because
        # ContextVars do not propagate to the polling thread.
        self._snapshot_log_path = None
        # Set True when the trip cascade ran but
        # ``_thread.interrupt_main`` raised — the main thread did not
        # receive the KeyboardInterrupt and will surface a downstream
        # error instead. Catch sites read this via
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

        When True, GPU protection ran successfully (subprocesses
        terminated, kill callbacks invoked) but the main thread did
        not receive a ``KeyboardInterrupt``. The pipeline's catch
        site checks this to reclassify a downstream exception caused
        by the now-dead subprocess.

        Returns:
            failed (bool): True only when the watchdog tripped and
                the interrupt delivery raised.
        """
        return self._interrupt_main_failed

    def used_pct_at_trip(self) -> Optional[float]:
        """Return the used-memory percent at the trip moment, or None."""
        return self._used_pct_at_trip

    def temperature_c_at_trip(self) -> Optional[float]:
        """Return the device temperature at the trip moment, or None."""
        return self._temp_c_at_trip

    def trip_kind(self) -> Optional[str]:
        """Return ``"memory"``, ``"thermal"``, or ``None`` if not tripped."""
        return self._tripped_kind

    def make_error(
        self, message: Optional[str] = None
    ) -> Union[GpuMemoryWatchdogError, GpuThermalWatchdogError]:
        """Build the trip-kind-appropriate watchdog error.

        Parameters:
            message (str or None): Override the default message.

        Returns:
            err: :class:`GpuMemoryWatchdogError` for VRAM trips,
                :class:`GpuThermalWatchdogError` for temperature
                trips. Falls back to a memory-shaped error when the
                trip kind is unset.
        """
        if self._tripped_kind == "thermal":
            if message is None:
                temp = (
                    f"{self._temp_c_at_trip:.1f}"
                    if self._temp_c_at_trip is not None
                    else "?"
                )
                abort_temp = (
                    f"{self.abort_temp_c:.1f}" if self.abort_temp_c is not None else "?"
                )
                message = (
                    f"GPU thermal watchdog tripped: device "
                    f"{self.device_index} at {temp} C "
                    f"(abort threshold {abort_temp} C)."
                )
            return GpuThermalWatchdogError(
                message,
                device_index=self.device_index,
                temperature_c_at_trip=self._temp_c_at_trip,
                abort_temp_c=self.abort_temp_c,
            )
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
            _logger.warning(
                "no GPU info available for device %d (no pynvml, no "
                "nvidia-smi). Disabled.",
                self.device_index,
            )
            self._enabled = False
            return self
        self._enabled = True
        # Publish the active watchdog so the per-recording
        # ``KeyboardInterrupt`` catch site can convert a
        # ``_thread.interrupt_main`` from this watchdog into a
        # classified error rather than letting it bubble up raw.
        self._token = _active_gpu_watchdog.set(self)
        used_pct, total_gb = info
        # Try to set up a long-lived pynvml session for the polling
        # thread. Falls back to per-poll ``read_gpu_memory`` (which
        # uses nvidia-smi) when pynvml is unavailable; in that case
        # thermal / throttle monitoring degrades silently because
        # nvidia-smi-only does not expose those signals here.
        session = _PynvmlSession(self.device_index)
        if session.start():
            self._session = session
        else:
            self._session = None
        thermal_str = ""
        if self._session is not None and (
            self.warn_temp_c is not None or self.abort_temp_c is not None
        ):
            initial_temp = self._session.read_temperature_c()
            if initial_temp is not None:
                thermal_str = (
                    f" temp_warn>={self.warn_temp_c} "
                    f"abort>={self.abort_temp_c} (now {initial_temp:.1f}C)"
                )
        _logger.info(
            "active: device=%d (%.1f GB) start=%.1f%% warn>=%.1f%% "
            "abort>=%.1f%% poll=%.1fs%s",
            self.device_index,
            total_gb,
            used_pct,
            self.warn_pct,
            self.abort_pct,
            self.poll_interval_s,
            thermal_str,
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
        if self._token is not None:
            try:
                _active_gpu_watchdog.reset(self._token)
            except (LookupError, ValueError, RuntimeError):
                # Another context modified the var between set/reset,
                # or the token was already consumed (Python 3.10+
                # raises RuntimeError on re-used tokens).
                pass
            self._token = None
        if self._session is not None:
            self._session.shutdown()
            self._session = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Polling loop: warn, then trip, then exit."""
        # Defer the first poll so __enter__ has time to return.
        if self._stop_event.wait(self.poll_interval_s):
            return
        blind_threshold_s = 5.0 * self.warn_repeat_s
        # VRAM and thermal/throttle have independent failure modes
        # (the same _session may succeed for one and fail for the
        # other on driver corner cases) so they get independent
        # blindness trackers.
        vram_blind_started_t: Optional[float] = None
        vram_blind_warned = False
        thermal_blind_started_t: Optional[float] = None
        thermal_blind_warned = False
        while not self._stop_event.is_set():
            now = time.time()
            # Memory: prefer the cached pynvml session, fall back
            # to the free-function reader (which uses nvidia-smi).
            if self._session is not None:
                info = self._session.read_memory()
            else:
                info = read_gpu_memory(self.device_index)
            if info is None:
                if vram_blind_started_t is None:
                    vram_blind_started_t = now
                elif (
                    not vram_blind_warned
                    and now - vram_blind_started_t >= blind_threshold_s
                ):
                    self._warn_blind_vram(now - vram_blind_started_t)
                    vram_blind_warned = True
            else:
                vram_blind_started_t = None
                vram_blind_warned = False
                used_pct, _total_gb = info
                if used_pct >= self.abort_pct:
                    self._on_abort(used_pct)
                    return
                if used_pct >= self.warn_pct:
                    self._maybe_warn(used_pct)

            # Thermal + throttle reasons require pynvml; skip when
            # the session is unavailable. When _session is present
            # but a configured sub-read returns None, that counts
            # as thermal blindness for the warning tracker.
            if self._session is not None:
                thermal_configured = (
                    self.warn_temp_c is not None
                    or self.abort_temp_c is not None
                    or self.monitor_throttle_reasons
                )
                thermal_unreadable = False
                if self.warn_temp_c is not None or self.abort_temp_c is not None:
                    temp_c = self._session.read_temperature_c()
                    if temp_c is None:
                        thermal_unreadable = True
                    else:
                        if (
                            self.abort_temp_c is not None
                            and temp_c >= self.abort_temp_c
                        ):
                            self._on_thermal_abort(temp_c)
                            return
                        if self.warn_temp_c is not None and temp_c >= self.warn_temp_c:
                            self._maybe_warn_temp(temp_c)
                if self.monitor_throttle_reasons:
                    mask = self._session.read_throttle_reasons()
                    if mask is None:
                        thermal_unreadable = True
                    elif mask & sum(bit for bit, _ in _THROTTLE_REASON_LABELS):
                        self._maybe_warn_throttle(mask)

                if thermal_configured and thermal_unreadable:
                    if thermal_blind_started_t is None:
                        thermal_blind_started_t = now
                    elif (
                        not thermal_blind_warned
                        and now - thermal_blind_started_t >= blind_threshold_s
                    ):
                        self._warn_blind_thermal(now - thermal_blind_started_t)
                        thermal_blind_warned = True
                else:
                    thermal_blind_started_t = None
                    thermal_blind_warned = False

            self._stop_event.wait(self.poll_interval_s)

    def _maybe_warn(self, used_pct: float) -> None:
        """Print a warning if enough time has passed since the last one."""
        now = time.time()
        if now - self._last_warn_t < self.warn_repeat_s:
            return
        self._last_warn_t = now
        _logger.warning(
            "device %d VRAM at %.1f%% (warn=%.1f%% / abort=%.1f%%).",
            self.device_index,
            used_pct,
            self.warn_pct,
            self.abort_pct,
        )
        append_audit_event(
            watchdog="gpu_memory",
            event="warn",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            used_pct=used_pct,
            warn_pct=self.warn_pct,
            abort_pct=self.abort_pct,
        )

    def _maybe_warn_temp(self, temp_c: float) -> None:
        """Print a thermal warning if rate-limit allows."""
        now = time.time()
        if now - self._last_temp_warn_t < self.warn_repeat_s:
            return
        self._last_temp_warn_t = now
        abort = f"{self.abort_temp_c:.1f}" if self.abort_temp_c is not None else "off"
        _logger.warning(
            "device %d temperature %.1f C (warn>=%.1f / abort>=%s).",
            self.device_index,
            temp_c,
            self.warn_temp_c,
            abort,
        )
        append_audit_event(
            watchdog="gpu_thermal",
            event="warn",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            temperature_c=temp_c,
            warn_temp_c=self.warn_temp_c,
            abort_temp_c=self.abort_temp_c,
        )

    def _maybe_warn_throttle(self, mask: int) -> None:
        """Print a throttle-reason warning if rate-limit allows."""
        now = time.time()
        if now - self._last_throttle_warn_t < self.warn_repeat_s:
            return
        self._last_throttle_warn_t = now
        reasons = _format_throttle_reasons(mask) or f"mask=0x{mask:x}"
        _logger.warning(
            "device %d throttling — %s.",
            self.device_index,
            reasons,
        )
        append_audit_event(
            watchdog="gpu_throttle",
            event="warn",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            throttle_mask=int(mask),
            throttle_reasons=reasons,
        )

    def _warn_blind_vram(self, blind_for: float) -> None:
        _logger.warning(
            "VRAM reading for device %d unreadable for %.1fs — "
            "watchdog is blind to VRAM-pressure aborts until readings "
            "recover.",
            self.device_index,
            blind_for,
        )
        append_audit_event(
            watchdog="gpu_memory",
            event="blind_warn",
            source="vram",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            blind_for_s=blind_for,
        )

    def _warn_blind_thermal(self, blind_for: float) -> None:
        _logger.warning(
            "thermal/throttle reading for device %d unreadable for "
            "%.1fs — watchdog is blind to thermal aborts until "
            "readings recover.",
            self.device_index,
            blind_for,
        )
        append_audit_event(
            watchdog="gpu_memory",
            event="blind_warn",
            source="thermal",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            blind_for_s=blind_for,
        )

    def _on_thermal_abort(self, temp_c: float) -> None:
        """Trip on thermal threshold; terminate, run callbacks, raise."""
        self._tripped = True
        self._tripped_kind = "thermal"
        self._temp_c_at_trip = temp_c
        abort = f"{self.abort_temp_c:.1f}" if self.abort_temp_c is not None else "?"
        _logger.error(
            "THERMAL ABORT: device %d at %.1f C (>= %s C). "
            "Terminating subprocesses and raising into main thread.",
            self.device_index,
            temp_c,
            abort,
        )
        append_audit_event(
            watchdog="gpu_thermal",
            event="abort",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            temperature_c=temp_c,
            abort_temp_c=self.abort_temp_c,
        )
        _try_capture_snapshot_to_results(
            self._snapshot_log_path,
            f"GPU thermal watchdog trip — device {self.device_index} at "
            f"{temp_c:.1f} C",
        )
        self._kill_targets_and_interrupt()

    def _on_abort(self, used_pct: float) -> None:
        """Record trip, terminate subprocesses, run callbacks, interrupt main."""
        self._tripped = True
        self._tripped_kind = "memory"
        self._used_pct_at_trip = used_pct
        _logger.error(
            "ABORT: device %d VRAM at %.1f%% (>= %.1f%%). Terminating "
            "subprocesses and raising into main thread.",
            self.device_index,
            used_pct,
            self.abort_pct,
        )
        append_audit_event(
            watchdog="gpu_memory",
            event="abort",
            log_path=self._snapshot_log_path,
            device_index=self.device_index,
            used_pct=used_pct,
            abort_pct=self.abort_pct,
        )
        _try_capture_snapshot_to_results(
            self._snapshot_log_path,
            f"GPU memory watchdog trip — device {self.device_index} at "
            f"{used_pct:.1f}%",
        )
        self._kill_targets_and_interrupt()

    def _kill_targets(self) -> None:
        """Terminate registered subprocesses and run kill callbacks.

        Side-effect-only: returns nothing, raises only the explicit
        ``SystemExit`` / ``KeyboardInterrupt`` propagation case.
        Split from :meth:`_kill_targets_and_interrupt` so a future
        telemetry-only abort path can run the destructive cascade
        without injecting a ``KeyboardInterrupt`` into the main
        thread.

        Notes:
            - When ``entries`` is empty (no subprocesses registered),
              the ``if entries:`` gate skips the grace sleep — there
              is nothing to terminate, so callbacks fire immediately.
              The ``default=self.kill_grace_s`` on the ``max(...)``
              call is unreachable in that case but kept for clarity.
        """
        with self._lock:
            entries = list(self._subprocesses)
            callbacks = list(self._kill_callbacks)
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
            time.sleep(max((g for _, g in entries), default=self.kill_grace_s))
        for popen, _grace in entries:
            try:
                if popen.poll() is None:
                    popen.kill()
            except Exception as exc:
                _logger.error(
                    "kill() failed for pid=%s: %s",
                    getattr(popen, "pid", "?"),
                    exc,
                )
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

    def _kill_targets_and_interrupt(self) -> None:
        """Common subprocess + callback termination + interrupt main.

        Used by the abort paths (``_on_abort``, ``_on_thermal_abort``)
        which both want destructive cleanup followed by
        ``_thread.interrupt_main`` to convert the daemon-thread trip
        into a classified main-thread error.
        """
        self._kill_targets()
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
                watchdog="gpu_memory",
                event="interrupt_delivery_failed",
                log_path=self._snapshot_log_path,
                device_index=self.device_index,
                error=repr(exc),
            )
