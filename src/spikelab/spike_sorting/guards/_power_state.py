"""Cross-platform: prevent the system from sleeping during a sort.

A laptop closing its lid mid-sort, or the OS kicking off a
"modern standby" / suspend cycle, can suspend the whole sort
process. Resume is often incomplete (CUDA contexts lost, file
handles invalidated, network mounts stale). The cleanest fix is
to ask the OS not to sleep while the sort is running.

Per-platform mechanism:

* Windows — ``SetThreadExecutionState(ES_CONTINUOUS |
  ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)``. Released on
  context exit by clearing the flags.
* macOS — spawns ``caffeinate -dims`` as a child process.
  ``-d`` prevents display sleep, ``-i`` prevents idle sleep,
  ``-m`` prevents disk idle, ``-s`` blocks system sleep on AC
  power. The child is terminated on context exit.
* Linux — spawns ``systemd-inhibit --what=sleep:idle ...
  sleep infinity`` as a child process; the inhibitor lock
  releases when the child exits. Requires systemd; non-systemd
  inits fall back to no-op with a one-line notice.

In every case the context manager is best-effort: when the
required tool is missing the context yields ``False`` rather than
raising.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from typing import Iterator, Optional

# Windows API constants — defined here to avoid the ctypes import
# on non-Windows hosts.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040  # avoid screensaver suspend on desktops


@contextlib.contextmanager
def prevent_system_sleep() -> Iterator[bool]:
    """Ask the OS not to enter sleep / hibernate during the context.

    Cross-platform: uses ``SetThreadExecutionState`` on Windows,
    ``caffeinate`` on macOS, and ``systemd-inhibit`` on Linux. On
    every other platform the context manager is a no-op and yields
    ``False``.

    Yields:
        active (bool): ``True`` when the safeguard is engaged,
            ``False`` when it is unavailable on this platform or
            could not be installed.

    Notes:
        - macOS / Linux variants spawn a child process that holds
          the inhibitor lock; the child is terminated on context
          exit so the lock releases promptly.
        - The display may still blank under all backends — only
          *system* sleep is prevented, not display blanking.
    """
    if sys.platform == "win32":
        yield from _prevent_sleep_windows()
        return
    if sys.platform == "darwin":
        yield from _prevent_sleep_macos()
        return
    if sys.platform.startswith("linux"):
        yield from _prevent_sleep_linux()
        return
    yield False


def _prevent_sleep_windows() -> Iterator[bool]:
    try:
        import ctypes  # noqa: WPS433
    except Exception:
        yield False
        return

    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED

    try:
        kernel32 = ctypes.windll.kernel32
        prev = kernel32.SetThreadExecutionState(flags)
        if prev == 0:
            print(
                "[power state] SetThreadExecutionState returned 0; "
                "treating sleep prevention as active but the OS may "
                "not have honoured the request."
            )
        else:
            print("[power state] active (Windows): system sleep prevented.")
    except Exception as exc:
        print(f"[power state] failed to engage sleep prevention: {exc!r}")
        yield False
        return

    try:
        yield True
    finally:
        try:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            pass


def _spawn_inhibitor(argv: list, label: str) -> Optional[subprocess.Popen]:
    """Spawn an inhibitor child or return ``None`` on failure."""
    try:
        return subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(f"[power state] {label} not found; sleep prevention " "unavailable.")
        return None
    except Exception as exc:
        print(f"[power state] failed to spawn {label}: {exc!r}")
        return None


def _terminate_inhibitor(proc: subprocess.Popen, label: str) -> None:
    """Best-effort terminate-then-kill of the inhibitor child."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass


def _prevent_sleep_macos() -> Iterator[bool]:
    proc = _spawn_inhibitor(
        ["caffeinate", "-dims"],
        "caffeinate",
    )
    if proc is None:
        yield False
        return
    print("[power state] active (macOS): caffeinate -dims engaged.")
    try:
        yield True
    finally:
        _terminate_inhibitor(proc, "caffeinate")


def _prevent_sleep_linux() -> Iterator[bool]:
    # ``systemd-inhibit`` holds the lock for the lifetime of its
    # child command; ``sleep infinity`` is the canonical "stay
    # alive until killed" placeholder.
    proc = _spawn_inhibitor(
        [
            "systemd-inhibit",
            "--what=sleep:idle",
            "--who=spikelab",
            "--why=spike sorting in progress",
            "--mode=block",
            "sleep",
            "infinity",
        ],
        "systemd-inhibit",
    )
    if proc is None:
        yield False
        return
    print("[power state] active (Linux): systemd-inhibit engaged.")
    try:
        yield True
    finally:
        _terminate_inhibitor(proc, "systemd-inhibit")
