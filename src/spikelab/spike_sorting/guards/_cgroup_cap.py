"""Linux cgroup v2 memory cap for the sort process.

On Linux, ``_bounded_host_memory`` uses ``RLIMIT_DATA`` to bound the
data segment, but anonymous mmap regions and shared memory used by
PyTorch/numpy/MATLAB can still escape the cap. The kernel-level
equivalent of the Windows Job Object is a cgroup v2 ``memory.max``
write — when the process commits beyond it, the kernel OOM-kills it
deterministically.

This module opportunistically writes ``memory.max`` to a writable
cgroup v2 file when one is reachable (typical when the user runs the
sort inside a ``systemd-run --user --scope`` wrapper). When no
writable cgroup is present, the context manager is a no-op and
yields ``False`` — the userspace :class:`HostMemoryWatchdog` is
still the primary protection in that case.

Best-effort: failures (no cgroup v2, no permission, no detectable
host RAM) print a one-line notice and yield ``False`` rather than
raising.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator, Optional


def _detect_cgroup_v2_memory_max() -> Optional[Path]:
    """Return the writable ``memory.max`` path for the current cgroup, or None.

    Reads ``/proc/self/cgroup`` to find the cgroup v2 path the
    current process is attached to, then resolves
    ``/sys/fs/cgroup/<path>/memory.max`` and returns it only when
    that file exists and is writable. cgroup v1 hierarchies and
    hybrid setups return ``None``.
    """
    proc_cgroup = Path("/proc/self/cgroup")
    if not proc_cgroup.is_file():
        return None
    try:
        text = proc_cgroup.read_text()
    except OSError:
        return None

    # cgroup v2 unified hierarchy lines look like ``0::/user.slice/...``.
    # The empty controller field after the first colon is the v2 marker.
    cgroup_path: Optional[str] = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[1] == "":
            cgroup_path = parts[2].strip()
            break
    if cgroup_path is None:
        return None

    target = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.max"
    if not target.is_file():
        return None
    if not os.access(target, os.W_OK):
        return None
    return target


def _read_total_ram_bytes() -> Optional[int]:
    """Return total host RAM in bytes via ``get_system_ram_bytes``."""
    try:
        from ..sorting_utils import get_system_ram_bytes

        return get_system_ram_bytes()
    except Exception:
        return None


@contextlib.contextmanager
def linux_cgroup_v2_memory_cap(frac: float = 0.8) -> Iterator[bool]:
    """Bound the current process's memory via cgroup v2 ``memory.max``.

    On Linux, when the calling process is attached to a writable
    cgroup v2 hierarchy (typical with ``systemd-run --user --scope``
    or a container runtime that delegates a sub-cgroup), writes
    ``frac * total_host_ram`` to ``memory.max``. The kernel
    OOM-kills the process if it commits beyond the cap.

    On non-Linux, when no cgroup v2 hierarchy is reachable, when the
    cgroup is not writable, or when host RAM cannot be detected, the
    context manager is a no-op and yields ``False``. The previous
    ``memory.max`` value is restored on exit so the cap does not
    leak into a longer-lived shell.

    Parameters:
        frac (float): Fraction of total host RAM to cap memory at.
            Defaults to ``0.8`` (80%).

    Yields:
        active (bool): ``True`` when the cap was successfully
            installed; ``False`` otherwise.

    Notes:
        - To enable the cap on a workstation without root, run the
          sort under ``systemd-run --user --scope -- python ...`` —
          systemd creates a delegated sub-cgroup writable by the
          calling user.
        - The cap complements (does not replace) the userspace
          :class:`HostMemoryWatchdog`, which provides early warn
          signals and a graceful abort before the kernel OOM-killer
          fires.
    """
    if sys.platform != "linux":
        yield False
        return

    target = _detect_cgroup_v2_memory_max()
    if target is None:
        yield False
        return

    ram_bytes = _read_total_ram_bytes()
    if ram_bytes is None or ram_bytes <= 0:
        print(
            "[cgroup cap] cgroup v2 memory.max writable but host RAM not "
            "detectable; cap not enforced."
        )
        yield False
        return

    cap_bytes = int(ram_bytes * float(frac))

    try:
        prev = target.read_text().strip()
    except OSError as exc:
        print(f"[cgroup cap] could not read existing memory.max: {exc!r}")
        yield False
        return

    try:
        target.write_text(str(cap_bytes))
    except OSError as exc:
        print(f"[cgroup cap] failed to set memory.max: {exc!r}")
        yield False
        return

    print(
        f"[cgroup cap] active: memory.max = {cap_bytes / 1e9:.1f} GB "
        f"(= {frac * 100:.0f}% of {ram_bytes / 1e9:.1f} GB host RAM, "
        f"cgroup v2)."
    )

    try:
        yield True
    finally:
        try:
            target.write_text(prev)
        except OSError:
            pass
