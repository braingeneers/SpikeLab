"""Windows-only: prevent the system from sleeping during a sort.

A laptop closing its lid mid-sort, or Windows kicking off its
"modern standby" cycle, can suspend the whole sort process. The
process resumes when the system wakes — but in many failure modes
the resume is incomplete (CUDA contexts lost, file handles
invalidated, network mounts stale). The cleanest fix is to ask
the OS not to sleep while the sort is running.

This module wraps Windows' ``SetThreadExecutionState`` API. The
call sets ``ES_CONTINUOUS | ES_SYSTEM_REQUIRED`` for the calling
thread for the duration of the context, then restores the prior
state on exit.

The context manager is a no-op on non-Windows. Failure to acquire
the API (e.g. ctypes problem) is also a no-op — the safeguard is
opportunistic.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Iterator

# Windows API constants — defined here to avoid the ctypes import
# on non-Windows hosts.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040  # avoid screensaver suspend on desktops


@contextlib.contextmanager
def prevent_system_sleep() -> Iterator[bool]:
    """Ask Windows not to enter sleep / hibernate during the context.

    On Windows, calls ``SetThreadExecutionState`` with the
    continuous + system-required flags so the OS treats the sort
    as work that should keep the system awake. On exit, the
    previous execution-state is restored by clearing all flags
    (``ES_CONTINUOUS`` only).

    On non-Windows the context manager is a no-op and yields
    ``False``.

    Yields:
        active (bool): ``True`` when the API was successfully
            engaged; ``False`` otherwise. Useful for telling the
            user whether the safeguard is in effect.

    Notes:
        - The call affects only the calling thread's execution
          state — other Python processes unrelated to this sort
          are unaffected.
        - If the user closes the lid, the screen still goes dark;
          we only prevent the *system* from sleeping, not the
          display from blanking. Modern laptops will continue to
          run with the lid closed under this state.
    """
    if sys.platform != "win32":
        yield False
        return

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
            # Some Windows builds return 0 even on success; we
            # cannot reliably detect failure here. Treat as active
            # but warn diagnostically.
            print(
                "[power state] SetThreadExecutionState returned 0; "
                "treating sleep prevention as active but the OS may "
                "not have honoured the request."
            )
        else:
            print("[power state] active: system sleep prevented for the sort.")
    except Exception as exc:
        print(f"[power state] failed to engage sleep prevention: {exc!r}")
        yield False
        return

    try:
        yield True
    finally:
        try:
            # Clear all flags by setting only ES_CONTINUOUS (the
            # MSDN-recommended way to release).
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            pass
