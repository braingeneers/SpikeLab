"""Shared audit log for watchdog warn/trip events.

Every watchdog can append timestamped events to a per-recording
``watchdog_events.jsonl`` file. Each line is a JSON object with at
least ``timestamp``, ``watchdog`` (e.g. ``"host_memory"``,
``"gpu_memory"``, ``"disk"``, ``"inactivity"``), and ``event``
(``"warn"``, ``"abort"``, ``"info"``). Watchdog-specific payload
fields (memory percent, free GB, etc.) live alongside.

The audit log enables three use cases:

* **Threshold tuning** — operators can review near-trips ("we hit
  87% RAM 6 times before the 92% abort fired — bump warn_pct").
* **Post-incident analysis** — when a recording fails, the
  events leading up to it are timestamped and structured.
* **Cross-batch trending** — accumulate events across many runs
  to detect drift (memory creep, slowdown).

Best-effort: append failures (disk full, permissions) are
swallowed so audit-side bugs never break a sort.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_AUDIT_FILENAME = "watchdog_events.jsonl"
_AUDIT_LOCK = threading.Lock()


def append_audit_event(
    *,
    watchdog: str,
    event: str,
    log_path: Optional[Path] = None,
    **payload: Any,
) -> None:
    """Append one event to the per-recording audit log.

    Resolves the audit file path from the per-recording log path:
    if *log_path* is provided it is used directly; otherwise the
    active log path published by ``sort_recording`` (via the
    ``get_active_log_path`` ContextVar) is consulted. When neither
    yields a path, the call is a silent no-op.

    Each event is a single line of JSON for easy ``jq``-style
    filtering. The line includes:

    * ``timestamp`` — ISO 8601 with seconds.
    * ``watchdog`` — short identifier (e.g. ``"host_memory"``).
    * ``event`` — short verb (e.g. ``"warn"``, ``"abort"``).
    * ``**payload`` — watchdog-specific fields.

    Parameters:
        watchdog (str): Short watchdog identifier.
        event (str): Event kind.
        log_path (Path or None): Optional explicit log path. When
            None, the active ContextVar is read. The audit file is
            written next to *log_path* (same parent folder).
        **payload: Additional JSON-friendly fields.

    Notes:
        - Writes are serialised through a module-level lock so
          multiple watchdog threads can append safely.
        - The event line is appended atomically (single ``write``
          call) — short JSON lines are atomic up to PIPE_BUF on
          POSIX and within a sector on Windows; corruption
          requires a system-level failure mid-write.
    """
    try:
        if log_path is None:
            from ._inactivity import get_active_log_path

            log_path = get_active_log_path()
        if log_path is None:
            return
        results_folder = Path(log_path).parent
        target = results_folder / _AUDIT_FILENAME
        line = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "watchdog": watchdog,
            "event": event,
        }
        line.update({k: _json_safe(v) for k, v in payload.items()})
        encoded = json.dumps(line, default=str) + "\n"
        with _AUDIT_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(encoded)
    except Exception:
        # Audit-side errors must never break a sort.
        pass


def _json_safe(value: Any) -> Any:
    """Coerce common non-JSON-friendly values to safe forms."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
