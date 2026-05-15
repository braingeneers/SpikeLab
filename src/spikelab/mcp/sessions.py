"""Wrappers over the Claude Agent SDK's session helpers + 35-day retention sweep.

A 'task' is exactly an SDK session; the ``task_id`` is the SDK session UUID.
Tasks are tagged by ``kind`` in the first turn's prompt prefix so
``list_spikelab_tasks(kind=...)`` can filter cleanly without out-of-band
metadata. The encoder runs once on a new task's first prompt; the decoder
accepts both the form we write (``]\n``) and the form the SDK normalises to
when stripping whitespace from the stored prompt (``] ``).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .config import PROJECT_DIR, RETENTION_DAYS

# Prefix prepended to the first prompt of each task so we can later read it
# back from the transcript and tell the five kinds apart.
TASK_KIND_PREFIX = "[oc-spikelab-mcp:kind="
TASK_KIND_SUFFIX = "]\n"
_TASK_KIND_CLOSER = "]"


def _project_dir() -> str:
    return str(PROJECT_DIR)


def encode_kind(kind: str, prompt: str) -> str:
    return f"{TASK_KIND_PREFIX}{kind}{TASK_KIND_SUFFIX}{prompt}"


def _decode_kind(first_prompt: str) -> tuple[str, str]:
    """Return ``(kind, visible_prompt)`` from a stored first prompt.

    Accepts both the form we write (``[...prefix...]<kind>]\\n``) and the
    normalised form the SDK stores (``[...prefix...]<kind>] `` with the
    newline collapsed to a space — or any single whitespace char).
    """
    if not first_prompt.startswith(TASK_KIND_PREFIX):
        return "unknown", first_prompt
    close = first_prompt.find(_TASK_KIND_CLOSER, len(TASK_KIND_PREFIX))
    if close == -1:
        return "unknown", first_prompt
    kind = first_prompt[len(TASK_KIND_PREFIX) : close]
    tail = first_prompt[close + 1 :]
    if tail and tail[0] in (" ", "\n", "\t", "\r"):
        tail = tail[1:]
    return kind, tail


def list_tasks(kind: str | None = None) -> list[dict[str, Any]]:
    """List all OC-spikelab tasks (most recently active first).

    If ``kind`` is provided, returns only tasks whose first prompt was tagged
    with that kind.
    """
    from claude_agent_sdk import list_sessions  # lazy: SDK is [mcp] extra

    sessions = list_sessions(directory=_project_dir(), include_worktrees=False)
    out: list[dict[str, Any]] = []
    for s in sessions:
        # Kind is encoded in the first prompt — SDK-generated custom_title /
        # summary fields would shadow our prefix if we looked at those first.
        task_kind, _ = _decode_kind(s.first_prompt or "")
        if kind is not None and task_kind != kind:
            continue
        display = s.custom_title or s.summary
        if not display:
            _, display = _decode_kind(s.first_prompt or "")
        out.append(
            {
                "task_id": s.session_id,
                "kind": task_kind,
                "created_at": s.created_at,
                "last_active": s.last_modified,
                "summary": display,
            }
        )
    return out


def get_task_status(task_id: str) -> dict[str, Any] | None:
    from claude_agent_sdk import (  # lazy: SDK is [mcp] extra
        get_session_info,
        get_session_messages,
    )

    info = get_session_info(task_id, directory=_project_dir())
    if info is None:
        return None
    try:
        messages = get_session_messages(task_id, directory=_project_dir())
    except Exception:
        messages = []
    kind, _ = _decode_kind(info.first_prompt or "")
    display = info.custom_title or info.summary
    if not display:
        _, display = _decode_kind(info.first_prompt or "")
    return {
        "task_id": info.session_id,
        "kind": kind,
        "created_at": info.created_at,
        "last_active": info.last_modified,
        "summary": display,
        "turn_count": len(messages),
        "transcript": [_summarize_message(m) for m in messages],
    }


def _summarize_message(msg: Any) -> dict[str, Any]:
    """Best-effort JSON-friendly view of a stored session message."""
    try:
        return asdict(msg)
    except TypeError:
        return {"repr": repr(msg)}


def sweep_old_tasks(retention_days: int = RETENTION_DAYS) -> list[str]:
    """Delete tasks whose ``last_active`` is older than ``retention_days``."""
    from claude_agent_sdk import (
        delete_session,
        list_sessions,
    )  # lazy: SDK is [mcp] extra

    cutoff_ms = (time.time() - retention_days * 86400) * 1000
    deleted: list[str] = []
    for s in list_sessions(directory=_project_dir(), include_worktrees=False):
        if s.last_modified < cutoff_ms:
            try:
                delete_session(s.session_id, directory=_project_dir())
                deleted.append(s.session_id)
            except (FileNotFoundError, ValueError):
                pass
    return deleted
