"""Read-only progress snapshot for in-flight spikelab tasks.

Used by the ``get_spikelab_task_progress`` MCP tool. Parallel-safe — does not
touch the daemon, only reads on-disk SDK session state and (when an associated
sorter log file can be located) tails that log file.

Resolution order for the returned tail:

1. **Explicit ``log_path`` argument.** Caller knows exactly which file the run
   is writing to. Read the last ``tail_lines`` lines.

2. **Scan the SDK transcript for log-file mentions.** Spikelab sorters use a
   well-defined Tee log convention — :file:`<output_folder>/kilosort4.log`,
   :file:`<output_folder>/kilosort2.log`, :file:`<output_folder>/rt_sort.log`,
   or :file:`<folder>/sorting_*.log`. The agent's Bash tool calls and tool
   results contain the absolute paths it passed to those runners; pull out
   the most-recently-mentioned path that points at an existing file.

3. **Transcript tail.** No log file located (educator / short analyzer turn /
   the run hasn't started writing yet). Return a summary of the last few
   transcript events instead.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PROJECT_DIR

# Recognised sorter log filenames. We search transcript text for paths that
# end in one of these. The set comes from:
#   spikelab.spike_sorting.ks4_runner    -> "kilosort4.log"
#   spikelab.spike_sorting.ks2_runner    -> "kilosort2.log"
#   spikelab.spike_sorting.rt_sort_runner -> "rt_sort.log"
#   spikelab.spike_sorting.report        -> "sorting_<N>.log"
_LOG_FILENAME_RE = re.compile(
    r"(?P<path>(?:/|~|\$HOME|\.{0,2}/)[^\s'\"`)\]]*?"
    r"(?:kilosort[24]\.log|rt_sort\.log|sorting_[^\s'\"`)\]]+\.log))"
)


def _project_dir() -> str:
    return str(PROJECT_DIR)


def _tail_lines(path: Path, n: int) -> str:
    """Return the last ``n`` lines of ``path``. Reads in 64 KB chunks from the
    end so multi-MB Kilosort logs don't slurp into memory."""
    chunk_size = 65536
    data = b""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            while end > 0 and data.count(b"\n") <= n:
                read = min(chunk_size, end)
                end -= read
                f.seek(end)
                data = f.read(read) + data
    except OSError as e:
        return f"<could not read {path}: {e}>"
    lines = data.splitlines()
    if len(lines) > n:
        lines = lines[-n:]
    return b"\n".join(lines).decode("utf-8", errors="replace")


def _extract_text_from_block(block: Any) -> str | None:
    """Best-effort text extraction from one SDK content block."""
    if isinstance(block, dict):
        if block.get("type") == "text":
            return str(block.get("text") or "")
        if block.get("type") == "tool_use":
            inp = block.get("input") or {}
            # Bash, Read, Glob, Grep, Write, Edit all carry their primary arg
            # in a single string field — concatenate everything string-valued.
            parts = [block.get("name", "")]
            for k, v in inp.items():
                if isinstance(v, str):
                    parts.append(f"{k}={v}")
            return " ".join(parts)
        if block.get("type") == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                out = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        out.append(str(c.get("text") or ""))
                return "\n".join(out)
    return None


def _scan_transcript_for_log_paths(messages: list[Any]) -> list[Path]:
    """Walk the transcript from newest to oldest, collect every recognised
    sorter log path. Returns paths in newest-first order, de-duplicated."""
    seen: set[str] = set()
    out: list[Path] = []
    for msg in reversed(messages):
        # Each message is a dataclass from claude_agent_sdk; convert to a dict
        # so we can introspect content blocks uniformly.
        try:
            d = asdict(msg)
        except TypeError:
            d = msg if isinstance(msg, dict) else {}
        # The transcript stores message content under "content" or similar;
        # the SDK's exact shape is a list of blocks. Walk any string we find.
        haystacks: list[str] = []
        if isinstance(d.get("content"), list):
            for block in d["content"]:
                t = _extract_text_from_block(block)
                if t:
                    haystacks.append(t)
        if isinstance(d.get("content"), str):
            haystacks.append(d["content"])
        # Fallback: repr the whole thing
        if not haystacks:
            haystacks.append(repr(d))
        for h in haystacks:
            for m in _LOG_FILENAME_RE.finditer(h):
                raw = m.group("path")
                expanded = Path(raw).expanduser()
                key = str(expanded)
                if key in seen:
                    continue
                seen.add(key)
                # Only consider paths that actually exist — agent output may
                # contain placeholder paths or paths to files that haven't
                # been created yet.
                if expanded.is_file():
                    out.append(expanded)
    return out


def _summarize_transcript_tail(messages: list[Any], n: int) -> list[dict[str, Any]]:
    """Return the last ``n`` transcript events as compact summaries."""
    out: list[dict[str, Any]] = []
    for msg in messages[-n:]:
        try:
            d = asdict(msg)
        except TypeError:
            d = msg if isinstance(msg, dict) else {"repr": repr(msg)}
        # Try to recognise the canonical shapes.
        content = d.get("content")
        if isinstance(content, list):
            for block in content:
                t = _extract_text_from_block(block)
                if t is None:
                    continue
                btype = block.get("type") if isinstance(block, dict) else "unknown"
                out.append({"type": btype, "summary": t[:800]})
        elif isinstance(content, str):
            out.append({"type": d.get("role", "msg"), "summary": content[:800]})
        else:
            out.append({"type": "msg", "summary": str(d)[:800]})
    return out


def get_task_progress(
    task_id: str,
    log_path: str | None = None,
    tail_lines: int = 100,
) -> dict[str, Any]:
    """Snapshot the recent activity of one task.

    Parameters:
        task_id: SDK session UUID.
        log_path: Optional explicit path to a sorter / analyzer log file. If
            given, the tail of this file is returned regardless of the
            transcript scan. Use this when the caller knows where the run
            is writing.
        tail_lines: Number of lines to return when reading a log file (when
            the source is ``"log_file"``). Capped at 1000 to avoid response
            bloat.

    Returns:
        dict with keys::

            task_id, kind, source ("log_file" | "transcript" | "missing"),
            log_path (when source=log_file), tail (str), turn_count (int),
            log_last_modified (unix-ms, when source=log_file),
            elapsed_since_last_event_s (float).
    """
    from claude_agent_sdk import (  # lazy: SDK is [mcp] extra
        get_session_info,
        get_session_messages,
    )

    tail_lines = max(1, min(int(tail_lines), 1000))

    info = get_session_info(task_id, directory=_project_dir())
    if info is None:
        return {
            "task_id": task_id,
            "kind": "unknown",
            "source": "missing",
            "tail": "",
            "turn_count": 0,
            "elapsed_since_last_event_s": None,
            "errors": [f"unknown task_id {task_id!r}"],
        }
    from .sessions import _decode_kind

    kind, _ = _decode_kind(info.first_prompt or "")

    try:
        messages = get_session_messages(task_id, directory=_project_dir())
    except Exception:
        messages = []
    turn_count = len(messages)

    last_modified_ms = info.last_modified or 0
    elapsed_s = (
        max(0.0, time.time() - last_modified_ms / 1000.0) if last_modified_ms else None
    )

    # ----- explicit log_path argument
    if log_path:
        p = Path(log_path).expanduser()
        if p.is_file():
            return {
                "task_id": task_id,
                "kind": kind,
                "source": "log_file",
                "log_path": str(p),
                "log_last_modified": int(p.stat().st_mtime * 1000),
                "tail": _tail_lines(p, tail_lines),
                "turn_count": turn_count,
                "elapsed_since_last_event_s": elapsed_s,
            }
        # Caller-supplied path didn't exist — fall through to scan-based
        # discovery, but record that we tried.
        explicit_miss = (
            f"log_path={log_path!r} does not exist; falling back to transcript scan"
        )
    else:
        explicit_miss = None

    # ----- scan transcript for known sorter log paths
    candidates = _scan_transcript_for_log_paths(messages)
    if candidates:
        p = candidates[0]
        return {
            "task_id": task_id,
            "kind": kind,
            "source": "log_file",
            "log_path": str(p),
            "log_last_modified": int(p.stat().st_mtime * 1000),
            "tail": _tail_lines(p, tail_lines),
            "turn_count": turn_count,
            "elapsed_since_last_event_s": elapsed_s,
            **({"note": explicit_miss} if explicit_miss else {}),
        }

    # ----- transcript tail fallback
    return {
        "task_id": task_id,
        "kind": kind,
        "source": "transcript",
        "tail_events": _summarize_transcript_tail(messages, n=10),
        "turn_count": turn_count,
        "elapsed_since_last_event_s": elapsed_s,
        **({"note": explicit_miss} if explicit_miss else {}),
    }
