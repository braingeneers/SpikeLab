"""Long-lived per-task_id child process for the OC-SpikeLab daemon.

One child per active SDK session. Spawned by the daemon, holds a single
``ClaudeSDKClient`` in memory across turns, reads prompt requests from stdin
and writes responses to stdout as newline-delimited JSON.

Invocation::

    python -m spikelab.mcp.child --kind <educator|sorter|analyzer|developer|map_updater>
                                 --task-id <uuid|none>

Wire protocol (stdin/stdout, JSON lines):

    daemon → child  {"prompt": "<text>"}
    child  → daemon {"task_id": "...", "kind": "...", "response": "...",
                     "is_error": false, "errors": [...]}

EOF on stdin = graceful shutdown after the current turn (if any) completes.
SIGTERM is treated identically: complete the in-flight turn, then exit.

Every kind is lock-free. Spikelab has no rig and no library-level session
lock to coordinate against — the daemon's per-task_id ``in_flight`` mutex
already serialises requests on a single child's pipe by construction. GPU
contention across concurrent sorter children on different task_ids is a
known follow-up; see the spec's Open decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from .config import (
    ANALYZER_SKILL,
    DEVELOPER_SKILL,
    EDUCATOR_SKILL,
    MAP_UPDATER_SKILL,
    PROJECT_DIR,
    SORTER_SKILL,
)
from .sessions import encode_kind

_KIND_TO_SKILL: dict[str, Path] = {
    "educator": EDUCATOR_SKILL,
    "sorter": SORTER_SKILL,
    "analyzer": ANALYZER_SKILL,
    "developer": DEVELOPER_SKILL,
    "map_updater": MAP_UPDATER_SKILL,
}


def _build_options(system_prompt_path: Path, resume_id: str | None):
    from claude_agent_sdk import ClaudeAgentOptions  # lazy: SDK is [mcp] extra

    return ClaudeAgentOptions(
        system_prompt=system_prompt_path.read_text(),
        cwd=str(PROJECT_DIR),
        allowed_tools=["Bash", "Read", "Glob", "Grep", "Write", "Edit"],
        permission_mode="bypassPermissions",
        resume=resume_id,
    )


# ---------------------------------------------------------------------------
# stdin/stdout helpers
# ---------------------------------------------------------------------------
async def _stdin_lines():
    """Yield JSON-line requests from stdin, one per line. Returns on EOF."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    while True:
        line = await reader.readline()
        if not line:
            return  # EOF
        yield line.decode("utf-8", errors="replace").rstrip("\n")


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------
async def _run_one_turn(
    client,
    prompt: str,
    kind: str,
    resume_task_id: str | None,
    is_first_turn: bool,
) -> dict[str, Any]:
    from claude_agent_sdk import (  # lazy
        AssistantMessage,
        ResultMessage,
        TextBlock,
    )

    # On the very first turn of a freshly spawned child (no resume), tag the
    # prompt with the kind so list_spikelab_tasks can recover it later.
    effective_prompt = (
        prompt if (resume_task_id or not is_first_turn) else encode_kind(kind, prompt)
    )

    response_text_parts: list[str] = []
    result_session_id: str | None = None
    is_error = False
    error_messages: list[str] = []

    await client.query(effective_prompt)
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    response_text_parts.append(block.text)
        elif isinstance(msg, ResultMessage):
            result_session_id = msg.session_id
            is_error = msg.is_error
            if msg.errors:
                error_messages.extend(msg.errors)

    return {
        "task_id": result_session_id or resume_task_id,
        "kind": kind,
        "response": "".join(response_text_parts).strip(),
        "is_error": is_error,
        "errors": error_messages,
    }


# ---------------------------------------------------------------------------
# Main child loop
# ---------------------------------------------------------------------------
async def child_main(kind: str, task_id_or_none: str | None) -> int:
    """One SDK session, many turns. Exits on stdin EOF or SIGTERM."""
    if kind not in _KIND_TO_SKILL:
        print(f"child: unknown kind {kind!r}", file=sys.stderr)
        return 2

    from claude_agent_sdk import ClaudeSDKClient  # lazy

    options = _build_options(
        system_prompt_path=_KIND_TO_SKILL[kind],
        resume_id=task_id_or_none,
    )

    # Handle SIGTERM as a request to exit after the current turn. We can't
    # easily interrupt the SDK mid-turn (and shouldn't — that would corrupt
    # in-flight tool calls); just remember the request and break the loop
    # after the current turn completes.
    shutdown_requested = False

    def _handle_sigterm(*_):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    is_first_turn = True
    resume_task_id = task_id_or_none

    async with ClaudeSDKClient(options=options) as client:
        async for line in _stdin_lines():
            if shutdown_requested:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                _write_response(
                    {
                        "task_id": resume_task_id,
                        "kind": kind,
                        "response": "",
                        "is_error": True,
                        "errors": [f"child: bad JSON request: {e!r}"],
                    }
                )
                continue

            prompt = req.get("prompt")
            if not isinstance(prompt, str):
                _write_response(
                    {
                        "task_id": resume_task_id,
                        "kind": kind,
                        "response": "",
                        "is_error": True,
                        "errors": ["child: missing or non-string 'prompt'"],
                    }
                )
                continue

            try:
                result = await _run_one_turn(
                    client=client,
                    prompt=prompt,
                    kind=kind,
                    resume_task_id=resume_task_id,
                    is_first_turn=is_first_turn,
                )
            except Exception as e:
                result = {
                    "task_id": resume_task_id,
                    "kind": kind,
                    "response": "",
                    "is_error": True,
                    "errors": [f"child: turn raised: {e!r}"],
                }

            # The SDK assigns a session UUID on the first turn of a new task.
            # Latch it so subsequent turns report a stable task_id.
            if result.get("task_id"):
                resume_task_id = result["task_id"]
            is_first_turn = False

            _write_response(result)

            if shutdown_requested:
                break

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=sorted(_KIND_TO_SKILL.keys()),
    )
    parser.add_argument(
        "--task-id",
        default="none",
        help="SDK session UUID to resume, or 'none' to start a new task",
    )
    args = parser.parse_args()
    task_id_or_none = None if args.task_id == "none" else args.task_id
    try:
        return asyncio.run(child_main(args.kind, task_id_or_none))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
