"""MCP stdio server exposing the OC-SpikeLab tool surface.

Architecture: this process is a thin per-connection proxy. The four (+1
dev-only) ``ask_spikelab_*`` endpoints forward to the long-lived daemon
(``daemon.py``) over a Unix socket; the daemon owns persistent per-task_id
child agents and amortises the SDK-session spawn cost across turns.

In-process tools (no daemon round-trip):

- ``list_spikelab_tasks(kind=None)``
- ``get_spikelab_task_status(task_id)``
- ``get_spikelab_task_progress(task_id, log_path=None, tail_lines=100)``

The first two read on-disk SDK session state via the helpers in
``sessions.py``. The third also reads the on-disk transcript and, when an
associated sorter log file (``kilosort{2,4}.log``, ``rt_sort.log``,
``sorting_*.log``) is referenced in the transcript, tails that file directly
— this gives live progress for long-running Kilosort / RT-Sort jobs.

Tool counts:

- User mode (default): 8 tools — 4 ``ask_spikelab_*`` + 3 lifecycle helpers
  (``list_*_tasks``, ``get_*_task_status``, ``get_*_task_progress``) +
  ``kill_spikelab_task``.
- Dev mode (``OC_SPIKELAB_MCP_DEV=1``): adds ``ask_spikelab_map_updater`` and
  ``run_test_suite`` → 10 tools.

Every kind is lock-free — there is no rig flock here and no operator-only
serialization. Cross-task_id GPU contention on the sorter endpoint is a
documented follow-up.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ._daemon_proxy import call_daemon
from .config import DEV_MODE
from .progress import get_task_progress
from .sessions import get_task_status, list_tasks, sweep_old_tasks

mcp = FastMCP(
    name="oc-spikelab",
    instructions=(
        "OC-SpikeLab spike-train analysis agent. Four always-on endpoints, "
        "each locked to one sub-skill:\n"
        "  • ask_spikelab_educator  — concept questions, interpretation of "
        "results, neuroscience background. Read-only.\n"
        "  • ask_spikelab_sorter    — run spike sorting (Kilosort 2/4, "
        "RT-Sort), curation, QC figures. Long-running.\n"
        "  • ask_spikelab_analyzer  — load sorted data, run analyses, "
        "produce figures, write analysis scripts.\n"
        "  • ask_spikelab_developer — integrate analysis code into the "
        "library, write tests, submit PRs.\n"
        "Pass task_id to continue a multi-turn conversation; omit it to "
        "start a new one. Persistent per-task_id child agents amortise the "
        "SDK session spawn cost across turns — the first turn pays ~10 s, "
        "subsequent turns on the same task_id are much faster.\n"
        "Use get_spikelab_task_progress(task_id) to peek at a long-running "
        "sort or analysis without blocking; it tails the sorter's log file "
        "when one can be located. Use list_spikelab_tasks / "
        "get_spikelab_task_status for full registry / transcript reads. "
        "All read-only tools are parallel-safe."
    ),
)


# ---------------------------------------------------------------------------
# ask_spikelab_educator
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ask_spikelab_educator",
    description=(
        "Ask the OC-SpikeLab educator agent a conceptual question — what an "
        "analysis does, how a method works, what a result means, neuroscience "
        "background, anything about the SpikeLab API and data structures. "
        "Read-only: this session cannot write or execute analysis scripts. "
        "Runs in parallel with sorter / analyzer / developer endpoints. Pass "
        "task_id to continue a multi-turn explanation."
    ),
)
async def ask_spikelab_educator(
    prompt: str, task_id: str | None = None
) -> dict[str, Any]:
    return await call_daemon(
        {
            "op": "ask",
            "kind": "educator",
            "task_id": task_id,
            "prompt": prompt,
        }
    )


# ---------------------------------------------------------------------------
# ask_spikelab_sorter
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ask_spikelab_sorter",
    description=(
        "Send a prompt to the OC-SpikeLab sorter agent. Use for running "
        "spike sorting pipelines (Kilosort2 / Kilosort4 / RT-Sort), "
        "configuring sorter parameters, curating units, generating QC "
        "figures, troubleshooting sort failures. Long-running — sorts can "
        "take many minutes. While a sort is running, call "
        "get_spikelab_task_progress(task_id) to peek at the Kilosort log "
        "without blocking the in-flight turn. Pass task_id to continue an "
        "existing conversation."
    ),
)
async def ask_spikelab_sorter(
    prompt: str, task_id: str | None = None
) -> dict[str, Any]:
    return await call_daemon(
        {
            "op": "ask",
            "kind": "sorter",
            "task_id": task_id,
            "prompt": prompt,
        }
    )


# ---------------------------------------------------------------------------
# ask_spikelab_analyzer
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ask_spikelab_analyzer",
    description=(
        "Send a prompt to the OC-SpikeLab analyzer agent. Use for loading "
        "sorted data, running spike-train analyses (firing rates, STTC, "
        "burst detection, pairwise correlations, event-aligned stacks, "
        "PCA / UMAP, etc.), producing figures, writing analysis scripts. "
        "Long-running analyses are common; use get_spikelab_task_progress "
        "to peek without blocking. Pass task_id to continue an existing "
        "session."
    ),
)
async def ask_spikelab_analyzer(
    prompt: str, task_id: str | None = None
) -> dict[str, Any]:
    return await call_daemon(
        {
            "op": "ask",
            "kind": "analyzer",
            "task_id": task_id,
            "prompt": prompt,
        }
    )


# ---------------------------------------------------------------------------
# ask_spikelab_developer
# ---------------------------------------------------------------------------
@mcp.tool(
    name="ask_spikelab_developer",
    description=(
        "Send a prompt to the OC-SpikeLab developer agent. Use for "
        "integrating analysis code into the library: auditing scripts "
        "against existing methods, replacing reimplementations, adding "
        "novel methods, writing tests, submitting PRs. Iterative multi-turn "
        "work — pass task_id to continue an existing integration session."
    ),
)
async def ask_spikelab_developer(
    prompt: str, task_id: str | None = None
) -> dict[str, Any]:
    return await call_daemon(
        {
            "op": "ask",
            "kind": "developer",
            "task_id": task_id,
            "prompt": prompt,
        }
    )


# ---------------------------------------------------------------------------
# Lifecycle helpers (in-process, parallel-safe, no daemon round-trip)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="list_spikelab_tasks",
    description=(
        "List all known OC-SpikeLab MCP tasks (most recently active first). "
        "Pass kind='educator', 'sorter', 'analyzer', 'developer', or "
        "'map_updater' to filter. Each entry includes task_id, kind, "
        "created_at, last_active, summary. Reads on-disk SDK session state "
        "directly — no daemon hop, parallel-safe."
    ),
)
async def list_spikelab_tasks(kind: str | None = None) -> list[dict[str, Any]]:
    return list_tasks(kind=kind)


@mcp.tool(
    name="get_spikelab_task_status",
    description=(
        "Return full metadata and message transcript for a single "
        "OC-SpikeLab task. Returns null if the task_id is unknown. Reads "
        "on-disk SDK session state directly — no daemon hop, parallel-safe. "
        "For a lightweight in-flight progress check (sorter log tail or "
        "recent activity only), prefer get_spikelab_task_progress."
    ),
)
async def get_spikelab_task_status(task_id: str) -> dict[str, Any] | None:
    return get_task_status(task_id)


@mcp.tool(
    name="get_spikelab_task_progress",
    description=(
        "Snapshot the recent activity of one task. Parallel-safe — does not "
        "block, does not touch the live child. Resolution order: (1) if "
        "log_path is given, tail that file directly; (2) else scan the SDK "
        "transcript for recognised sorter log paths (kilosort2.log / "
        "kilosort4.log / rt_sort.log / sorting_*.log) and tail the most "
        "recently mentioned one; (3) else return the last few transcript "
        "events. Use during long sort or analysis runs to check progress "
        "without interrupting the agent."
    ),
)
async def get_spikelab_task_progress(
    task_id: str,
    log_path: str | None = None,
    tail_lines: int = 100,
) -> dict[str, Any]:
    return get_task_progress(task_id=task_id, log_path=log_path, tail_lines=tail_lines)


@mcp.tool(
    name="kill_spikelab_task",
    description=(
        "Terminate the live child agent for a task_id, if any. Conversation "
        "history on disk is preserved — the next ask_spikelab_* call on this "
        "task_id respawns a fresh child that resumes from disk. Use to free "
        "memory when a conversation is known to be done (e.g. bench-companion "
        "closing a thread). Returns {task_id, killed, was_alive}."
    ),
)
async def kill_spikelab_task(task_id: str) -> dict[str, Any]:
    return await call_daemon({"op": "kill_task", "task_id": task_id})


# ---------------------------------------------------------------------------
# Dev-only tools (registered conditionally when OC_SPIKELAB_MCP_DEV=1)
# ---------------------------------------------------------------------------
def _register_dev_tools() -> None:
    """Add map_updater + test-runner."""

    @mcp.tool(
        name="ask_spikelab_map_updater",
        description=(
            "DEV-ONLY: regenerate SpikeLab repo maps (REPO_MAP.md, "
            "REPO_MAP_DETAILED.md). Maintenance only — typically single-shot. "
            "Daemon-side idle watchdog is short (600s) since follow-ups are "
            "rare. Available only when the server is launched with "
            "OC_SPIKELAB_MCP_DEV=1."
        ),
    )
    async def ask_spikelab_map_updater(
        prompt: str, task_id: str | None = None
    ) -> dict[str, Any]:
        return await call_daemon(
            {
                "op": "ask",
                "kind": "map_updater",
                "task_id": task_id,
                "prompt": prompt,
            }
        )

    from .test_runner import run_test_suite as _run_test_suite

    @mcp.tool(
        name="run_test_suite",
        description=(
            "DEV-ONLY: run the OC-SpikeLab MCP test suite. Categories: "
            "INFRA, SEC, SCOPE, EDU, SORT. Defaults to all. Available only "
            "when the server is launched with OC_SPIKELAB_MCP_DEV=1. Takes "
            "5-15 minutes depending on categories (SORT-1 drives a real "
            "synthetic-recording sort end-to-end)."
        ),
    )
    async def run_test_suite(
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(_run_test_suite, categories)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    if DEV_MODE:
        _register_dev_tools()
    try:
        sweep_old_tasks()
    except Exception:
        # Never let a sweep failure prevent the server from starting.
        pass
    mcp.run()


if __name__ == "__main__":
    main()
