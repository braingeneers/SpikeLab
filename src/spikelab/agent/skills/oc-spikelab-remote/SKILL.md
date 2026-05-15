---
name: OC-spikelab-remote
description: Drive OC-SpikeLab spike sorting + electrophysiological analysis on a different host via its MCP server. Use when an orchestrator or bench-companion session running off-host needs to sort recordings, run analyses, ask conceptual questions, or integrate analysis code into the library without direct access to the spikelab Python package.
---

<!-- GENERATED FILE — DO NOT EDIT. Source: src/spikelab/agent/skills/_sources/oc-spikelab-client.md.in. Regenerate with `python scripts/gen_client_skills.py`. -->

# OC-SpikeLab (remote)

You are operating **OC-SpikeLab** through an MCP server on a different host. You never SSH in interactively, never `import spikelab` locally, and never read sub-skill files yourself — all spikelab interaction goes through MCP tools.

---

## MCP tool surface

A long-lived MCP server exposes spikelab through these tools. The daemon owns persistent per-`task_id` child agents — the first turn on a new task pays the SDK session spawn cost (~10 s), subsequent turns on the same `task_id` are much faster.

**Always-on endpoints (4):**

| Tool | When to use |
|---|---|
| `ask_spikelab_educator(prompt, task_id=None)` | Concept questions, interpretation of results, neuroscience background, SpikeLab API explanations. Read-only — cannot write files or execute analyses. |
| `ask_spikelab_sorter(prompt, task_id=None)` | Run spike sorting (Kilosort 2, Kilosort 4, RT-Sort), configure sorters, curate units, generate QC figures. Long-running — sorts can take many minutes. |
| `ask_spikelab_analyzer(prompt, task_id=None)` | Load sorted data, run analyses (firing rates, STTC, burst detection, pairwise correlations, event-aligned stacks, PCA / UMAP), produce figures, write analysis scripts. |
| `ask_spikelab_developer(prompt, task_id=None)` | Integrate analysis code into the library: audit scripts, replace reimplementations, add novel methods, write tests, submit PRs. |

**Lifecycle and progress helpers (3, parallel-safe — read on-disk SDK state, no daemon hop, no blocking):**

| Tool | Purpose |
|---|---|
| `list_spikelab_tasks(kind=None)` | List all known tasks, most recently active first. Filter by `kind`. |
| `get_spikelab_task_status(task_id)` | Full metadata + message transcript for one task. |
| `get_spikelab_task_progress(task_id, log_path=None, tail_lines=100)` | Lightweight progress snapshot for a long-running sort or analysis. Resolution order: (1) explicit `log_path` argument → tail that file; (2) scan the SDK transcript for `kilosort2.log` / `kilosort4.log` / `rt_sort.log` / `sorting_*.log` mentions and tail the most recent existing one; (3) fall back to recent transcript events. Use during multi-minute sorts to check iteration counts, spike-detection progress, etc., without interrupting the agent. |

**Explicit lifecycle:**

| Tool | Purpose |
|---|---|
| `kill_spikelab_task(task_id)` | Terminate the live child agent for `task_id`. Conversation history on disk is preserved; the next `ask_spikelab_*` call respawns and resumes. Use to free memory when a session is known to be done. |

### Multi-turn conversations

Pass `task_id=None` on the first call; the returned `task_id` is the SDK session UUID — feed it back to subsequent calls on the same endpoint to keep one persistent child agent in memory. Each endpoint enforces a kind-mismatch check: calling `ask_spikelab_analyzer(task_id=<a-sorter-task>)` errors out, because resuming with a different system prompt would silently corrupt context.

### Progress checks during long runs

When the sorter or analyzer agent is mid-turn (sort_recording running, GPLVM fit going, etc.), the `ask_spikelab_*` call blocks until that turn returns. Use `get_spikelab_task_progress(task_id)` from a parallel session to peek at recent activity without blocking. For sorter runs in particular, the spikelab library writes Kilosort logs to known paths (`<output_folder>/kilosort{2,4}.log`, `<output_folder>/rt_sort.log`); the progress tool tails them directly when it can find them in the transcript.

### Concurrency model

- Same `task_id` → calls queue at the daemon's per-task `in_flight` mutex (intrinsic — a child serves one turn at a time on its single stdin pipe).
- Different `task_id`s on the same kind, or any two different kinds → run in parallel. Educator + analyzer + sorter can be active simultaneously.
- GPU contention across concurrent sorter calls on different `task_id`s is **not** coordinated cross-process — the spikelab library has an intra-process VRAM watchdog only. Running two simultaneous sorts on the same GPU may trip the watchdog mid-run. Avoid until cross-process coordination lands.

### Installation on the host

The MCP daemon must be running on the spikelab host before any `ask_spikelab_*` call works. Host setup (clone path, conda env, `[mcp]` extra, systemd user unit, `OC_SPIKELAB_PROJECT_ROOT`) is documented in the host operator's `INSTALL.md` shipped with the OC-SpikeLab repository — link it to whoever administers the spikelab host. From this side, if `ask_spikelab_*` returns `"oc-spikelab daemon not reachable: ..."`, the daemon is down; coordinate with the host operator.

---

