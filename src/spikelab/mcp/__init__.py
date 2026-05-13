"""MCP server + daemon bridging OC-SpikeLab to external orchestrator/companion sessions.

Three-layer design (mirrors `OC_ephys_tools.mcp`):

- :mod:`spikelab.mcp.server` — short-lived per-connection MCP stdio server. Thin
  proxy: each ``ask_spikelab_*`` tool forwards to the daemon over a Unix socket.
- :mod:`spikelab.mcp.daemon` — long-lived systemd-managed process owning the
  per-task_id child registry, watchdog reaper, kill op.
- :mod:`spikelab.mcp.child` — long-lived per-task_id agent process. One per
  active SDK session. Holds a single ``ClaudeSDKClient`` in memory across turns.

There is no operator concept here — every kind is lock-free. The daemon's
per-task_id :class:`asyncio.Lock` is the only serialization primitive, and it
only prevents two concurrent turns on the *same* task_id (intrinsic to the
single-pipe child design). Cross-task GPU contention on the sorter endpoint is
documented as a follow-up; the library's intra-process VRAM watchdog catches
runaway memory but does not coordinate across processes.
"""
