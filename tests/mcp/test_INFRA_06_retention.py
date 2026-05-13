"""INFRA-6 — the 35-day retention sweep runs without error.

Strategy: call ``sweep_old_tasks`` directly. Functional verification that an
actually-old session gets removed is delegated to claude_agent_sdk's own test
suite (it owns the on-disk layout). This is a unit-level smoke test — no
Claude credentials required.
"""

from __future__ import annotations

import sys


def main() -> int:
    from spikelab.mcp.sessions import sweep_old_tasks  # type: ignore

    deleted = sweep_old_tasks(retention_days=35)
    print(f"sweep_old_tasks deleted {len(deleted)} sessions: {deleted}")
    print("INFRA-6 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
