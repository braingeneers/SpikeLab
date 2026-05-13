"""Paths and constants for the OC-SpikeLab MCP daemon + server.

Install-specific values come from ``system_config.json`` and ``system_params.json``
at the *project root* — not at the OC-SpikeLab clone root. On a typical install,
the project root is the orchestrator workspace (e.g. ``Habitat-MaxOne-test``) and
the OC-SpikeLab repo is a sibling. Because of that, ``$OC_SPIKELAB_PROJECT_ROOT``
must be set explicitly in most deployments — the upward-walk fallback only works
when this package happens to be installed under the project tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from spikelab import __path__ as _SPIKELAB_PATH

# ---------------------------------------------------------------------------
# Sub-skill paths — resolved at import time from the installed package.
# An editable install picks these up from the source tree; a wheel install
# from site-packages. Either way the SKILL.md files ship as package data.
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(_SPIKELAB_PATH[0])
_SKILLS_BASE = _PKG_ROOT / "agent" / "skills"

EDUCATOR_SKILL = _SKILLS_BASE / "spikelab-educator" / "SKILL.md"
SORTER_SKILL = _SKILLS_BASE / "spikelab-spikesorter" / "SKILL.md"
ANALYZER_SKILL = _SKILLS_BASE / "spikelab-analysis-implementer" / "SKILL.md"
DEVELOPER_SKILL = _SKILLS_BASE / "spikelab-developer" / "SKILL.md"
MAP_UPDATER_SKILL = _SKILLS_BASE / "spikelab-map-updater" / "SKILL.md"


# ---------------------------------------------------------------------------
# Project root + system_config.json
# ---------------------------------------------------------------------------
def _discover_project_root() -> Path:
    """Find the project root.

    ``$OC_SPIKELAB_PROJECT_ROOT`` first (the normal path — set by the systemd
    unit / launch.sh / orchestrator's MCP-server config). Falls back to walking
    upward from this file looking for ``system_config.json`` — only useful when
    the package happens to live under the project tree.
    """
    explicit = os.environ.get("OC_SPIKELAB_PROJECT_ROOT")
    if explicit:
        p = Path(explicit).resolve()
        if (p / "system_config.json").exists():
            return p
        raise RuntimeError(
            f"OC_SPIKELAB_PROJECT_ROOT={explicit} but no system_config.json there"
        )
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "system_config.json").exists():
            return parent
    raise RuntimeError(
        "could not locate system_config.json from "
        f"{here} or any ancestor; set OC_SPIKELAB_PROJECT_ROOT"
    )


PROJECT_ROOT = _discover_project_root()
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "system_config.json"
SYSTEM_PARAMS_PATH = PROJECT_ROOT / "system_params.json"


def load_system_config() -> dict[str, Any]:
    return json.loads(SYSTEM_CONFIG_PATH.read_text())


def load_system_params() -> dict[str, Any]:
    """Load ``system_params.json``. Returns ``{}`` if the file is missing —
    tunables fall back to compiled-in defaults below."""
    if not SYSTEM_PARAMS_PATH.exists():
        return {}
    return json.loads(SYSTEM_PARAMS_PATH.read_text())


# Cached at import time. Tests that mutate the on-disk file can call
# ``load_system_config()`` / ``load_system_params()`` directly to re-read.
SYSTEM_CONFIG = load_system_config()
SYSTEM_PARAMS = load_system_params()


# ---------------------------------------------------------------------------
# SDK session storage — per-specialist subdir under $HOME so ephys and spikelab
# sessions don't bleed into each other's ``list_sessions`` results.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path.home() / "var" / "workspace" / "spikelab"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

# How long a task remains resumable after its last interaction.
RETENTION_DAYS = 35

# Dev-mode gate. Set by ``deploy/launch.sh --dev`` (or directly in the env).
DEV_MODE = os.environ.get("OC_SPIKELAB_MCP_DEV") == "1"


# ---------------------------------------------------------------------------
# Daemon: idle-watchdog windows per kind. ``system_params.json`` keys override
# the defaults; an absent file just uses the compiled-in values.
#
# Spikelab tunings (heavier than ephys):
# - sorter   7200 s (2 h) — Kilosort runs are long; we don't want to reap a
#                           child mid-sort when the customer steps away.
# - analyzer 3600 s (1 h) — multi-step analysis sessions iterate slowly.
# - developer 3600 s (1 h) — integration work is iterative, multi-turn.
# - educator 1800 s (30 min) — usually one-shot conceptual questions.
# - map_updater 600 s (10 min) — single-shot maintenance, dev-only.
# ---------------------------------------------------------------------------
_DEFAULT_WATCHDOG_SECONDS = {
    "educator": 1800,
    "sorter": 7200,
    "analyzer": 3600,
    "developer": 3600,
    "map_updater": 600,
}


def idle_watchdog_seconds(kind: str) -> int:
    """Return the configured idle-watchdog window for a kind, or the default."""
    key = f"ask_spikelab_{kind}_idle_watchdog_seconds"
    val = SYSTEM_PARAMS.get(key)
    if val is None:
        return _DEFAULT_WATCHDOG_SECONDS[kind]
    return int(val)


# ---------------------------------------------------------------------------
# Daemon: Unix-socket path. Per-user runtime dir if available, falling back to
# /tmp. Resolved at runtime so the daemon and the MCP server proxy stay in sync
# without a shared config file.
# ---------------------------------------------------------------------------
def daemon_socket_path() -> Path:
    """Resolve the per-user daemon socket path. Tries
    ``$XDG_RUNTIME_DIR/spikelab-daemon.sock`` first, then ``/tmp/...``."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        p = Path(xdg)
        if p.is_dir() and os.access(p, os.W_OK):
            return p / "spikelab-daemon.sock"
    return Path(f"/tmp/spikelab-daemon-{os.getuid()}.sock")


# Per-call timeout the MCP-server proxy uses when talking to the daemon.
# Long enough that even a multi-minute sorter turn (which the daemon waits on
# inside one ``ask`` request) doesn't blow the timeout. Sorter children have
# their own much longer idle watchdog (2 h); this 10-minute budget is for one
# turn's reasoning + tool calls.
DAEMON_REQUEST_TIMEOUT_SECONDS = 600

# Daemon watchdog tick — how often the daemon scans the registry for idle
# children to reap.
DAEMON_WATCHDOG_TICK_SECONDS = 30
