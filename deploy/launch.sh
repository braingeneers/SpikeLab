#!/bin/bash
# Launcher for SSH-stdio access to the OC-SpikeLab MCP server.
#
# Used in two scenarios:
#
# 1. Customer device (default). The customer's own account hosts the MCP.
#    External callers SSH in to the customer's account with a key whose
#    authorized_keys entry pins them to this launcher via forced-command.
#    No dedicated UNIX user required.
#
#      command="/path/to/OC-SpikeLab/deploy/launch.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 AAAA... external@machine
#
# 2. Appliance install. A dedicated `oc-spikelab-mcp` UNIX account isolates
#    the MCP from the operator's normal account. Same launcher, just
#    installed under that user's home. Uncommon for spikelab; see
#    persistent_ask_services_spec.md "Process-ownership model" for context.
#
# Local callers (orchestrator + bench-companion on the same workstation as
# the rig) skip SSH entirely and spawn the MCP server directly via their
# Claude Code mcpServers config:
#
#      "oc-spikelab": {
#          "type": "stdio",
#          "command": "/home/<user>/anaconda3/envs/OC_env/bin/python",
#          "args": ["-m", "spikelab.mcp"],
#          "env": {"OC_SPIKELAB_PROJECT_ROOT": "/path/to/Habitat-MaxOne-test"}
#      }
#
# Either way, this launcher only runs the per-connection MCP-server proxy.
# The actual SDK sessions live in long-lived child agents owned by the
# spikelab-daemon (a separate systemd --user unit) over a Unix socket. If
# the daemon is down, ask_spikelab_* calls return
# {"is_error": true, "errors": ["oc-spikelab daemon not reachable: ..."]}.

set -euo pipefail

# Resolve paths relative to this script so a redeploy doesn't restale them.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Project root (where system_config.json + system_params.json live). The
# config-discovery walk in config.py would find it if the package source
# lived under the project tree, but typically OC-SpikeLab is cloned as a
# sibling, so set this explicitly per-install. Override by exporting
# OC_SPIKELAB_PROJECT_ROOT in the environment before sshd hands off.
if [[ -z "${OC_SPIKELAB_PROJECT_ROOT:-}" ]]; then
    echo "launch.sh: OC_SPIKELAB_PROJECT_ROOT is not set. Set it in" >&2
    echo "  ~/.config/environment.d/oc-spikelab.conf (systemd-user, recommended)" >&2
    echo "  or in ~/.profile / ~/.bashrc for SSH-stdio sessions." >&2
    exit 2
fi
export OC_SPIKELAB_PROJECT_ROOT

# Python with the [mcp] extra installed. Override per-install if the
# launcher should use a venv at a non-default path.
#   - PYTHON env var (highest priority)
#   - $REPO_ROOT/.venv/bin/python (if a venv lives next to the repo)
#   - /usr/bin/env python (fall through)
VENV_PYTHON="${PYTHON:-}"
if [[ -z "$VENV_PYTHON" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$VENV_PYTHON" ]]; then
    VENV_PYTHON="$(command -v python || true)"
fi
if [[ -z "$VENV_PYTHON" || ! -x "$VENV_PYTHON" ]]; then
    echo "launch.sh: no usable Python interpreter found (set PYTHON env)" >&2
    exit 1
fi

# Minimal env.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

if [[ "${1:-}" == "--dev" ]]; then
    export OC_SPIKELAB_MCP_DEV=1
fi

exec "$VENV_PYTHON" -m spikelab.mcp
