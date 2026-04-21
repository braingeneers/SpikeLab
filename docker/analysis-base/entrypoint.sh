#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SPIKELAB_DEBUG:-}" ]]; then
  echo "SPIKELAB_DEBUG enabled"
  python -V
fi

exec "$@"
