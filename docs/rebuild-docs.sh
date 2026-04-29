#!/usr/bin/env bash
# Pull the SpikeLab repo and rebuild the Sphinx HTML if the upstream branch
# has advanced AND any path under docs/, src/, or pyproject.toml changed. The
# served directory is updated atomically via a symlink swap so nginx never
# serves a half-written tree. On build failure, the previous HTML is kept.

set -euo pipefail

REPO_DIR="${SPIKELAB_REPO_DIR:-/opt/spikelab}"
SERVE_DIR="${SPIKELAB_SERVE_DIR:-/srv/spikelab-docs}"
BRANCH="${SPIKELAB_BRANCH:-main}"

cd "${REPO_DIR}"

old_head="$(git rev-parse HEAD)"
git fetch --quiet origin "${BRANCH}"
new_head="$(git rev-parse "origin/${BRANCH}")"

if [ "${old_head}" = "${new_head}" ]; then
    exit 0
fi

# Anything doc-relevant changed?
if git diff --quiet "${old_head}" "${new_head}" -- docs/ src/ pyproject.toml; then
    # Upstream advanced but nothing relevant — fast-forward and exit.
    git reset --hard "${new_head}"
    exit 0
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
echo "[${ts}] rebuild: ${old_head:0:7} -> ${new_head:0:7}"

git reset --hard "${new_head}"

# Reinstall the package only if pyproject.toml moved (deps may have changed).
if ! git diff --quiet "${old_head}" "${new_head}" -- pyproject.toml; then
    echo "[${ts}] pyproject.toml changed, reinstalling package"
    pip install --no-cache-dir -e ".[docs]" >/dev/null
fi

build_dir="${SERVE_DIR}/build-${ts}"
if ! python -m sphinx -b html docs/source "${build_dir}"; then
    echo "[${ts}] sphinx build FAILED, keeping previous HTML"
    rm -rf "${build_dir}"
    # Roll the working tree back so cron retries cleanly next tick.
    git reset --hard "${old_head}"
    exit 1
fi

# Atomic symlink swap. ln -sfn writes a temp name; mv -Tf renames it onto the
# existing symlink in a single atomic syscall (same filesystem).
ln -sfn "${build_dir}" "${SERVE_DIR}/current.tmp"
mv -Tf "${SERVE_DIR}/current.tmp" "${SERVE_DIR}/current"

# Keep the most recent 3 builds for trivial rollback; prune the rest.
ls -1dt "${SERVE_DIR}"/build-* 2>/dev/null | tail -n +4 | xargs -r rm -rf

echo "[${ts}] rebuild complete"
