#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <cpu|gpu> <image-tag>"
  echo "Example: $0 cpu ghcr.io/acme/spikelab-analysis-base:dev-abc1234"
  exit 1
fi

profile="$1"
image_tag="$2"

case "$profile" in
  cpu) dockerfile="docker/analysis-base/Dockerfile.cpu" ;;
  gpu) dockerfile="docker/analysis-base/Dockerfile.gpu" ;;
  *)
    echo "Error: profile must be 'cpu' or 'gpu', got '$profile'"
    exit 1
    ;;
esac

docker build \
  -f "${dockerfile}" \
  -t "${image_tag}" \
  .

echo "BUILT_IMAGE=${image_tag}"
