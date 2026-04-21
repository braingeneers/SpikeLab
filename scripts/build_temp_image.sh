#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <cpu|gpu> <image-tag>"
  echo "Example: $0 gpu ghcr.io/acme/spikelab-analysis-temp:run123"
  exit 1
fi

profile="$1"
image_tag="$2"

if [[ "$profile" == "gpu" ]]; then
  base_image="${BASE_IMAGE_GPU:-spikelab/analysis-base:gpu}"
else
  base_image="${BASE_IMAGE_CPU:-spikelab/analysis-base:cpu}"
fi

docker build \
  -f docker/analysis-temp/Dockerfile.temp \
  --build-arg "BASE_IMAGE=${base_image}" \
  -t "${image_tag}" \
  .

echo "BUILT_IMAGE=${image_tag}"
