#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image-tag>"
  exit 1
fi

image_tag="$1"
docker push "${image_tag}"
echo "PUSHED_IMAGE=${image_tag}"
