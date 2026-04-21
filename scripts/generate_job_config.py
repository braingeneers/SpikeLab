"""Generate batch job config YAML from an image and profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True, help="Container image tag for the job"
    )
    parser.add_argument(
        "--profile",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Resource profile",
    )
    parser.add_argument("--output", required=True, help="Output YAML path")
    parser.add_argument("--name-prefix", default="analysis-temp")
    parser.add_argument("--namespace", default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resources = {
        "requests_cpu": "2",
        "requests_memory": "8Gi",
        "limits_cpu": "2",
        "limits_memory": "8Gi",
        "requests_gpu": 0,
        "limits_gpu": 0,
        "node_selector": {},
    }
    if args.profile == "gpu":
        resources["requests_gpu"] = 1
        resources["limits_gpu"] = 1
        resources["requests_memory"] = "16Gi"
        resources["limits_memory"] = "16Gi"

    payload = {
        "name_prefix": args.name_prefix,
        "namespace": args.namespace,
        "labels": {
            "analysis": "spikelab",
            "workflow": "batch-temp",
            "image_profile": args.profile,
        },
        "container": {
            "image": args.image,
            "image_pull_policy": "IfNotPresent",
            "command": ["python", "/opt/spikelab/run_analysis.py"],
            "args": [],
            "env": {
                "OUTPUT_PREFIX": "s3://YOUR-BUCKET/YOUR-PREFIX/",
            },
        },
        "resources": resources,
        "volumes": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"JOB_CONFIG={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
