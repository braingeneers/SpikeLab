"""Temporary analysis entrypoint for batch jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    run_id = os.getenv("RUN_ID", "local-run")
    output_dir = Path(os.getenv("OUTPUT_DIR", "/tmp/spikelab-output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "status": "ok",
        "message": "Temporary analysis container executed.",
        "output_prefix": os.getenv("OUTPUT_PREFIX", ""),
    }
    (output_dir / "analysis_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
