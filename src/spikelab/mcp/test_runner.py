"""Subprocess-based runner for the OC-SpikeLab MCP test suite.

Called by the dev-only ``run_test_suite`` MCP tool. Test scripts live under
``tests/mcp/`` in this repo; categories map to filename prefixes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Tests live next to the package install. For an editable install they're at
# <clone>/tests/mcp; for a wheel install they aren't shipped — the dev tool
# is only useful from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = _REPO_ROOT / "tests" / "mcp"

CATEGORIES = ("INFRA", "SEC", "SCOPE", "EDU", "SORT")


def _discover(categories: list[str] | None) -> list[Path]:
    if not TESTS_DIR.exists():
        return []
    want = tuple(c.upper() for c in (categories or CATEGORIES))
    out: list[Path] = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        # Filename pattern: test_<CATEGORY>_<id>_<slug>.py
        parts = p.stem.split("_", 2)
        if len(parts) < 2:
            continue
        cat = parts[1].upper()
        if cat in want:
            out.append(p)
    return out


def run_test_suite(categories: list[str] | None = None) -> dict[str, Any]:
    """Run the discovered test scripts in sequence. Returns per-script results."""
    scripts = _discover(categories)
    results: list[dict[str, Any]] = []
    for script in scripts:
        t0 = time.time()
        try:
            # SORT-1 may take a few minutes; rest are sub-30s. 30 min ceiling
            # per script covers a real Kilosort run on synthetic data.
            r = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            results.append(
                {
                    "script": script.name,
                    "returncode": r.returncode,
                    "elapsed_s": round(time.time() - t0, 2),
                    "stdout_tail": r.stdout[-2000:],
                    "stderr_tail": r.stderr[-2000:],
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "script": script.name,
                    "returncode": -1,
                    "elapsed_s": round(time.time() - t0, 2),
                    "error": "timeout",
                }
            )

    passed = sum(1 for r in results if r.get("returncode") == 0)
    failed = sum(1 for r in results if r.get("returncode") not in (0, None))
    return {
        "categories_requested": list(categories or CATEGORIES),
        "scripts_run": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
