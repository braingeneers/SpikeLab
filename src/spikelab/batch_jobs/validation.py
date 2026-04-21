"""Validation helpers for CLI/API job inputs."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import ValidationError

from .models import JobSpec, RunConfig


def validate_job_spec(payload: Dict[str, Any]) -> JobSpec:
    """Parse and validate a raw job spec payload."""
    return JobSpec.model_validate(payload)


def validate_run_config(payload: Dict[str, Any]) -> RunConfig:
    """Parse and validate a run config payload."""
    return RunConfig.model_validate(payload)


def summarize_validation_error(exc: ValidationError) -> str:
    """Return a concise human-readable validation summary."""
    parts = []
    for issue in exc.errors():
        loc = ".".join(str(item) for item in issue.get("loc", []))
        msg = issue.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)
