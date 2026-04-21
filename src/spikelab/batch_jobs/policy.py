"""Cluster policy preflight checks for job specs.

Thresholds are read from the active :class:`ClusterProfile` so that
different clusters can enforce different rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

from .models import ClusterProfile, JobSpec, PolicyConfig

Level = Literal["PASS", "WARN", "BLOCK"]


@dataclass
class PolicyFinding:
    code: str
    level: Level
    message: str


_SLEEP_THRESHOLD = 86_400  # 24 hours in seconds


def _contains_disallowed_sleep(command: Sequence[str], args: Sequence[str]) -> bool:
    """Detect idle-placeholder sleep patterns in batch job commands.

    This is a best-effort heuristic, not a security boundary. It catches
    common idle patterns (``sleep infinity``, ``sleep inf``, bare
    ``sleep``, and ``sleep <large_number>``) but cannot detect arbitrary
    constructs like ``while true; do sleep 60; done`` or obfuscated
    variants. The goal is to flag accidental misuse, not to prevent
    determined circumvention.
    """
    all_tokens = [*command, *args]
    joined = " ".join(all_tokens).lower()

    # Exact patterns
    if "sleep infinity" in joined or "sleep inf" in joined:
        return True

    # Bare "sleep" as the sole command (no duration argument)
    if joined.strip() == "sleep":
        return True

    # "sleep <large_number>" — check each token pair
    for i, tok in enumerate(all_tokens):
        if tok.lower() == "sleep" and i + 1 < len(all_tokens):
            try:
                duration = float(all_tokens[i + 1])
                if duration >= _SLEEP_THRESHOLD:
                    return True
            except (ValueError, IndexError):
                pass

    return False


def evaluate_policy(
    job_spec: JobSpec,
    profile: ClusterProfile,
) -> List[PolicyFinding]:
    """Evaluate policy checks using profile-driven thresholds."""
    findings: List[PolicyFinding] = []
    res = job_spec.resources
    cfg: PolicyConfig = profile.policy

    if res.requests_gpu > cfg.max_interactive_gpus:
        findings.append(
            PolicyFinding(
                "interactive_gpu_limit",
                "WARN",
                f"Requested GPUs exceed interactive limit guidance "
                f"({cfg.max_interactive_gpus} GPUs).",
            )
        )
    else:
        findings.append(
            PolicyFinding(
                "interactive_gpu_limit",
                "PASS",
                "GPU request is within interactive guidance.",
            )
        )

    if cfg.block_sleep_infinity and _contains_disallowed_sleep(
        job_spec.container.command, job_spec.container.args
    ):
        findings.append(
            PolicyFinding(
                "sleep_in_batch_job",
                "BLOCK",
                "Batch jobs containing 'sleep infinity' or trailing sleep "
                "are disallowed.",
            )
        )
    else:
        findings.append(
            PolicyFinding(
                "sleep_in_batch_job",
                "PASS",
                "No forbidden sleep patterns detected in command/args.",
            )
        )

    if cfg.warn_request_limit_mismatch and (
        res.requests_cpu != res.limits_cpu or res.requests_memory != res.limits_memory
    ):
        findings.append(
            PolicyFinding(
                "request_limit_mismatch",
                "WARN",
                "Cluster recommends requests close to limits; tune with " "monitoring.",
            )
        )
    else:
        findings.append(
            PolicyFinding(
                "request_limit_mismatch",
                "PASS",
                "CPU/memory requests and limits are aligned.",
            )
        )

    if (
        job_spec.active_deadline_seconds
        and job_spec.active_deadline_seconds > cfg.max_runtime_seconds
    ):
        findings.append(
            PolicyFinding(
                "long_runtime",
                "WARN",
                f"Runtime exceeds configured maximum " f"({cfg.max_runtime_seconds}s).",
            )
        )
    return findings


def summarize_preflight(findings: Iterable[PolicyFinding]) -> tuple[Level, str]:
    """Return aggregate level and text summary."""
    levels = {finding.level for finding in findings}
    if "BLOCK" in levels:
        status: Level = "BLOCK"
    elif "WARN" in levels:
        status = "WARN"
    else:
        status = "PASS"
    text = "\n".join(
        f"[{finding.level}] {finding.code}: {finding.message}" for finding in findings
    )
    return status, text
