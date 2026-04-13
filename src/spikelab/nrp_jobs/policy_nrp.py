"""NRP policy preflight checks for job specs.

References:
- https://nrp.ai/documentation/userdocs/start/policies/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

from .models import JobSpec

Level = Literal["PASS", "WARN", "BLOCK"]


@dataclass
class PolicyFinding:
    code: str
    level: Level
    message: str


def _contains_disallowed_sleep(command: Sequence[str], args: Sequence[str]) -> bool:
    tokens = " ".join([*command, *args]).lower()
    return "sleep infinity" in tokens or tokens.endswith(" sleep")


def evaluate_nrp_policy(job_spec: JobSpec) -> List[PolicyFinding]:
    """Evaluate policy checks and return findings."""
    findings: List[PolicyFinding] = []
    res = job_spec.resources

    if res.requests_gpu > 2:
        findings.append(
            PolicyFinding(
                "interactive_gpu_limit",
                "WARN",
                "Requested GPUs exceed interactive limit guidance (2 GPUs).",
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

    if _contains_disallowed_sleep(job_spec.container.command, job_spec.container.args):
        findings.append(
            PolicyFinding(
                "sleep_in_batch_job",
                "BLOCK",
                "Batch jobs containing 'sleep infinity' or trailing sleep are disallowed.",
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

    if res.requests_gpu > 0 and res.limits_gpu == 0:
        findings.append(
            PolicyFinding(
                "gpu_requests_without_limits",
                "BLOCK",
                "GPU request must include matching GPU limit.",
            )
        )

    if res.requests_cpu != res.limits_cpu or res.requests_memory != res.limits_memory:
        findings.append(
            PolicyFinding(
                "request_limit_mismatch",
                "WARN",
                "NRP recommends requests close to limits; tune with monitoring.",
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

    if job_spec.active_deadline_seconds and job_spec.active_deadline_seconds > 14 * 24 * 3600:
        findings.append(
            PolicyFinding(
                "long_runtime",
                "WARN",
                "Runtime exceeds 2-week workload purging window.",
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
