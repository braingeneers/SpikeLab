"""Cluster policy preflight checks for job specs.

Thresholds are read from the active :class:`ClusterProfile` so that
different clusters can enforce different rules.
"""

from __future__ import annotations

import math
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

    Notes:
        - The bare-``sleep`` check fires only when ``sleep`` is the sole
          token across ``command + args``. A trailing token that
          happens to be the literal string ``"sleep"`` (e.g.
          ``["python", "-c", "sleep"]``, where ``"sleep"`` is a Python
          snippet, not a shell command) is intentionally NOT flagged.
          This avoids false positives on commands that legitimately
          pass the string ``"sleep"`` as an argument. The downside is
          that a determined operator can sneak in a real
          ``exec("sleep infinity")``-style payload; this is documented
          as out-of-scope for the heuristic.
    """
    # Flatten multi-word tokens (e.g., ["sleep infinity"] from sh -c)
    # into individual words for consistent token-pair matching.
    all_tokens = []
    for tok in [*command, *args]:
        all_tokens.extend(tok.split())

    # Bare "sleep" as the sole command (no duration argument)
    if len(all_tokens) == 1 and all_tokens[0].lower() == "sleep":
        return True

    # Check token pairs: "sleep infinity", "sleep inf", "sleep <large_number>"
    for i, tok in enumerate(all_tokens):
        if tok.lower() != "sleep" or i + 1 >= len(all_tokens):
            continue
        next_tok = all_tokens[i + 1].lower()
        if next_tok in ("infinity", "inf"):
            return True
        try:
            duration = float(next_tok)
            # Flag any non-finite duration (NaN, +inf, -inf) as suspicious —
            # the actual sleep binary rejects these, and a job spec with
            # such a token is almost certainly a bug or an obfuscation
            # attempt around the literal "inf" / "infinity" check above.
            if not math.isfinite(duration) or duration >= _SLEEP_THRESHOLD:
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

    sleep_present = _contains_disallowed_sleep(
        job_spec.container.command, job_spec.container.args
    )
    if cfg.block_sleep_infinity and sleep_present:
        findings.append(
            PolicyFinding(
                "sleep_in_batch_job",
                "BLOCK",
                "Batch jobs containing 'sleep infinity' or trailing sleep "
                "are disallowed.",
            )
        )
    elif sleep_present:
        # ``block_sleep_infinity`` is disabled in the profile, but a
        # sleep pattern *was* detected. Surfacing this as a WARN keeps
        # the audit trail honest — the previous code emitted a PASS
        # ("No forbidden sleep patterns detected") even though the
        # pattern was present, just not blocked.
        findings.append(
            PolicyFinding(
                "sleep_in_batch_job",
                "WARN",
                "Sleep pattern detected but block_sleep_infinity is "
                "disabled in this profile; the job will be permitted "
                "but the pattern is recorded for audit.",
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
                "Cluster recommends requests close to limits; tune with monitoring.",
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

    if not job_spec.active_deadline_seconds:
        # No deadline set; the cluster's own max applies via Kubernetes.
        # Emit a PASS finding so the audit trail is complete.
        findings.append(
            PolicyFinding(
                "long_runtime",
                "PASS",
                "No active_deadline_seconds set; cluster default applies.",
            )
        )
    elif job_spec.active_deadline_seconds > cfg.max_runtime_seconds:
        findings.append(
            PolicyFinding(
                "long_runtime",
                "WARN",
                f"Runtime ({job_spec.active_deadline_seconds}s) exceeds "
                f"configured maximum ({cfg.max_runtime_seconds}s).",
            )
        )
    else:
        findings.append(
            PolicyFinding(
                "long_runtime",
                "PASS",
                f"Runtime ({job_spec.active_deadline_seconds}s) within "
                f"configured maximum ({cfg.max_runtime_seconds}s).",
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
