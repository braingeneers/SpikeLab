"""Credential resolution and redaction utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ResolvedCredentials:
    kubeconfig: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_session_token: Optional[str]


def resolve_credentials(
    *,
    kubeconfig: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
) -> ResolvedCredentials:
    """Resolve credentials with explicit args first, then environment."""
    return ResolvedCredentials(
        kubeconfig=kubeconfig or os.getenv("KUBECONFIG"),
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
    )


def redact_sensitive_map(values: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Redact common secret values before logging."""
    redacted: Dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            redacted[key] = ""
            continue
        key_upper = key.upper()
        if "SECRET" in key_upper or "TOKEN" in key_upper or "KEY" in key_upper:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted
