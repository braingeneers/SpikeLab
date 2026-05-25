"""Credential resolution and redaction utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

#: Sensitive substrings matched as word-boundary tokens in upper-cased
#: keys. The previous substring check redacted ``SECRETS_PATH`` (and
#: similar non-secret keys that happened to contain ``SECRET``) as a
#: false positive. Word-boundary matching restricts the heuristic to
#: keys that actually name a secret credential.
_SENSITIVE_PATTERNS = tuple(
    re.compile(rf"(^|[^A-Z]){tok}([^A-Z]|$)") for tok in ("SECRET", "TOKEN", "PASSWORD")
)


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
    """Redact common secret values before logging.

    Notes:
        - Keys are matched against word-boundary patterns for
          ``SECRET``, ``TOKEN``, and ``PASSWORD``. Previously the
          substring check redacted ``SECRETS_PATH`` (and similar
          non-secret keys that happened to contain ``SECRET``) as a
          false positive — the value of ``SECRETS_PATH`` is a
          filesystem path, not a credential.
    """
    redacted: Dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            redacted[key] = ""
            continue
        key_upper = key.upper()
        if any(pat.search(key_upper) for pat in _SENSITIVE_PATTERNS):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted
