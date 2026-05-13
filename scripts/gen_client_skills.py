#!/usr/bin/env python3
"""Generate the two OC-SpikeLab client skill files from a single canonical source.

Usage:
    python scripts/gen_client_skills.py                       # remote variant only
    python scripts/gen_client_skills.py --local-dest <path>   # both variants

The canonical source is at:
    src/spikelab/agent/skills/_sources/oc-spikelab-client.md.in

It contains markdown blocks marked by HTML-style comment markers:
    <!-- variant: both -->     ...content...                 <!-- endvariant -->
    <!-- variant: remote -->   ...remote-only content...     <!-- endvariant -->
    <!-- variant: local -->    ...local-only content...      <!-- endvariant -->

The generator emits one file per variant with only the matching blocks kept.
The leading comment block (lines starting with ``#``) of the source is
dropped from generated outputs.

Outputs:
    <repo>/src/spikelab/agent/skills/oc-spikelab-remote/SKILL.md   (always)
    <local-dest>                                                    (when --local-dest set)

The local destination is install-specific (the consuming project's
``.claude/skills/OC-spikelab/SKILL.md`` is the canonical target on this rig).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = (
    REPO_ROOT
    / "src"
    / "spikelab"
    / "agent"
    / "skills"
    / "_sources"
    / "oc-spikelab-client.md.in"
)
REMOTE_DEST = (
    REPO_ROOT
    / "src"
    / "spikelab"
    / "agent"
    / "skills"
    / "oc-spikelab-remote"
    / "SKILL.md"
)

_BLOCK_RE = re.compile(
    r"<!--\s*variant:\s*(?P<name>[\w-]+)\s*-->\n(?P<body>.*?)\n?<!--\s*endvariant\s*-->\n?",
    re.DOTALL,
)

_GENERATED_NOTICE = (
    "<!-- GENERATED FILE — DO NOT EDIT. Source: src/spikelab/agent/"
    "skills/_sources/oc-spikelab-client.md.in. Regenerate with "
    "`python scripts/gen_client_skills.py`. -->"
)

_FRONT_MATTER_RE = re.compile(
    r"\A(?P<fm>---\n.*?\n---\n)(?P<rest>.*)",
    re.DOTALL,
)


def _strip_source_preamble(text: str) -> str:
    """Drop the leading ``# `` comment lines from the source file."""
    out: list[str] = []
    in_preamble = True
    for line in text.splitlines(keepends=True):
        if in_preamble:
            stripped = line.lstrip()
            if stripped == "" or stripped.startswith("#"):
                continue
            in_preamble = False
        out.append(line)
    return "".join(out)


def render(source_text: str, variant: str) -> str:
    """Keep only ``<!-- variant: both -->`` and ``<!-- variant: <variant> -->``
    blocks. Strip the markers themselves."""
    body = _strip_source_preamble(source_text)

    def _replace(m: re.Match[str]) -> str:
        name = m.group("name")
        if name in ("both", variant):
            return m.group("body").rstrip() + "\n"
        return ""

    rendered = _BLOCK_RE.sub(_replace, body)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).lstrip()

    # The generated-file notice must come AFTER the YAML front matter so
    # skill loaders that parse `---`-bounded metadata at the top of the
    # file still see the front matter unaltered.
    m = _FRONT_MATTER_RE.match(rendered)
    if m:
        return (
            m.group("fm") + "\n" + _GENERATED_NOTICE + "\n\n" + m.group("rest").lstrip()
        )
    return _GENERATED_NOTICE + "\n\n" + rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dest",
        type=Path,
        default=None,
        help=(
            "Where to write the co-located variant — typically the host "
            "project's .claude/skills/OC-spikelab/SKILL.md. Skipped if omitted."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Render outputs and exit nonzero if they differ from what's on "
            "disk. Useful in CI to catch un-regenerated changes."
        ),
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"ERROR: source missing: {SOURCE}", file=sys.stderr)
        return 2

    source_text = SOURCE.read_text()
    targets: list[tuple[str, Path]] = [("remote", REMOTE_DEST)]
    if args.local_dest is not None:
        targets.append(("local", args.local_dest))

    drift = False
    for variant, dest in targets:
        rendered = render(source_text, variant)
        if args.check:
            existing = dest.read_text() if dest.exists() else ""
            if existing != rendered:
                drift = True
                print(f"DRIFT: {dest}", file=sys.stderr)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(rendered)
            print(f"wrote {dest} ({variant}, {len(rendered)} bytes)")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
