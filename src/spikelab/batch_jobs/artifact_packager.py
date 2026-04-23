"""Create uploadable analysis bundles for batch job execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Literal

SupportedFormat = Literal["workspace", "sorting", "custom"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_analysis_bundle(
    *,
    input_paths: Iterable[str],
    run_id: str,
    output_dir: str,
    output_format: SupportedFormat,
    metadata: Dict[str, object] | None = None,
) -> str:
    """Create a run zip bundle and return its absolute path."""
    if output_format not in {"workspace", "sorting", "custom"}:
        raise ValueError("output_format must be one of: workspace, sorting, custom")

    output_base = Path(output_dir).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{run_id}-bundle-") as temp_dir:
        bundle_dir = Path(temp_dir) / run_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        payload_files: List[Dict[str, str]] = []

        for item in input_paths:
            src = Path(item)
            if not src.exists():
                raise FileNotFoundError(f"Input file not found: {src}")
            dest = bundle_dir / src.name
            shutil.copy2(src, dest)
            payload_files.append(
                {
                    "name": dest.name,
                    "sha256": _sha256(dest),
                    "size_bytes": str(dest.stat().st_size),
                }
            )

        manifest = {
            "run_id": run_id,
            "output_format": output_format,
            "files": payload_files,
            "metadata": metadata or {},
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        zip_base = output_base / run_id
        zip_path = shutil.make_archive(
            str(zip_base), "zip", root_dir=temp_dir, base_dir=run_id
        )
    return str(Path(zip_path).resolve())
