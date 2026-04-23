"""Container entrypoint for workspace-centric batch jobs.

Invoked as ``python -m spikelab.batch_jobs.entrypoints.workspace``
inside a Kubernetes job container.

Environment variables:
    INPUT_URI: S3 URI of the input bundle zip.
    OUTPUT_PREFIX: S3 URI prefix for uploading the updated workspace.
    SCRIPT_NAME: Filename of the analysis script inside the bundle.
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def main() -> None:
    """Download workspace bundle, run analysis script, upload results."""
    input_uri = _require_env("INPUT_URI")
    output_prefix = _require_env("OUTPUT_PREFIX")
    script_name = _require_env("SCRIPT_NAME")

    from spikelab.batch_jobs.storage_s3 import S3StorageClient
    from spikelab.data_loaders.s3_utils import parse_s3_url
    from spikelab.workspace.workspace import AnalysisWorkspace

    # Build a minimal storage client from environment
    storage = S3StorageClient(
        prefix=output_prefix,
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    with tempfile.TemporaryDirectory(prefix="spikelab-workspace-") as work_dir:
        work = Path(work_dir)

        # --- Download and extract input bundle ---
        bundle_zip = str(work / "input.zip")
        storage.download_file(s3_uri=input_uri, local_path=bundle_zip)

        extract_dir = work / "input"
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            zf.extractall(extract_dir)

        # Find the workspace base path (look for *.h5 inside the bundle)
        h5_files = list(extract_dir.rglob("*.h5"))
        if not h5_files:
            raise FileNotFoundError("No .h5 workspace file found in input bundle")
        workspace_h5 = h5_files[0]
        workspace_base = str(workspace_h5.with_suffix(""))

        # Find the analysis script
        script_candidates = list(extract_dir.rglob(script_name))
        if not script_candidates:
            raise FileNotFoundError(
                f"Analysis script {script_name!r} not found in input bundle"
            )
        script_path = str(script_candidates[0])

        # --- Load workspace ---
        workspace = AnalysisWorkspace.load(workspace_base)

        # --- Run analysis script ---
        # The script receives the workspace as a global variable named
        # 'workspace'. It can modify it freely; the modified workspace
        # is saved and uploaded after the script completes.
        run_globals = {
            "workspace": workspace,
            "__name__": "__main__",
        }
        runpy.run_path(script_path, init_globals=run_globals, run_name="__main__")

        # In case the script replaced the workspace object
        workspace = run_globals.get("workspace", workspace)

        # --- Save and upload results ---
        output_dir = work / "output"
        output_dir.mkdir()
        output_base = str(output_dir / "workspace")
        workspace.save(output_base)

        # Upload .h5 and .json
        for ext in (".h5", ".json"):
            local_file = f"{output_base}{ext}"
            s3_uri = output_prefix + f"workspace{ext}"
            storage.upload_file(local_path=local_file, s3_uri=s3_uri)

    print("Workspace job completed successfully.")


if __name__ == "__main__":
    main()
