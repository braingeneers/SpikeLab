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
from typing import List


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def _find_workspace_h5(extract_dir: Path) -> Path:
    """Locate the workspace .h5 by content signature.

    Every SpikeLab workspace .h5 written by ``AnalysisWorkspace.save``
    has the ``__workspace_id__`` attribute at the file root (see
    ``workspace.hdf5_io.dump_workspace``). Filename is not canonical
    because ``submit_workspace_job`` preserves the user's chosen base
    path when they pass a string instead of an AnalysisWorkspace
    object, so any .h5 in the bundle could carry an arbitrary name —
    including names that collide with extra ``bundle_input_paths``
    .h5 files like recordings.

    Parameters:
        extract_dir (Path): Directory containing the extracted bundle.

    Returns:
        Path: Path to the unique workspace .h5 in the bundle.

    Raises:
        FileNotFoundError: If no .h5 in the bundle has the
            ``__workspace_id__`` attribute.
        RuntimeError: If more than one .h5 in the bundle matches —
            the bundle layout is ambiguous and the entrypoint refuses
            to guess.
    """
    import h5py

    candidates: List[Path] = []
    for h5_path in extract_dir.rglob("*.h5"):
        try:
            with h5py.File(h5_path, "r") as f:
                if "__workspace_id__" in f.attrs:
                    candidates.append(h5_path)
        except OSError:
            # Not a valid HDF5 file (e.g. truncated, wrong format) —
            # skip; clearly not a workspace.
            continue
    if not candidates:
        raise FileNotFoundError(
            "No SpikeLab workspace .h5 found in input bundle. "
            "Expected a file with the __workspace_id__ attribute "
            "(written by AnalysisWorkspace.save)."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple workspace .h5 candidates in bundle: "
            f"{[str(p) for p in candidates]}. Each bundle should "
            f"contain exactly one workspace."
        )
    return candidates[0]


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

        # Find the workspace .h5 by its content signature (the
        # __workspace_id__ attribute), not by filename — bundles can
        # contain other .h5 files (recordings, intermediate outputs)
        # via ``bundle_input_paths``, and the workspace's name follows
        # whatever base path the caller chose.
        workspace_h5 = _find_workspace_h5(extract_dir)
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
