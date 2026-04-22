"""Container entrypoint for spike sorting batch jobs.

Invoked as ``python -m spikelab.batch_jobs.entrypoints.sorting``
inside a Kubernetes job container.

Environment variables:
    INPUT_URI: S3 URI of the input bundle zip containing recordings +
        sorting_config.json.
    OUTPUT_PREFIX: S3 URI prefix for uploading sorted results.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import zipfile
from dataclasses import fields
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def _reconstruct_config(config_dict: dict):
    """Rebuild a SortingPipelineConfig from a nested dict.

    Parameters:
        config_dict (dict): Nested dict as produced by
            ``dataclasses.asdict(config)``.

    Returns:
        config (SortingPipelineConfig): Reconstructed config.
    """
    from spikelab.spike_sorting.config import SortingPipelineConfig

    sub_configs = {}
    for f in fields(SortingPipelineConfig):
        sub_dict = config_dict.get(f.name, {})
        sub_configs[f.name] = f.type(**sub_dict)
    return SortingPipelineConfig(**sub_configs)


def main() -> None:
    """Download recordings, run sorting, upload results."""
    input_uri = _require_env("INPUT_URI")
    output_prefix = _require_env("OUTPUT_PREFIX")

    from spikelab.batch_jobs.storage_s3 import S3StorageClient
    from spikelab.spike_sorting.pipeline import sort_recording

    storage = S3StorageClient(
        prefix=output_prefix,
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    with tempfile.TemporaryDirectory(prefix="spikelab-sorting-") as work_dir:
        work = Path(work_dir)

        # --- Download and extract input bundle ---
        bundle_zip = str(work / "input.zip")
        storage.download_file(s3_uri=input_uri, local_path=bundle_zip)

        extract_dir = work / "input"
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            zf.extractall(extract_dir)

        # --- Load sorting config ---
        config_files = list(extract_dir.rglob("sorting_config.json"))
        if not config_files:
            raise FileNotFoundError(
                "sorting_config.json not found in input bundle"
            )
        with open(config_files[0], "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = _reconstruct_config(config_dict)

        # --- Identify recording files ---
        # Everything in the bundle that is not sorting_config.json or
        # manifest.json is a recording file.
        recording_files = []
        for path in sorted(extract_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name in {"sorting_config.json", "manifest.json"}:
                continue
            recording_files.append(str(path))

        if not recording_files:
            raise FileNotFoundError("No recording files found in input bundle")

        # --- Set up output folders ---
        results_dir = work / "results"
        inter_dir = work / "intermediate"
        results_folders = [
            str(results_dir / Path(r).stem) for r in recording_files
        ]
        inter_folders = [
            str(inter_dir / Path(r).stem) for r in recording_files
        ]
        for folder in results_folders + inter_folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

        # --- Run sorting ---
        spikedata_results = sort_recording(
            recording_files=recording_files,
            config=config,
            intermediate_folders=inter_folders,
            results_folders=results_folders,
        )

        # --- Save and upload curated SpikeData pickles ---
        for i, sd in enumerate(spikedata_results):
            rec_name = Path(recording_files[min(i, len(recording_files) - 1)]).stem
            pkl_name = f"{rec_name}_curated.pkl"
            pkl_path = str(results_dir / pkl_name)
            with open(pkl_path, "wb") as f:
                pickle.dump(sd, f)

            s3_uri = output_prefix + pkl_name
            storage.upload_file(local_path=pkl_path, s3_uri=s3_uri)

        # --- Upload sorting metadata ---
        meta = {
            "n_recordings": len(recording_files),
            "n_results": len(spikedata_results),
            "recording_names": [Path(r).stem for r in recording_files],
            "sorter": config.sorter.sorter_name,
        }
        meta_path = str(results_dir / "sorting_report.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        storage.upload_file(
            local_path=meta_path, s3_uri=output_prefix + "sorting_report.json"
        )

        # --- Upload QC figures if generated ---
        for res_folder in results_folders:
            for fig_path in Path(res_folder).rglob("*.png"):
                relative = fig_path.relative_to(results_dir)
                s3_uri = output_prefix + str(relative).replace("\\", "/")
                storage.upload_file(local_path=str(fig_path), s3_uri=s3_uri)

    print("Sorting job completed successfully.")


if __name__ == "__main__":
    main()
