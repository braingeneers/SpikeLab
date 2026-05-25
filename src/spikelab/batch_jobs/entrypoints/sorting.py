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
from typing import Any, Dict


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def _reconstruct_config(config_dict: dict):
    """Rebuild a SortingPipelineConfig from a nested dict.

    Host-side ``_resolve_sorting_config`` serialises via
    ``dataclasses.asdict(config)`` + ``json.dumps(..., default=str)``,
    which flattens ``Path`` fields to plain strings. The naive
    reconstruction ``f.type(**sub_dict)`` would then leave ``Path``-
    annotated fields as ``str`` at runtime (dataclasses do not coerce
    from type hints), and downstream code that does
    ``cfg.recording.intermediate_dir / "subdir"`` would crash with
    ``TypeError: unsupported operand type(s) for /: 'str' and 'str'``.

    This loop re-wraps any field whose static type annotation is
    ``pathlib.Path`` back into a ``Path`` after the
    sub-dataclass is constructed.

    Parameters:
        config_dict (dict): Nested dict as produced by
            ``dataclasses.asdict(config)``.

    Returns:
        config (SortingPipelineConfig): Reconstructed config.
    """
    from pathlib import Path

    from spikelab.spike_sorting.config import SortingPipelineConfig

    def _is_path_annotation(annotation: Any) -> bool:
        # Match ``Path`` and ``Optional[Path]`` (the latter shows up as
        # a typing union with NoneType in the get_type_hints result).
        if annotation is Path:
            return True
        args = getattr(annotation, "__args__", None)
        if args:
            return any(a is Path for a in args)
        return False

    import typing

    sub_configs = {}
    for f in fields(SortingPipelineConfig):
        sub_dict = config_dict.get(f.name, {})
        sub_instance = f.type(**sub_dict)
        # Restore Path fields on the just-constructed sub-config.
        try:
            sub_hints = typing.get_type_hints(f.type)
        except Exception:
            sub_hints = {}
        for sub_field_name, sub_annotation in sub_hints.items():
            if not _is_path_annotation(sub_annotation):
                continue
            val = getattr(sub_instance, sub_field_name, None)
            if isinstance(val, str) and val:
                setattr(sub_instance, sub_field_name, Path(val))
        sub_configs[f.name] = sub_instance
    return SortingPipelineConfig(**sub_configs)


def main() -> None:
    """Download recordings, run sorting, upload results."""
    input_uri = _require_env("INPUT_URI")
    output_prefix = _require_env("OUTPUT_PREFIX")

    from spikelab.batch_jobs.storage_s3 import S3StorageClient
    from spikelab.spike_sorting.pipeline import sort_recording

    # Container receives fully-formed S3 URIs (INPUT_URI, OUTPUT_PREFIX)
    # from env vars — the host did the prefix templating before
    # submission. The entrypoint only needs the raw download/upload
    # primitives, not the prefix-templating methods, so prefix=None
    # preserves the class invariant: ``S3StorageClient.prefix`` always
    # means the bucket-level base, never a templated output URI.
    storage = S3StorageClient(
        prefix=None,
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )

    with tempfile.TemporaryDirectory(prefix="spikelab-sorting-") as work_dir:
        work = Path(work_dir)

        # --- Download and extract input bundle ---
        bundle_zip = str(work / "input.zip")
        storage.download_file(s3_uri=input_uri, local_path=bundle_zip)

        extract_dir = work / "input"
        # Validate zip members against the target directory before
        # extracting. Without this, a malicious bundle (compromised S3,
        # hostile workspace job) could write files outside extract_dir
        # via ``../`` segments — full RCE inside the container on
        # Python <3.12 which does not validate ZipInfo paths.
        from spikelab.batch_jobs.artifact_packager import _safe_extractall

        with zipfile.ZipFile(bundle_zip, "r") as zf:
            _safe_extractall(zf, extract_dir)

        # --- Load sorting config ---
        config_files = list(extract_dir.rglob("sorting_config.json"))
        if not config_files:
            raise FileNotFoundError("sorting_config.json not found in input bundle")
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

        # Reject stem-level collisions across recording_files. Two
        # recordings with the same stem (e.g. ``rec.bin`` and
        # ``rec.h5``, or ``dir_a/rec.bin`` and ``dir_b/rec.bin`` in a
        # legacy bundle) would both produce ``pkl_name="rec_curated.pkl"``
        # and silently overwrite each other on S3. The host-side
        # packager already rejects duplicate basenames at packaging
        # time, but this defensive check catches bundles constructed
        # externally or via non-spikelab tooling.
        stem_to_path: Dict[str, str] = {}
        for rec_path in recording_files:
            stem = Path(rec_path).stem
            if stem in stem_to_path:
                raise RuntimeError(
                    f"Recording basename collision on stem {stem!r}: "
                    f"{stem_to_path[stem]!r} and {rec_path!r} both produce "
                    f"output filename {stem}_curated.pkl. Rename the inputs "
                    "so each recording has a unique stem before bundling."
                )
            stem_to_path[stem] = rec_path

        # --- Set up output folders ---
        results_dir = work / "results"
        inter_dir = work / "intermediate"
        results_folders = [str(results_dir / Path(r).stem) for r in recording_files]
        inter_folders = [str(inter_dir / Path(r).stem) for r in recording_files]
        for folder in results_folders + inter_folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

        # --- Run sorting ---
        # Construct a SortRunReport so per-recording status, wall time,
        # and any failure classification can be uploaded alongside the
        # bundle. Previously the entrypoint only saw the list of
        # SpikeData and reported "Sorting job completed successfully"
        # even when some recordings had degraded or partial results.
        from spikelab.spike_sorting.pipeline import SortRunReport

        sort_report = SortRunReport()
        spikedata_results = sort_recording(
            recording_files=recording_files,
            config=config,
            intermediate_folders=inter_folders,
            results_folders=results_folders,
            out_report=sort_report,
        )

        # --- Save and upload curated SpikeData pickles ---
        # ``sort_recording`` must return exactly one SpikeData per
        # recording. The previous ``min(i, len(recording_files) - 1)``
        # saturation silently clobbered earlier pickles to the same
        # S3 key on length-too-long, and silently dropped trailing
        # recordings on length-too-short — both modes still reported
        # success. Fail loudly instead, then iterate by zip so the
        # mapping is structurally enforced.
        if len(spikedata_results) != len(recording_files):
            raise RuntimeError(
                f"sort_recording returned {len(spikedata_results)} "
                f"SpikeData but bundle contains {len(recording_files)} "
                "recordings; cannot map results to recording names."
            )
        for sd, rec_path in zip(spikedata_results, recording_files):
            rec_name = Path(rec_path).stem
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
            "all_succeeded": sort_report.all_succeeded,
            "n_succeeded": len(sort_report.succeeded),
            "n_failed": len(sort_report.failed),
        }
        meta_path = str(results_dir / "sorting_report.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        storage.upload_file(
            local_path=meta_path, s3_uri=output_prefix + "sorting_report.json"
        )

        # Detailed per-recording report — JSON-serialisable and
        # contains status, wall time, classified errors, etc. — so the
        # operator can audit which recordings succeeded without
        # parsing per-recording log files.
        detailed_report_path = str(results_dir / "sort_run_report.json")
        with open(detailed_report_path, "w", encoding="utf-8") as f:
            json.dump(sort_report.to_dict(), f, indent=2, default=str)
        storage.upload_file(
            local_path=detailed_report_path,
            s3_uri=output_prefix + "sort_run_report.json",
        )

        # --- Upload QC figures if generated ---
        for res_folder in results_folders:
            for fig_path in Path(res_folder).rglob("*.png"):
                relative = fig_path.relative_to(results_dir)
                s3_uri = output_prefix + str(relative).replace("\\", "/")
                storage.upload_file(local_path=str(fig_path), s3_uri=s3_uri)

    if sort_report.all_succeeded:
        print(
            f"Sorting job completed successfully ({len(sort_report.succeeded)}/"
            f"{len(sort_report.records)} recordings)."
        )
    else:
        # Surface the failed recordings in the final log line so the
        # operator notices that "job exit 0" does not necessarily
        # mean every recording was processed cleanly.
        failed_names = ", ".join(r.rec_path for r in sort_report.failed)
        print(
            f"Sorting job completed with partial failures: "
            f"{len(sort_report.failed)}/{len(sort_report.records)} "
            f"recordings failed ({failed_names}). See "
            f"sort_run_report.json for per-recording status."
        )


if __name__ == "__main__":
    main()
