"""CLI entrypoint for NRP job lifecycle commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .backend_k8s import KubernetesBatchJobBackend
from .credentials import resolve_credentials
from .models import ClusterProfile, JobSpec
from .profiles import load_cluster_profile, load_profile_from_name
from .session import RunSession
from .storage_s3 import S3StorageClient
from .validation import summarize_validation_error, validate_job_spec


def _load_payload(path: str) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return payload


def _load_profile(name: str | None, path: str | None) -> ClusterProfile:
    if path:
        return load_cluster_profile(path)
    return load_profile_from_name(name or "nrp")


def _build_session(profile: ClusterProfile, kubeconfig: str | None) -> RunSession:
    creds = resolve_credentials(kubeconfig=kubeconfig)
    backend = KubernetesBatchJobBackend(
        namespace=profile.namespace,
        kubeconfig=creds.kubeconfig,
        use_kubectl_fallback=True,
    )
    storage = S3StorageClient(
        prefix=profile.default_s3_prefix,
        endpoint_url=profile.endpoint_url,
        region_name=profile.region_name,
        aws_access_key_id=creds.aws_access_key_id,
        aws_secret_access_key=creds.aws_secret_access_key,
        aws_session_token=creds.aws_session_token,
    )
    return RunSession(profile=profile, backend=backend, storage_client=storage, credentials=creds)


def _cmd_deploy(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile, args.profile_file)
    session = _build_session(profile, args.kubeconfig)
    payload = _load_payload(args.job_config)
    try:
        job_spec = validate_job_spec(payload)
    except Exception as exc:
        msg = summarize_validation_error(exc) if hasattr(exc, "errors") else str(exc)
        raise SystemExit(f"Invalid job config: {msg}") from exc

    if args.render_only:
        manifest = session.render_manifest(
            job_name=f"{job_spec.name_prefix}-dry-run",
            job_spec=job_spec,
            run_id="dry-run",
        )
        if args.output_manifest:
            Path(args.output_manifest).write_text(manifest, encoding="utf-8")
            print(f"MANIFEST_PATH={args.output_manifest}")
        else:
            print(manifest)
        return 0

    result = session.submit_prepared_job(
        job_spec=job_spec,
        allow_policy_risk=args.allow_policy_risk,
    )
    print(f"JOB_NAME={result.job_name}")
    print(f"OUTPUT_PREFIX={result.output_prefix}")
    print(f"LOGS_PREFIX={result.logs_prefix}")
    if args.wait:
        final_state = session.wait_for_completion(
            job_name=result.job_name,
            max_wait_seconds=args.max_wait_seconds,
        )
        print(f"FINAL_STATUS={final_state}")
        if args.follow_logs:
            pods = session.backend.pods_for_job(result.job_name)
            if pods:
                for line in session.backend.stream_logs(pods[0], follow=False):
                    print(line)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    args.render_only = True
    args.wait = False
    args.follow_logs = False
    args.max_wait_seconds = 0
    args.allow_policy_risk = False
    return _cmd_deploy(args)


def _cmd_status(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile, args.profile_file)
    session = _build_session(profile, args.kubeconfig)
    state = session.backend.job_status(args.job_name)
    print(f"JOB_STATUS={state}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile, args.profile_file)
    session = _build_session(profile, args.kubeconfig)
    pods = session.backend.pods_for_job(args.job_name)
    if not pods:
        raise SystemExit("No pods found for job")
    pod_name = pods[0]
    for line in session.backend.stream_logs(pod_name, follow=args.follow):
        print(line)
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile, args.profile_file)
    session = _build_session(profile, args.kubeconfig)
    session.backend.delete_job(args.job_name)
    print(f"DELETED_JOB={args.job_name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spikelab-nrp-jobs")
    parser.add_argument("--profile", default="nrp")
    parser.add_argument("--profile-file")
    parser.add_argument("--kubeconfig")
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy-job")
    deploy.add_argument("--job-config", required=True)
    deploy.add_argument("--allow-policy-risk", action="store_true")
    deploy.add_argument("--render-only", action="store_true")
    deploy.add_argument("--output-manifest")
    deploy.add_argument("--wait", action="store_true")
    deploy.add_argument("--max-wait-seconds", type=int, default=3600)
    deploy.add_argument("--follow-logs", action="store_true")
    deploy.set_defaults(func=_cmd_deploy)

    status = sub.add_parser("job-status")
    status.add_argument("job_name")
    status.set_defaults(func=_cmd_status)

    logs = sub.add_parser("job-logs")
    logs.add_argument("job_name")
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=_cmd_logs)

    delete = sub.add_parser("job-delete")
    delete.add_argument("job_name")
    delete.set_defaults(func=_cmd_delete)

    render = sub.add_parser("render-job")
    render.add_argument("--job-config", required=True)
    render.add_argument("--output-manifest")
    render.set_defaults(func=_cmd_render)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
