# Batch Jobs — Remote Cluster Execution Instructions

These instructions describe how to deploy SpikeLab analysis or spike sorting
jobs to a remote Kubernetes cluster using the `spikelab-batch-jobs` CLI. Read
this file when a user requests remote/cluster execution (e.g., "run this on
NRP", "deploy to cluster", "submit a batch job").

Works with any Kubernetes cluster — use `--profile nrp` for Nautilus or
create a custom profile YAML for other clusters.

## Required Inputs

Ask the user for:
- Job config path (`--job-config`) with image, command/args, resources, and optional volumes.
- Target profile: `--profile defaults` (generic), `--profile nrp` (Nautilus), or `--profile-file /path/to/custom.yaml`.
- Image strategy (`--image-profile cpu|gpu` or explicit `--image`).
- Namespace/context confirmation (`kubectl config current-context` + namespace).
- Whether they want to wait for completion and stream logs.

Never ask users to paste secrets in chat.

## Profiles

Profiles control namespace, default images, S3 prefix, credential mounts, and policy thresholds. Two built-in profiles ship with the package:

- **`defaults`** — Generic defaults, no org-specific values. Requires the user to specify image, S3 prefix, and volumes explicitly.
- **`nrp`** — Pre-configured for the Nautilus Research Platform (braingeneers namespace, NRP images, S3 credentials, kube-config mounts).

**Custom profiles:** Create a YAML file with any subset of `ClusterProfile` fields. Key fields:

```yaml
name: my-cluster
namespace: my-team
default_images:
  cpu: registry.example.com/analysis:cpu
  gpu: registry.example.com/analysis:gpu
default_s3_prefix: "s3://my-bucket/my-prefix/"
namespace_hooks:
  my-team:
    image_pull_policy: Always
    env_defaults:
      AWS_SHARED_CREDENTIALS_FILE: /etc/spikelab/aws/credentials
    required_volumes:
      - name: aws-creds
        mount_path: /etc/spikelab/aws/credentials
        sub_path: credentials
        secret_name: aws-credentials
        read_only: true
policy:
  max_interactive_gpus: 4
  max_runtime_seconds: 604800
```

Use with: `spikelab-batch-jobs deploy-job --profile-file /path/to/my-cluster.yaml --job-config ...`

## Credentials and Secrets

- Credentials must come from user environment or files they already manage (`KUBECONFIG`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional session token).
- Never print secret values.
- Never commit credentials into files.
- Reference Kubernetes secrets by name only.
- For private registries, reference image pull secret names in Kubernetes, never raw credentials.

**Namespace hooks and credentials:** When a profile has `namespace_hooks` configured for the target namespace, the batch system automatically:
- Mounts credential secrets as volumes at configured paths (e.g., `/etc/spikelab/aws/`)
- Injects environment variables (`AWS_SHARED_CREDENTIALS_FILE`, `KUBECONFIG`, etc.) pointing to those mounts

This means containers do not need to be root or have credentials at `/root/.aws/`. If a user reports "credentials not found" errors, check that:
1. The profile's `namespace_hooks` match the target namespace
2. The K8s secrets referenced in `required_volumes` exist in the namespace
3. The env vars point to the correct mount paths

## Container Prep (for compute-intensive workflows)

These scripts are in the SpikeLab repository under `scripts/` and `docker/`. They are not globally installed — run them from the repo root.

1. Choose base image path:
   - CPU: `docker/analysis-base/Dockerfile.cpu`
   - GPU: `docker/analysis-base/Dockerfile.gpu`
2. Build a temporary run image using:
   - `bash scripts/build_temp_image.sh <cpu|gpu> <image-tag>`
3. Push the image:
   - `bash scripts/push_temp_image.sh <image-tag>`
4. Generate a job config:
   - `python scripts/generate_job_config.py --image <image-tag> --profile <cpu|gpu> --output configs/batch-temp-job.yaml`
5. Confirm image is pullable from target cluster/namespace before deploy.

## Fixed Workflow

1. **Preflight checks**
   - Run `kubectl version --client`.
   - Run `kubectl config current-context`.
   - Validate registry/image tag exists and is pushed.
   - Optionally verify S3 access if asked by the user.
2. **Validate inputs**
   - Ensure `--job-config` is present.
   - Run a dry render first:
     - `spikelab-batch-jobs render-job --profile <profile> --job-config <path> --image-profile <cpu|gpu> --output-manifest /tmp/job.yaml`
   - Inspect the rendered YAML for correctness before submitting.
3. **Submit**
   - Run `spikelab-batch-jobs deploy-job --profile <profile> --job-config <path> --image-profile <cpu|gpu>`.
   - If user requested explicit image, pass `--image <image-tag>`.
   - Add `--wait --max-wait-seconds <N>` if user wants to block until completion.
   - Capture the machine-parseable line: `JOB_NAME=<value>`.
4. **Observe**
   - If user wants status: `spikelab-batch-jobs job-status --profile <profile> <job_name>`.
   - If user wants logs: `spikelab-batch-jobs job-logs --profile <profile> <job_name> --follow`.
5. **Failure triage**
   - Show `spikelab-batch-jobs job-status --profile <profile> <job_name>`.
   - Suggest `kubectl describe job <job_name> -n <namespace>` and pod logs.
6. **Teardown guidance**
   - Suggest deleting completed/failed jobs:
     - `spikelab-batch-jobs job-delete --profile <profile> <job_name>`
   - Remind user to clean up temporary image tags no longer needed.

**Important:** Always include `--profile <name>` in commands. The default is `defaults` (generic, no cluster-specific config). For NRP, always use `--profile nrp`.

## Policy Safety Rails

- Default behavior is policy-safe. Thresholds are set per-profile.
- Do not use `--allow-policy-risk` unless user explicitly asks.
- If a policy warning/block appears, explain it and request confirmation before continuing.
- Reject patterns that resemble batch `sleep infinity` placeholders.
- Policy checks: GPU limits, sleep detection, request/limit alignment, runtime caps.

## Python API

For programmatic job submission (e.g., from analysis scripts):

```python
from spikelab.batch_jobs import RunSession, ClusterProfile, JobSpec
from spikelab.batch_jobs.profiles import load_profile_from_name
from spikelab.batch_jobs.backend_k8s import KubernetesBatchJobBackend
from spikelab.batch_jobs.storage_s3 import S3StorageClient
from spikelab.batch_jobs.credentials import resolve_credentials

profile = load_profile_from_name("nrp")
creds = resolve_credentials()
backend = KubernetesBatchJobBackend(namespace=profile.namespace)
storage = S3StorageClient(
    prefix=profile.default_s3_prefix,
    path_templates=profile.storage,
)

session = RunSession(
    profile=profile, backend=backend,
    storage_client=storage, credentials=creds,
)

job_spec = JobSpec(
    name_prefix="my-analysis",
    namespace=profile.namespace,
    container={"image": "my-image:latest", "command": ["python", "run.py"]},
    resources={"requests_cpu": "2", "requests_memory": "8Gi",
               "limits_cpu": "2", "limits_memory": "8Gi"},
)

result = session.submit_prepared_job(job_spec=job_spec)
print(result.job_name, result.output_prefix)
```

## CLI Command Reference

```bash
# Deploy a job (NRP profile)
spikelab-batch-jobs deploy-job --profile nrp --job-config configs/job.yaml --image-profile gpu --wait --max-wait-seconds 3600

# Dry-run render
spikelab-batch-jobs render-job --profile nrp --job-config configs/job.yaml --image-profile cpu --output-manifest ./rendered-job.yaml

# Check status
spikelab-batch-jobs job-status --profile nrp analysis-job-abc123

# Stream logs
spikelab-batch-jobs job-logs --profile nrp analysis-job-abc123 --follow

# Delete completed job
spikelab-batch-jobs job-delete --profile nrp analysis-job-abc123
```
