---
name: spikelab-batch-jobs
description: Deploys and monitors SpikeLab analysis jobs on Kubernetes clusters using the spikelab-batch-jobs CLI workflow. Use when the user wants to submit batch jobs, stream logs, enforce cluster policy-safe defaults, upload bundles to S3, or clean up completed jobs.
---

# SpikeLab Batch Jobs

Use this skill for repeatable batch job execution with explicit validation and safety checks.

## Required Inputs

Ask the user for:
- Job config path (`--job-config`) with image, command/args, resources, and optional volumes.
- Target profile (`defaults` by default, or `nrp` for Nautilus) or `--profile-file`.
- Image strategy (`--image-profile cpu|gpu` or explicit `--image`).
- Namespace/context confirmation (`kubectl config current-context` + namespace).
- Whether they want to wait for completion and stream logs.

Never ask users to paste secrets in chat.

## Credentials and Secrets

- Credentials must come from user environment or files they already manage (`KUBECONFIG`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional session token).
- Never print secret values.
- Never commit credentials into files.
- Reference Kubernetes secrets by name only.
- For private registries, reference image pull secret names in Kubernetes, never raw credentials.

## Container Prep (for compute-intensive workflows)

1. Choose base image path:
   - CPU: `docker/analysis-base/Dockerfile.cpu`
   - GPU: `docker/analysis-base/Dockerfile.gpu`
2. Build a temporary run image using:
   - `scripts/build_temp_image.sh <cpu|gpu> <image-tag>`
3. Push the image:
   - `scripts/push_temp_image.sh <image-tag>`
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
     - `spikelab-batch-jobs render-job --job-config <path> --image-profile <cpu|gpu> --output-manifest /tmp/job.yaml`
3. **Submit**
   - Run `spikelab-batch-jobs deploy-job --job-config <path> --image-profile <cpu|gpu>`.
   - If user requested explicit image, pass `--image <image-tag>`.
   - Capture the machine-parseable line: `JOB_NAME=<value>`.
4. **Observe**
   - If user wants status: `spikelab-batch-jobs job-status <job_name>`.
   - If user wants logs: `spikelab-batch-jobs job-logs <job_name> --follow`.
5. **Failure triage**
   - Show `spikelab-batch-jobs job-status <job_name>`.
   - Suggest `kubectl describe job <job_name> -n <namespace>` and pod logs.
6. **Teardown guidance**
   - Suggest deleting completed/failed jobs:
     - `spikelab-batch-jobs job-delete <job_name>`
   - Remind user to clean up temporary tags no longer needed.

## Policy Safety Rails

- Default behavior is policy-safe.
- Do not use risk override flags unless user explicitly asks.
- If a policy warning/block appears, explain it and request confirmation before continuing.
- Reject patterns that resemble batch `sleep infinity` placeholders.

## Command Examples

```bash
spikelab-batch-jobs deploy-job --job-config configs/job.yaml --image-profile gpu --wait --max-wait-seconds 3600
```

```bash
spikelab-batch-jobs render-job --job-config configs/job.yaml --image-profile cpu --output-manifest ./rendered-job.yaml
```

```bash
spikelab-batch-jobs job-logs analysis-job-abc123 --follow
```
