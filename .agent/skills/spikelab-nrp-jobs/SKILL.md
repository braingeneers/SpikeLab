---
name: spikelab-nrp-jobs
description: Deploys and monitors SpikeLab analysis jobs on NRP using the spikelab-nrp-jobs CLI workflow. Use when the user wants to submit Kubernetes batch jobs, stream logs, enforce NRP policy-safe defaults, upload bundles to S3, or clean up completed jobs.
---

# SpikeLab NRP Jobs

Use this skill for repeatable NRP job execution with explicit validation and safety checks.

## Required Inputs

Ask the user for:
- Job config path (`--job-config`) with image, command/args, resources, and optional volumes.
- Target profile (`nrp` by default) or `--profile-file`.
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
   - `scripts/nrp_build_temp_image.sh <cpu|gpu> <image-tag>`
3. Push the image:
   - `scripts/nrp_push_temp_image.sh <image-tag>`
4. Generate a job config:
   - `python scripts/nrp_generate_job_config.py --image <image-tag> --profile <cpu|gpu> --output configs/nrp-temp-job.yaml`
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
     - `spikelab-nrp-jobs render-job --job-config <path> --image-profile <cpu|gpu> --output-manifest /tmp/job.yaml`
3. **Submit**
   - Run `spikelab-nrp-jobs deploy-job --job-config <path> --image-profile <cpu|gpu>`.
   - If user requested explicit image, pass `--image <image-tag>`.
   - Capture the machine-parseable line: `JOB_NAME=<value>`.
4. **Observe**
   - If user wants status: `spikelab-nrp-jobs job-status <job_name>`.
   - If user wants logs: `spikelab-nrp-jobs job-logs <job_name> --follow`.
5. **Failure triage**
   - Show `spikelab-nrp-jobs job-status <job_name>`.
   - Suggest `kubectl describe job <job_name> -n <namespace>` and pod logs.
6. **Teardown guidance**
   - Suggest deleting completed/failed jobs:
     - `spikelab-nrp-jobs job-delete <job_name>`
   - Remind user to clean up temporary tags no longer needed.

## Policy Safety Rails

- Default behavior is policy-safe for NRP.
- Do not use risk override flags unless user explicitly asks.
- If a policy warning/block appears, explain it and request confirmation before continuing.
- Reject patterns that resemble batch `sleep infinity` placeholders.

## Command Examples

```bash
spikelab-nrp-jobs deploy-job --profile nrp --job-config configs/job.yaml --image-profile gpu --wait --max-wait-seconds 3600
```

```bash
spikelab-nrp-jobs render-job --profile nrp --job-config configs/job.yaml --image-profile cpu --output-manifest ./rendered-job.yaml
```

```bash
spikelab-nrp-jobs job-logs analysis-job-abc123 --follow
```
