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
- Namespace/context confirmation (`kubectl config current-context` + namespace).
- Whether they want to wait for completion and stream logs.

Never ask users to paste secrets in chat.

## Credentials and Secrets

- Credentials must come from user environment or files they already manage (`KUBECONFIG`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional session token).
- Never print secret values.
- Never commit credentials into files.
- Reference Kubernetes secrets by name only.

## Fixed Workflow

1. **Preflight checks**
   - Run `kubectl version --client`.
   - Run `kubectl config current-context`.
   - Optionally verify S3 access if asked by the user.
2. **Validate inputs**
   - Ensure `--job-config` is present.
   - Run a dry render first:
     - `spikelab-nrp-jobs render-job --job-config <path> --output-manifest /tmp/job.yaml`
3. **Submit**
   - Run `spikelab-nrp-jobs deploy-job --job-config <path>`.
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

## Policy Safety Rails

- Default behavior is policy-safe for NRP.
- Do not use risk override flags unless user explicitly asks.
- If a policy warning/block appears, explain it and request confirmation before continuing.
- Reject patterns that resemble batch `sleep infinity` placeholders.

## Command Examples

```bash
spikelab-nrp-jobs deploy-job --profile nrp --job-config configs/job.yaml --wait --max-wait-seconds 3600
```

```bash
spikelab-nrp-jobs render-job --profile nrp --job-config configs/job.yaml --output-manifest ./rendered-job.yaml
```

```bash
spikelab-nrp-jobs job-logs analysis-job-abc123 --follow
```
